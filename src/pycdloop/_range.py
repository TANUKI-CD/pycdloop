"""Rescaling by powers of two for values near the ends of the floating-point range.

Squaring a coefficient above ``2**512`` overflows and squaring one below
``2**-512`` underflows, although the result of the operation that needs the
square may be representable. Multiplying by a power of two is exact, so an
operation of known degree can be evaluated on rescaled operands and its
result scaled back, without changing the value it gives in the ordinary
range.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np

# Records whose largest coefficient lies within 2**-WINDOW .. 2**WINDOW are
# left as they are, so that results in the ordinary range are not affected
WINDOW = 400
# A shift beyond this takes every finite number to zero or infinity
_LIMIT = 5000
# With keep, the smallest coefficient of a record stays above 2**-_KEPT
# where the largest can stay below 2**_KEPT
_KEPT = 1000


def exponents(x: np.ndarray, *, always: bool = False, keep: bool = False) -> np.ndarray:
    """Return the power of two by which to divide each record along the last axis.

    The power brings the largest coefficient of a record into ``[1, 2)``.
    It is 0 for a record whose largest coefficient is within the window,
    unless *always* is set, and for a record of zeros or non-finite values.
    With *keep* the power is lowered so that dividing by it does not take
    the smallest nonzero coefficient of the record below the normal range,
    as far as that leaves the largest one inside it.
    """

    sizes = np.maximum(np.abs(x.real), np.abs(x.imag))
    largest = sizes.max(axis=-1, keepdims=True)
    found: np.ndarray = np.frexp(largest)[1] - 1
    usable = np.isfinite(largest) & (largest > 0)
    if not always:
        usable &= np.abs(found) > WINDOW
    if keep:
        # The smallest nonzero part, real or imaginary, of the record
        parts = np.minimum(
            np.where(x.real != 0, np.abs(x.real), np.inf),
            np.where(x.imag != 0, np.abs(x.imag), np.inf),
        )
        smallest = np.frexp(parts.min(axis=-1, keepdims=True))[1] - 1
        lowered = np.maximum(np.where(usable, smallest, 0) + _KEPT, found - _KEPT)
        found = np.minimum(found, lowered)
    return np.where(usable, found, 0).astype(np.intc)


def shift(x: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Return ``x * 2**powers``, exactly where the result is representable."""

    if not np.any(powers):
        return x
    powers = np.clip(powers, -_LIMIT, _LIMIT).astype(np.intc)
    if not np.iscomplexobj(x):
        real: np.ndarray = np.ldexp(x, powers)
        return real
    result = np.empty(np.broadcast_shapes(x.shape, powers.shape), dtype=x.dtype)
    result.real = np.ldexp(x.real, powers)
    result.imag = np.ldexp(x.imag, powers)
    return result


def unit(
    x: np.ndarray, *, always: bool = False, keep: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return records rescaled to unit size, with the powers of two removed."""

    powers = exponents(x, always=always, keep=keep)
    return shift(x, -powers), powers


def scale_power(powers: np.ndarray, degree: int) -> np.ndarray:
    """Return ``degree * powers``, limited to the shifts that matter."""

    bounded = max(-_LIMIT, min(_LIMIT, degree))
    return np.clip(powers.astype(np.int64) * bounded, -_LIMIT, _LIMIT)


# Arithmetic with an exponent that cannot leave the range

# A mantissa and its exponent
_Part = tuple[np.ndarray, np.ndarray]


# The exponent that stands for zero, and the size beyond which an exponent
# is that of an infinity or of zero whatever is done with it afterwards
_NOTHING = -(1 << 44)
_FAR = 1 << 40


def _settle(mantissa: np.ndarray, exponent: np.ndarray) -> _Part:
    """Return a mantissa in ``[1/2, 1)`` and the exponent that goes with it."""

    fraction, moved = np.frexp(mantissa)
    exponent = exponent + moved
    far = np.isfinite(fraction) & (fraction != 0) & (exponent > _FAR)
    if far.any():
        fraction = np.where(far, np.copysign(np.inf, fraction), fraction)
    gone = (fraction == 0) | (exponent < -_FAR)
    fraction = np.where(gone & np.isfinite(fraction), 0.0, fraction)
    return fraction, np.where(gone, _NOTHING, np.minimum(exponent, _FAR + 1))


class Wide:
    """Real or complex numbers ``mantissa * 2**exponent`` with integer exponents.

    The four operations round as floating-point operations do, and their
    intermediate values cannot overflow or underflow, so a computation
    whose result is representable gives it whatever the sizes of the
    values on the way. A complex number is a pair of real parts, each with
    its own exponent. :meth:`to_array` rounds to ordinary floats, where a
    value beyond the range becomes an infinity or zero.
    """

    __slots__ = ("imaginary", "real")

    def __init__(
        self,
        real: _Part,
        imaginary: _Part | None = None,
    ) -> None:
        """Hold the mantissas and exponents of the real and imaginary parts."""

        self.real = real
        self.imaginary = imaginary

    @classmethod
    def of(cls, values: Any, exponents: Any = 0) -> Wide:
        """Return ``values * 2**exponents`` for an array of floats."""

        array = np.asarray(values)
        extra = np.asarray(exponents, dtype=np.int64)
        if not np.iscomplexobj(array):
            return cls(_settle(array.astype(np.float64), extra))
        return cls(_settle(array.real, extra), _settle(array.imag, extra))

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array of numbers."""

        return tuple(np.broadcast_shapes(self.real[0].shape, self.real[1].shape))

    def to_array(self) -> np.ndarray:
        """Return the numbers as floats, rounded where they are subnormal."""

        def part(value: _Part) -> np.ndarray:
            mantissa, exponent = value
            bounded = np.clip(exponent, -_LIMIT, _LIMIT).astype(np.intc)
            rounded: np.ndarray = np.ldexp(mantissa, bounded)
            return rounded

        if self.imaginary is None:
            return part(self.real)
        real, imaginary = part(self.real), part(self.imaginary)
        result = np.empty(np.broadcast_shapes(real.shape, imaginary.shape), complex)
        result.real = real
        result.imag = imaginary
        return result

    def reshaped(self, shape: tuple[int, ...]) -> Wide:
        """Return the numbers in another shape."""

        def part(value: _Part) -> _Part:
            mantissa, exponent = np.broadcast_arrays(*value)
            return mantissa.reshape(shape), exponent.reshape(shape)

        return Wide(
            part(self.real), None if self.imaginary is None else part(self.imaginary)
        )

    @classmethod
    def joined(cls, pieces: list[Wide]) -> Wide:
        """Return one-dimensional arrays of numbers joined end to end."""

        def part(values: list[_Part]) -> _Part:
            flat = [np.broadcast_arrays(*value) for value in values]
            return (
                np.concatenate([np.atleast_1d(pair[0]) for pair in flat]),
                np.concatenate([np.atleast_1d(pair[1]) for pair in flat]),
            )

        if all(piece.imaginary is None for piece in pieces):
            return cls(part([piece.real for piece in pieces]))
        pairs = [piece._complex() for piece in pieces]
        return cls(part([pair[0] for pair in pairs]), part([pair[1] for pair in pairs]))

    def padded(self, before: int, after: int) -> Wide:
        """Return a one-dimensional array with zeros before and after it."""

        return Wide.joined([Wide.of(np.zeros(before)), self, Wide.of(np.zeros(after))])

    def pick(self, index: Any) -> Wide:
        """Return the numbers at an index, as indexing an array does."""

        def part(value: _Part) -> _Part:
            mantissa, exponent = np.broadcast_arrays(*value)
            return mantissa[index], exponent[index]

        return Wide(
            part(self.real), None if self.imaginary is None else part(self.imaginary)
        )

    def modulus(self) -> _Part:
        """Return the absolute values, as mantissas in ``[1/2, 1)`` and exponents."""

        if self.imaginary is None:
            return np.abs(self.real[0]), self.real[1]
        # The parts are brought to the larger of their exponents, where
        # their squares neither overflow nor underflow together
        exponent = np.maximum(self.real[1], self.imaginary[1])
        down = np.clip(self.real[1] - exponent, -_LIMIT, 0).astype(np.intc)
        other = np.clip(self.imaginary[1] - exponent, -_LIMIT, 0).astype(np.intc)
        with np.errstate(invalid="ignore"):
            size = np.hypot(
                np.ldexp(self.real[0], down), np.ldexp(self.imaginary[0], other)
            )
        return _settle(size, exponent)

    def over(self, other: Wide) -> Wide:
        """Return the quotients by real or complex numbers.

        A complex quotient is Smith's division: it divides by the larger
        part of the divisor, and by a divisor with one part it is one
        division of each part.
        """

        if other.imaginary is None:
            return self / other
        (a, b), (c, d) = self._complex(), other._complex()
        # z / s = (-i z) / (-i s), which puts the larger part of s first
        turned = exceeds(d, c)
        c, d = choose(turned, d, c), choose(turned, _negative(c), d)
        a, b = choose(turned, b, a), choose(turned, _negative(a), b)
        ratio = _divide(d, c)
        below = _add(c, _multiply(d, ratio))
        return Wide(
            _divide(_add(a, _multiply(b, ratio)), below),
            _divide(_add(b, _negative(_multiply(a, ratio))), below),
        )

    def expanded(self) -> Wide:
        """Return the numbers with a trailing axis of length 1."""

        def part(value: _Part) -> _Part:
            return value[0][..., None], value[1][..., None]

        return Wide(
            part(self.real), None if self.imaginary is None else part(self.imaginary)
        )

    def at(self, index: Any, other: Wide) -> Wide:
        """Return the numbers with those at a Boolean index taken from *other*."""

        def part(mine: _Part, theirs: _Part) -> _Part:
            return (
                np.where(index, theirs[0], mine[0]),
                np.where(index, theirs[1], mine[1]),
            )

        if self.imaginary is None or other.imaginary is None:
            assert self.imaginary is None and other.imaginary is None
            return Wide(part(self.real, other.real))
        return Wide(part(self.real, other.real), part(self.imaginary, other.imaginary))

    def is_zero(self) -> np.ndarray:
        """Return where the numbers are zero."""

        zero: np.ndarray = self.real[0] == 0
        if self.imaginary is not None:
            zero = zero & (self.imaginary[0] == 0)
        return zero

    def __neg__(self) -> Wide:
        """Return the negatives."""

        real = (-self.real[0], self.real[1])
        if self.imaginary is None:
            return Wide(real)
        return Wide(real, (-self.imaginary[0], self.imaginary[1]))

    def __add__(self, other: Wide) -> Wide:
        """Return the sums."""

        if self.imaginary is None and other.imaginary is None:
            return Wide(_add(self.real, other.real))
        mine, theirs = self._complex(), other._complex()
        return Wide(_add(mine[0], theirs[0]), _add(mine[1], theirs[1]))

    def __sub__(self, other: Wide) -> Wide:
        """Return the differences."""

        return self + (-other)

    def __mul__(self, other: Wide) -> Wide:
        """Return the products."""

        if self.imaginary is None and other.imaginary is None:
            return Wide(_multiply(self.real, other.real))
        (a, b), (c, d) = self._complex(), other._complex()
        return Wide(
            _add(_multiply(a, c), _negative(_multiply(b, d))),
            _add(_multiply(a, d), _multiply(b, c)),
        )

    def __truediv__(self, other: Wide) -> Wide:
        """Return the quotients by real numbers."""

        assert other.imaginary is None
        real = _divide(self.real, other.real)
        if self.imaginary is None:
            return Wide(real)
        return Wide(real, _divide(self.imaginary, other.real))

    def _complex(self) -> tuple[_Part, _Part]:
        """Return the real and imaginary parts, the second zero for a real number."""

        if self.imaginary is not None:
            return self.real, self.imaginary
        zero = (np.zeros_like(self.real[0]), np.full_like(self.real[1], _NOTHING))
        return self.real, zero


def rational(value: Fraction) -> tuple[float, int]:
    """Return a rational number as the nearest mantissa of 53 bits and its exponent."""

    top, bottom = abs(value.numerator), value.denominator
    if not top:
        return 0.0, _NOTHING
    # A quotient of 64 or 65 bits, with a last bit that records whether
    # anything was cut off, rounds to 53 bits as the fraction itself does
    moved = bottom.bit_length() - top.bit_length() + 64
    if moved >= 0:
        whole, rest = divmod(top << moved, bottom)
    else:
        whole, rest = divmod(top, bottom << -moved)
    mantissa, exponent = math.frexp(float(whole | bool(rest)))
    return (-mantissa if value < 0 else mantissa), exponent - moved


def exceeds(x: _Part, y: _Part) -> np.ndarray:
    """Return where real numbers with wide exponents are larger in absolute value."""

    first, second = np.abs(x[0]), np.abs(y[0])
    larger: np.ndarray = (x[1] > y[1]) | ((x[1] == y[1]) & (first > second))
    return larger & (first > 0)


def choose(index: np.ndarray, x: _Part, y: _Part) -> _Part:
    """Return real numbers with wide exponents taken from *x* at an index, else *y*."""

    return np.where(index, x[0], y[0]), np.where(index, x[1], y[1])


def _negative(value: _Part) -> _Part:
    """Return the negative of real numbers with wide exponents."""

    return -value[0], value[1]


def _multiply(x: _Part, y: _Part) -> _Part:
    """Return the product of real numbers with wide exponents."""

    with np.errstate(invalid="ignore"):
        return _settle(x[0] * y[0], x[1] + y[1])


def _divide(x: _Part, y: _Part) -> _Part:
    """Return the quotient of real numbers with wide exponents."""

    with np.errstate(divide="ignore", invalid="ignore"):
        return _settle(x[0] / y[0], x[1] - np.where(y[0] == 0, 0, y[1]))


def _add(x: _Part, y: _Part) -> _Part:
    """Return the sum of real numbers with wide exponents."""

    # Both mantissas are brought to the larger exponent. One that is moved
    # down by more than the precision does not change the sum, as in a sum
    # of floats, so it does not matter that it is rounded on the way
    exponent = np.maximum(x[1], y[1])
    down = np.clip(x[1] - exponent, -_LIMIT, 0).astype(np.intc)
    other = np.clip(y[1] - exponent, -_LIMIT, 0).astype(np.intc)
    with np.errstate(invalid="ignore"):
        return _settle(np.ldexp(x[0], down) + np.ldexp(y[0], other), exponent)


def total(values: Wide) -> Wide:
    """Return the sums of wide numbers along the last axis."""

    def part(value: _Part) -> _Part:
        mantissa, exponent = np.broadcast_arrays(*value)
        top = exponent.max(axis=-1, keepdims=True)
        down = np.clip(exponent - top, -_LIMIT, 0).astype(np.intc)
        with np.errstate(invalid="ignore"):
            return _settle(np.ldexp(mantissa, down).sum(axis=-1), top[..., 0])

    if values.imaginary is None:
        return Wide(part(values.real))
    return Wide(part(values.real), part(values.imaginary))


def spread(x: np.ndarray) -> tuple[int, int]:
    """Return the least and greatest binary exponents of the nonzero parts of floats.

    Both are 0 for an array without nonzero finite parts. A part that is
    not finite gives the widest spread there is.
    """

    parts = np.abs(x.real), np.abs(x.imag) if np.iscomplexobj(x) else None
    low, high = 0, 0
    first = True
    for part in parts:
        if part is None or not part.size:
            continue
        if not np.isfinite(part).all():
            return -_LIMIT, _LIMIT
        top = float(part.max())
        if top == 0:
            continue
        bottom = float(part.min(where=part > 0, initial=np.inf))
        least, greatest = int(np.frexp(bottom)[1]) - 1, int(np.frexp(top)[1]) - 1
        low, high = (
            (least, greatest) if first else (min(low, least), max(high, greatest))
        )
        first = False
    return low, high


# Division of complex numbers

# NumPy divides complex numbers by Smith's method with a reciprocal, whose
# ratio and reciprocal overflow or underflow for operands near the ends of
# the range. It is used while the divisor lies within 2**-ROOM/2 ..
# 2**ROOM/2 and the numerator within 2**-ROOM .. 2**ROOM, where every
# value on the way is far inside the range
ROOM = 400


def quotient(z: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Return ``z / s`` for real or complex arrays, at any size of the values.

    A real quotient is one correctly rounded division. A complex quotient
    outside the ordinary range is Smith's division on numbers whose
    exponents cannot leave the range: it divides by the larger part of the
    divisor, and for a real or an imaginary divisor it is one correctly
    rounded division of each part.
    """

    if not (np.iscomplexobj(z) or np.iscomplexobj(s)):
        real: np.ndarray = z / s
        return real
    low, high = spread(s)
    if max(-low, high) <= ROOM // 2:
        low, high = spread(z)
        if max(-low, high) <= ROOM:
            plain: np.ndarray = z / s
            return plain
    shape = np.broadcast_shapes(np.shape(z), np.shape(s))
    z = np.broadcast_to(np.asarray(z, dtype=complex), shape)
    s = np.broadcast_to(np.asarray(s, dtype=complex), shape)
    with np.errstate(all="ignore"):
        result = Wide.of(z).over(Wide.of(s)).to_array()
        # A divisor with one part divides each part of the numerator once,
        # which a division of floats rounds correctly at every size, also
        # where the quotient is below the normal range
        for single, turned in ((s.imag == 0, False), (s.real == 0, True)):
            if not single.any():
                continue
            if turned:
                # z / (d i) = (b - a i) / d
                real, imaginary = z.imag / s.imag, -z.real / s.imag
            else:
                real, imaginary = z.real / s.real, z.imag / s.real
            result.real[single] = real[single]
            result.imag[single] = imaginary[single]
    return result


def power(values: np.ndarray, exponent: int) -> np.ndarray:
    """Return a positive integer power of each float, at any size on the way.

    A part of a complex product is a difference of two products, which
    can overflow although the part does not. Here a power is lost only
    where it is itself beyond the range.
    """

    base = Wide.of(values)
    result = base if exponent & 1 else None
    exponent >>= 1
    while exponent:
        base = base * base
        if exponent & 1:
            result = base if result is None else result * base
        exponent >>= 1
    assert result is not None
    with np.errstate(over="ignore"):
        return result.to_array()
