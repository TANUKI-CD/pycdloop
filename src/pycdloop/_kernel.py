"""Private loop-algebra arithmetic on raw coefficient arrays.

An element of the loop algebra R[L] has two coordinate systems where 2 is
invertible in R, and only the first of them in characteristic 2.

Loop basis
    One array of shape ``batch_shape + (2, D) + field.shape``: row 0 holds
    the coefficients ``a_t`` of the basis elements ``[e_t]`` and row 1 the
    coefficients ``b_t`` of ``[-e_t]``, where ``D`` is the quotient order of
    the loop.

Split form
    A pair ``(spectral, signed)`` of arrays of shape
    ``batch_shape + (D,) + field.shape``. With ``c = a + b`` and
    ``d = a - b``, ``spectral`` is the Walsh--Hadamard transform ``H c`` and
    ``signed`` is ``d``.

The map from the loop basis to the split form is an algebra isomorphism onto
``R^D (+) S``, where ``R^D`` has coordinatewise operations and ``S`` is the
Cayley--Dickson algebra of the loop, or the tensor product of the
Cayley--Dickson algebras of the factors of a central product. A product in
the split form is therefore ``D`` coordinatewise products and one product in
``S``; the transforms are needed only to enter and leave it.

The arithmetic functions take the loop algebra, a mode (``"loop"`` or
``"split"``) and operands in that mode, and return a result in the same
mode. Sums and scalings are evaluated in either mode. Products, and the
operations built on them, are evaluated in split form; in mode ``"loop"``
they are the routines of characteristic 2. The conversions, the
augmentation, and the table and workspace products take no mode.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from . import _range
from ._validation import BINARY_MESSAGE, CD_ONLY_MESSAGE

# Coefficients of intermediate arrays that the table product may hold at once
_TABLE_BUDGET = 1 << 22

# Largest temporary, in bytes, that a thread keeps between products
_WORKSPACE_LIMIT = 1 << 23


class _Workspace(threading.local):
    """Temporaries of the loop-basis product, kept by each thread between calls.

    A product of arrays in the loop basis needs six temporaries of the size
    of a batch of split forms. Allocating them afresh for every product
    costs more than the arithmetic once they outgrow the allocator's
    recycled blocks, because the operating system hands out new pages each
    time. Each thread therefore keeps one array for each role and reuses it
    while the requested shape and type stay the same. The arrays never
    leave :func:`loop_product`, and arrays above ``_WORKSPACE_LIMIT`` bytes
    are not kept.
    """

    def __init__(self) -> None:
        """Start with no arrays."""

        self.arrays: dict[int, np.ndarray] = {}

    def get(self, slot: int, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        """Return an uninitialized array for one role of the product."""

        kept = self.arrays.get(slot)
        if kept is not None and kept.shape == shape and kept.dtype == dtype:
            return kept
        array = np.empty(shape, dtype=dtype)
        if array.nbytes <= _WORKSPACE_LIMIT:
            self.arrays[slot] = array
        else:
            self.arrays.pop(slot, None)
        return array


_WORKSPACE = _Workspace()

Split = tuple[np.ndarray, np.ndarray]


def _rows(algebra: Any, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows of ``[e_t]`` and ``[-e_t]`` coefficients."""

    selector = (slice(None),) * (1 + algebra._field.tail)
    return coefficients[(Ellipsis, 0) + selector], coefficients[
        (Ellipsis, 1) + selector
    ]


def _join(algebra: Any, plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    """Stack the two rows of loop-basis coefficients."""

    return np.stack([plus, minus], axis=plus.ndim - algebra._field.tail - 1)


def to_split(algebra: Any, coefficients: np.ndarray) -> Split:
    """Return the split form of loop-basis coefficients."""

    field = algebra._field
    if field.binary:
        raise TypeError(BINARY_MESSAGE.format(name="the split form"))
    if field.native and coefficients.dtype == field.dtype:
        split: Split = field.split(coefficients, algebra._size)
        return split
    # Other element types take the same steps as whole-array operations
    plus, minus = _rows(algebra, coefficients)
    return field.wht(field.add(plus, minus)), field.subtract(plus, minus)


def from_split(algebra: Any, split: Split) -> np.ndarray:
    """Return the loop-basis coefficients of a split form."""

    field = algebra._field
    if field.binary:
        raise TypeError(BINARY_MESSAGE.format(name="the split form"))
    spectral, signed = split
    if field.native and spectral.dtype == signed.dtype == field.dtype:
        coefficients: np.ndarray = field.unsplit(spectral, signed, algebra._size)
        return coefficients
    group = field.divide_int(field.wht(spectral), algebra._size)
    plus = field.halve(field.add(group, signed))
    minus = field.halve(field.subtract(group, signed))
    return _join(algebra, plus, minus)


def _zero_spectral(algebra: Any, signed: np.ndarray) -> np.ndarray:
    """Return a zero group component matching a signed component."""

    return np.zeros(signed.shape, dtype=signed.dtype)


def _scalar_axis(algebra: Any, scalar: np.ndarray, count: int) -> np.ndarray:
    """Insert index axes so a batch of field scalars broadcasts over coefficients."""

    result: np.ndarray = algebra._field.expand(scalar, count)
    return result


# Linear operations, available in both modes


def add(algebra: Any, mode: str, a: Any, b: Any) -> Any:
    """Return ``a + b``."""

    field = algebra._field
    if mode == "loop":
        return field.add(a, b)
    return field.add(a[0], b[0]), field.add(a[1], b[1])


def subtract(algebra: Any, mode: str, a: Any, b: Any) -> Any:
    """Return ``a - b``."""

    field = algebra._field
    if mode == "loop":
        return field.subtract(a, b)
    return field.subtract(a[0], b[0]), field.subtract(a[1], b[1])


def negate(algebra: Any, mode: str, a: Any) -> Any:
    """Return ``-a``."""

    field = algebra._field
    if mode == "loop":
        return field.negate(a)
    return field.negate(a[0]), field.negate(a[1])


def scale(algebra: Any, mode: str, a: Any, scalar: np.ndarray) -> Any:
    """Return ``a`` times a field scalar, or a batch of field scalars."""

    field = algebra._field
    if mode == "loop":
        return field.scale(a, _scalar_axis(algebra, scalar, 2))
    factor = _scalar_axis(algebra, scalar, 1)
    return field.scale(a[0], factor), field.scale(a[1], factor)


def add_scalar(algebra: Any, mode: str, a: Any, scalar: np.ndarray) -> Any:
    """Return ``a`` plus a field scalar times the identity."""

    field = algebra._field
    first = (Ellipsis, 0) + (slice(None),) * field.tail
    if mode == "loop":
        # The identity is the basis element [e_0]
        shape = np.broadcast_shapes(a.shape, _scalar_axis(algebra, scalar, 2).shape)
        result = np.array(np.broadcast_to(a, shape))
        corner = (Ellipsis, 0, 0) + (slice(None),) * field.tail
        result[corner] = field.add(result[corner], scalar)
        return result
    # The identity has every spectral coordinate 1 and signed component e_0
    spectral, signed = a
    shape = np.broadcast_shapes(signed.shape, _scalar_axis(algebra, scalar, 1).shape)
    signed = np.array(np.broadcast_to(signed, shape))
    signed[first] = field.add(signed[first], scalar)
    return field.add(spectral, _scalar_axis(algebra, scalar, 1)), signed


def conjugate(algebra: Any, mode: str, a: Any) -> Any:
    """Return the image under the involution ``[x] -> [x^-1]``."""

    if mode == "split":
        # Every element of the quotient group is its own inverse, so the
        # group component is fixed
        return a[0].copy(), algebra._signed.conjugate(a[1])
    # The inverse of s e_t is s sigma(t, t) e_t, so the involution swaps the
    # two coefficients of every index whose basis element squares to -1
    plus, minus = _rows(algebra, a)
    swap = algebra._swap[..., None] if algebra._field.tail else algebra._swap
    return _join(algebra, np.where(swap, minus, plus), np.where(swap, plus, minus))


# Multiplicative operations, in split form


def multiply(
    algebra: Any, mode: str, a: Any, b: Any, signed_algorithm: str = "auto"
) -> Any:
    """Return the ordered product ``a b``."""

    if mode == "loop":
        return algebra._loop_multiply(a, b, signed_algorithm)
    spectral = algebra._field.multiply(a[0], b[0])
    return spectral, algebra._signed.multiply(a[1], b[1], signed_algorithm)


# In the loop basis the operations below are written with products alone.
# That is how they are evaluated in characteristic 2, where the algebra has
# no split form.


def square(algebra: Any, mode: str, a: Any) -> Any:
    """Return ``a a``."""

    if mode == "loop":
        return multiply(algebra, mode, a, a)
    return algebra._field.multiply(a[0], a[0]), algebra._signed.square(a[1])


def commutator(algebra: Any, mode: str, a: Any, b: Any) -> Any:
    """Return ``a b - b a``; its group component vanishes."""

    if mode == "loop":
        products = multiply(algebra, mode, a, b), multiply(algebra, mode, b, a)
        return algebra._field.subtract(*products)
    signed = algebra._signed.commutator(a[1], b[1])
    return _zero_spectral(algebra, signed), signed


def anticommutator(algebra: Any, mode: str, a: Any, b: Any) -> Any:
    """Return ``a b + b a``."""

    field = algebra._field
    if mode == "loop":
        return field.add(multiply(algebra, mode, a, b), multiply(algebra, mode, b, a))
    spectral = field.multiply_int(field.multiply(a[0], b[0]), 2)
    return spectral, algebra._signed.anticommutator(a[1], b[1])


def jordan_product(algebra: Any, mode: str, a: Any, b: Any) -> Any:
    """Return ``(a b + b a) / 2``."""

    field = algebra._field
    if mode == "loop":
        return field.halve(anticommutator(algebra, mode, a, b))
    signed = field.halve(algebra._signed.anticommutator(a[1], b[1]))
    return field.multiply(a[0], b[0]), signed


def associator(algebra: Any, mode: str, a: Any, b: Any, c: Any) -> Any:
    """Return ``(a b) c - a (b c)``; its group component vanishes."""

    if mode == "loop":
        left = multiply(algebra, mode, multiply(algebra, mode, a, b), c)
        right = multiply(algebra, mode, a, multiply(algebra, mode, b, c))
        return algebra._field.subtract(left, right)
    signed = algebra._signed.associator(a[1], b[1], c[1])
    return _zero_spectral(algebra, signed), signed


def left_operator_commutator(algebra: Any, mode: str, a: Any, b: Any, x: Any) -> Any:
    """Return ``a (b x) - b (a x)``; its group component vanishes."""

    if mode == "loop":
        first = multiply(algebra, mode, a, multiply(algebra, mode, b, x))
        second = multiply(algebra, mode, b, multiply(algebra, mode, a, x))
        return algebra._field.subtract(first, second)
    signed = algebra._signed.left_operator_commutator(a[1], b[1], x[1])
    return _zero_spectral(algebra, signed), signed


def right_operator_commutator(algebra: Any, mode: str, a: Any, b: Any, x: Any) -> Any:
    """Return ``(x b) a - (x a) b``; its group component vanishes."""

    if mode == "loop":
        first = multiply(algebra, mode, multiply(algebra, mode, x, b), a)
        second = multiply(algebra, mode, multiply(algebra, mode, x, a), b)
        return algebra._field.subtract(first, second)
    signed = algebra._signed.right_operator_commutator(a[1], b[1], x[1])
    return _zero_spectral(algebra, signed), signed


def _require_powers(algebra: Any, name: str) -> Any:
    """Return the signed component algebra where powers are well defined.

    Raise for a central product outside characteristic 2. In characteristic
    2 there is no signed component and the result is ``None``.
    """

    if not algebra._has_powers:
        raise TypeError(CD_ONLY_MESSAGE.format(name=name))
    return algebra._signed


def augmentation(algebra: Any, a: np.ndarray) -> np.ndarray:
    """Return the sum of the loop-basis coefficients of every record."""

    field = algebra._field
    result: np.ndarray = field.sum(field.sum(a, -1), -1)
    return result


def _binary_inverse(algebra: Any, a: np.ndarray) -> np.ndarray:
    """Return the inverse in characteristic 2."""

    # Every element satisfies x^4 = alpha^4 with alpha its augmentation, so
    # the inverse of an element with alpha != 0 is x^3 / alpha^4
    field = algebra._field
    alpha = augmentation(algebra, a)
    if not bool(np.all(field.nonzero(alpha))):
        raise ZeroDivisionError("the element has augmentation zero")
    cube = multiply(algebra, "loop", multiply(algebra, "loop", a, a), a)
    return scale(algebra, "loop", cube, field.reciprocal(field.power(alpha, 4)))


def _binary_power(algebra: Any, a: np.ndarray, exponent: int) -> np.ndarray:
    """Return an integer power in characteristic 2."""

    field = algebra._field
    alpha = augmentation(algebra, a)
    if exponent < 0 and not bool(np.all(field.nonzero(alpha))):
        raise ZeroDivisionError("the element has augmentation zero")
    # x^4 = alpha^4, so x^(4q + r) = alpha^(4q) x^r with 0 <= r < 4, also
    # for negative q when alpha is invertible: at most two products are
    # needed
    quotient, remainder = divmod(exponent, 4)
    if remainder == 0:
        result = np.zeros(a.shape, dtype=a.dtype)
        result[(Ellipsis, 0, 0) + (slice(None),) * field.tail] = field.scalar(1)
    elif remainder == 1:
        result = a.copy()
    else:
        result = multiply(algebra, "loop", a, a)
        if remainder == 3:
            result = multiply(algebra, "loop", result, a)
    if quotient:
        factor = field.power(alpha, 4 * abs(quotient))
        if quotient < 0:
            factor = field.reciprocal(factor)
        result = scale(algebra, "loop", result, factor)
    return result


def inverse(algebra: Any, mode: str, a: Any) -> Any:
    """Return the inverse that is a polynomial in the element."""

    signed = _require_powers(algebra, "inverse()")
    field = algebra._field
    if mode == "loop":
        return _binary_inverse(algebra, a)
    # Check every denominator before producing any part of the result
    if not bool(np.all(field.nonzero(a[0]))):
        raise ZeroDivisionError("the element has a zero spectral coordinate")
    if not bool(np.all(signed.has_norm(a[1]))):
        raise ZeroDivisionError("the signed component has quadratic norm zero")
    return field.reciprocal(a[0]), signed.reciprocal(a[1])


def power(algebra: Any, mode: str, a: Any, exponent: int) -> Any:
    """Return an integer power, where powers are well defined."""

    signed = _require_powers(algebra, "the power operator")
    field = algebra._field
    if mode == "loop":
        return _binary_power(algebra, a, exponent)
    if exponent < 0:
        a = inverse(algebra, mode, a)
        exponent = -exponent
    spectral, d = a
    # Write d = d_0 + w, with w the part of d without the identity. Then
    # w^2 = -Q(w) is a scalar, so the powers of d lie in the commutative
    # ring F[w] = F[z] / (z^2 + Q(w)), where binary exponentiation of
    # d_0 + z gives d^k = a + b w with O(log k) field operations, and
    # d^k costs O(N) more. Over R this ring is a copy of C, in which
    # products do not cancel, and for a scalar d it is the power of d_0.
    first = (Ellipsis, 0) + (slice(None),) * field.tail
    scalar = d[first]
    rest = d.copy()
    rest[first] = 0
    if exponent > 1 and not field.exact and not _fits_power(algebra, d, exponent):
        out = _wide_power(signed, field, scalar, rest, exponent)
        return field.power(spectral, exponent), out
    # The first power needs no product, and so no norm, which for a large
    # operand is out of range
    norm = signed.quadratic_norm(rest) if exponent > 1 else None
    one = np.broadcast_to(field.scalar(1), scalar.shape).astype(field.dtype)

    def times(x: Split, y: Split) -> Split:
        """Multiply a + b z by a' + b' z modulo z^2 + Q(w)."""

        assert norm is not None
        tip = field.multiply(field.multiply(x[1], y[1]), norm)
        low = field.subtract(field.multiply(x[0], y[0]), tip)
        return low, field.add(field.multiply(x[0], y[1]), field.multiply(x[1], y[0]))

    if exponent == 0:
        result: Split = (one, np.zeros(scalar.shape, dtype=field.dtype))
    else:
        # From the leading bit down, so that the first power is the operand
        # itself and no product involves the norm before one is needed
        base: Split = (scalar, one)
        result = base
        for bit in bin(exponent)[3:]:
            result = times(result, result)
            if bit == "1":
                result = times(result, base)
    out = field.scale(rest, field.expand(result[1], 1))
    out[first] = result[0]
    return field.power(spectral, exponent), out


# The recurrence of a power is evaluated with plain floats while every
# value on the way stays within 2**-_ROOM .. 2**_ROOM, far enough inside the
# range that nothing is lost at its ends
_ROOM = _range.ROOM


def _fits_power(algebra: Any, d: np.ndarray, exponent: int) -> bool:
    """Return whether the power recurrence of *d* stays in the ordinary range."""

    low, high = _range.spread(d)
    # The coefficients of a power k are below (2 N)^(k/2) times the k-th
    # power of the largest coefficient, times k
    size = max(-low, high, 0) + algebra._size.bit_length() // 2 + 2
    return exponent * size + 2 * exponent.bit_length() <= _ROOM


def _wide_power(
    signed: Any, field: Any, scalar: np.ndarray, rest: np.ndarray, exponent: int
) -> np.ndarray:
    """Return the signed part of a power over R or C, at any size of the values.

    The recurrence is that of :func:`power`, on numbers whose exponents
    cannot leave the range, so a coefficient of the result is lost only
    where it is itself beyond the range.
    """

    wide = _range.Wide
    norm = signed.wide_norm(rest)
    flat = norm.is_zero()
    nothing = wide.of(np.zeros(scalar.shape, dtype=field.dtype))

    def times(
        x: tuple[_range.Wide, _range.Wide], y: tuple[_range.Wide, _range.Wide]
    ) -> tuple[_range.Wide, _range.Wide]:
        """Multiply a + b z by a' + b' z modulo z^2 + Q(w)."""

        # A norm of zero removes the term whatever the size of b b'
        tip = (x[1] * y[1] * norm).at(flat, nothing)
        return x[0] * y[0] - tip, x[0] * y[1] + x[1] * y[0]

    base = (wide.of(scalar), wide.of(np.ones(scalar.shape, dtype=field.dtype)))
    result = base
    for bit in bin(exponent)[3:]:
        result = times(result, result)
        if bit == "1":
            result = times(result, base)
    with np.errstate(over="ignore"):
        out = (wide.of(rest) * result[1].expanded()).to_array()
        # A coefficient of zero stays zero beside a factor beyond the range
        out[rest == 0] = 0
        out[..., 0] = result[0].to_array()
    return out


def loop_product(
    algebra: Any,
    left: tuple[str, Any],
    right: tuple[str, Any],
    signed_algorithm: str = "auto",
) -> np.ndarray:
    """Return a product in the loop basis, evaluated through the split form.

    Each operand is a pair of a mode and data in that mode: loop-basis
    coefficients, which are split into the thread's workspace, or a split
    form, which is read and never modified. The arithmetic is that of
    :func:`to_split`, :func:`multiply` and :func:`from_split`, so the result
    is identical; only the temporaries are reused.
    """

    field = algebra._field
    size = algebra._size
    record = (size,) + field.shape
    parts: list[Split] = []
    for slot, (mode, data) in enumerate((left, right)):
        if mode == "split":
            parts.append(data)
            continue
        lead = data.shape[: data.ndim - field.tail - 2]
        out = (
            _WORKSPACE.get(2 * slot, lead + record, field.dtype),
            _WORKSPACE.get(2 * slot + 1, lead + record, field.dtype),
        )
        parts.append(field.split(data, size, out))
    (first_spectral, first_signed), (second_spectral, second_signed) = parts
    shape = np.broadcast_shapes(first_spectral.shape, second_spectral.shape)
    if field.kind == "finite":
        spectral = field.multiply(first_spectral, second_spectral)
    else:
        spectral = _WORKSPACE.get(4, shape, field.dtype)
        np.multiply(first_spectral, second_spectral, out=spectral)
    signed = algebra._signed.multiply_into(
        first_signed,
        second_signed,
        signed_algorithm,
        _WORKSPACE.get(5, shape, field.dtype),
    )
    result: np.ndarray = field.unsplit(spectral, signed, size)
    return result


# The reference product in the loop basis


def table_multiply(algebra: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the product computed directly from the multiplication table.

    The product of the basis elements ``[s e_p]`` and ``[t e_q]`` is the
    basis element ``[s t sigma(p, q) e_(p ^ q)]``, so the coefficient of
    ``[e_k]`` in ``x y`` collects, over all ``p`` and with ``q = p ^ k``,
    the two products of equal signs where ``sigma(p, q) = 1`` and the two
    products of opposite signs where ``sigma(p, q) = -1``; the coefficient
    of ``[-e_k]`` collects the rest. This takes ``4 D^2`` multiplications.
    """

    field = algebra._field
    size = algebra._size
    tail = field.tail
    lead = np.broadcast_shapes(
        x.shape[: x.ndim - tail - 2], y.shape[: y.ndim - tail - 2]
    )
    record = (2, size) + field.shape
    xs = np.ascontiguousarray(np.broadcast_to(x, lead + record)).reshape((-1,) + record)
    ys = np.ascontiguousarray(np.broadcast_to(y, lead + record)).reshape((-1,) + record)
    out = np.empty(xs.shape, dtype=field.dtype)
    partner, positive = algebra._table()
    mask = positive[..., None] if tail else positive
    step = max(1, _TABLE_BUDGET // (size * size * max(1, field.deg)))
    for start in range(0, xs.shape[0], step):
        block = slice(start, start + step)
        x_plus, x_minus = _rows(algebra, xs[block])
        y_plus, y_minus = _rows(algebra, ys[block])
        # Axis 1 runs over p and axis 2 over the output index k
        left_plus = field.expand(x_plus, 1)
        left_minus = field.expand(x_minus, 1)
        right_plus = y_plus[:, partner]
        right_minus = y_minus[:, partner]
        same = field.add(
            field.multiply(left_plus, right_plus),
            field.multiply(left_minus, right_minus),
        )
        opposite = field.add(
            field.multiply(left_plus, right_minus),
            field.multiply(left_minus, right_plus),
        )
        plus = field.sum(np.where(mask, same, opposite), 1)
        minus = field.sum(np.where(mask, opposite, same), 1)
        out[block] = _join(algebra, plus, minus)
    return out.reshape(lead + record)


def divide(algebra: Any, mode: str, a: Any, scalar: np.ndarray) -> Any:
    """Return ``a`` divided by a nonzero field scalar."""

    field = algebra._field
    if not bool(np.all(field.nonzero(scalar))):
        raise ZeroDivisionError("division by zero in the coefficient field")
    if field.exact:
        return scale(algebra, mode, a, field.reciprocal(scalar))
    if mode == "loop":
        return _range.quotient(a, _scalar_axis(algebra, scalar, 2))
    divisor = _scalar_axis(algebra, scalar, 1)
    return _range.quotient(a[0], divisor), _range.quotient(a[1], divisor)


def subtract_from_scalar(algebra: Any, mode: str, a: Any, scalar: np.ndarray) -> Any:
    """Return a field scalar times the identity, minus ``a``."""

    return add_scalar(algebra, mode, negate(algebra, mode, a), scalar)


def _duplicate(mode: str, a: Any) -> Any:
    """Return a copy of an operand, so that no result shares its storage."""

    if mode == "loop":
        return a.copy()
    return a[0].copy(), a[1].copy()


def left_power(algebra: Any, mode: str, a: Split, exponent: int) -> Split:
    """Return ``a (a (... (a a)))`` with *exponent* factors, at least one."""

    result = _duplicate(mode, a)
    for _ in range(exponent - 1):
        result = multiply(algebra, mode, a, result)
    return result


def right_power(algebra: Any, mode: str, a: Split, exponent: int) -> Split:
    """Return ``(((a a) a) ...) a`` with *exponent* factors, at least one."""

    result = _duplicate(mode, a)
    for _ in range(exponent - 1):
        result = multiply(algebra, mode, result, a)
    return result
