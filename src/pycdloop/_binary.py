"""Finite fields of characteristic 2 and their scalars."""

from __future__ import annotations

import operator
from typing import Any, final

import numpy as np

from ._validation import CACHE_LOCK, _index

MAX_DEGREE = 12

# The Conway polynomials of F_(2^d) for d = 2..12, constant coefficient first.
# They are the default presentations of GAP and SageMath, so residues written
# by this package mean the same field elements there.
_CONWAY: dict[int, tuple[int, ...]] = {
    2: (1, 1, 1),
    3: (1, 1, 0, 1),
    4: (1, 1, 0, 0, 1),
    5: (1, 0, 1, 0, 0, 1),
    6: (1, 1, 0, 1, 1, 0, 1),
    7: (1, 1, 0, 0, 0, 0, 0, 1),
    8: (1, 0, 1, 1, 1, 0, 0, 0, 1),
    9: (1, 0, 0, 0, 1, 0, 0, 0, 0, 1),
    10: (1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1),
    11: (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    12: (1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1),
}

# At degree one the polynomial plays no part in arithmetic
_PRIME_MODULUS = (0, 1)


def _mask(modulus: tuple[int, ...]) -> int:
    """Return a polynomial over F_2 as an integer with one bit for each coefficient."""

    return sum(bit << i for i, bit in enumerate(modulus))


def _reduce(value: int, modulus: int, degree: int) -> int:
    """Reduce a polynomial over F_2 modulo a polynomial of the given degree."""

    for shift in range(value.bit_length() - 1 - degree, -1, -1):
        if (value >> (degree + shift)) & 1:
            value ^= modulus << shift
    return value


def _multiply(a: int, b: int, modulus: int, degree: int) -> int:
    """Multiply two residues, given as bit masks, modulo the field polynomial."""

    product = 0
    while b:
        if b & 1:
            product ^= a
        a <<= 1
        b >>= 1
    return _reduce(product, modulus, degree)


def _is_irreducible(modulus: int, degree: int) -> bool:
    """Return whether a polynomial over F_2 of the given degree is irreducible."""

    # A polynomial of degree d is irreducible exactly when t^(2^d) = t modulo
    # it and, for every prime q dividing d, t^(2^(d/q)) - t is coprime to it
    def frobenius(count: int) -> int:
        """Return t^(2^count) modulo the polynomial."""

        value = 2
        for _ in range(count):
            value = _multiply(value, value, modulus, degree)
        return value

    def gcd(a: int, b: int) -> int:
        """Return the greatest common divisor of two polynomials over F_2."""

        while b:
            while a and a.bit_length() >= b.bit_length():
                a ^= b << (a.bit_length() - b.bit_length())
            a, b = b, a
        return a

    if degree == 1:
        return True
    if frobenius(degree) != _reduce(2, modulus, degree):
        return False
    primes = [q for q in range(2, degree + 1) if degree % q == 0]
    primes = [q for q in primes if all(q % r for r in range(2, q))]
    return all(gcd(modulus, frobenius(degree // q) ^ 2) == 1 for q in primes)


@final
class BinaryField:
    """An immutable coefficient parent for the finite field F_(2**deg).

    pyfastcd provides the finite fields of odd characteristic; this class
    provides those of characteristic 2 for loop algebras. The attributes
    listed below have the names and meaning they have in
    ``pyfastcd.FiniteField``. ``LoopAlgebra(loop, char=2, deg=d)`` and
    ``LoopAlgebra(loop, field=BinaryField(deg=d))`` select the same algebra.

    Parameters
    ----------
    deg : int, optional
        Extension degree in ``1..12``, defaulting to the prime field F_2.

    Attributes
    ----------
    char : int
        The characteristic, 2.
    deg : int
        The extension degree.
    order : int
        ``2**deg``.
    modulus : tuple[int, ...]
        The monic irreducible polynomial that presents the field, constant
        coefficient first; ``(0, 1)`` at degree one.
    dtype : numpy.dtype
        ``uint32`` storage for the coordinates of a scalar, each 0 or 1.
    coefficient_shape : tuple[int, ...]
        ``()`` at degree one, otherwise ``(deg,)`` for one field scalar.

    Raises
    ------
    TypeError
        If the degree is not integer-like or is a boolean.
    ValueError
        If the degree is outside the supported range.

    Notes
    -----
    An extension field is represented by polynomials over F_2 modulo
    ``modulus``. The constructor uses the Conway polynomial of the degree,
    which is also the default of GAP and SageMath. Use :meth:`from_modulus`
    for another presentation. Fields are cached by their polynomial, and two
    presentations of the same field are different parents.

    Calling a field creates a scalar: ``F(1)`` embeds an integer in the
    prime subfield, and ``F([c0, c1, ...])`` reads all ``deg`` polynomial
    coordinates. Scalars support ``+``, ``-``, ``*``, ``/`` and ``**``.

    Examples
    --------
    >>> import pycdloop as cdl
    >>> F = cdl.BinaryField(deg=2)
    >>> F.modulus
    (1, 1, 1)
    >>> t = F([0, 1])
    >>> t * t == t + 1
    True
    >>> t ** 3 == F.one()
    True
    """

    __slots__ = ("deg", "modulus", "_modulus_mask", "_exp", "_log", "__weakref__")

    deg: int
    modulus: tuple[int, ...]
    _modulus_mask: int
    _exp: np.ndarray
    _log: np.ndarray
    _cache: dict[tuple[int, ...], BinaryField] = {}

    def __new__(cls, *, deg: Any = 1) -> BinaryField:
        """Validate the degree and retrieve the cached default presentation."""

        degree = _index(deg, "deg")
        if not 1 <= degree <= MAX_DEGREE:
            raise ValueError(f"deg must be in 1..{MAX_DEGREE}")
        modulus = _PRIME_MODULUS if degree == 1 else _CONWAY[degree]
        return cls._create(modulus)

    def __init__(self, *, deg: Any = 1) -> None:
        """Initialization is completed once by the cached constructor."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject subclasses so cached field identity remains unambiguous."""

        raise TypeError("BinaryField cannot be subclassed")

    @classmethod
    def _create(cls, modulus: tuple[int, ...]) -> BinaryField:
        """Return the cached field with the given polynomial, building it once."""

        if modulus not in cls._cache:
            with CACHE_LOCK:
                if modulus not in cls._cache:
                    self = object.__new__(cls)
                    degree = len(modulus) - 1
                    mask = _mask(modulus)
                    store = object.__setattr__
                    store(self, "deg", degree)
                    store(self, "modulus", modulus)
                    store(self, "_modulus_mask", mask)
                    exp, log = cls._tables(mask, degree)
                    store(self, "_exp", exp)
                    store(self, "_log", log)
                    cls._cache[modulus] = self
        return cls._cache[modulus]

    @staticmethod
    def _tables(mask: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the tables of powers and of logarithms of a generator."""

        # The nonzero residues form a cyclic group of order 2^d - 1. With g a
        # generator, exp[i] = g^i and log[g^i] = i turn a product into an
        # addition of exponents; exp has two periods so that a sum of two
        # logarithms indexes it without a reduction.
        size = (1 << degree) - 1
        powers = [1]
        for generator in range(2, 1 << degree):
            powers = [1]
            value = generator
            while value != 1 and len(powers) < size:
                powers.append(value)
                value = _multiply(value, generator, mask, degree)
            if len(powers) == size:
                break
        exp = np.array(powers + powers, dtype=np.int64)
        log = np.zeros(1 << degree, dtype=np.int64)
        log[exp[:size]] = np.arange(size)
        exp.setflags(write=False)
        log.setflags(write=False)
        return exp, log

    @classmethod
    def from_modulus(cls, *, modulus: Any) -> BinaryField:
        """Return the field presented by an explicit polynomial.

        ``modulus`` is an ordered integer sequence of length ``deg + 1``,
        constant coefficient first; entries are reduced modulo 2 and the
        leading coefficient must be one. A reducible polynomial, or a degree
        outside ``1..12``, raises ``ValueError``. Every monic linear
        polynomial gives the prime field, whose modulus is ``(0, 1)``.
        """

        if isinstance(modulus, (str, bytes, set, frozenset, dict)):
            raise TypeError("modulus must be an ordered sequence of integers")
        try:
            entries = tuple(
                _index(entry, "a modulus coefficient") % 2 for entry in modulus
            )
        except TypeError as error:
            raise TypeError(
                "modulus must be an ordered sequence of integers"
            ) from error
        degree = len(entries) - 1
        if not 1 <= degree <= MAX_DEGREE:
            raise ValueError(f"the degree of the modulus must be in 1..{MAX_DEGREE}")
        if entries[-1] != 1:
            raise ValueError("the leading coefficient of the modulus must be one")
        if degree == 1:
            return cls._create(_PRIME_MODULUS)
        if not _is_irreducible(_mask(entries), degree):
            raise ValueError("the modulus is not irreducible")
        return cls._create(entries)

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation of a field shared by scalars and loop algebras."""

        raise AttributeError("BinaryField objects are immutable")

    def __delattr__(self, name: str) -> None:
        """Reject deletion of immutable cached field state."""

        raise AttributeError("BinaryField objects are immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the field by its polynomial."""

        return (_rebuild_field, (self.modulus,))

    def __repr__(self) -> str:
        """Return the constructor expression of the field."""

        default = _PRIME_MODULUS if self.deg == 1 else _CONWAY[self.deg]
        if self.modulus == default:
            return f"BinaryField(deg={self.deg})"
        return f"BinaryField.from_modulus(modulus={self.modulus})"

    @property
    def char(self) -> int:
        """The characteristic, 2."""

        return 2

    @property
    def order(self) -> int:
        """The number of elements, ``2**deg``."""

        return 1 << self.deg

    @property
    def dtype(self) -> np.dtype[Any]:
        """The storage type of the coordinates of a scalar."""

        return np.dtype(np.uint32)

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        """The shape of the coordinates of one scalar."""

        return () if self.deg == 1 else (self.deg,)

    # Scalars

    def __call__(self, value: Any = 0) -> BinaryFieldElement:
        """Create a scalar from an integer, a scalar or polynomial coordinates."""

        if isinstance(value, BinaryFieldElement):
            if value.algebra is not self:
                raise ValueError("the scalar belongs to a different field")
            return value
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("a boolean is not a scalar of a finite field")
        if isinstance(value, (int, np.integer)):
            return BinaryFieldElement._make(self, int(value) % 2)
        if isinstance(value, (str, bytes, set, frozenset, dict)):
            raise TypeError("coordinates must be an ordered sequence of integers")
        try:
            entries = [_index(entry, "a coordinate") % 2 for entry in value]
        except TypeError as error:
            raise TypeError(
                "a scalar is an integer or a sequence of polynomial coordinates"
            ) from error
        if len(entries) != self.deg:
            raise ValueError(f"a scalar has {self.deg} polynomial coordinates")
        return BinaryFieldElement._make(self, _mask(tuple(entries)))

    def scalar(self, value: Any) -> BinaryFieldElement:
        """Return a scalar; the same as calling the field."""

        return self(value)

    def zero(self) -> BinaryFieldElement:
        """Return the additive identity."""

        return BinaryFieldElement._make(self, 0)

    def one(self) -> BinaryFieldElement:
        """Return the multiplicative identity."""

        return BinaryFieldElement._make(self, 1)

    # Array arithmetic on bit masks, used by the coefficient-field layer

    def _pack(self, bits: np.ndarray) -> np.ndarray:
        """Return coordinate arrays as arrays of bit masks."""

        if self.deg == 1:
            return bits.astype(np.int64)
        weights = np.int64(1) << np.arange(self.deg, dtype=np.int64)
        packed: np.ndarray = (bits.astype(np.int64) * weights).sum(axis=-1)
        return packed

    def _unpack(self, masks: np.ndarray) -> np.ndarray:
        """Return arrays of bit masks as coordinate arrays."""

        if self.deg == 1:
            return masks.astype(np.uint32)
        shifts = np.arange(self.deg, dtype=np.int64)
        return ((masks[..., None] >> shifts) & 1).astype(np.uint32)

    def _multiply_masks(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply arrays of bit masks in the field."""

        product = self._exp[self._log[a] + self._log[b]]
        result: np.ndarray = np.where((a == 0) | (b == 0), 0, product)
        return result

    def _reciprocal_masks(self, a: np.ndarray) -> np.ndarray:
        """Invert arrays of nonzero bit masks in the field."""

        size = self.order - 1
        result: np.ndarray = self._exp[(size - self._log[a]) % size]
        return result

    def _power_masks(self, a: np.ndarray, exponent: int) -> np.ndarray:
        """Raise arrays of bit masks to a nonnegative integer power."""

        if exponent == 0:
            return np.ones(a.shape, dtype=np.int64)
        size = self.order - 1
        powers = self._exp[(self._log[a] * (exponent % size)) % size]
        result: np.ndarray = np.where(a == 0, 0, powers)
        return result


@final
class BinaryFieldElement:
    """One immutable scalar of a :class:`BinaryField`.

    Create scalars by calling a field. Scalars are hashable and combine with
    scalars of the same field and with integers, which stand for elements
    of the prime subfield and are reduced modulo 2 in arithmetic.

    Attributes
    ----------
    algebra : BinaryField
        The field that contains the scalar, named as for pyfastcd scalars.
    coefficients : numpy.ndarray
        A read-only ``uint32`` array of the polynomial coordinates, of shape
        ``algebra.coefficient_shape``.

    Notes
    -----
    Comparison with an integer does not reduce the integer, so that equality
    agrees with hashing: ``F(1) == 1`` is true and ``F(1) == 3`` is false.

    Examples
    --------
    >>> import pycdloop as cdl
    >>> F = cdl.BinaryField(deg=3)
    >>> x = F([1, 1, 0])
    >>> x + x == 0
    True
    >>> x * x.reciprocal() == 1
    True
    """

    __slots__ = ("_field", "_value")

    _field: BinaryField
    _value: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject direct construction."""

        raise TypeError("create scalars by calling a BinaryField")

    @classmethod
    def _make(cls, field: BinaryField, value: int) -> BinaryFieldElement:
        """Create a scalar from a field and a bit mask."""

        self = object.__new__(cls)
        object.__setattr__(self, "_field", field)
        object.__setattr__(self, "_value", value)
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation."""

        raise AttributeError("BinaryFieldElement objects are immutable")

    def __delattr__(self, name: str) -> None:
        """Reject deletion of immutable state."""

        raise AttributeError("BinaryFieldElement objects are immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the field and the bit mask."""

        return (_rebuild_scalar, (self._field, self._value))

    @property
    def algebra(self) -> BinaryField:
        """The field that contains the scalar."""

        return self._field

    @property
    def coefficients(self) -> np.ndarray:
        """A read-only array of the polynomial coordinates."""

        field = self._field
        bits = [(self._value >> i) & 1 for i in range(field.deg)]
        array = np.array(bits if field.deg > 1 else bits[0], dtype=np.uint32)
        array.setflags(write=False)
        return array

    def _partner(self, other: Any) -> int | None:
        """Return the bit mask of an operand, or ``None`` if it is not one."""

        if isinstance(other, BinaryFieldElement):
            if other._field is not self._field:
                raise ValueError("operands belong to different fields")
            return other._value
        if isinstance(other, (bool, np.bool_)):
            return None
        if isinstance(other, (int, np.integer)):
            return int(other) % 2
        return None

    def _wrap(self, value: int) -> BinaryFieldElement:
        """Return a scalar of the same field."""

        return BinaryFieldElement._make(self._field, value)

    def __add__(self, other: Any) -> Any:
        """Add; addition and subtraction coincide in characteristic 2."""

        partner = self._partner(other)
        return NotImplemented if partner is None else self._wrap(self._value ^ partner)

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__

    def __neg__(self) -> BinaryFieldElement:
        """Return the scalar itself, which is its own negative."""

        return self

    def __pos__(self) -> BinaryFieldElement:
        """Return the scalar itself."""

        return self

    def __mul__(self, other: Any) -> Any:
        """Multiply in the field."""

        partner = self._partner(other)
        if partner is None:
            return NotImplemented
        field = self._field
        return self._wrap(
            _multiply(self._value, partner, field._modulus_mask, field.deg)
        )

    __rmul__ = __mul__

    def reciprocal(self) -> BinaryFieldElement:
        """Return the multiplicative inverse; zero raises ``ZeroDivisionError``."""

        if self._value == 0:
            raise ZeroDivisionError("division by zero in the coefficient field")
        mask = np.array(self._value, dtype=np.int64)
        return self._wrap(int(self._field._reciprocal_masks(mask)))

    def __truediv__(self, other: Any) -> Any:
        """Divide by a nonzero scalar."""

        partner = self._partner(other)
        if partner is None:
            return NotImplemented
        return self * self._wrap(partner).reciprocal()

    def __rtruediv__(self, other: Any) -> Any:
        """Divide an integer by the scalar."""

        partner = self._partner(other)
        if partner is None:
            return NotImplemented
        return self._wrap(partner) * self.reciprocal()

    def __pow__(self, exponent: Any) -> BinaryFieldElement:
        """Return an integer power; negative exponents use the reciprocal."""

        if isinstance(exponent, (bool, np.bool_)):
            raise TypeError("a power requires an integer exponent")
        count = operator.index(exponent)
        base = self.reciprocal() if count < 0 else self
        mask = np.array(base._value, dtype=np.int64)
        return self._wrap(int(self._field._power_masks(mask, abs(count))))

    def __eq__(self, other: object) -> Any:
        """Compare with a scalar of the same field or an unreduced integer."""

        if isinstance(other, BinaryFieldElement):
            return other._field is self._field and other._value == self._value
        if isinstance(other, (bool, np.bool_)):
            return NotImplemented
        if isinstance(other, (int, np.integer)):
            return int(other) in (0, 1) and self._value == int(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Hash an element of the prime subfield like the integer it equals."""

        if self._value < 2:
            return hash(self._value)
        return hash((self._field, self._value))

    def __bool__(self) -> bool:
        """Return whether the scalar is nonzero."""

        return self._value != 0

    def __repr__(self) -> str:
        """Return the coordinates and the field."""

        field = self._field
        if field.deg == 1:
            return f"BinaryFieldElement({self._value}, field={field!r})"
        bits = [(self._value >> i) & 1 for i in range(field.deg)]
        return f"BinaryFieldElement({bits}, field={field!r})"


def _rebuild_field(modulus: tuple[int, ...]) -> BinaryField:
    """Recreate a pickled field."""

    return BinaryField._create(tuple(modulus))


def _rebuild_scalar(field: BinaryField, value: int) -> BinaryFieldElement:
    """Recreate a pickled scalar."""

    return BinaryFieldElement._make(field, value)
