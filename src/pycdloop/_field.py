"""Private coefficient-field arithmetic on raw NumPy arrays."""

from __future__ import annotations

import numbers
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pyfastcd as fcd

from . import _core, _range
from ._binary import BinaryField, BinaryFieldElement
from ._transform import _hadamard, _hadamard_mod
from ._validation import BINARY_MESSAGE, CACHE_LOCK

_CACHE: dict[tuple[int, int, Any], _CoefficientField] = {}


def _is_two(char: Any) -> bool:
    """Return whether a characteristic selector is the integer 2."""

    if isinstance(char, (bool, np.bool_)):
        return False
    return isinstance(char, (int, np.integer)) and int(char) == 2


def _is_default(selector: Any, value: int) -> bool:
    """Return whether a selector was left at its default integer value."""

    if isinstance(selector, (bool, np.bool_)):
        return False
    return isinstance(selector, (int, np.integer)) and int(selector) == value


def _coefficient_field(
    *, char: Any = 0, deg: Any = 1, field: Any = None
) -> _CoefficientField:
    """Return the cached arithmetic of the field chosen by the selectors."""

    binary: BinaryField | None = None
    if isinstance(field, BinaryField):
        # As in pyfastcd, a field object is passed on its own
        if not (_is_default(char, 0) and _is_default(deg, 1)):
            raise TypeError("field cannot be combined with char or deg")
        binary = field
    elif field is None and _is_two(char):
        binary = BinaryField(deg=deg)
    elif field is not None and not isinstance(field, fcd.FiniteField):
        raise TypeError("field must be a pyfastcd.FiniteField or a BinaryField")
    if binary is not None:
        binary_key = (2, binary.deg, binary)
        if binary_key not in _CACHE:
            with CACHE_LOCK:
                if binary_key not in _CACHE:
                    _CACHE[binary_key] = _BinaryCoefficientField(binary)
        return _CACHE[binary_key]
    # pyfastcd owns the selector rules of every other field, so a level-zero
    # algebra validates them. It is created under the lock: two threads that
    # ask pyfastcd for a field for the first time at once can be given two
    # field objects, which would be the keys of two parents
    with CACHE_LOCK:
        parent = fcd.Algebra(level=0, char=char, deg=deg, field=field)
        key = (parent.char, parent.deg, parent.field)
        if key not in _CACHE:
            _CACHE[key] = _CoefficientField(parent)
        return _CACHE[key]


def _reduce_integers(values: np.ndarray, char: int) -> np.ndarray:
    """Return an integer array reduced modulo the characteristic, as ``uint32``."""

    # An unsigned value of 2^63 or more does not fit the signed type that the
    # other integers are reduced in
    if values.dtype.kind == "u":
        reduced = np.remainder(values.astype(np.uint64, copy=False), np.uint64(char))
    else:
        reduced = np.remainder(values.astype(np.int64, copy=False), char)
    result: np.ndarray = reduced.astype(np.uint32)
    return result


def _is_integer(value: Any) -> bool:
    """Return whether a value is a Python or NumPy integer, and not a Boolean."""

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, np.integer))


def _booleans(values: Any) -> bool:
    """Return whether values are an array or a nested sequence of Booleans."""

    # A Boolean is not a scalar of a finite field, so Booleans are not taken
    # for its coefficients either, although pyfastcd reads them as 0 and 1
    if not isinstance(values, np.ndarray):
        try:
            values = np.asarray(values)
        except (TypeError, ValueError):
            return False
    return bool(values.dtype.kind == "b")


def _kernel_input(values: np.ndarray, dtype: Any) -> np.ndarray:
    """Return an array as the compiled kernels read it: contiguous and aligned."""

    result = np.ascontiguousarray(values, dtype=dtype)
    if result.flags.aligned:
        return result
    # Storage at an unaligned address is copied to memory that is aligned
    return np.require(result, requirements=["C", "A"])


class _CoefficientField:
    """Arithmetic of one coefficient field on arrays of coefficients.

    A coefficient array has shape ``index_shape + self.shape``, where
    ``self.shape`` is ``()`` over R, C and prime fields and ``(deg,)`` over
    an extension field. Real and complex coefficients are ``float64`` and
    ``complex128`` values; finite-field coefficients are canonical
    ``uint32`` residues, as in pyfastcd. Arithmetic methods return new
    arrays; ``view`` and ``expand`` return views of their argument.
    """

    __slots__ = ("char", "deg", "field", "dtype", "shape", "kind", "tail")

    def __init__(self, parent: Any) -> None:
        """Record the field of a level-zero pyfastcd algebra."""

        self.char: int = parent.char
        self.deg: int = parent.deg
        self.field: Any = parent.field
        self.dtype: np.dtype[Any] = np.dtype(parent.dtype)
        if self.field is None:
            self.kind = "real" if self.deg == 1 else "complex"
            self.shape: tuple[int, ...] = ()
        else:
            self.kind = "finite"
            self.shape = tuple(self.field.coefficient_shape)
        self.tail = len(self.shape)

    @property
    def selectors(self) -> dict[str, Any]:
        """Return the keyword arguments that select this field in a loop algebra."""

        if self.field is not None:
            return {"field": self.field}
        return {"char": 0, "deg": self.deg}

    @property
    def exact(self) -> bool:
        """Return whether arithmetic in this field is exact."""

        return self.kind in ("finite", "binary")

    @property
    def binary(self) -> bool:
        """Return whether the field has characteristic 2."""

        return self.kind == "binary"

    # Conversion

    def coerce(self, values: Any) -> np.ndarray:
        """Copy coefficients into a new array of this field's storage type."""

        if self.kind == "finite":
            if isinstance(values, np.ndarray) and values.dtype.kind in "ui":
                result = _reduce_integers(values, self.char)
            elif _booleans(values):
                raise TypeError("coefficients over a finite field must be integers")
            else:
                result = np.array(self.field.array(values).coefficients, copy=True)
            if result.ndim < self.tail or result.shape[result.ndim - self.tail :] != (
                self.shape
            ):
                raise ValueError("coefficients do not have the field's shape")
            return np.ascontiguousarray(result)
        array = np.asarray(values)
        if array.dtype.kind == "O":
            flat = [self._number(value) for value in array.ravel()]
            array = np.array(flat, dtype=self.dtype).reshape(array.shape)
        if array.dtype.kind not in "biufc":
            raise TypeError("coefficients must be numeric")
        if array.dtype.kind == "c" and self.kind == "real":
            raise TypeError("complex coefficients are not supported over R")
        return np.array(array, dtype=self.dtype, order="C", copy=True)

    def _number(self, value: Any) -> Any:
        """Convert one real or complex scalar to a Python number."""

        if isinstance(value, (Decimal, Fraction)):
            return float(value)
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, numbers.Complex):
            if self.kind == "real":
                raise TypeError("complex scalars are not supported over R")
            return complex(value)
        raise TypeError("coefficients must be numeric")

    def is_scalar(self, value: Any) -> bool:
        """Return whether *value* is one scalar of this field."""

        if self.kind == "finite":
            if isinstance(value, fcd.Element):
                return value.algebra is self.field
            if isinstance(value, (bool, np.bool_)):
                return False
            return isinstance(value, (numbers.Integral, np.integer))
        if isinstance(value, np.ndarray):
            if value.ndim != 0 or value.dtype.kind not in "biufc":
                return False
            return self.kind == "complex" or value.dtype.kind != "c"
        if isinstance(value, (Decimal, Fraction, numbers.Real)):
            return True
        return self.kind == "complex" and isinstance(value, numbers.Complex)

    def scalar(self, value: Any) -> np.ndarray:
        """Return one field scalar as a coefficient array of shape ``self.shape``."""

        if not self.is_scalar(value):
            raise TypeError("operand is not a scalar of the coefficient field")
        if self.kind == "finite":
            if isinstance(value, fcd.Element):
                return np.array(value.coefficients, dtype=np.uint32).reshape(self.shape)
            result = np.zeros(self.shape, dtype=np.uint32)
            result[(0,) * self.tail] = int(value) % self.char
            return result
        if isinstance(value, np.ndarray):
            value = value[()]
        return np.array(self._number(value), dtype=self.dtype)

    def python(self, coefficient: np.ndarray) -> Any:
        """Return one coefficient as a ``float``, ``complex`` or field scalar."""

        if self.kind == "finite":
            if self.tail:
                return self.field(coefficient.tolist())
            return self.field(int(coefficient))
        return coefficient.item()

    def view(self, coefficients: np.ndarray) -> Any:
        """Return coefficients as the public array type of this field."""

        if self.kind == "finite":
            return self.field.asarray(coefficients)
        return coefficients

    def zeros(self, index_shape: tuple[int, ...]) -> np.ndarray:
        """Return an array of zero coefficients."""

        return np.zeros(tuple(index_shape) + self.shape, dtype=self.dtype)

    def index_shape(self, coefficients: np.ndarray) -> tuple[int, ...]:
        """Return the shape of an array without its polynomial axis."""

        return coefficients.shape[: coefficients.ndim - self.tail]

    def expand(self, coefficients: np.ndarray, count: int) -> np.ndarray:
        """Insert *count* index axes of length one before the polynomial axis."""

        if count == 0:
            return coefficients
        shape = self.index_shape(coefficients) + (1,) * count + self.shape
        return coefficients.reshape(shape)

    # Linear arithmetic

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add coefficients."""

        if self.kind == "finite":
            total = a.astype(np.int64) + b
            return np.remainder(total, self.char).astype(np.uint32)
        return a + b

    def subtract(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Subtract coefficients."""

        if self.kind == "finite":
            difference = a.astype(np.int64) - b
            return np.remainder(difference, self.char).astype(np.uint32)
        return a - b

    def negate(self, a: np.ndarray) -> np.ndarray:
        """Negate coefficients."""

        if self.kind == "finite":
            return np.remainder(-a.astype(np.int64), self.char).astype(np.uint32)
        return -a

    def multiply_int(self, a: np.ndarray, factor: int) -> np.ndarray:
        """Multiply coefficients by a fixed integer."""

        if self.kind == "finite":
            scaled = a.astype(np.uint64) * np.uint64(factor % self.char)
            return np.remainder(scaled, np.uint64(self.char)).astype(np.uint32)
        return a * float(factor)

    def divide_int(self, a: np.ndarray, divisor: int) -> np.ndarray:
        """Divide coefficients by a fixed integer that is invertible in the field."""

        if self.kind == "finite":
            return self.multiply_int(a, pow(divisor, -1, self.char))
        return a / float(divisor)

    def halve(self, a: np.ndarray) -> np.ndarray:
        """Divide coefficients by two."""

        if self.kind == "finite":
            return self.multiply_int(a, (self.char + 1) // 2)
        return a * 0.5

    def sum(self, a: np.ndarray, axis: int) -> np.ndarray:
        """Sum coefficients along one index axis, counted before the polynomial axis."""

        position = axis % (a.ndim - self.tail)
        if self.kind == "finite":
            total = a.sum(axis=position, dtype=np.uint64)
            return np.remainder(total, np.uint64(self.char)).astype(np.uint32)
        return a.sum(axis=position)

    def signed_sum(self, a: np.ndarray, signs: np.ndarray, axis: int) -> np.ndarray:
        """Sum coefficients along one index axis after applying fixed signs.

        *signs* holds +1 and -1 and broadcasts against the index axes of *a*.
        """

        position = axis % (a.ndim - self.tail)
        weights = np.asarray(signs)[..., None] if self.tail else np.asarray(signs)
        if self.kind == "finite":
            total = (a.astype(np.int64) * weights).sum(axis=position)
            return np.remainder(total, self.char).astype(np.uint32)
        return (a * weights).sum(axis=position)

    def apply_signs(self, a: np.ndarray, signs: np.ndarray) -> np.ndarray:
        """Multiply coefficients by fixed signs that broadcast over the index axes."""

        weights = np.asarray(signs)[..., None] if self.tail else np.asarray(signs)
        if self.kind == "finite":
            return np.where(weights < 0, self.negate(a), a)
        return a * weights

    # Multiplicative arithmetic

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply coefficients in the field, coordinate by coordinate."""

        if self.kind != "finite":
            return a * b
        if not self.tail:
            product = a.astype(np.uint64) * b
            return np.remainder(product, np.uint64(self.char)).astype(np.uint32)
        shape = np.broadcast_shapes(a.shape, b.shape)
        left = self.field.asarray(np.ascontiguousarray(np.broadcast_to(a, shape)))
        right = self.field.asarray(np.ascontiguousarray(np.broadcast_to(b, shape)))
        return np.asarray(self.field.multiply(left, right).coefficients).reshape(shape)

    def scale(self, a: np.ndarray, scalar: np.ndarray) -> np.ndarray:
        """Multiply every coefficient by one field scalar or an array of them."""

        return self.multiply(a, scalar)

    def reciprocal(self, a: np.ndarray) -> np.ndarray:
        """Invert coefficients, raising ``ZeroDivisionError`` at a zero."""

        if not bool(np.all(self.nonzero(a))):
            raise ZeroDivisionError("division by zero in the coefficient field")
        if self.kind == "finite":
            values = self.field.asarray(np.ascontiguousarray(a))
            return np.asarray(values.reciprocal().coefficients).reshape(a.shape)
        return _range.quotient(np.ones((), dtype=a.dtype), a)

    def power(self, a: np.ndarray, exponent: int) -> np.ndarray:
        """Raise coefficients to a nonnegative integer power."""

        if self.kind == "finite":
            values = self.field.asarray(np.ascontiguousarray(a))
            return np.asarray((values**exponent).coefficients).reshape(a.shape)
        # Binary exponentiation by multiplication gives the exact power
        # whenever that power is representable, which the power function of
        # NumPy does not
        result = a.copy() if exponent & 1 else None
        base = a
        count = exponent
        exponent >>= 1
        while exponent:
            base = base * base
            if exponent & 1:
                result = base if result is None else result * base
            exponent >>= 1
        if result is None:
            return np.ones_like(a)
        if self.kind == "complex" and count > 1:
            # A part of a complex product is a difference of two products,
            # which can overflow although the part does not. A power that
            # is not finite is formed again where that cannot happen
            lost = ~np.isfinite(result) & np.isfinite(a)
            if lost.any():
                result[lost] = _range.power(a[lost], count)
        return result

    # Predicates

    def nonzero(self, a: np.ndarray) -> np.ndarray:
        """Return a Boolean array marking the nonzero coefficients."""

        marks = a != 0
        return marks.any(axis=-1) if self.tail else marks

    def equal(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return a Boolean array marking equal coefficients."""

        marks = a == b
        return marks.all(axis=-1) if self.tail else marks

    # Transform and extension

    def wht(self, a: np.ndarray) -> np.ndarray:
        """Return the Walsh--Hadamard transform along the last index axis."""

        if self.kind != "finite":
            return _hadamard(a)
        return _hadamard_mod(a, self.char, self.deg if self.tail else 1)

    @property
    def native(self) -> bool:
        """Return whether coefficients are stored as machine words."""

        return self.dtype.kind in "fcu"

    @property
    def lanes(self) -> int:
        """The number of storage words of the compiled kernels in one coefficient."""

        if self.kind == "complex":
            return 2
        return self.deg if self.tail else 1

    def split(
        self,
        coefficients: np.ndarray,
        size: int,
        out: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the split form of loop-basis records, by the compiled kernel.

        The records have shape ``(2, size) + self.shape`` after any leading
        axes. The result is the pair of the Walsh--Hadamard transform of the
        row sums and the row differences, written to *out* when it is given.
        """

        lead = coefficients.shape[: coefficients.ndim - self.tail - 2]
        source = _kernel_input(coefficients, self.dtype)
        if out is None:
            spectral = np.empty(lead + (size,) + self.shape, dtype=self.dtype)
            signed = np.empty(spectral.shape, dtype=self.dtype)
        else:
            spectral, signed = out
        if spectral.size:
            if self.kind == "finite":
                _core.split_mod(source, spectral, signed, size, self.lanes, self.char)
            else:
                _core.split(source, spectral, signed, size, self.lanes)
        return spectral, signed

    def unsplit(
        self, spectral: np.ndarray, signed: np.ndarray, size: int
    ) -> np.ndarray:
        """Return the loop-basis records of split forms, by the compiled kernel."""

        if spectral.shape != signed.shape:
            shape = np.broadcast_shapes(spectral.shape, signed.shape)
            spectral = np.broadcast_to(spectral, shape)
            signed = np.broadcast_to(signed, shape)
        first = _kernel_input(spectral, self.dtype)
        second = _kernel_input(signed, self.dtype)
        lead = first.shape[: first.ndim - self.tail - 1]
        coefficients = np.empty(lead + (2, size) + self.shape, dtype=self.dtype)
        if coefficients.size:
            if self.kind == "finite":
                inverse = pow(size, -1, self.char)
                _core.unsplit_mod(
                    first, second, coefficients, size, self.lanes, self.char, inverse
                )
            else:
                _core.unsplit(first, second, coefficients, size, self.lanes)
        return coefficients

    def gaussian_extension(self) -> _CoefficientField | None:
        """Return the field obtained by adjoining a square root of -1, if supported.

        The result is C over R, and F_(p^2) with modulus ``t^2 + 1`` over a
        prime field with ``p = 3 (mod 4)``. Other fields return ``None``.
        """

        if self.kind == "real":
            return _coefficient_field(char=0, deg=2)
        if self.kind == "finite" and self.deg == 1 and self.char % 4 == 3:
            extension = fcd.FiniteField.from_modulus(char=self.char, modulus=(1, 0, 1))
            return _coefficient_field(field=extension)
        return None

    def pack_gaussian(self, real: np.ndarray, imaginary: np.ndarray) -> np.ndarray:
        """Combine two coefficient arrays into one array over the Gaussian extension."""

        if self.kind == "real":
            return real + 1j * imaginary
        return np.ascontiguousarray(np.stack([real, imaginary], axis=-1))

    def unpack_gaussian(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split an array over the Gaussian extension into its two parts."""

        if self.kind == "real":
            return np.ascontiguousarray(values.real), np.ascontiguousarray(values.imag)
        return (
            np.ascontiguousarray(values[..., 0]),
            np.ascontiguousarray(values[..., 1]),
        )


class _BinaryCoefficientField(_CoefficientField):
    """Arithmetic of a field of characteristic 2 on arrays of coefficients.

    A coefficient is stored as its polynomial coordinates, each 0 or 1, in
    ``uint32`` words: one word at degree one and ``deg`` words otherwise, the
    layout of the finite fields of odd characteristic. Addition is XOR, and
    products go through the logarithm tables of the :class:`BinaryField`
    after packing each coefficient into one integer.

    Division by two does not exist here, so the Walsh--Hadamard transform
    and the split form are not available, and the methods that need them
    raise ``TypeError``.
    """

    __slots__ = ()

    def __init__(self, field: BinaryField) -> None:
        """Record a field of characteristic 2."""

        self.char = 2
        self.deg = field.deg
        self.field = field
        self.dtype = np.dtype(np.uint32)
        self.kind = "binary"
        self.shape = field.coefficient_shape
        self.tail = len(self.shape)

    # Conversion

    def _nested(self, values: Any) -> Any:
        """Replace the scalars in a nested sequence by their coordinates."""

        if isinstance(values, BinaryFieldElement):
            if values.algebra is not self.field:
                raise ValueError("a scalar belongs to a different field")
            return values.coefficients.tolist()
        if isinstance(values, np.ndarray):
            return values
        if isinstance(values, (list, tuple)):
            # In a list that holds scalars every entry is one coefficient, so
            # an integer beside them stands for an element of the prime subfield
            if any(isinstance(value, BinaryFieldElement) for value in values):
                return [
                    self._nested(value)
                    if isinstance(value, BinaryFieldElement)
                    else self.scalar(value).tolist()
                    for value in values
                ]
            return [self._nested(value) for value in values]
        return values

    def coerce(self, values: Any) -> np.ndarray:
        """Copy coefficients into a new array, reducing integers modulo 2."""

        nested = self._nested(values)
        array = np.asarray(nested)
        if array.dtype.kind == "f" and not isinstance(nested, np.ndarray):
            # NumPy reads integers of mixed sign beyond the range of int64 as
            # floats; the entries themselves decide whether they are integers
            array = np.array(nested, dtype=object)
        if array.dtype.kind == "O":
            # Python integers beyond the range of the NumPy types
            entries = array.ravel().tolist()
            if all(_is_integer(entry) for entry in entries):
                parities = [int(entry) % 2 for entry in entries]
                array = np.array(parities, dtype=np.uint32).reshape(array.shape)
        if array.dtype.kind not in "iu":
            raise TypeError("coefficients over a finite field must be integers")
        result = _reduce_integers(array, 2)
        if result.ndim < self.tail or result.shape[result.ndim - self.tail :] != (
            self.shape
        ):
            raise ValueError("coefficients do not have the field's shape")
        return np.ascontiguousarray(result)

    def is_scalar(self, value: Any) -> bool:
        """Return whether *value* is one scalar of this field."""

        if isinstance(value, BinaryFieldElement):
            return value.algebra is self.field
        if isinstance(value, (bool, np.bool_)):
            return False
        return isinstance(value, (numbers.Integral, np.integer))

    def scalar(self, value: Any) -> np.ndarray:
        """Return one field scalar as a coefficient array of shape ``self.shape``."""

        if not self.is_scalar(value):
            raise TypeError("operand is not a scalar of the coefficient field")
        if isinstance(value, BinaryFieldElement):
            return np.array(value.coefficients, dtype=np.uint32).reshape(self.shape)
        result = np.zeros(self.shape, dtype=np.uint32)
        result[(0,) * self.tail] = int(value) % 2
        return result

    def python(self, coefficient: np.ndarray) -> Any:
        """Return one coefficient as a scalar of the field."""

        return self.field(np.asarray(coefficient).tolist())

    def view(self, coefficients: np.ndarray) -> Any:
        """Return coefficients as they are: there is no array type of scalars."""

        return coefficients

    # Linear arithmetic

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add coefficients, by XOR."""

        return np.bitwise_xor(a, b)

    def subtract(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Subtract coefficients, which is addition."""

        return np.bitwise_xor(a, b)

    def negate(self, a: np.ndarray) -> np.ndarray:
        """Negate coefficients, which leaves them unchanged."""

        return a.copy()

    def multiply_int(self, a: np.ndarray, factor: int) -> np.ndarray:
        """Multiply coefficients by a fixed integer."""

        return a.copy() if factor % 2 else np.zeros(a.shape, dtype=a.dtype)

    def divide_int(self, a: np.ndarray, divisor: int) -> np.ndarray:
        """Divide coefficients by a fixed odd integer."""

        if divisor % 2 == 0:
            raise TypeError(BINARY_MESSAGE.format(name="division by an even integer"))
        return a.copy()

    def halve(self, a: np.ndarray) -> np.ndarray:
        """Division by two does not exist in characteristic 2."""

        raise TypeError(BINARY_MESSAGE.format(name="division by two"))

    def sum(self, a: np.ndarray, axis: int) -> np.ndarray:
        """Sum coefficients along one index axis, by XOR."""

        position = axis % (a.ndim - self.tail)
        return np.bitwise_xor.reduce(a, axis=position)

    def signed_sum(self, a: np.ndarray, signs: np.ndarray, axis: int) -> np.ndarray:
        """Sum coefficients along one index axis; signs are trivial here."""

        return self.sum(a, axis)

    def apply_signs(self, a: np.ndarray, signs: np.ndarray) -> np.ndarray:
        """Apply fixed signs, which leaves coefficients unchanged."""

        return a.copy()

    # Multiplicative arithmetic

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply coefficients in the field, coordinate by coordinate."""

        if not self.tail:
            return np.bitwise_and(a, b)
        field = self.field
        return field._unpack(field._multiply_masks(field._pack(a), field._pack(b)))

    def reciprocal(self, a: np.ndarray) -> np.ndarray:
        """Invert coefficients, raising ``ZeroDivisionError`` at a zero."""

        if not bool(np.all(self.nonzero(a))):
            raise ZeroDivisionError("division by zero in the coefficient field")
        field = self.field
        return field._unpack(field._reciprocal_masks(field._pack(a)))

    def power(self, a: np.ndarray, exponent: int) -> np.ndarray:
        """Raise coefficients to a nonnegative integer power."""

        field = self.field
        return field._unpack(field._power_masks(field._pack(a), exponent))

    # Transform and extension

    def wht(self, a: np.ndarray) -> np.ndarray:
        """The Walsh--Hadamard transform is not invertible in characteristic 2."""

        raise TypeError(BINARY_MESSAGE.format(name="the Walsh--Hadamard transform"))

    @property
    def native(self) -> bool:
        """Return ``False``: the compiled coordinate change needs division by two."""

        return False

    def split(
        self,
        coefficients: np.ndarray,
        size: int,
        out: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The split form does not exist in characteristic 2."""

        raise TypeError(BINARY_MESSAGE.format(name="the split form"))

    def unsplit(
        self, spectral: np.ndarray, signed: np.ndarray, size: int
    ) -> np.ndarray:
        """The split form does not exist in characteristic 2."""

        raise TypeError(BINARY_MESSAGE.format(name="the split form"))

    def gaussian_extension(self) -> _CoefficientField | None:
        """Return ``None``: no method here adjoins a square root of -1."""

        return None
