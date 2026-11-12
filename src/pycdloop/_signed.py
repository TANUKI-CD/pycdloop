"""Private arithmetic of the signed component of a loop algebra.

The loop algebra of a Cayley--Dickson loop splits into a group algebra and
the Cayley--Dickson algebra itself; for a central product the second summand
is a tensor product of Cayley--Dickson algebras. The classes here multiply
in that second summand on raw coefficient arrays of shape
``batch_shape + (size,) + field.shape``, where ``size`` is the quotient
order of the loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyfastcd as fcd

from ._field import _CoefficientField
from ._loop import _cd_twist
from ._schedule import schedule_multiply
from ._validation import _choice

# The largest outer level multiplied row by row under "auto"; the schedule
# takes over above it
ROWS_MAX_LEVEL = 4

_CACHE: dict[tuple[tuple[int, ...], _CoefficientField], _SignedAlgebra] = {}


def _signed_algebra(
    levels: tuple[int, ...], field: _CoefficientField
) -> _SignedAlgebra:
    """Return the cached signed component for the given factor levels."""

    key = (levels, field)
    if key not in _CACHE:
        if len(levels) == 1:
            _CACHE[key] = _CayleyDickson(levels[0], field)
        else:
            _CACHE[key] = _Tensor(levels, field)
    return _CACHE[key]


class _SignedAlgebra:
    """Operations shared by both kinds of signed component."""

    field: _CoefficientField
    size: int
    algorithms: tuple[str, ...]
    # The pyfastcd algebra, when the component is one Cayley--Dickson algebra
    cd: Any = None

    def multiply(
        self, x: np.ndarray, y: np.ndarray, algorithm: str = "auto"
    ) -> np.ndarray:
        """Return the ordered product, broadcasting leading axes."""

        raise NotImplementedError

    def multiply_into(
        self, x: np.ndarray, y: np.ndarray, algorithm: str, out: np.ndarray
    ) -> np.ndarray:
        """Return the ordered product; implementations may write it to *out*."""

        return self.multiply(x, y, algorithm)

    def conjugate(self, x: np.ndarray) -> np.ndarray:
        """Return the involution induced by inversion in the loop."""

        raise NotImplementedError

    def square(self, x: np.ndarray) -> np.ndarray:
        """Return ``x x``."""

        return self.multiply(x, x)

    def commutator(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return ``x y - y x``."""

        return self.field.subtract(self.multiply(x, y), self.multiply(y, x))

    def anticommutator(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return ``x y + y x``."""

        return self.field.add(self.multiply(x, y), self.multiply(y, x))

    def associator(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return ``(x y) z - x (y z)``."""

        left = self.multiply(self.multiply(x, y), z)
        return self.field.subtract(left, self.multiply(x, self.multiply(y, z)))

    def left_operator_commutator(
        self, a: np.ndarray, b: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Return ``a (b x) - b (a x)``."""

        first = self.multiply(a, self.multiply(b, x))
        return self.field.subtract(first, self.multiply(b, self.multiply(a, x)))

    def right_operator_commutator(
        self, a: np.ndarray, b: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Return ``(x b) a - (x a) b``."""

        first = self.multiply(self.multiply(x, b), a)
        return self.field.subtract(first, self.multiply(self.multiply(x, a), b))


class _CayleyDickson(_SignedAlgebra):
    """The Cayley--Dickson algebra ``A_n``, delegated to pyfastcd."""

    def __init__(self, level: int, field: _CoefficientField) -> None:
        """Create the pyfastcd algebra of the given level over the field."""

        self.field = field
        self.level = level
        self.size = 1 << level
        self._record_size = self.size * field.deg if field.tail else self.size
        self.cd = fcd.Algebra(level=level, **field.selectors)
        names = ["auto", "direct", "quasilinear"]
        if field.kind == "real" and self.size <= 1024:
            names.insert(2, "cariow_cariowa")
        self.algorithms = tuple(names)

    def _wrap(self, *arrays: np.ndarray) -> tuple[list[Any], tuple[int, ...] | None]:
        """View coefficient arrays as pyfastcd operands of one kind.

        Over R, when every operand holds a single record the operands become
        pyfastcd elements, whose arithmetic skips the validation and
        broadcasting of the array interface, and the second result is the
        batch shape to restore. Otherwise they become pyfastcd arrays, which
        share the given storage where its layout allows. Over C and finite
        fields the array interface is the faster one for single records too.
        """

        record = self.field.tail + 1
        single = self.field.kind == "real"
        if single and all(array.size == self._record_size for array in arrays):
            lead = np.broadcast_shapes(
                *(array.shape[: array.ndim - record] for array in arrays)
            )
            shape = (self.size,) + self.field.shape
            return [self.cd(array.reshape(shape)) for array in arrays], lead
        return [self.cd.asarray(array) for array in arrays], None

    def _raw(self, result: Any, lead: tuple[int, ...] | None) -> np.ndarray:
        """Return the coefficients of a pyfastcd result as a writable array."""

        coefficients = np.asarray(result.coefficients)
        if lead is None:
            return coefficients
        return coefficients.reshape(lead + coefficients.shape).copy()

    def multiply(
        self, x: np.ndarray, y: np.ndarray, algorithm: str = "auto"
    ) -> np.ndarray:
        """Return the ordered product, broadcasting leading axes."""

        (left, right), lead = self._wrap(x, y)
        return self._raw(self.cd.multiply(left, right, algorithm=algorithm), lead)

    def multiply_into(
        self, x: np.ndarray, y: np.ndarray, algorithm: str, out: np.ndarray
    ) -> np.ndarray:
        """Return the ordered product, written to *out* where pyfastcd allows it.

        *out* has the broadcast shape of the operands and overlaps neither.
        A product of single records goes through pyfastcd elements, which
        take no output storage, and is returned in a new array.
        """

        (left, right), lead = self._wrap(x, y)
        if lead is not None:
            return self._raw(self.cd.multiply(left, right, algorithm=algorithm), lead)
        self.cd.multiply(left, right, out=out, algorithm=algorithm)
        return out

    def conjugate(self, x: np.ndarray) -> np.ndarray:
        """Return the Cayley--Dickson conjugate."""

        (value,), lead = self._wrap(x)
        return self._raw(value.conjugate(), lead)

    def square(self, x: np.ndarray) -> np.ndarray:
        """Return ``x x`` from the scalar and imaginary parts."""

        (value,), lead = self._wrap(x)
        return self._raw(value.square(), lead)

    def commutator(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return ``x y - y x``."""

        (left, right), lead = self._wrap(x, y)
        return self._raw(left.commutator(right), lead)

    def anticommutator(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return ``x y + y x`` from the scalar and imaginary parts."""

        (left, right), lead = self._wrap(x, y)
        return self._raw(left.anticommutator(right), lead)

    def associator(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return ``(x y) z - x (y z)``."""

        (first, second, third), lead = self._wrap(x, y, z)
        return self._raw(first.associator(second, third), lead)

    def left_operator_commutator(
        self, a: np.ndarray, b: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Return ``a (b x) - b (a x)``."""

        operands, lead = self._wrap(a, b, x)
        return self._raw(self.cd.left_operator_commutator(*operands), lead)

    def right_operator_commutator(
        self, a: np.ndarray, b: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Return ``(x b) a - (x a) b``."""

        operands, lead = self._wrap(a, b, x)
        return self._raw(self.cd.right_operator_commutator(*operands), lead)

    def quadratic_norm(self, x: np.ndarray) -> np.ndarray:
        """Return the quadratic form ``Q(x) = sum of x_i^2`` as field scalars."""

        if self.field.kind == "finite":
            values = self.cd.asarray(x).quadratic_norm().coefficients
            return np.asarray(values).reshape(
                x.shape[: -1 - self.field.tail] + self.field.shape
            )
        # A plain sum of squares is exact on small integers, which the
        # scaled accumulation of pyfastcd is not
        norm: np.ndarray = (x * x).sum(axis=-1)
        return norm

    def reciprocal(self, x: np.ndarray) -> np.ndarray:
        """Return ``conjugate(x) / Q(x)``; raise ``ZeroDivisionError`` at ``Q = 0``."""

        (value,), lead = self._wrap(x)
        return self._raw(value.reciprocal(), lead)


class _Tensor(_SignedAlgebra):
    """A tensor product ``A_n1 (x) A_n2 (x) ...`` of Cayley--Dickson algebras.

    Coefficients are stored flat, in the order of the flat basis index of
    the central product. The product treats the factor of smallest level as
    an outer Cayley--Dickson algebra with coefficients in the tensor product
    of the remaining factors, and multiplies it by one of three methods:

    ``"native"``
        For an outer factor of level 1 over R or over a prime field with
        ``p = 3 (mod 4)``: the outer factor is the field extension by a
        square root of -1, so the product is one product of the inner
        algebra over that extension.
    ``"rows"``
        All ``N^2`` products of rows of the two operands, as one batch.
    ``"schedule"``
        The quasilinear schedule of :mod:`pycdloop._schedule`.
    """

    def __init__(self, levels: tuple[int, ...], field: _CoefficientField) -> None:
        """Order the factors by level and prepare the inner algebra."""

        self.field = field
        self.levels = levels
        self.sizes = tuple(1 << level for level in levels)
        self.size = int(np.prod(self.sizes))
        # A stable sort keeps equal levels in their given order
        self.order = tuple(sorted(range(len(levels)), key=lambda i: levels[i]))
        ordered = tuple(levels[i] for i in self.order)
        self.outer_level = ordered[0]
        self.outer_size = 1 << self.outer_level
        self.inner = _signed_algebra(ordered[1:], field)
        self.extension = field.gaussian_extension() if self.outer_level == 1 else None
        self.native = (
            _signed_algebra(ordered[1:], self.extension)
            if self.extension is not None
            else None
        )
        names = ["auto", "rows", "schedule"]
        if self.native is not None:
            names.insert(1, "native")
        self.algorithms = tuple(names)
        indices = np.arange(self.outer_size, dtype=np.int64)
        partner = indices[None, :] ^ indices[:, None]
        # Row k of these tables lists, for each p, the index p ^ k and the
        # sign sigma(p, p ^ k) of the term that lands on output row k
        self._partner = partner
        self._signs = _cd_twist(self.outer_level, indices[None, :], partner)
        self._conjugation = self._conjugation_signs()

    def _conjugation_signs(self) -> np.ndarray:
        """Return the signs of the tensor product of the factor conjugations."""

        signs = np.ones(1, dtype=np.int8)
        for size in self.sizes:
            factor = -np.ones(size, dtype=np.int8)
            factor[0] = 1
            signs = np.multiply.outer(signs, factor).ravel()
        return signs

    def conjugate(self, x: np.ndarray) -> np.ndarray:
        """Return the tensor product of the Cayley--Dickson conjugations."""

        return self.field.apply_signs(x, self._conjugation)

    def default_algorithm(self) -> str:
        """Return the method that ``"auto"`` selects for the outer factor."""

        if self.native is not None:
            return "native"
        return "rows" if self.outer_level <= ROWS_MAX_LEVEL else "schedule"

    def multiply(
        self, x: np.ndarray, y: np.ndarray, algorithm: str = "auto"
    ) -> np.ndarray:
        """Return the ordered product, broadcasting leading axes."""

        name = _choice(algorithm, "signed_algorithm", self.algorithms)
        if name == "auto":
            name = self.default_algorithm()
        tail = self.field.tail
        lead = np.broadcast_shapes(
            x.shape[: x.ndim - tail - 1], y.shape[: y.ndim - tail - 1]
        )
        u = self._to_rows(np.broadcast_to(x, lead + x.shape[x.ndim - tail - 1 :]))
        v = self._to_rows(np.broadcast_to(y, lead + y.shape[y.ndim - tail - 1 :]))
        if name == "native":
            product = self._multiply_native(u, v)
        elif name == "rows":
            product = self._multiply_rows(u, v)
        else:
            flat = (-1,) + u.shape[len(lead) :]
            product = schedule_multiply(
                self.field,
                self.inner.multiply,
                self.outer_level,
                np.ascontiguousarray(u).reshape(flat),
                np.ascontiguousarray(v).reshape(flat),
            ).reshape(u.shape)
        return self._from_rows(product)

    def _to_rows(self, x: np.ndarray) -> np.ndarray:
        """Reshape flat coefficients to rows: one inner element per outer index."""

        tail = self.field.tail
        lead = x.shape[: x.ndim - tail - 1]
        factors = x.reshape(lead + self.sizes + self.field.shape)
        base = len(lead)
        axes = tuple(base + i for i in self.order)
        every = tuple(range(base)) + axes + tuple(range(base + len(axes), factors.ndim))
        rows = lead + (self.outer_size, self.size // self.outer_size) + self.field.shape
        return np.ascontiguousarray(factors.transpose(every)).reshape(rows)

    def _from_rows(self, rows: np.ndarray) -> np.ndarray:
        """Undo :meth:`_to_rows`."""

        tail = self.field.tail
        lead = rows.shape[: rows.ndim - tail - 2]
        ordered = tuple(self.sizes[i] for i in self.order)
        factors = rows.reshape(lead + ordered + self.field.shape)
        base = len(lead)
        inverse = np.argsort(self.order)
        axes = tuple(base + int(i) for i in inverse)
        every = tuple(range(base)) + axes + tuple(range(base + len(axes), factors.ndim))
        flat = lead + (self.size,) + self.field.shape
        return np.ascontiguousarray(factors.transpose(every)).reshape(flat)

    def _row(self, rows: np.ndarray, index: Any) -> np.ndarray:
        """Select along the outer axis of an array of rows."""

        position = rows.ndim - self.field.tail - 2
        return rows[(slice(None),) * position + (index,)]

    def _multiply_native(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Multiply through the extension of the field by a square root of -1."""

        # A_1 (x) B is B with coefficients extended by i = e_1, since e_1 is
        # central in the tensor product and squares to -1
        assert self.native is not None
        field = self.field
        left = field.pack_gaussian(self._row(u, 0), self._row(u, 1))
        right = field.pack_gaussian(self._row(v, 0), self._row(v, 1))
        parts = field.unpack_gaussian(self.native.multiply(left, right))
        return np.stack(parts, axis=parts[0].ndim - field.tail - 1)

    def _multiply_rows(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Multiply by forming every product of a row of *u* with a row of *v*."""

        # U V = sum over p, r of sigma(p, r) e_(p ^ r) (x) (U_p V_r). The
        # products U_p V_r form one batch over (p, r); output row k then
        # collects the terms with r = p ^ k.
        field = self.field
        position = u.ndim - field.tail - 2
        left = np.expand_dims(u, position + 1)
        right = np.expand_dims(v, position)
        products = self.inner.multiply(left, right)
        size = self.outer_size
        own = np.broadcast_to(np.arange(size)[None, :], (size, size))
        gathered = products[(slice(None),) * position + (own, self._partner)]
        return field.signed_sum(gathered, self._signs[..., None], position + 1)
