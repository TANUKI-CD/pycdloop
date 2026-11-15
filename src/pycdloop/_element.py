"""Immutable loop-algebra elements and the arithmetic shared with element arrays."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, final

import numpy as np

from . import _kernel, _structure
from ._loop import LoopElement
from ._validation import (
    CD_ONLY_MESSAGE,
    DIFFERENT_ALGEBRA_MESSAGE,
    NO_COPY_MESSAGE,
    NO_SPLIT_MESSAGE,
    POWER_MESSAGE,
    STORAGE_MESSAGE,
    _exponent,
    _index,
    _stored,
)

_Implementation = Callable[..., Any]


class _Arithmetic:
    """Operators and methods shared by elements and both kinds of array.

    A subclass says in which coordinates an operation runs (``_mode``),
    supplies its data in those coordinates (``_data``), wraps a raw result
    (``_result``), and names the operand types it combines with
    (``_accepts``). Everything else is written once, here, on top of
    :mod:`pycdloop._kernel`.
    """

    __slots__ = ()

    # NumPy defers to the reflected operators below instead of broadcasting
    # over the coefficient storage
    __array_ufunc__ = None

    _algebra: Any

    # Hooks

    def _mode(self, linear: bool, operands: tuple[Any, ...]) -> str:
        """Return the coordinates in which an operation is evaluated."""

        raise NotImplementedError

    def _data(self, mode: str) -> Any:
        """Return raw data in the given coordinates."""

        raise NotImplementedError

    def _result(self, mode: str, data: Any) -> Any:
        """Wrap raw data as an object of the receiver's type."""

        raise NotImplementedError

    def _accepts(self, other: Any) -> bool:
        """Return whether *other* is an operand type of the receiver."""

        raise NotImplementedError

    def _truth(self, marks: np.ndarray) -> Any:
        """Return a predicate result: a ``bool`` or a Boolean array."""

        raise NotImplementedError

    # Dispatch

    def _operand(self, other: Any) -> Any:
        """Validate the operand of a named method."""

        if not self._accepts(other):
            raise TypeError(
                f"operand must be an element or array of the same loop algebra, "
                f"not {type(other).__name__}"
            )
        if other._algebra is not self._algebra:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
        return other

    def _run(
        self,
        implementation: _Implementation,
        operands: tuple[Any, ...],
        *,
        linear: bool = False,
        **options: Any,
    ) -> Any:
        """Evaluate a kernel routine on operands and wrap the result."""

        mode = self._mode(linear, operands)
        data = [operand._data(mode) for operand in operands]
        return self._result(mode, implementation(self._algebra, mode, *data, **options))

    def _with_scalar(self, implementation: _Implementation, scalar: Any) -> Any:
        """Evaluate a kernel routine that combines the receiver with a field scalar."""

        value = self._algebra._field.scalar(scalar)
        mode = self._mode(True, (self,))
        data = implementation(self._algebra, mode, self._data(mode), value)
        return self._result(mode, data)

    def _binary(
        self,
        other: Any,
        implementation: _Implementation,
        scalar_implementation: _Implementation | None,
        *,
        linear: bool,
        reflected: bool = False,
    ) -> Any:
        """Evaluate a binary operator with an element, array, or field scalar."""

        if self._algebra._field.is_scalar(other):
            if scalar_implementation is None:
                return NotImplemented
            return self._with_scalar(scalar_implementation, other)
        if not self._accepts(other):
            return NotImplemented
        if other._algebra is not self._algebra:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
        operands = (other, self) if reflected else (self, other)
        return self._run(implementation, operands, linear=linear)

    # Operators

    def __add__(self, other: Any) -> Any:
        """Add an element or array, or a field scalar times the identity."""

        return self._binary(other, _kernel.add, _kernel.add_scalar, linear=True)

    def __radd__(self, other: Any) -> Any:
        """Add with the receiver on the right."""

        return self._binary(
            other, _kernel.add, _kernel.add_scalar, linear=True, reflected=True
        )

    def __sub__(self, other: Any) -> Any:
        """Subtract an element or array, or a field scalar times the identity."""

        if self._algebra._field.is_scalar(other):
            field = self._algebra._field
            negated = field.python(field.negate(field.scalar(other)))
            return self._with_scalar(_kernel.add_scalar, negated)
        return self._binary(other, _kernel.subtract, None, linear=True)

    def __rsub__(self, other: Any) -> Any:
        """Subtract the receiver from an element, array, or field scalar."""

        return self._binary(
            other,
            _kernel.subtract,
            _kernel.subtract_from_scalar,
            linear=True,
            reflected=True,
        )

    def __mul__(self, other: Any) -> Any:
        """Return the ordered product, or scale by a field scalar."""

        return self._binary(other, _kernel.multiply, _kernel.scale, linear=False)

    def __rmul__(self, other: Any) -> Any:
        """Return the ordered product with the receiver on the right."""

        return self._binary(
            other, _kernel.multiply, _kernel.scale, linear=False, reflected=True
        )

    def __truediv__(self, other: Any) -> Any:
        """Divide by a nonzero field scalar."""

        if not self._algebra._field.is_scalar(other):
            return NotImplemented
        return self._with_scalar(_kernel.divide, other)

    def __neg__(self) -> Any:
        """Return the additive inverse."""

        return self._run(_kernel.negate, (self,), linear=True)

    def __pow__(self, exponent: Any) -> Any:
        """Return an integer power, where powers do not depend on bracketing.

        Negative exponents use the inverse. In the loop algebra of a central
        product over R, C, or a field of odd characteristic, powers can depend
        on bracketing, and this operator raises ``TypeError`` for every such
        algebra; use :meth:`left_power` or :meth:`right_power`. In
        characteristic 2 the square of an element is central for every loop,
        and every element satisfies ``x**4 == aug(x)**4``, with ``aug(x)`` the
        sum of its coefficients, so a power needs at most two products.
        """

        if not self._algebra._has_powers:
            raise TypeError(POWER_MESSAGE)
        return self._run(_kernel.power, (self,), exponent=_exponent(exponent))

    # Products and defects

    def conjugate(self) -> Any:
        """Return the image under the involution that inverts every loop element.

        The linear map ``[x] -> [x^-1]`` reverses products. It exchanges the
        coefficients of ``[e_t]`` and ``[-e_t]`` for every basis element
        that squares to -1; in split form it fixes the group component and
        conjugates the signed component.
        """

        return self._run(_kernel.conjugate, (self,), linear=True)

    def square(self) -> Any:
        """Return ``x x``."""

        return self._run(_kernel.square, (self,))

    def commutator(self, other: Any) -> Any:
        """Return ``x y - y x``."""

        return self._run(_kernel.commutator, (self, self._operand(other)))

    def anticommutator(self, other: Any) -> Any:
        """Return ``x y + y x``."""

        return self._run(_kernel.anticommutator, (self, self._operand(other)))

    def jordan_product(self, other: Any) -> Any:
        """Return ``(x y + y x) / 2``; not defined in characteristic 2."""

        return self._run(_kernel.jordan_product, (self, self._operand(other)))

    def associator(self, y: Any, z: Any) -> Any:
        """Return ``(x y) z - x (y z)``."""

        operands = (self, self._operand(y), self._operand(z))
        return self._run(_kernel.associator, operands)

    def _bracketed_power(self, implementation: _Implementation, exponent: Any) -> Any:
        """Return a power with a fixed bracketing, for a nonnegative exponent."""

        count = _index(exponent, "exponent")
        if count < 0:
            raise ValueError("exponent must be nonnegative")
        if count == 0:
            return self._with_scalar(_kernel.scale, 0)._with_scalar(
                _kernel.add_scalar, 1
            )
        return self._run(implementation, (self,), exponent=count)

    def left_power(self, exponent: Any) -> Any:
        """Return ``x (x (... (x x)))`` with *exponent* factors.

        The exponent is a nonnegative integer; exponent zero gives the
        identity. The product is evaluated with ``exponent - 1``
        multiplications.
        """

        return self._bracketed_power(_kernel.left_power, exponent)

    def right_power(self, exponent: Any) -> Any:
        """Return ``(((x x) x) ...) x`` with *exponent* factors.

        The exponent is a nonnegative integer; exponent zero gives the
        identity. The product is evaluated with ``exponent - 1``
        multiplications.
        """

        return self._bracketed_power(_kernel.right_power, exponent)

    def inverse(self) -> Any:
        """Return the inverse that is a polynomial in the element.

        It is a two-sided inverse. Over R, C, and fields of odd characteristic
        it is defined for a Cayley--Dickson loop, is the pair of the
        coordinatewise reciprocal and ``conjugate(d) / Q(d)``, and exists
        exactly when every spectral coordinate is nonzero and the signed
        component has nonzero quadratic norm. From level 4 on an element can
        have other two-sided inverses, and over C and finite fields also where
        this one does not exist; they are not computed. In characteristic 2 the
        inverse is defined for every loop, exists when the augmentation
        ``aug(x)`` is nonzero, and equals ``x**3 / aug(x)**4``. An element
        without this inverse raises ``ZeroDivisionError``, and a central
        product outside characteristic 2 raises ``TypeError``. For an array,
        every record is checked before any result is produced.
        """

        return self._run(_kernel.inverse, (self,))

    # Structure

    def _powers(self, name: str) -> Any:
        """Return the signed component algebra where powers are well defined.

        Raise for a central product outside characteristic 2.
        """

        if not self._algebra._has_powers:
            raise TypeError(CD_ONLY_MESSAGE.format(name=name))
        return self._algebra._signed

    def _augmentation_marks(self) -> np.ndarray:
        """Mark the records whose augmentation is nonzero."""

        algebra = self._algebra
        alpha = _kernel.augmentation(algebra, self._data("loop"))
        marks: np.ndarray = algebra._field.nonzero(alpha)
        return marks

    def has_inverse(self) -> Any:
        """Return whether :meth:`inverse` exists."""

        signed = self._powers("has_inverse()")
        field = self._algebra._field
        if field.binary:
            return self._truth(self._augmentation_marks())
        spectral, d = self._data("split")
        marks = field.nonzero(spectral).all(axis=-1)
        return self._truth(marks & signed.has_norm(d))

    def is_idempotent(self) -> Any:
        """Return whether ``x x == x``, compared exactly.

        The comparison is made in split coordinates, and in characteristic 2
        in the loop basis, where the only idempotents are 0 and 1.
        """

        algebra = self._algebra
        field = algebra._field
        if field.binary:
            values = self._data("loop")
            squared = _kernel.square(algebra, "loop", values)
            return self._truth(field.equal(squared, values).all(axis=(-2, -1)))
        split = self._data("split")
        squared = _kernel.square(algebra, "split", split)
        marks = field.equal(squared[0], split[0]).all(axis=-1)
        return self._truth(marks & field.equal(squared[1], split[1]).all(axis=-1))

    def is_nilpotent(self) -> Any:
        """Return whether some power vanishes, where powers are well defined.

        For a Cayley--Dickson loop over R, C, or a field of odd
        characteristic, the group component must be zero, and the signed
        component ``d`` must satisfy ``d_0 = 0`` and ``Q(d) = 0``, in which
        case ``d d = 0``. In characteristic 2, for every loop, the elements
        of augmentation zero are the nilpotent ones, and their fourth power
        vanishes.
        """

        signed = self._powers("is_nilpotent()")
        field = self._algebra._field
        if field.binary:
            return self._truth(~self._augmentation_marks())
        spectral, d = self._data("split")
        first = (Ellipsis, 0) + (slice(None),) * field.tail
        marks = ~field.nonzero(spectral).any(axis=-1)
        marks &= ~field.nonzero(d[first])
        return self._truth(marks & ~signed.has_norm(d))

    def is_central(self) -> Any:
        """Return whether the element commutes and associates with every element.

        This holds exactly when the coefficients of ``[e_t]`` and ``[-e_t]``
        agree for every basis element ``e_t`` that is not central in the
        loop; in split form, when the signed component is supported on the
        indices of the central elements of the loop.
        """

        field = self._algebra._field
        if field.binary:
            marks = _structure.central_marks(self._algebra, self._data("loop"))
            return self._truth(marks)
        d = self._data("split")[1]
        outside = field.nonzero(d) & ~self._algebra._central
        return self._truth(~outside.any(axis=-1))

    def is_zero_divisor(self, side: str | None = None) -> Any:
        """Return whether a nonzero element multiplies with this one to zero.

        *side* is ``"left"`` to ask for a nonzero ``y`` with ``x y = 0``,
        ``"right"`` for ``y x = 0``, and ``None`` for either. The zero
        element counts as a zero divisor.

        An element is a zero divisor when a spectral coordinate vanishes or
        its signed component is a zero divisor. For a Cayley--Dickson loop
        of level at most 3 the second condition is ``Q(d) = 0``; otherwise
        it is decided from the rank of a multiplication matrix, which costs
        ``O(D^3)`` operations for each record.

        In characteristic 2 the zero divisors on either side are the
        elements of augmentation zero, for every loop.
        """

        algebra = self._algebra
        if algebra._field.binary:
            marks = _structure.binary_zero_divisor_marks(
                algebra, self._data("loop"), side
            )
        else:
            marks = _structure.zero_divisor_marks(algebra, self._data("split"), side)
        return self._truth(marks)

    def weight(self) -> Any:
        """Return the number of nonzero coefficients in the loop basis."""

        marks = self._algebra._field.nonzero(self._data("loop"))
        counts = marks.sum(axis=(-2, -1))
        return int(counts) if counts.ndim == 0 else counts

    def distance(self, other: Any) -> Any:
        """Return the number of loop-basis coefficients that differ from *other*."""

        field = self._algebra._field
        partner = self._operand(other)
        marks = ~field.equal(self._data("loop"), partner._data("loop"))
        counts = marks.sum(axis=(-2, -1))
        return int(counts) if counts.ndim == 0 else counts


def _freeze(array: np.ndarray) -> np.ndarray:
    """Return an array as read-only storage."""

    array.setflags(write=False)
    return array


def _immutable(array: np.ndarray) -> np.ndarray:
    """Return a copy of an array that cannot be made writable."""

    # A read-only array that owns its memory can be made writable again. A
    # view of a bytes object cannot, so this is the form in which an element
    # hands out its data, while its own arrays stay private
    return _over(array.tobytes(), array.dtype, array.shape)


def _over(data: bytes, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Return a new read-only array over a bytes object."""

    array: np.ndarray = np.frombuffer(data, dtype=dtype)
    return array.reshape(shape)


@final
class Element(_Arithmetic):
    """One immutable element of a loop algebra.

    Create elements with a :class:`LoopAlgebra`: ``A(values)``,
    ``A.basis(x)``, ``A.from_terms(...)`` and the other constructors.
    Constructing the class directly raises ``TypeError``.

    An element is a formal combination of loop elements. Its coefficients
    form an array of shape ``A.coefficient_shape``: row 0 holds the
    coefficients of the basis elements ``[e_t]`` and row 1 those of
    ``[-e_t]``. Note that ``[-e_t]`` is a basis element of its own, not the
    negative of ``[e_t]``.

    Attributes
    ----------
    algebra : LoopAlgebra
        The parent algebra.
    dimension : int
        The dimension of the parent algebra.
    shape : tuple[int, ...]
        ``algebra.coefficient_shape``.
    dtype : numpy.dtype
        The coefficient storage type of the parent.
    coefficients : numpy.ndarray
        A read-only array of the loop-basis coefficients.

    Notes
    -----
    Elements are hashable and combine with elements and arrays of the same
    algebra and with scalars of the coefficient field. A scalar ``r`` in
    ``x + r`` stands for ``r`` times the identity; in ``r * x`` it scales
    every coefficient.

    Internally an element keeps its loop-basis coefficients, its split form
    (see :meth:`spectrum` and :meth:`epsilon`), or both. Whichever it was
    created with is never replaced; the other is derived from it on first
    use and kept. A product is computed in split form and its result is
    created in split form, so a chain of products performs no transform
    until coefficients are read.

    In characteristic 2 the algebra has no split form. An element then
    keeps its loop-basis coefficients and, after its first lifted product,
    the transforms of their lifts, which later products reuse.

    Examples
    --------
    >>> import pycdloop as cdl
    >>> Q = cdl.CDLoop(2)
    >>> A = cdl.LoopAlgebra(Q)
    >>> x = A.basis(Q.basis(1)) + 2 * A.basis(-Q.basis(2))
    >>> (x * x).terms()
    {e_3: 2.0, -e_0: 5.0, -e_3: 2.0}
    """

    __slots__ = (
        "_algebra",
        "_loop",
        "_split",
        "_born",
        "_hash",
        "_native",
        "_lift",
        "_shown",
        "__weakref__",
    )

    _loop: np.ndarray | None
    _split: tuple[np.ndarray, np.ndarray] | None
    _born: str
    _hash: int | None
    _native: Any
    _lift: Any
    # The bytes of the coefficients that have been handed out. Each request
    # gets a new array over them, so that nothing a caller does to the shape
    # or type of one array reaches the next
    _shown: bytes | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject direct construction."""

        raise TypeError("create elements with a LoopAlgebra, not with Element()")

    @classmethod
    def _make(cls, algebra: Any, mode: str, data: Any) -> Element:
        """Create an element from raw data that the element will own."""

        self = object.__new__(cls)
        store = object.__setattr__
        store(self, "_algebra", algebra)
        store(self, "_born", mode)
        store(self, "_hash", None)
        store(self, "_native", None)
        store(self, "_lift", None)
        store(self, "_shown", None)
        if mode == "loop":
            if data.shape != algebra.coefficient_shape:
                raise ValueError("coefficients do not have the shape of one element")
            store(self, "_loop", _freeze(np.ascontiguousarray(data)))
            store(self, "_split", None)
        else:
            spectral, signed = data
            pair = (
                _freeze(np.ascontiguousarray(spectral)),
                _freeze(np.ascontiguousarray(signed)),
            )
            store(self, "_loop", None)
            store(self, "_split", pair)
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation."""

        raise AttributeError("Element objects are immutable")

    def __delattr__(self, name: str) -> None:
        """Reject deletion of immutable state."""

        raise AttributeError("Element objects are immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the algebra and the form the element was created with."""

        data: Any
        if self._born == "loop":
            data = _immutable(self._data("loop"))
        else:
            assert self._split is not None
            data = (_immutable(self._split[0]), _immutable(self._split[1]))
        return (_rebuild, (self._algebra, self._born, data))

    def __copy__(self) -> Element:
        """Return the element itself, which is immutable."""

        return self

    def __deepcopy__(self, memo: Any) -> Element:
        """Return an equal element with independent storage."""

        return self.copy()

    # Hooks

    def _mode(self, linear: bool, operands: tuple[Any, ...]) -> str:
        """Evaluate in the loop basis only when every operand was created in it."""

        # The choice depends on how the operands were created, never on what
        # they have cached, so an operation always rounds the same way. In
        # characteristic 2 there is no split form and every element is
        # created in the loop basis.
        if self._algebra._field.binary:
            return "loop"
        if linear and all(operand._born == "loop" for operand in operands):
            return "loop"
        return "split"

    def _run(
        self,
        implementation: _Implementation,
        operands: tuple[Any, ...],
        *,
        linear: bool = False,
        **options: Any,
    ) -> Any:
        """Evaluate a kernel routine; products in characteristic 2 reuse the lifts."""

        algebra = self._algebra
        if implementation is _kernel.multiply and algebra._field.binary:
            values = algebra._binary_multiply(operands, "auto", **options)
            return Element._make(algebra, "loop", values)
        return super()._run(implementation, operands, linear=linear, **options)

    def _data(self, mode: str) -> Any:
        """Return the requested form, deriving and keeping it on first use."""

        if mode == "lift":
            # The transformed lifts of the coefficients, kept for products in
            # characteristic 2 as the split form is kept elsewhere
            if self._lift is None:
                lifts = self._algebra._lift.split_limbs(self._data("loop"))
                frozen = [(_freeze(first), _freeze(second)) for first, second in lifts]
                object.__setattr__(self, "_lift", frozen)
            return self._lift
        if mode == "loop":
            if self._loop is None:
                assert self._split is not None
                coefficients = _kernel.from_split(self._algebra, self._split)
                object.__setattr__(self, "_loop", _freeze(coefficients))
            return self._loop
        if self._split is None:
            assert self._loop is not None
            spectral, signed = _kernel.to_split(self._algebra, self._loop)
            object.__setattr__(self, "_split", (_freeze(spectral), _freeze(signed)))
        return self._split

    def _result(self, mode: str, data: Any) -> Element:
        """Wrap raw data as an element."""

        return Element._make(self._algebra, mode, data)

    def _accepts(self, other: Any) -> bool:
        """Combine with elements only; arrays handle elements themselves."""

        return isinstance(other, Element)

    def _truth(self, marks: np.ndarray) -> bool:
        """Return a predicate result as a ``bool``."""

        return bool(marks)

    # Data

    @property
    def algebra(self) -> Any:
        """The parent loop algebra."""

        return self._algebra

    @property
    def dimension(self) -> int:
        """The dimension of the parent algebra."""

        dimension: int = self._algebra.dimension
        return dimension

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the coefficient array."""

        shape: tuple[int, ...] = self._algebra.coefficient_shape
        return shape

    @property
    def dtype(self) -> np.dtype[Any]:
        """The coefficient storage type."""

        dtype: np.dtype[Any] = self._algebra.dtype
        return dtype

    @property
    def coefficients(self) -> np.ndarray:
        """A read-only array of the loop-basis coefficients."""

        if self._shown is None:
            object.__setattr__(self, "_shown", self._data("loop").tobytes())
        assert self._shown is not None
        algebra = self._algebra
        return _over(self._shown, algebra.dtype, algebra.coefficient_shape)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Return the loop-basis coefficients to NumPy."""

        coefficients: np.ndarray = self._data("loop")
        if dtype is not None and np.dtype(dtype) != coefficients.dtype:
            if copy is False:
                raise ValueError(NO_COPY_MESSAGE)
            return coefficients.astype(dtype)
        return coefficients.copy() if copy else self.coefficients

    def to_numpy(self, *, copy: bool = True) -> np.ndarray:
        """Return the loop-basis coefficients; a read-only view with ``copy=False``."""

        if not isinstance(copy, bool):
            raise TypeError("copy must be a bool")
        return self._data("loop").copy() if copy else self.coefficients

    def copy(self) -> Element:
        """Return an equal element with independent storage."""

        data: Any
        if self._born == "loop":
            assert self._loop is not None
            data = self._loop.copy()
        else:
            assert self._split is not None
            data = (self._split[0].copy(), self._split[1].copy())
        return Element._make(self._algebra, self._born, data)

    def __pos__(self) -> Element:
        """Return the element itself."""

        return self

    def _location(self, x: Any) -> tuple[int, int]:
        """Return the row and index of a loop element's coefficient."""

        element = self._algebra.loop._own(x)
        return (0 if element.sign > 0 else 1), element.index

    def coefficient(self, x: Any) -> Any:
        """Return the coefficient of the basis element ``[x]``.

        The result is a ``float`` over R, a ``complex`` over C, and a
        scalar of the field over a finite field.
        """

        field = self._algebra._field
        return field.python(self._data("loop")[self._location(x)])

    def terms(self) -> dict[LoopElement, Any]:
        """Return the nonzero coefficients, keyed by loop element."""

        field = self._algebra._field
        coefficients = self._data("loop")
        loop = self._algebra.loop
        marks = field.nonzero(coefficients)
        result = {}
        for row, index in zip(*np.nonzero(marks), strict=True):
            element = LoopElement(loop, int(index), 1 if row == 0 else -1)
            result[element] = field.python(coefficients[row, index])
        return result

    def support(self) -> frozenset[LoopElement]:
        """Return the loop elements with a nonzero coefficient."""

        return frozenset(self.terms())

    def to_bytes(self) -> bytes:
        """Return a canonical encoding of the loop-basis coefficients.

        Coefficients are written in the order of ``coefficients.ravel()``.
        Over R and C each is a little-endian IEEE binary64 value, or two for
        a complex coefficient; over a finite field each residue is a
        big-endian unsigned integer of the smallest whole number of bytes
        that holds ``char - 1``. :meth:`LoopAlgebra.from_bytes` reads the
        encoding back.
        """

        result: bytes = self._algebra._encode(self._data("loop"))
        return result

    # Decomposition

    def kappa(self) -> Any:
        """Return the group component ``(a_t + b_t)``.

        The map to the group component is an algebra homomorphism onto the
        group algebra of the quotient of the loop by ``{+1, -1}``, in which
        products are dyadic convolutions. The result is an ndarray over R
        and C and in characteristic 2, and a pyfastcd array of field scalars
        over a finite field of odd characteristic.
        """

        algebra = self._algebra
        field = algebra._field
        if self._born == "loop":
            plus, minus = _kernel._rows(algebra, self._data("loop"))
            return field.view(field.add(plus, minus))
        spectral = self._data("split")[0]
        return field.view(field.divide_int(field.wht(spectral), algebra._size))

    def spectrum(self) -> Any:
        """Return the Walsh--Hadamard transform of the group component.

        Coordinate ``a`` is the value at this element of the character
        ``e_t -> (-1)**popcount(a & t)``, extended linearly. Products act
        coordinatewise on spectra. The result has the type of
        :meth:`kappa`. Not defined in characteristic 2, where the algebra
        has no decomposition.
        """

        self._algebra._require_split("spectrum()")
        return self._algebra._field.view(self._data("split")[0].copy())

    def epsilon(self) -> Any:
        """Return the signed component ``(a_t - b_t)``.

        The map to the signed component is an algebra homomorphism. For a
        Cayley--Dickson loop the result is the pyfastcd ``Element`` of the
        Cayley--Dickson algebra with those coefficients. For a central
        product it is a read-only coefficient array with one axis for each
        factor, followed by the polynomial axis of an extension field, read
        as an element of the tensor product of the factors' Cayley--Dickson
        algebras. Not defined in characteristic 2, where it would coincide
        with the group component.
        """

        algebra = self._algebra
        algebra._require_split("epsilon()")
        signed = self._data("split")[1]
        if algebra._signed.cd is not None:
            if self._native is None:
                object.__setattr__(self, "_native", algebra._signed.cd(signed))
            return self._native
        if self._native is None:
            object.__setattr__(self, "_native", signed.tobytes())
        shape = algebra._signed.sizes + algebra._field.shape
        return _over(self._native, algebra.dtype, shape)

    def augmentation(self) -> Any:
        """Return the sum of all coefficients, a scalar of the field.

        The augmentation is an algebra homomorphism to the field. Where the
        spectrum exists it is spectral coordinate 0.
        """

        algebra = self._algebra
        field = algebra._field
        if field.binary:
            return field.python(_kernel.augmentation(algebra, self._data("loop")))
        return field.python(self._data("split")[0][(0,) + (slice(None),) * field.tail])

    def minimal_polynomial(self) -> np.ndarray:
        """Return the monic polynomial of least degree that vanishes at the element.

        Defined for the loop algebra of a Cayley--Dickson loop, and for
        every loop in characteristic 2, where the powers of an element do
        not depend on bracketing. The result is an array of field
        coefficients, constant term first. Over R and C, coincidences
        between spectral values are detected by exact comparison of
        floating-point numbers. In characteristic 2 the polynomial is
        ``(t - aug(x))**j`` for some ``j`` from 1 to 4.
        """

        self._powers("minimal_polynomial()")
        algebra = self._algebra
        if algebra._field.binary:
            return _structure.binary_minimal_polynomial(algebra, self._data("loop"))
        return _structure.minimal_polynomial(algebra, self._data("split"))

    def _annihilator(self, side: str) -> Any:
        """Return a basis of one annihilator as an array."""

        algebra = self._algebra
        if algebra._field.binary:
            basis = _structure.binary_annihilator(algebra, self._data("loop"), side)
        else:
            basis = _structure.annihilator(algebra, self._data("split"), side)
        return algebra._wrap_array(basis)

    def left_annihilator(self) -> Any:
        """Return a basis of the elements ``y`` with ``y x = 0``, as an array."""

        return self._annihilator("left")

    def right_annihilator(self) -> Any:
        """Return a basis of the elements ``y`` with ``x y = 0``, as an array."""

        return self._annihilator("right")

    # Comparison

    def _scalar_value(self) -> Any:
        """Return the scalar this element equals, or ``None`` if it is not one."""

        field = self._algebra._field
        coefficients = self._data("loop")
        marks = field.nonzero(coefficients)
        marks[0, 0] = False
        if marks.any():
            return None
        return field.python(coefficients[0, 0])

    def __eq__(self, other: object) -> Any:
        """Compare loop-basis coefficients exactly.

        An element equals a field scalar ``r`` when it is ``r`` times the
        identity. Elements of different algebras are unequal. Over a finite
        field an integer is not reduced modulo the characteristic, so that
        equality agrees with hashing.
        """

        if isinstance(other, Element):
            if other._algebra is not self._algebra:
                return False
            return bool(np.array_equal(self._data("loop"), other._data("loop")))
        field = self._algebra._field
        if field.is_scalar(other):
            value = self._scalar_value()
            if value is None:
                return False
            if field.exact and not hasattr(other, "algebra"):
                assert isinstance(other, (int, np.integer))
                # Integers are not reduced, so equality agrees with hashing
                return bool(value == other) and 0 <= int(other) < field.char
            return bool(value == other)
        return NotImplemented

    def __hash__(self) -> int:
        """Hash consistently with equality, including equality with scalars."""

        if self._hash is None:
            value = self._scalar_value()
            if value is not None:
                result = hash(value)
            else:
                coefficients = self._data("loop")
                if not self._algebra._field.exact:
                    # Adding zero maps -0.0 to +0.0, which compare equal
                    coefficients = coefficients + 0.0
                result = hash((self._algebra, coefficients.tobytes()))
            object.__setattr__(self, "_hash", result)
        assert self._hash is not None
        return self._hash

    def __bool__(self) -> bool:
        """Return whether the element is nonzero."""

        if self._born == "loop":
            return bool(np.any(self._loop))
        assert self._split is not None
        return bool(np.any(self._split[0]) or np.any(self._split[1]))

    def __repr__(self) -> str:
        """Return the coefficients and the parent."""

        body = np.array2string(self._data("loop"), separator=", ", threshold=16)
        body = " ".join(body.split())
        return f"Element({body}, algebra={self._algebra!r})"


def _rebuild(algebra: Any, mode: str, data: Any) -> Element:
    """Recreate a pickled element, whose data is checked as storage is."""

    field = algebra._field
    if mode == "loop":
        values = _stored(field, np.array(data), algebra.coefficient_shape)
        return Element._make(algebra, mode, values)
    if field.binary:
        raise ValueError(NO_SPLIT_MESSAGE)
    record = (algebra._size,) + field.shape
    spectral, signed = (_stored(field, np.array(part), record) for part in data)
    if spectral.shape != record or signed.shape != record:
        raise ValueError(STORAGE_MESSAGE)
    return Element._make(algebra, mode, (spectral, signed))
