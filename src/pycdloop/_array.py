"""NumPy-backed batches of loop-algebra elements in either coordinate system."""

from __future__ import annotations

from typing import Any, final

import numpy as np

from . import _kernel
from ._element import Element, _Arithmetic
from ._validation import (
    ARRAY_TRUTH_MESSAGE,
    MIXED_BASIS_MESSAGE,
    NO_COPY_MESSAGE,
    STORAGE_MESSAGE,
    _stored,
)


def _check(algebra: Any, values: np.ndarray, record: tuple[int, ...]) -> np.ndarray:
    """Return the storage of an array after checking what it holds.

    Storage that has been handed out is writable from outside the array, so
    its dtype, its shape and, over a finite field, its residues are checked
    again wherever an operation reads it. Storage that only the package has
    held needs no check, and an array records which kind it has in a flag
    that it shares with every array that shares its storage.
    """

    return _stored(algebra._field, values, record)


def _batch_key(key: Any, batch_ndim: int, record_ndim: int) -> tuple[Any, ...]:
    """Restrict an index to the batch axes of an array.

    An index that consumes more axes than the batch has is rejected. An
    ellipsis is kept, because where it stands decides how NumPy arranges
    the axes of advanced indices, and is followed by one full slice for
    each coefficient axis so that it cannot reach them.
    """

    items = key if isinstance(key, tuple) else (key,)
    consumed = 0
    for item in items:
        if item is None or item is Ellipsis:
            continue
        if isinstance(item, (bool, np.bool_)):
            # Like None, a Boolean scalar adds an axis and takes none
            continue
        if isinstance(item, (slice, int, np.integer)):
            consumed += 1
            continue
        # Anything else is read by NumPy as an array: a list, a tuple
        # inside the index, an array or an object that converts to one. A
        # Boolean array takes one axis for each of its own
        try:
            mask = np.asarray(item)
        except (TypeError, ValueError):
            consumed += 1
            continue
        consumed += mask.ndim if mask.dtype == np.bool_ else 1
    if consumed > batch_ndim:
        raise IndexError("too many indices for the batch axes of an array of elements")
    if sum(item is Ellipsis for item in items) > 1:
        raise IndexError("an index can only have a single ellipsis")
    if any(item is Ellipsis for item in items):
        return tuple(items) + (slice(None),) * record_ndim
    return tuple(items)


class _Batch(_Arithmetic):
    """Behaviour common to both array types."""

    __slots__ = ()

    def _truth(self, marks: np.ndarray) -> np.ndarray:
        """Return a predicate result as a Boolean array over the batch."""

        return marks

    def __bool__(self) -> bool:
        """Reject the ambiguous truth value of an array."""

        raise ValueError(ARRAY_TRUTH_MESSAGE)

    __hash__ = None  # type: ignore[assignment]

    @property
    def algebra(self) -> Any:
        """The parent loop algebra."""

        return self._algebra

    @property
    def dtype(self) -> np.dtype[Any]:
        """The coefficient storage type."""

        dtype: np.dtype[Any] = self._algebra.dtype
        return dtype

    @property
    def dimension(self) -> int:
        """The dimension of the parent algebra."""

        dimension: int = self._algebra.dimension
        return dimension

    def __len__(self) -> int:
        """Return the length of the first batch axis."""

        shape = self.batch_shape
        if not shape:
            raise TypeError("an array with an empty batch shape has no length")
        return int(shape[0])

    def __iter__(self) -> Any:
        """Iterate over the first batch axis."""

        for index in range(len(self)):
            yield self[index]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The leading axes that index the elements of the array."""

        raise NotImplementedError

    def __getitem__(self, key: Any) -> Any:
        """Select elements along the batch axes."""

        raise NotImplementedError

    def __ne__(self, other: object) -> Any:
        """Return the negation of ``==`` over the batch."""

        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return ~result


@final
class ElementArray(_Batch):
    """A batch of loop-algebra elements stored by loop-basis coefficients.

    Create arrays with :meth:`LoopAlgebra.array` and
    :meth:`LoopAlgebra.asarray`. Constructing the class directly raises
    ``TypeError``.

    The coefficient storage has shape ``batch_shape + coefficient_shape``:
    after the batch axes come the axis of length 2 that separates the
    coefficients of ``[e_t]`` from those of ``[-e_t]``, the axis over ``t``,
    and over an extension field the polynomial axis. Only the batch axes
    broadcast.

    Attributes
    ----------
    algebra : LoopAlgebra
        The parent algebra.
    shape : tuple[int, ...]
        ``batch_shape + algebra.coefficient_shape``.
    batch_shape : tuple[int, ...]
        The leading axes.
    dtype : numpy.dtype
        The coefficient storage type.
    coefficients : numpy.ndarray
        The coefficient storage itself, not a copy.

    Notes
    -----
    Arrays are writable when their storage is, and are not hashable. They
    combine with arrays and elements of the same algebra and with scalars of
    the coefficient field; every operation returns a new array. A product
    transforms both operands to split form and the result back. To run
    several operations without those transforms, convert once with
    :meth:`split` and work with the resulting :class:`SplitArray`. Combining
    an ``ElementArray`` with a ``SplitArray`` raises ``TypeError``. In
    characteristic 2 there is no split form, and ``ElementArray`` is the
    only array type.

    Indexing and iteration act on the batch axes: ``X[i]`` is an
    :class:`Element` copy when it selects one record, and otherwise an
    array that shares storage where NumPy's indexing does.

    Examples
    --------
    >>> import numpy as np
    >>> import pycdloop as cdl
    >>> A = cdl.LoopAlgebra(cdl.CDLoop(2))
    >>> X = A.asarray(np.ones((5, 2, 4)))
    >>> (X * X).batch_shape
    (5,)
    >>> X[0] == A(np.ones((2, 4)))
    True
    """

    __slots__ = ("_algebra", "_values", "_shown", "__weakref__")

    _values: np.ndarray
    # One flag for all arrays over the same storage: whether it has been
    # handed out, and so must be checked before it is read
    _shown: list[bool]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject direct construction."""

        raise TypeError(
            "create arrays with LoopAlgebra.array() or LoopAlgebra.asarray()"
        )

    @classmethod
    def _make(
        cls, algebra: Any, values: np.ndarray, shown: list[bool] | None = None
    ) -> ElementArray:
        """Wrap coefficient storage without copying it.

        *shown* is the flag of the array whose storage this one shares; a
        new flag is made for storage that nothing else holds.
        """

        record = algebra.coefficient_shape
        if (
            values.ndim < len(record)
            or values.shape[values.ndim - len(record) :] != record
        ):
            raise ValueError(
                f"coefficients must have shape batch_shape + {record}, "
                f"not {values.shape}"
            )
        self = object.__new__(cls)
        self._algebra = algebra
        self._values = values
        self._shown = [False] if shown is None else shown
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the algebra and a copy of the coefficients."""

        return (_rebuild_array, (self._algebra, np.array(self._checked())))

    def __copy__(self) -> ElementArray:
        """Return an array with independent storage."""

        return self.copy()

    def __deepcopy__(self, memo: Any) -> ElementArray:
        """Return an array with independent storage."""

        return self.copy()

    # Hooks

    def _mode(self, linear: bool, operands: tuple[Any, ...]) -> str:
        """Add in the loop basis and multiply in split form.

        In characteristic 2 there is no split form, and everything is
        evaluated in the loop basis.
        """

        if self._algebra._field.binary:
            return "loop"
        return "loop" if linear else "split"

    def _data(self, mode: str) -> Any:
        """Return the coefficients, or a temporary split form."""

        if mode == "loop":
            return self._checked()
        return _kernel.to_split(self._algebra, self._checked())

    def _checked(self) -> np.ndarray:
        """Return the storage, checked if it has been handed out."""

        if self._shown[0]:
            algebra = self._algebra
            return _check(algebra, self._values, algebra.coefficient_shape)
        return self._values

    def _expose(self) -> np.ndarray:
        """Hand out the storage, which every later operation then checks."""

        self._shown[0] = True
        return self._values

    def _result(self, mode: str, data: Any) -> ElementArray:
        """Wrap a raw result, returning it to the loop basis."""

        if mode == "split":
            data = _kernel.from_split(self._algebra, data)
        return ElementArray._make(self._algebra, data)

    def _run(
        self,
        implementation: Any,
        operands: tuple[Any, ...],
        *,
        linear: bool = False,
        **options: Any,
    ) -> Any:
        """Evaluate a kernel routine; products reuse the thread's workspace."""

        algebra = self._algebra
        if implementation is _kernel.multiply and algebra._field.binary:
            values = algebra._binary_multiply(operands, "auto", **options)
            return ElementArray._make(algebra, values)
        if implementation is _kernel.multiply and algebra._field.native:
            # An array is split into the workspace; an element contributes
            # the split form it already keeps
            pairs = [
                ("loop", operand._checked())
                if isinstance(operand, ElementArray)
                else ("split", operand._data("split"))
                for operand in operands
            ]
            values = _kernel.loop_product(algebra, pairs[0], pairs[1], **options)
            return ElementArray._make(algebra, values)
        return super()._run(implementation, operands, linear=linear, **options)

    def _accepts(self, other: Any) -> bool:
        """Combine with elements and loop-basis arrays."""

        if isinstance(other, SplitArray):
            raise TypeError(MIXED_BASIS_MESSAGE)
        return isinstance(other, (Element, ElementArray))

    # Data

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the coefficient storage."""

        return tuple(self._values.shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The leading axes that index the elements of the array."""

        record = len(self._algebra.coefficient_shape)
        return tuple(self._values.shape[: self._values.ndim - record])

    @property
    def coefficients(self) -> np.ndarray:
        """The coefficient storage, shared with the array."""

        return self._expose()

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Return the coefficient storage to NumPy."""

        if dtype is not None and np.dtype(dtype) != self._values.dtype:
            if copy is False:
                raise ValueError(NO_COPY_MESSAGE)
            return self._values.astype(dtype)
        return self._values.copy() if copy else self._expose()

    def to_numpy(self, *, copy: bool = True) -> np.ndarray:
        """Return the coefficients; the storage itself with ``copy=False``."""

        if not isinstance(copy, bool):
            raise TypeError("copy must be a bool")
        return self._values.copy() if copy else self._expose()

    def copy(self) -> ElementArray:
        """Return an array with independent storage."""

        return ElementArray._make(self._algebra, self._checked().copy())

    def __pos__(self) -> ElementArray:
        """Return a copy."""

        return self.copy()

    def split(self) -> SplitArray:
        """Return the array in split form.

        The result holds the Walsh--Hadamard transform of the group
        components and the signed components. Operations on it need no
        transforms; :meth:`SplitArray.to_loop` converts back. Not defined in
        characteristic 2, where the algebra has no split form.
        """

        self._algebra._require_split("split()")
        spectral, signed = _kernel.to_split(self._algebra, self._checked())
        return SplitArray._make(self._algebra, spectral, signed)

    # Indexing

    def __getitem__(self, key: Any) -> Any:
        """Select along the batch axes.

        One selected record is returned as an :class:`Element` copy; any
        other selection is an array that shares storage where NumPy's
        indexing does.
        """

        algebra = self._algebra
        record_ndim = len(algebra.coefficient_shape)
        selected = self._values[_batch_key(key, len(self.batch_shape), record_ndim)]
        if selected.shape == algebra.coefficient_shape:
            if self._shown[0]:
                _check(algebra, selected, algebra.coefficient_shape)
            return Element._make(algebra, "loop", selected.copy())
        return ElementArray._make(algebra, selected, self._shown)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Assign a field scalar, an element, an array or coefficients."""

        algebra = self._algebra
        field = algebra._field
        if isinstance(value, SplitArray):
            raise TypeError(MIXED_BASIS_MESSAGE)
        if field.is_scalar(value):
            data = algebra.scalar(value).coefficients
        elif isinstance(value, (Element, ElementArray)):
            if value._algebra is not algebra:
                raise ValueError("operands belong to different loop algebras")
            data = value._data("loop")
        else:
            data = field.coerce(value)
        record_ndim = len(algebra.coefficient_shape)
        self._values[_batch_key(key, len(self.batch_shape), record_ndim)] = data

    # Decomposition

    def kappa(self) -> Any:
        """Return the group components ``(a_t + b_t)`` of every record."""

        field = self._algebra._field
        plus, minus = _kernel._rows(self._algebra, self._checked())
        return field.view(field.add(plus, minus))

    def spectrum(self) -> Any:
        """Return the Walsh--Hadamard transforms of the group components.

        Not defined in characteristic 2.
        """

        self._algebra._require_split("spectrum()")
        field = self._algebra._field
        return field.view(_kernel.to_split(self._algebra, self._checked())[0])

    def epsilon(self) -> Any:
        """Return the signed components ``(a_t - b_t)`` of every record.

        The result is a ``pyfastcd.ElementArray`` for a Cayley--Dickson
        loop, and a coefficient array of shape ``batch_shape + (D,)`` for a
        central product, with the polynomial axis of an extension field
        appended. Not defined in characteristic 2.
        """

        self._algebra._require_split("epsilon()")
        signed = _kernel.to_split(self._algebra, self._checked())[1]
        return self._algebra._signed_view(signed)

    # Comparison

    def __eq__(self, other: object) -> Any:
        """Compare records exactly, giving a Boolean array over the batch.

        A record equals a field scalar ``r`` when it is ``r`` times the
        identity. An array or element of another algebra is unequal
        everywhere. Over a finite field an integer is not reduced modulo
        the characteristic.
        """

        algebra = self._algebra
        field = algebra._field
        if isinstance(other, SplitArray):
            raise TypeError(MIXED_BASIS_MESSAGE)
        if isinstance(other, (Element, ElementArray)):
            if other._algebra is not algebra:
                partner = getattr(other, "batch_shape", ())
                shape = np.broadcast_shapes(self.batch_shape, partner)
                return np.zeros(shape, dtype=bool)
            data = other._data("loop")
        elif field.is_scalar(other):
            if field.exact and not hasattr(other, "algebra"):
                assert isinstance(other, (int, np.integer))
                if not 0 <= int(other) < field.char:
                    return np.zeros(self.batch_shape, dtype=bool)
            data = algebra.scalar(other).coefficients
        else:
            return NotImplemented
        return field.equal(self._checked(), data).all(axis=(-2, -1))

    def __repr__(self) -> str:
        """Return the batch shape and the parent."""

        return (
            f"ElementArray(batch_shape={self.batch_shape}, algebra={self._algebra!r})"
        )


@final
class SplitArray(_Batch):
    """A batch of loop-algebra elements stored in split form.

    Create one with :meth:`ElementArray.split`, or from components with
    :meth:`LoopAlgebra.split_array`. Constructing the class directly raises
    ``TypeError``.

    The split form of an element with loop-basis coefficients ``a_t`` of
    ``[e_t]`` and ``b_t`` of ``[-e_t]`` is the pair of the Walsh--Hadamard
    transform of ``(a_t + b_t)`` and the array ``(a_t - b_t)``. Addition,
    scaling and multiplication act on the two parts separately, and a
    product needs no transform: it is one coordinatewise product and one
    product of signed components.

    Attributes
    ----------
    algebra : LoopAlgebra
        The parent algebra.
    batch_shape : tuple[int, ...]
        The leading axes.
    spectral
        The transformed group components, of shape ``batch_shape + (D,)``:
        an ndarray over R and C, a pyfastcd array of field scalars over a
        finite field. It shares the array's storage.
    signed
        The signed components: a ``pyfastcd.ElementArray`` for a
        Cayley--Dickson loop, an ndarray of shape ``batch_shape + (D,)``
        for a central product. It shares the array's storage.

    Notes
    -----
    A ``SplitArray`` combines with other ``SplitArray`` objects and elements
    of the same algebra and with scalars of the coefficient field, and
    returns ``SplitArray`` results. Combining it with an
    :class:`ElementArray` raises ``TypeError``. Indexing and iteration act
    on the batch axes; one selected record is returned as an
    :class:`Element`.

    Examples
    --------
    >>> import numpy as np
    >>> import pycdloop as cdl
    >>> A = cdl.LoopAlgebra(cdl.CDLoop(3))
    >>> X = A.random(0, batch_shape=(4,))
    >>> S = X.split()
    >>> T = (S * S + S) * S
    >>> bool(np.allclose(T.to_loop().coefficients, ((X * X + X) * X).coefficients))
    True
    """

    __slots__ = ("_algebra", "_spectral", "_signed", "_shown", "__weakref__")

    _spectral: np.ndarray
    _signed: np.ndarray
    _shown: list[bool]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject direct construction."""

        raise TypeError("create split arrays with ElementArray.split()")

    @classmethod
    def _make(
        cls,
        algebra: Any,
        spectral: np.ndarray,
        signed: np.ndarray,
        shown: list[bool] | None = None,
    ) -> SplitArray:
        """Wrap the two parts, copying them only to broadcast them together.

        *shown* is the flag of the array whose storage this one shares.
        """

        record = (algebra._size,) + algebra._field.shape
        for part in (spectral, signed):
            if (
                part.ndim < len(record)
                or part.shape[part.ndim - len(record) :] != record
            ):
                raise ValueError(f"each part must have shape batch_shape + {record}")
        if spectral.shape != signed.shape:
            shape = np.broadcast_shapes(spectral.shape, signed.shape)
            spectral = np.array(np.broadcast_to(spectral, shape))
            signed = np.array(np.broadcast_to(signed, shape))
        self = object.__new__(cls)
        self._algebra = algebra
        self._spectral = spectral
        self._signed = signed
        self._shown = [False] if shown is None else shown
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the algebra and copies of both parts."""

        spectral, signed = self._checked()
        return (_rebuild_split, (self._algebra, np.array(spectral), np.array(signed)))

    def __copy__(self) -> SplitArray:
        """Return an array with independent storage."""

        return self.copy()

    def __deepcopy__(self, memo: Any) -> SplitArray:
        """Return an array with independent storage."""

        return self.copy()

    # Hooks

    def _mode(self, linear: bool, operands: tuple[Any, ...]) -> str:
        """Evaluate everything in split form."""

        return "split"

    def _data(self, mode: str) -> Any:
        """Return the two parts."""

        if mode != "split":
            raise TypeError(MIXED_BASIS_MESSAGE)
        return self._checked()

    def _checked(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the two parts, checked if they have been handed out."""

        if not self._shown[0]:
            return self._spectral, self._signed
        algebra = self._algebra
        record = (algebra._size,) + algebra._field.shape
        spectral = _check(algebra, self._spectral, record)
        signed = _check(algebra, self._signed, record)
        if spectral.shape != signed.shape:
            raise ValueError(STORAGE_MESSAGE)
        return spectral, signed

    def _result(self, mode: str, data: Any) -> SplitArray:
        """Wrap a raw result."""

        return SplitArray._make(self._algebra, *data)

    def _accepts(self, other: Any) -> bool:
        """Combine with elements and split arrays."""

        if isinstance(other, ElementArray):
            raise TypeError(MIXED_BASIS_MESSAGE)
        return isinstance(other, (Element, SplitArray))

    # Data

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The leading axes that index the elements of the array."""

        record = 1 + self._algebra._field.tail
        return tuple(self._spectral.shape[: self._spectral.ndim - record])

    @property
    def spectral(self) -> Any:
        """The transformed group components, sharing the array's storage."""

        self._shown[0] = True
        return self._algebra._field.view(self._spectral)

    @property
    def signed(self) -> Any:
        """The signed components, sharing the array's storage."""

        self._shown[0] = True
        return self._algebra._signed_view(self._signed)

    def copy(self) -> SplitArray:
        """Return an array with independent storage."""

        spectral, signed = self._checked()
        return SplitArray._make(self._algebra, spectral.copy(), signed.copy())

    def __pos__(self) -> SplitArray:
        """Return a copy."""

        return self.copy()

    def to_loop(self) -> ElementArray:
        """Return the array in the loop basis."""

        values = _kernel.from_split(self._algebra, self._checked())
        return ElementArray._make(self._algebra, values)

    def weight(self) -> Any:
        """Return the number of nonzero loop-basis coefficients of every record."""

        return self.to_loop().weight()

    def distance(self, other: Any) -> Any:
        """Return the number of loop-basis coefficients that differ from *other*."""

        partner = self._operand(other)
        if isinstance(partner, SplitArray):
            partner = partner.to_loop()
        return self.to_loop().distance(partner)

    def kappa(self) -> Any:
        """Return the group components ``(a_t + b_t)`` of every record."""

        field = self._algebra._field
        size = self._algebra._size
        return field.view(field.divide_int(field.wht(self._checked()[0]), size))

    def spectrum(self) -> Any:
        """Return a copy of the transformed group components."""

        return self._algebra._field.view(self._checked()[0].copy())

    def epsilon(self) -> Any:
        """Return a copy of the signed components, typed as :attr:`signed`."""

        return self._algebra._signed_view(self._checked()[1].copy())

    # Indexing

    def __getitem__(self, key: Any) -> Any:
        """Select along the batch axes.

        One selected record is returned as an :class:`Element` copy; any
        other selection is an array that shares storage where NumPy's
        indexing does.
        """

        algebra = self._algebra
        index = _batch_key(key, len(self.batch_shape), 1 + algebra._field.tail)
        spectral, signed = self._spectral[index], self._signed[index]
        record = (algebra._size,) + algebra._field.shape
        if spectral.shape == record and signed.shape == record:
            if self._shown[0]:
                _check(algebra, spectral, record)
                _check(algebra, signed, record)
            return Element._make(algebra, "split", (spectral.copy(), signed.copy()))
        return SplitArray._make(algebra, spectral, signed, self._shown)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Assign a field scalar, an element or a split array."""

        algebra = self._algebra
        if isinstance(value, ElementArray):
            raise TypeError(MIXED_BASIS_MESSAGE)
        if algebra._field.is_scalar(value):
            value = algebra.scalar(value)
        if not isinstance(value, (Element, SplitArray)):
            raise TypeError("assign a field scalar, an element or a split array")
        if value._algebra is not algebra:
            raise ValueError("operands belong to different loop algebras")
        spectral, signed = value._data("split")
        index = _batch_key(key, len(self.batch_shape), 1 + algebra._field.tail)
        self._spectral[index] = spectral
        self._signed[index] = signed

    # Comparison

    def __eq__(self, other: object) -> Any:
        """Compare records exactly in split form, giving a Boolean array.

        A record equals a field scalar ``r`` when it is ``r`` times the
        identity. An array or element of another algebra is unequal
        everywhere. Over a finite field an integer is not reduced modulo
        the characteristic.
        """

        algebra = self._algebra
        field = algebra._field
        if isinstance(other, ElementArray):
            raise TypeError(MIXED_BASIS_MESSAGE)
        if field.is_scalar(other):
            if field.exact and not hasattr(other, "algebra"):
                assert isinstance(other, (int, np.integer))
                if not 0 <= int(other) < field.char:
                    return np.zeros(self.batch_shape, dtype=bool)
            other = algebra.scalar(other)
        if not isinstance(other, (Element, SplitArray)):
            return NotImplemented
        if other._algebra is not algebra:
            partner = getattr(other, "batch_shape", ())
            shape = np.broadcast_shapes(self.batch_shape, partner)
            return np.zeros(shape, dtype=bool)
        spectral, signed = other._data("split")
        mine = self._checked()
        marks = field.equal(mine[0], spectral).all(axis=-1)
        return marks & field.equal(mine[1], signed).all(axis=-1)

    def __repr__(self) -> str:
        """Return the batch shape and the parent."""

        return f"SplitArray(batch_shape={self.batch_shape}, algebra={self._algebra!r})"


def _rebuild_array(algebra: Any, values: np.ndarray) -> ElementArray:
    """Recreate a pickled loop-basis array, whose data is checked as storage is."""

    return ElementArray._make(
        algebra, _check(algebra, values, algebra.coefficient_shape)
    )


def _rebuild_split(
    algebra: Any, spectral: np.ndarray, signed: np.ndarray
) -> SplitArray:
    """Recreate a pickled split array, whose data is checked as storage is."""

    record = (algebra._size,) + algebra._field.shape
    parts = _check(algebra, spectral, record), _check(algebra, signed, record)
    if parts[0].shape != parts[1].shape:
        raise ValueError(STORAGE_MESSAGE)
    return SplitArray._make(algebra, *parts)
