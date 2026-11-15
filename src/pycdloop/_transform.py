"""The Walsh--Hadamard transform and dyadic convolution."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core

# Stages of unreduced butterflies that keep residues below 2**31 inside int64
_UNREDUCED_STAGES = 30


def _butterflies(work: np.ndarray, modulus: int | None = None) -> None:
    """Apply the Walsh--Hadamard butterflies in place along the last axis.

    This is the reference evaluation of the transform, written with array
    operations so that it accepts any element type. The compiled kernel in
    ``_core.c`` performs the same additions and subtractions on each
    coordinate in the same order, and is used for ``float64``,
    ``complex128``, and residue arrays. The array must be C-contiguous, so
    that each stage can address the two halves of every block as views.
    With *modulus*, entries are reduced often enough to stay inside the
    integer range of the array.
    """

    # The transform H_N of length N = 2^n is evaluated in n stages. Stage j
    # splits the axis into blocks of length 2^(j+1) and replaces the two
    # halves (u, v) of every block by (u + v, u - v). Every output coordinate
    # is therefore a signed sum of all N inputs formed by a balanced tree of
    # depth n, which is why the transform needs no separate summation order
    # to keep floating-point error growth logarithmic in N.
    size = work.shape[-1]
    lead = work.shape[:-1]
    half = 1
    stage = 0
    while half < size:
        view = work.reshape(lead + (size // (2 * half), 2, half))
        upper = view[..., 0, :]
        lower = view[..., 1, :]
        total = upper + lower
        np.subtract(upper, lower, out=lower)
        upper[...] = total
        half *= 2
        stage += 1
        if modulus is not None and stage % _UNREDUCED_STAGES == 0:
            np.remainder(work, modulus, out=work)
    if modulus is not None:
        np.remainder(work, modulus, out=work)


def _is_power_of_two(size: int) -> bool:
    """Return whether *size* is a positive power of two."""

    return size > 0 and size & (size - 1) == 0


def _hadamard(values: np.ndarray) -> np.ndarray:
    """Return the transform of an array along its last axis.

    ``float64`` and ``complex128`` arrays go to the compiled kernel, a
    complex coefficient being two adjacent ``float64`` words. Any other
    element type is transformed by :func:`_butterflies`.
    """

    work = np.array(values, order="C", copy=True)
    if work.dtype == np.float64:
        if work.size:
            _core.transform(work, work.shape[-1], 1)
    elif work.dtype == np.complex128:
        if work.size:
            _core.transform(work, work.shape[-1], 2)
    else:
        _butterflies(work)
    return work


def _hadamard_mod(values: np.ndarray, modulus: int, lanes: int = 1) -> np.ndarray:
    """Return the transform of canonical residues.

    The transformed axis is the last one when *lanes* is 1. Otherwise the
    array has a trailing polynomial axis of length *lanes* and the
    transformed axis is the one before it.
    """

    work = np.array(values, dtype=np.int64, order="C", copy=True)
    if work.size:
        dimension = work.shape[-1] if lanes == 1 else work.shape[-2]
        _core.transform_mod(work, dimension, lanes, modulus)
    return work.astype(np.uint32)


def wht(a: Any, *, axis: int = -1, field: Any = None) -> np.ndarray:
    """Return the Walsh--Hadamard transform of *a* along one axis.

    The transform is ``H_N a`` with ``(H_N)[i, j] = (-1)**popcount(i & j)``.
    It is not normalized, so applying it twice multiplies by ``N``.

    Parameters
    ----------
    a : array_like
        Coefficients. Without *field* they are converted to ``float64``, or
        to ``complex128`` when complex. With *field* they are residues of
        that finite field, with a trailing polynomial axis of length
        ``field.deg`` for an extension field.
    axis : int, optional
        The transformed axis, whose length must be a power of two. For an
        extension field the trailing polynomial axis is not counted, so the
        default selects the last axis before it.
    field : pyfastcd.FiniteField, optional
        The coefficient field of exact residues, of odd characteristic.

    Returns
    -------
    numpy.ndarray
        A new array with the shape of the input: ``float64`` or
        ``complex128`` without *field*, canonical ``uint32`` residues with
        it.

    Raises
    ------
    ValueError
        If the length of the transformed axis is not a power of two.
    TypeError
        For a :class:`BinaryField`: modulo 2 every entry of the transform
        matrix is 1, so beyond length 1 the transform is not invertible.

    Examples
    --------
    >>> import pycdloop as cdl
    >>> cdl.wht([1, 0, 0, 0])
    array([1., 1., 1., 1.])
    >>> cdl.wht([1, 2, 3, 4])
    array([10., -2., -4.,  0.])
    """

    # Imported here because the coefficient-field module uses the routines above
    from ._field import _coefficient_field

    domain = _coefficient_field(field=field) if field is not None else None
    if domain is not None and domain.binary:
        raise TypeError(
            "the Walsh--Hadamard transform is not defined in characteristic 2"
        )
    if domain is None:
        values = np.asarray(a)
        if values.dtype.kind not in "biufc":
            raise TypeError("wht requires numeric coefficients")
        kind = np.complex128 if values.dtype.kind == "c" else np.float64
        values = values.astype(kind)
        tail = 0
    else:
        values = domain.coerce(a)
        tail = len(domain.shape)
    index_ndim = values.ndim - tail
    if index_ndim < 1:
        raise ValueError("wht requires at least one coefficient axis")
    if not -index_ndim <= axis < index_ndim:
        raise ValueError("axis is out of range")
    position = axis % index_ndim
    if not _is_power_of_two(values.shape[position]):
        raise ValueError("the transformed axis must have power-of-two length")
    moved = np.moveaxis(values, position, -1)
    if domain is None:
        result = _hadamard(moved)
    else:
        result = _hadamard_mod(moved, domain.char)
    return np.ascontiguousarray(np.moveaxis(result, -1, position))


def dyadic_convolve(a: Any, b: Any, *, field: Any = None) -> np.ndarray:
    """Return the dyadic convolution of *a* and *b* along the last axis.

    The result is ``c[k] = sum(a[i] * b[i ^ k] for i in range(N))``, the
    product of the two inputs in the group algebra of the elementary abelian
    2-group of order ``N``. It is computed from three Walsh--Hadamard
    transforms and ``N`` coordinatewise products.

    In characteristic 2 the transform cannot be used, and the convolution
    is computed on coefficients read as integers over a large prime field,
    or over an extension of it when the field is not F_2, and then reduced.

    Parameters
    ----------
    a, b : array_like
        Operands of shape ``(..., N)``, or ``(..., N, field.deg)`` over an
        extension field, with ``N`` a power of two. Leading axes broadcast.
    field : pyfastcd.FiniteField or BinaryField, optional
        The coefficient field of exact residues. Without it the operands are
        real or complex floating-point coefficients.

    Returns
    -------
    numpy.ndarray
        The convolution, with the broadcast shape of the operands.

    Examples
    --------
    >>> import pycdloop as cdl
    >>> cdl.dyadic_convolve([1, 2, 0, 0], [0, 1, 0, 0])
    array([2., 1., 0., 0.])
    """

    from ._field import _coefficient_field

    if field is not None:
        domain = _coefficient_field(field=field)
        x, y = domain.coerce(a), domain.coerce(b)
        if domain.binary:
            return _binary_convolve(domain, x, y)
    else:
        x, y = np.asarray(a), np.asarray(b)
        for values in (x, y):
            if values.dtype.kind not in "biufc":
                raise TypeError("dyadic_convolve requires numeric coefficients")
        is_complex = x.dtype.kind == "c" or y.dtype.kind == "c"
        domain = _coefficient_field(char=0, deg=2 if is_complex else 1)
        x, y = x.astype(domain.dtype), y.astype(domain.dtype)
    tail = len(domain.shape)
    if x.ndim <= tail or y.ndim <= tail:
        raise ValueError("dyadic_convolve requires a coefficient axis")
    size = x.shape[x.ndim - tail - 1]
    if size != y.shape[y.ndim - tail - 1] or not _is_power_of_two(size):
        raise ValueError("operands must have the same power-of-two length")
    product = domain.multiply(domain.wht(x), domain.wht(y))
    return domain.divide_int(domain.wht(product), size)


def _binary_convolve(domain: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a dyadic convolution in characteristic 2, by lifting."""

    # Imported here because the lifting module builds on the kernel layer
    from ._field import _coefficient_field
    from ._lift import _Codec

    tail = domain.tail
    size = x.shape[x.ndim - tail - 1]
    if size != y.shape[y.ndim - tail - 1] or not _is_power_of_two(size):
        raise ValueError("operands must have the same power-of-two length")
    codec = _Codec(domain)
    codec.check(size)
    container = _coefficient_field(field=codec.container)
    products = []
    for first, second in zip(codec.limbs(x), codec.limbs(y), strict=True):
        spectrum = container.multiply(container.wht(first), container.wht(second))
        products.append(container.divide_int(container.wht(spectrum), size))
    return codec.combine(products)
