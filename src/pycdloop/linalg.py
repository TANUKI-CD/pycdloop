"""Rank, row reduction and null spaces of matrices over a coefficient field.

Without ``field`` a matrix is real or complex and the results are numerical:
ranks and null spaces come from the singular value decomposition, with a
tolerance. With a finite field, a ``pyfastcd.FiniteField`` or a
:class:`~pycdloop.BinaryField`, the entries are residues, with a trailing
polynomial axis for an extension field, and every result is exact.

Examples
--------
>>> import numpy as np
>>> import pyfastcd as fcd
>>> from pycdloop import linalg
>>> M = np.array([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
>>> linalg.rank(M, field=fcd.FiniteField(char=5))
2
>>> linalg.nullspace(M, field=fcd.FiniteField(char=5))
array([[4, 4, 1]], dtype=uint32)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._field import _coefficient_field, _CoefficientField

__all__ = ["nullspace", "rank", "row_reduce"]


def _prepare(matrix: Any, field: Any) -> tuple[_CoefficientField, np.ndarray]:
    """Return the field arithmetic and the matrix in its storage type."""

    if field is not None:
        domain = _coefficient_field(field=field)
        values = domain.coerce(matrix)
    else:
        values = np.asarray(matrix)
        if values.dtype.kind not in "biufc":
            raise TypeError("a matrix must be numeric")
        domain = _coefficient_field(char=0, deg=2 if values.dtype.kind == "c" else 1)
        values = values.astype(domain.dtype)
    if values.ndim != 2 + domain.tail:
        raise ValueError("a matrix must have two index axes")
    return domain, values


def _tolerance(singular: np.ndarray, shape: tuple[int, ...], tol: Any) -> float:
    """Return the threshold below which a singular value counts as zero."""

    if tol is not None:
        return float(tol)
    largest = float(singular.max()) if singular.size else 0.0
    return largest * max(shape) * float(np.finfo(np.float64).eps)


def _reduce(
    domain: _CoefficientField, matrix: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Return the reduced row echelon form over an exact field, with its pivots."""

    # Gauss--Jordan elimination, one pivot column at a time. Clearing a
    # column subtracts a multiple of the pivot row from every other row, and
    # that update is a single array operation over the whole matrix.
    work = matrix.copy()
    rows, columns = work.shape[0], work.shape[1]
    pivots: list[int] = []
    row = 0
    for column in range(columns):
        if row == rows:
            break
        candidates = np.flatnonzero(domain.nonzero(work[row:, column]))
        if candidates.size == 0:
            continue
        chosen = row + int(candidates[0])
        if chosen != row:
            work[[row, chosen]] = work[[chosen, row]]
        work[row] = domain.scale(work[row], domain.reciprocal(work[row, column]))
        factors = work[:, column].copy()
        factors[row] = 0
        work = domain.subtract(work, domain.multiply(factors[:, None], work[row][None]))
        pivots.append(column)
        row += 1
    return work, pivots


def _reduce_float(matrix: np.ndarray, tol: Any) -> tuple[np.ndarray, list[int]]:
    """Return the reduced row echelon form of a real or complex matrix."""

    work = matrix.copy()
    rows, columns = work.shape
    scale = float(np.abs(work).max()) if work.size else 0.0
    threshold = (
        float(tol)
        if tol is not None
        else scale * max(rows, columns) * float(np.finfo(np.float64).eps)
    )
    pivots: list[int] = []
    row = 0
    for column in range(columns):
        if row == rows:
            break
        # Partial pivoting: take the largest remaining entry of the column
        chosen = row + int(np.argmax(np.abs(work[row:, column])))
        if abs(work[chosen, column]) <= threshold:
            work[row:, column] = 0
            continue
        if chosen != row:
            work[[row, chosen]] = work[[chosen, row]]
        work[row] = work[row] / work[row, column]
        factors = work[:, column].copy()
        factors[row] = 0
        work -= factors[:, None] * work[row][None, :]
        pivots.append(column)
        row += 1
    return work, pivots


def rank(matrix: Any, *, field: Any = None, tol: Any = None) -> int:
    """Return the rank of a matrix.

    Parameters
    ----------
    matrix : array_like
        A matrix of shape ``(rows, columns)``, or
        ``(rows, columns, field.deg)`` over an extension field.
    field : pyfastcd.FiniteField or BinaryField, optional
        The field of exact residues. Without it the matrix is real or
        complex.
    tol : float, optional
        Singular values at or below this count as zero. The default is the
        largest singular value times ``max(rows, columns)`` times the
        machine epsilon. Ignored over a finite field.

    Returns
    -------
    int
        The rank.
    """

    domain, values = _prepare(matrix, field)
    if domain.exact:
        return len(_reduce(domain, values)[1])
    if values.size == 0:
        return 0
    singular = np.linalg.svd(values, compute_uv=False)
    return int((singular > _tolerance(singular, values.shape, tol)).sum())


def row_reduce(
    matrix: Any, *, field: Any = None, tol: Any = None
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return the reduced row echelon form of a matrix and its pivot columns.

    Parameters
    ----------
    matrix : array_like
        A matrix, as for :func:`rank`.
    field : pyfastcd.FiniteField or BinaryField, optional
        The field of exact residues.
    tol : float, optional
        Over R and C, entries of a pivot column at or below this magnitude
        count as zero. The default is the largest entry of the matrix times
        ``max(rows, columns)`` times the machine epsilon.

    Returns
    -------
    reduced : numpy.ndarray
        The reduced row echelon form, with the shape of the input. Its
        nonzero rows come first and form a basis of the row space.
    pivots : tuple[int, ...]
        The pivot column of each nonzero row.
    """

    domain, values = _prepare(matrix, field)
    if domain.exact:
        reduced, pivots = _reduce(domain, values)
    else:
        reduced, pivots = _reduce_float(values, tol)
    return reduced, tuple(pivots)


def nullspace(matrix: Any, *, field: Any = None, tol: Any = None) -> np.ndarray:
    """Return a basis of the vectors ``v`` with ``matrix @ v = 0``.

    Parameters
    ----------
    matrix : array_like
        A matrix, as for :func:`rank`.
    field : pyfastcd.FiniteField or BinaryField, optional
        The field of exact residues.
    tol : float, optional
        The singular value threshold of :func:`rank`.

    Returns
    -------
    numpy.ndarray
        An array whose rows are the basis vectors, of shape
        ``(nullity, columns)`` with the polynomial axis of an extension
        field appended. Over R and C the rows are orthonormal.
    """

    domain, values = _prepare(matrix, field)
    columns = values.shape[1]
    if not domain.exact:
        if values.shape[0] == 0:
            return np.eye(columns, dtype=domain.dtype)
        _, singular, vh = np.linalg.svd(values, full_matrices=True)
        kept = int((singular > _tolerance(singular, values.shape, tol)).sum())
        return np.ascontiguousarray(vh[kept:].conj())
    reduced, pivots = _reduce(domain, values)
    free = [column for column in range(columns) if column not in pivots]
    basis = domain.zeros((len(free), columns))
    one = domain.scalar(1)
    for number, column in enumerate(free):
        basis[number, column] = one
        for row, pivot in enumerate(pivots):
            basis[number, pivot] = domain.negate(reduced[row, column])
    return basis
