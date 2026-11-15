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

import math
from typing import Any

import numpy as np

from . import _range
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
    return largest * (max(shape) * float(np.finfo(np.float64).eps))


# The singular value decomposition multiplies and divides entries of a
# matrix by one another. Entries above 2**_HEADROOM are first brought down
# to it, which is exact for every entry that is not smaller than the
# largest by nearly the whole range, far below what the decomposition can
# tell from zero. A matrix whose largest entry is below 2**-_HEADROOM is
# brought up to unit size, which is exact for every entry
_HEADROOM = 900


def _in_range(values: np.ndarray, tol: Any) -> tuple[np.ndarray, Any]:
    """Return a real or complex matrix and a tolerance away from the ends of the range.

    Rank and null space do not change when a matrix is multiplied by a
    power of two. A matrix with an entry near the top of the range is
    divided by the smallest power that makes room for the products of the
    computation, a matrix near the bottom is multiplied up, and the
    tolerance is scaled with it.
    """

    if values.size == 0:
        return values, tol
    largest = float(np.maximum(np.abs(values.real), np.abs(values.imag)).max())
    if not np.isfinite(largest) or largest == 0:
        return values, tol
    power = int(np.frexp(largest)[1])
    if power > _HEADROOM:
        power -= _HEADROOM
    elif power >= -_HEADROOM:
        return values, tol
    if tol is not None:
        tol = float(np.ldexp(float(tol), -power))
    return _range.shift(values, np.asarray(-power)), tol


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


# An exponent below that of every float, for a threshold of zero
_NOTHING = -(1 << 50)


def _threshold(tol: Any) -> tuple[float, int]:
    """Return a tolerance as a mantissa in ``[1/2, 1)`` and an exponent."""

    given = float(tol)
    if math.isinf(given) and given > 0:
        return 0.5, -_NOTHING
    mantissa, exponent = math.frexp(given)
    return (mantissa, exponent) if mantissa > 0 else (0.0, _NOTHING)


def _reduce_float(matrix: np.ndarray, tol: Any) -> tuple[np.ndarray, list[int]]:
    """Return the reduced row echelon form of a real or complex matrix.

    Elimination uses scaled partial pivoting: of the entries of a column
    that count, the pivot is the largest relative to the largest entry of
    its row in the matrix as given. The rounding left in a row of large
    entries is then not taken before an entry of a row of small ones, and
    the choice does not depend on the scale of each row.

    With a tolerance, an entry counts when it is above the tolerance.
    Without one, an entry counts when it is above the rounding that
    elimination can have left in its place. A row that depends on the
    pivot rows cancels to its own rounding, about ``max(rows, columns)``
    times the machine epsilon times its largest entry, carried through
    the reduction: multiplied by the entries of the reduced pivot rows,
    which are large where the pivot columns are nearly dependent, and no
    smaller than the rounding of the largest product that was subtracted
    from the row. So the rounding of a row is taken to be that multiple
    of the machine epsilon, times one plus the largest entry of the pivot
    rows reduced so far, times the larger of the largest entry of the row
    as given and the largest product subtracted from it. This does not
    depend on the scale of the row. A pivot has in addition to be above
    that multiple of the machine epsilon times the largest entry of the
    matrix.

    An entry of the pivot column that does not count is zero in the
    elimination as well: it is set to zero and its row is left as it is,
    since as a factor it would be multiplied by the pivot row, and could
    come out as a pivot in a later column. An entry that counts is
    eliminated, also where it is too small to be a pivot.

    The entries of a finite reduced form can be reached through values
    beyond the range of the floats: a row divided by a small pivot, or the
    product of two small entries. Elimination in floats is therefore
    abandoned as soon as an operation overflows or underflows, and done
    again on numbers whose exponents cannot leave the range. Both give the
    same result where the first one completes.
    """

    if not matrix.size:
        return matrix.copy(), []
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            return _reduce_plain(matrix, tol)
    except FloatingPointError:
        with np.errstate(all="ignore"):
            return _reduce_wide(matrix, tol)


def _slack(shape: tuple[int, ...]) -> float:
    """Return the multiple of a size below which rounding is not told from zero."""

    return max(shape) * float(np.finfo(np.float64).eps)


def _reduce_plain(matrix: np.ndarray, tol: Any) -> tuple[np.ndarray, list[int]]:
    """Reduce in floats; the caller turns overflow and underflow into errors."""

    work = matrix.copy()
    rows, columns = work.shape
    # The size of each row as given, relative to which its entries are
    # compared when a pivot is chosen
    scales = np.abs(work).max(axis=1)
    slack = _slack(work.shape)
    # The largest entry of each row as given or product subtracted from
    # it, and the largest entry of the reduced pivot rows
    reach = scales.copy()
    grown = 0.0
    floor = slack * float(scales.max())
    scales[scales == 0] = 1.0
    pivots: list[int] = []
    row = 0
    for column in range(columns):
        if row == rows:
            break
        sizes = np.abs(work[:, column])
        limits = (slack * (1.0 + grown)) * reach if tol is None else float(tol)
        # The rows above hold pivot rows, whose entries are quotients and
        # are not compared with the tolerance
        counted = ((sizes > limits) | (np.arange(rows) < row)) & (sizes > 0)
        above = counted[row:]
        if tol is None:
            above = above & (sizes[row:] > floor)
        if not above.any():
            work[row:, column] = 0
            continue
        relative = np.where(above, sizes[row:] / scales[row:], -1.0)
        chosen = row + int(np.argmax(relative))
        if chosen != row:
            for values in (work, scales, reach, counted):
                values[[row, chosen]] = values[[chosen, row]]
        normalized = work[row] / work[row, column]
        factors = np.where(counted, work[:, column], 0)
        work -= factors[:, None] * normalized[None, :]
        reach = np.maximum(reach, np.abs(factors) * np.abs(normalized).max())
        work[row] = normalized
        # A complex quotient z / z need not be exactly 1, so the pivot
        # column is set to what it is in a reduced row echelon form
        work[:, column] = 0
        work[row, column] = 1
        grown = max(grown, float(np.abs(work[: row + 1]).max()))
        pivots.append(column)
        row += 1
    return work, pivots


def _largest(sizes: _range.Wide, axis: int | None = None) -> _range.Wide:
    """Return the largest of nonnegative numbers with wide exponents."""

    mantissas, exponents = np.broadcast_arrays(*sizes.real)
    exponents = np.where(mantissas > 0, exponents, _NOTHING)
    tops = exponents.max(axis=axis, keepdims=True)
    leading = np.where(exponents == tops, mantissas, 0.0).max(axis=axis)
    return _range.Wide((leading, tops.reshape(leading.shape)))


def _larger(x: _range.Wide, y: _range.Wide) -> _range.Wide:
    """Return the larger of nonnegative numbers with wide exponents, place by place."""

    return _range.Wide(_range.choose(_range.exceeds(x.real, y.real), x.real, y.real))


def _reduce_wide(matrix: np.ndarray, tol: Any) -> tuple[np.ndarray, list[int]]:
    """Reduce on numbers whose exponents cannot leave the range.

    Every entry, and each part of a complex entry, carries its own
    exponent, so rows and columns of any sizes stand beside one another
    and only an entry of the reduced form that is itself beyond the range
    is lost. The steps are those of :func:`_reduce_plain`, and so are the
    sizes that the tolerance is formed from.
    """

    wide = _range.Wide
    rows, columns = matrix.shape
    work = wide.of(matrix)
    zero, one = wide.of(np.zeros((), matrix.dtype)), wide.of(np.ones((), matrix.dtype))
    unit = wide.of(np.ones(()))
    places = np.arange(rows)[:, None], np.arange(columns)[None, :]
    # The size of each row as given, as a mantissa and an exponent
    given = _largest(wide(work.modulus()), 1)
    scales = np.where(given.real[0] > 0, given.real[0], 1.0)
    tops = given.real[1]
    reach = given
    grown = wide.of(np.zeros(()))
    slack = wide.of(np.asarray(_slack(matrix.shape)))
    floor = slack * _largest(given)
    fixed = None
    if tol is not None:
        mantissa, exponent = _threshold(tol)
        fixed = wide((np.asarray(mantissa), np.asarray(exponent, dtype=np.int64)))
    pivots: list[int] = []
    row = 0
    for column in range(columns):
        if row == rows:
            break
        sizes = wide(work.pick((slice(None), column)).modulus())
        limits = slack * (unit + grown) * reach if fixed is None else fixed
        over = _range.exceeds(sizes.real, limits.real)
        counted = (over | (np.arange(rows) < row)) & (sizes.real[0] > 0)
        above = counted[row:]
        if fixed is None:
            above = above & _range.exceeds(sizes.real, floor.real)[row:]
        if not above.any():
            work = work.at((places[0] >= row) & (places[1] == column), zero)
            continue
        # A size relative to that of its row is a float, or zero or an
        # infinity where it is beyond the range, which orders it as well
        fractions, heights = sizes.real
        apart = np.clip(heights[row:] - tops[row:], -5000, 5000).astype(np.intc)
        relative = np.where(
            above, np.ldexp(fractions[row:] / scales[row:], apart), -1.0
        )
        chosen = row + int(np.argmax(relative))
        if chosen != row:
            order = np.arange(rows)
            order[[row, chosen]] = order[[chosen, row]]
            work, reach = work.pick(order), reach.pick(order)
            tops, scales, counted = tops[order], scales[order], counted[order]
        normalized = work.pick(row).over(work.pick((row, column)))
        factors = work.pick((slice(None), column)).at(~counted, zero)
        work = work - factors.pick(np.s_[:, None]) * normalized.pick(None)
        widest = _largest(wide(normalized.modulus()))
        reach = _larger(reach, wide(factors.modulus()) * widest)
        work = work.at(places[0] == row, normalized.pick(None))
        work = work.at(places[1] == column, zero)
        work = work.at((places[0] == row) & (places[1] == column), one)
        grown = _larger(grown, _largest(wide(work.pick(np.s_[: row + 1]).modulus())))
        pivots.append(column)
        row += 1
    return work.to_array().astype(matrix.dtype), pivots


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
        machine epsilon. The decomposition resolves singular values down
        to about the machine epsilon times the largest, so a smaller value
        does not separate more. Ignored over a finite field.

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
    values, tol = _in_range(values, tol)
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
        count as zero, in the choice of a pivot and in the elimination.
        Without it, an entry counts as zero when it is no larger than the
        rounding that elimination can have left in its place:
        ``max(rows, columns)`` times the machine epsilon times the largest
        entry of its row, multiplied by the growth of the reduction; and a
        pivot has also to be above that multiple of the machine epsilon
        times the largest entry of the matrix. Of the entries that can be
        pivots, the pivot is the largest relative to the largest entry of
        its row, so the result does not otherwise depend on the scale of
        each row.

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
        values, tol = _in_range(values, tol)
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
