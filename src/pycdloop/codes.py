"""Linear codes in loop algebras.

A code in a loop algebra ``F[L]`` is a subspace, read as a linear code of
length ``len(L)`` through the loop-basis coefficients, so that the weight of
an element is its Hamming weight. The codes of interest are the ideals:
subspaces closed under multiplication by every element on the left, on the
right, or on both sides, or equivalently under translation by every loop
element on those sides.

Matrices here have one row for each codeword and one column for each loop
element, in the iteration order of the loop, with a trailing polynomial axis
over an extension field.

Examples
--------
>>> import pycdloop as cdl
>>> from pycdloop import codes
>>> Q = cdl.CDLoop(2)
>>> A = cdl.LoopAlgebra(Q, char=3)
>>> x = A.basis(Q.basis(0)) + A.basis(Q.basis(1))
>>> codes.generator_matrix(x).shape
(6, 8)
>>> y = x + A.basis(Q.basis(2))
>>> L = codes.generator_matrix(y)
>>> L.shape, codes.generator_matrix(y, side="two-sided").shape
((5, 8), (7, 8))
>>> codes.is_ideal(L, A), codes.is_ideal(L, A, side="right")
(True, False)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from . import _range, linalg
from ._algebra import LoopAlgebra
from ._element import Element
from ._field import _coefficient_field, _reduce_integers
from ._validation import DIFFERENT_ALGEBRA_MESSAGE, _choice

__all__ = ["generator_matrix", "is_ideal", "to_gap", "to_sage", "translate_matrix"]

_SIDES = ("left", "right")
_IDEAL_SIDES = ("left", "right", "two-sided")

# The number of coefficients in the translates that are reduced at one time,
# which bounds the memory of a closure whatever the order of the loop
_BLOCK = 1 << 20


def _sources(loop: Any, side: str) -> np.ndarray:
    """Return, for each loop element, where its translation takes coefficients from."""

    # Translating by a loop element permutes the basis: the coefficient of
    # [y] in x moves to [g y] under left translation by g and to [y g] under
    # right translation. Entry [g, z] is the position y that moves to z, so
    # that indexing the columns of a matrix by row g of the table translates
    # its rows by g
    positions = np.arange(loop.order, dtype=np.intp)
    if side == "left":
        targets = loop._multiply_positions(positions[:, None], positions[None, :])
    else:
        targets = loop._multiply_positions(positions[None, :], positions[:, None])
    sources = np.empty((loop.order, loop.order), dtype=np.intp)
    np.put_along_axis(sources, targets, positions[None, :], axis=1)
    return sources


def _tables(loop: Any, side: str) -> list[np.ndarray]:
    """Return the translation tables of one side, or of both."""

    names = _SIDES if side == "two-sided" else (side,)
    return [_sources(loop, name) for name in names]


def _translates(rows: np.ndarray, tables: list[np.ndarray]) -> np.ndarray:
    """Return every translate of every row, as rows of one matrix."""

    moved = [rows[:, table].reshape((-1,) + rows.shape[1:]) for table in tables]
    return moved[0] if len(moved) == 1 else np.concatenate(moved)


def _step(rows: np.ndarray, tables: list[np.ndarray]) -> int:
    """Return how many rows to translate at one time."""

    entries = int(np.prod(rows.shape[1:]))
    return max(1, _BLOCK // (len(tables) * rows.shape[1] * entries))


# Over R and C a row of norm 1 counts as lying in a space when its distance
# from the space is at most this. The rounding errors that build up over the
# steps of a closure stay many orders of magnitude below it
_TOLERANCE = float(np.finfo(np.float64).eps) ** 0.5


def _extend(
    frame: np.ndarray, rows: np.ndarray, tol: Any
) -> tuple[list[int], np.ndarray]:
    """Return the rows that extend the span of an orthonormal frame, and the new frame.

    The rows are scaled to norm 1 and taken one at a time, each time the one
    farthest from the span found so far, until none is farther than the
    tolerance. The frame is extended by the direction of each row taken.
    """

    threshold = _TOLERANCE if tol is None else float(tol)
    # A row is brought to unit size by a power of two before its norm is
    # formed, so that the squares neither underflow nor overflow
    sized = _range.unit(rows, always=True)[0]
    norms = np.linalg.norm(sized, axis=1)
    index = np.flatnonzero(norms > 0)
    rest = sized[index] / norms[index, None]
    for _ in range(2):
        rest = rest - (rest @ frame.conj().T) @ frame
    chosen: list[int] = []
    while len(rest) and len(frame) < frame.shape[1]:
        distances = np.linalg.norm(rest, axis=1)
        best = int(np.argmax(distances))
        if distances[best] <= threshold:
            break
        direction = rest[best] / distances[best]
        direction = direction - (direction @ frame.conj().T) @ frame
        direction = direction / np.linalg.norm(direction)
        chosen.append(int(index[best]))
        frame = np.concatenate([frame, direction[None]])
        rest = rest - np.outer(rest @ direction.conj(), direction)
    return chosen, frame


def _echelon(frame: np.ndarray, basis: np.ndarray, tol: Any) -> np.ndarray:
    """Return the reduced row echelon form of a real or complex row space.

    The space is given twice: by an orthonormal frame, and by a basis whose
    rows are translates of the generators.
    """

    kept, columns = frame.shape
    if kept == columns:
        return np.eye(columns, dtype=frame.dtype)
    if kept == 0:
        return frame
    # A column of the frame is a pivot column when it is not in the span of
    # the columns before it, which a Gram--Schmidt sweep decides. The
    # distance that counts as zero is lowered while the sweep finds fewer
    # pivot columns than rows
    epsilon = float(np.finfo(np.float64).eps)
    threshold = _TOLERANCE if tol is None else float(tol)
    while True:
        directions = np.empty((kept, kept), dtype=frame.dtype)
        pivots: list[int] = []
        for column in range(columns):
            count = len(pivots)
            if count == kept:
                break
            found = directions[:, :count]
            rest = frame[:, column]
            for _ in range(2):
                rest = rest - found @ (found.conj().T @ rest)
            norm = float(np.linalg.norm(rest))
            if norm > threshold:
                directions[:, count] = rest / norm
                pivots.append(column)
        if len(pivots) == kept or threshold < columns * epsilon:
            break
        threshold /= 256
    if len(pivots) == kept:
        # The reduced basis is the basis multiplied by the inverse of its
        # pivot columns. The rows of the basis hold the coefficients of the
        # generators, each row rescaled here by a power of two that keeps
        # its largest and its smallest coefficient in range, so the entries
        # of the result carry the rounding of this one solution only
        scaled = _range.unit(basis, always=True, keep=True)[0]
        reduced: np.ndarray = np.linalg.solve(scaled[:, pivots], scaled)
    else:
        adjoint = directions[:, : len(pivots)].conj().T
        reduced = np.linalg.solve(adjoint @ frame[:, pivots], adjoint @ frame)
    # The entries that are exactly 0 or 1 in a reduced row echelon form
    for row, column in enumerate(pivots):
        reduced[row, :column] = 0
        reduced[:, column] = 0
        reduced[row, column] = 1
    return reduced


def _span_exact(
    pending: np.ndarray, tables: list[np.ndarray], field: Any, closure: bool
) -> np.ndarray:
    """Return the reduced basis of a code over a finite field."""

    step = _step(pending, tables)
    basis = pending[:0]
    known: tuple[int, ...] = ()
    # The translates of a few rows at a time are reduced together with the
    # basis found so far. A row of the new basis whose pivot column is new
    # is one of a set of rows that, with the old basis, spans the new one,
    # so for the closure it is enough to translate those rows in their turn
    while len(pending) and len(known) < pending.shape[1]:
        block, pending = pending[:step], pending[step:]
        stacked = np.concatenate([basis, _translates(block, tables)])
        reduced, pivots = linalg.row_reduce(stacked, field=field)
        basis = reduced[: len(pivots)]
        if closure:
            fresh = [row for row, column in enumerate(pivots) if column not in known]
            pending = np.concatenate([pending, basis[fresh]])
        known = pivots
    return basis


def _span_float(
    pending: np.ndarray, tables: list[np.ndarray], closure: bool, tol: Any
) -> np.ndarray:
    """Return the reduced basis of a code over R or C."""

    step = _step(pending, tables)
    frame = pending[:0]
    basis = pending[:0]
    # Elimination with a tolerance can take rounding noise for a pivot, so
    # the code found so far is held as an orthonormal frame, against which
    # the translates of a few rows at a time are measured. The translates
    # that extend it join the basis, and for the closure they are translated
    # in their turn. Every row of the basis is then a generator translated
    # some number of times, with its coefficients unchanged
    while len(pending) and len(frame) < pending.shape[1]:
        block, pending = pending[:step], pending[step:]
        moved = _translates(block, tables)
        chosen, frame = _extend(frame, moved, tol)
        basis = np.concatenate([basis, moved[chosen]])
        if closure:
            pending = np.concatenate([pending, moved[chosen]])
    return _echelon(frame, basis, tol)


def _flatten(element: Any) -> np.ndarray:
    """Return the loop-basis coefficients of one element as a row."""

    algebra = element._algebra
    return np.asarray(element.coefficients).reshape(
        (1, algebra.dimension) + algebra._field.shape
    )


def translate_matrix(x: Any, *, side: str = "left") -> np.ndarray:
    """Return the matrix whose rows are the translates of one element.

    Parameters
    ----------
    x : Element
        The element.
    side : str, optional
        ``"left"`` for the rows ``[g] x`` and ``"right"`` for ``x [g]``,
        with ``g`` running over the loop in iteration order.

    Returns
    -------
    numpy.ndarray
        A matrix of shape ``(dimension, dimension)``, with the polynomial
        axis of an extension field appended.
    """

    if not isinstance(x, Element):
        raise TypeError("x must be one element of a loop algebra")
    name = _choice(side, "side", _SIDES)
    return np.ascontiguousarray(
        _translates(_flatten(x), _tables(x._algebra.loop, name))
    )


def generator_matrix(
    elements: Any, *, side: str = "left", closure: bool = True, tol: Any = None
) -> np.ndarray:
    """Return a generator matrix of the code generated by some elements.

    Parameters
    ----------
    elements : Element or iterable of Element
        The generators, all from one loop algebra.
    side : str, optional
        ``"left"`` for the left ideal, ``"right"`` for the right ideal, and
        ``"two-sided"`` for the ideal closed under translation on both
        sides.
    closure : bool, optional
        With ``True``, the default, the result spans the ideal generated by
        the elements. Because the algebra need not be associative, the span
        of the translates of the generators need not be closed under
        translation, so translates are added until the dimension stops
        growing. With ``False`` the result spans the translates of the
        generators only, on both sides for ``"two-sided"``.
    tol : float, optional
        Over R and C, a vector of norm 1 counts as lying in a space when its
        distance from the space is at most this, which decides the dimension
        of the code. The default is the square root of the machine epsilon.
        Ignored over a finite field.

    Returns
    -------
    numpy.ndarray
        A matrix in reduced row echelon form whose rows are a basis of the
        code, of shape ``(k, dimension)`` with the polynomial axis of an
        extension field appended.
    """

    name = _choice(side, "side", _IDEAL_SIDES)
    items = [elements] if isinstance(elements, Element) else list(_checked(elements))
    if not items:
        raise ValueError("at least one generator is required")
    algebra = items[0]._algebra
    for item in items:
        if item._algebra is not algebra:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
    tables = _tables(algebra.loop, name)
    rows = np.concatenate([_flatten(item) for item in items])
    if algebra.field is None:
        basis = _span_float(rows, tables, closure, tol)
    else:
        basis = _span_exact(rows, tables, algebra.field, closure)
    return np.ascontiguousarray(basis)


def is_ideal(matrix: Any, algebra: Any, *, side: str = "left", tol: Any = None) -> bool:
    """Return whether the rows of a matrix span an ideal of a loop algebra.

    Parameters
    ----------
    matrix : array_like
        A matrix with one column for each loop element, of shape
        ``(k, dimension)`` with the polynomial axis of an extension field
        appended. Its rows need not be independent. The entries are read
        as coefficients of the algebra, so a complex matrix for an algebra
        over R raises ``TypeError``.
    algebra : LoopAlgebra
        The loop algebra.
    side : str, optional
        ``"left"``, ``"right"``, or ``"two-sided"``: the side on which the
        row space is to be closed under translation by every loop element.
    tol : float, optional
        Over R and C, a vector of norm 1 counts as lying in a space when its
        distance from the space is at most this. The default is the square
        root of the machine epsilon. Ignored over a finite field.

    Returns
    -------
    bool
        ``True`` when no translate of a row lies outside the row space. The
        matrix without rows spans the zero ideal.
    """

    if not isinstance(algebra, LoopAlgebra):
        raise TypeError("algebra must be a loop algebra")
    name = _choice(side, "side", _IDEAL_SIDES)
    # The coefficients are those of elements of the algebra, so a complex
    # matrix is not one of an algebra over R
    values = algebra._field.coerce(matrix)
    if values.ndim != 2 + algebra._field.tail:
        raise ValueError("a matrix must have two index axes")
    if values.shape[1] != algebra.dimension:
        raise ValueError("the matrix must have one column for each loop element")
    tables = _tables(algebra.loop, name)
    step = _step(values, tables)
    if algebra.field is None:
        chosen, frame = _extend(values[:0], values, tol)
        basis = values[chosen]
        for start in range(0, len(basis), step):
            moved = _translates(basis[start : start + step], tables)
            if _extend(frame, moved, tol)[0]:
                return False
        return True
    reduced, pivots = linalg.row_reduce(values, field=algebra.field)
    basis = reduced[: len(pivots)]
    for start in range(0, len(basis), step):
        moved = _translates(basis[start : start + step], tables)
        grown = linalg.row_reduce(np.concatenate([basis, moved]), field=algebra.field)
        if len(grown[1]) != len(pivots):
            return False
    return True


def _checked(elements: Iterable[Any]) -> Iterable[Any]:
    """Yield generators after checking that each is an element."""

    for item in elements:
        if not isinstance(item, Element):
            raise TypeError("generators must be elements of a loop algebra")
        yield item


def _residues(matrix: Any, field: Any) -> np.ndarray:
    """Return the entries of a matrix over a finite field, reduced."""

    if field is None:
        raise ValueError("export is available over finite fields only")
    domain = _coefficient_field(field=field)
    values = np.asarray(matrix)
    if (
        values.dtype.kind not in "iu"
        or values.ndim != 2 + domain.tail
        or values.shape[2:] != domain.shape
    ):
        raise ValueError("the matrix must be an array of residues with two index axes")
    return _reduce_integers(values, field.char).astype(np.int64)


def _listing(rows: list[Any], compact: bool) -> str:
    """Return nested lists as source text, one row to a line."""

    text = [str(row).replace(" ", "") if compact else str(row) for row in rows]
    return "[\n" + ",\n".join("  " + row for row in text) + "\n]"


def to_gap(matrix: Any, *, field: Any) -> str:
    """Return a GAP expression for the code with the given generator matrix.

    The expression calls ``GeneratorMatCode`` of the GUAVA package, or
    ``NullCode`` for a matrix without rows. *field* is the field of the
    entries, a ``pyfastcd.FiniteField`` or a :class:`~pycdloop.BinaryField`.

    Over an extension field the entries are the polynomials of the matrix in
    a root ``t`` of ``field.modulus``. GAP presents the field by its Conway
    polynomial, with root ``Z(q)``, so the expression takes ``t = Z(q)`` when
    that is a root of the modulus, as it is for the default moduli of
    :class:`~pycdloop.BinaryField`, and the first root that GAP finds
    otherwise. The roots differ by automorphisms of the field, and the codes
    they give have the same length, dimension, and weights.
    """

    values = _residues(matrix, field)
    char, order = field.char, field.char**field.deg
    count, length = values.shape[:2]
    if count == 0:
        return f"NullCode({length}, GF({order}));"
    listing = _listing(values.tolist(), True)
    if field.deg == 1:
        return f"GeneratorMatCode({listing}*Z({char})^0, GF({char}));"
    modulus = str(list(field.modulus)).replace(" ", "")
    return (
        "CallFuncList(function()\n"
        "local f, t, powers;\n"
        f"f := UnivariatePolynomial(GF({char}), {modulus}*Z({char})^0);\n"
        f"t := First(Concatenation([Z({order})], RootsOfUPol(GF({order}), f)),\n"
        "  r -> IsZero(Value(f, r)));\n"
        f"powers := List([0..{field.deg - 1}], i -> t^i);\n"
        f"return GeneratorMatCode(List({listing},\n"
        f"  row -> List(row, c -> c*powers)), GF({order}));\n"
        "end, []);"
    )


def to_sage(matrix: Any, *, field: Any) -> str:
    """Return a SageMath expression for the code with the given generator matrix.

    The expression calls ``LinearCode`` on a matrix with the dimensions of
    the given one, so that a matrix without rows gives the zero code of the
    right length. *field* is the field of the entries, a
    ``pyfastcd.FiniteField`` or a :class:`~pycdloop.BinaryField`. Over an
    extension field the expression builds the field with ``field.modulus``
    and the generator ``t``, and the entries are the polynomials of the
    matrix in ``t``.
    """

    values = _residues(matrix, field)
    char, order = field.char, field.char**field.deg
    count, length = values.shape[:2]
    listing = _listing(values.tolist(), False) if count else "[]"
    if field.deg == 1:
        return f"LinearCode(matrix(GF({char}), {count}, {length}, {listing}))"
    rows = f"[[F(c) for c in row] for row in {listing}]"
    return (
        f"(lambda F: LinearCode(matrix(F, {count}, {length}, {rows})))"
        f"(GF({order}, 't', modulus={list(field.modulus)}))"
    )
