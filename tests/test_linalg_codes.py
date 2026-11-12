"""Tests of linear algebra over the coefficient fields and of codes in loop algebras."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pyfastcd as fcd
import pytest

import pycdloop as cdl
from pycdloop import codes, linalg

try:
    from . import reference
except ImportError:  # Support collection from inside the tests directory.
    import reference

EXACT_FIELDS = [
    fcd.FiniteField(char=3),
    fcd.FiniteField(char=13),
    fcd.FiniteField(char=3, deg=2),
    fcd.FiniteField(char=7, deg=2),
]
EXACT_IDS = ["F3", "F13", "F3^2", "F7^2"]


def _ring(field: Any) -> Any:
    """Return the reference ring of a finite field."""

    return reference.FiniteField(field.char, field.modulus)


def _random_matrix(field: Any, rows: int, columns: int, rng: Any) -> np.ndarray:
    """Return a random matrix of residues."""

    tail = () if field.deg == 1 else (field.deg,)
    return rng.integers(0, field.char, (rows, columns) + tail).astype(np.uint32)


def _apply(ring: Any, matrix: Any, vector: Any) -> list[Any]:
    """Return a matrix times a vector over a reference ring."""

    image = []
    for row in matrix:
        total = ring.zero
        for entry, value in zip(row, vector, strict=True):
            total = ring.add(total, ring.mul(entry, value))
        image.append(total)
    return image


def _is_exact_zero(field: Any, matrix: np.ndarray, vectors: np.ndarray) -> bool:
    """Return whether every row of *vectors* is killed by the matrix."""

    ring = _ring(field)
    entries = reference.to_ring(ring, matrix, 2)
    for vector in vectors:
        image = _apply(ring, entries, reference.to_ring(ring, vector, 1))
        if any(value != ring.zero for value in image):
            return False
    return True


@pytest.mark.parametrize("field", EXACT_FIELDS, ids=EXACT_IDS)
def test_exact_null_space_against_exhaustive_search(field: Any) -> None:
    """The null space has the size found by trying every vector."""

    ring = _ring(field)
    rng = np.random.default_rng(field.order)
    order = field.order
    columns = 3 if order <= 13 else 2
    elements = list(itertools.product(range(field.char), repeat=field.deg))
    for rows in (1, 2, 3):
        for _ in range(6):
            matrix = _random_matrix(field, rows, columns, rng)
            if rng.integers(2):
                # Repeat a row so that singular matrices are common
                matrix[-1] = matrix[0]
            entries = reference.to_ring(ring, matrix, 2)
            kernel = 0
            for vector in itertools.product(elements, repeat=columns):
                image = _apply(ring, entries, list(vector))
                kernel += all(value == ring.zero for value in image)
            rank = linalg.rank(matrix, field=field)
            assert kernel == order ** (columns - rank)
            basis = linalg.nullspace(matrix, field=field)
            assert basis.shape[:2] == (columns - rank, columns)
            assert basis.dtype == np.uint32
            assert _is_exact_zero(field, matrix, basis)
            assert linalg.rank(basis, field=field) == columns - rank


@pytest.mark.parametrize("field", EXACT_FIELDS, ids=EXACT_IDS)
def test_exact_row_reduction_properties(field: Any) -> None:
    """The reduced form has unit pivots, cleared pivot columns and equal row space."""

    rng = np.random.default_rng(11)
    tail = () if field.deg == 1 else (field.deg,)
    one = np.zeros(tail, dtype=np.uint32)
    one[(0,) * len(tail)] = 1
    for rows, columns, rank_bound in ((4, 6, 2), (5, 5, 5), (6, 3, 3), (1, 4, 1)):
        left = _random_matrix(field, rows, rank_bound, rng)
        right = _random_matrix(field, rank_bound, columns, rng)
        domain_matrix = np.zeros((rows, columns) + tail, dtype=np.uint32)
        ring = _ring(field)
        a = reference.to_ring(ring, left, 2)
        b = reference.to_ring(ring, right, 2)
        for i in range(rows):
            for j in range(columns):
                total = ring.zero
                for k in range(rank_bound):
                    total = ring.add(total, ring.mul(a[i][k], b[k][j]))
                domain_matrix[i, j] = total if field.deg > 1 else total[0]
        reduced, pivots = linalg.row_reduce(domain_matrix, field=field)
        rank = linalg.rank(domain_matrix, field=field)
        assert reduced.shape == domain_matrix.shape and len(pivots) == rank
        assert rank <= rank_bound
        assert list(pivots) == sorted(set(pivots))
        for row, pivot in enumerate(pivots):
            assert np.array_equal(reduced[row, pivot], one)
            column = reduced[:, pivot].copy()
            column[row] = 0
            assert not column.any()
            assert not reduced[row, :pivot].any()
        assert not reduced[rank:].any()
        again, same = linalg.row_reduce(reduced, field=field)
        assert np.array_equal(again, reduced) and same == pivots
        stacked = np.concatenate([domain_matrix, reduced])
        assert linalg.rank(stacked, field=field) == rank
        kernel = linalg.nullspace(domain_matrix, field=field)
        assert kernel.shape[0] == columns - rank
        assert _is_exact_zero(field, reduced, kernel)


def test_exact_rank_of_known_matrices() -> None:
    """Identity, zero, rank-one and modular examples have the expected rank."""

    field = fcd.FiniteField(char=5)
    assert linalg.rank(np.eye(6, dtype=np.int64), field=field) == 6
    assert linalg.rank(np.zeros((3, 4), dtype=np.int64), field=field) == 0
    assert linalg.rank(np.outer([1, 2, 3], [4, 1, 2, 3]), field=field) == 1
    # The rows are independent over the rationals but not modulo 5
    matrix = np.array([[1, 2], [3, 1]])
    assert linalg.rank(matrix) == 2 and linalg.rank(matrix, field=field) == 1
    assert linalg.rank([[6, -4], [0, 10]], field=field) == 1
    documented = np.array([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
    assert linalg.rank(documented, field=field) == 2
    assert linalg.nullspace(documented, field=field).tolist() == [[4, 4, 1]]
    reduced, pivots = linalg.row_reduce(documented, field=field)
    assert reduced.tolist() == [[1, 0, 1], [0, 1, 1], [0, 0, 0]] and pivots == (0, 1)
    rng = np.random.default_rng(0)
    wide = fcd.FiniteField(char=13)
    product = rng.integers(0, 13, (40, 7)) @ rng.integers(0, 13, (7, 40))
    assert linalg.rank(product, field=wide) == 7
    assert linalg.nullspace(product, field=wide).shape == (33, 40)


@pytest.mark.parametrize("kind", [np.float64, np.complex128])
def test_numerical_rank_null_space_and_row_reduction(kind: Any) -> None:
    """Over R and C the results hold up to rounding and the tolerance applies."""

    rng = np.random.default_rng(4)

    def draw(*shape: int) -> np.ndarray:
        """Return a random real or complex matrix."""

        values = rng.standard_normal(shape)
        if kind is np.complex128:
            values = values + 1j * rng.standard_normal(shape)
        return values.astype(kind)

    for rows, columns, rank_bound in ((5, 7, 3), (6, 6, 6), (7, 4, 2), (3, 3, 0)):
        matrix = draw(rows, rank_bound) @ draw(rank_bound, columns)
        assert linalg.rank(matrix) == rank_bound
        basis = linalg.nullspace(matrix)
        assert basis.shape == (columns - rank_bound, columns) and basis.dtype == kind
        assert np.allclose(matrix @ basis.T, 0, atol=1e-10)
        gram = basis @ basis.conj().T
        assert np.allclose(gram, np.eye(columns - rank_bound), atol=1e-12)
        reduced, pivots = linalg.row_reduce(matrix)
        assert len(pivots) == rank_bound and reduced.dtype == kind
        for row, pivot in enumerate(pivots):
            assert np.isclose(reduced[row, pivot], 1)
            column = reduced[:, pivot].copy()
            column[row] = 0
            assert np.allclose(column, 0, atol=1e-12)
        assert np.allclose(reduced[rank_bound:], 0, atol=1e-10)
        assert np.allclose(reduced @ basis.T, 0, atol=1e-9)
    nearly = np.diag([1.0, 1e-10]).astype(kind)
    assert linalg.rank(nearly) == 2 and linalg.rank(nearly, tol=1e-6) == 1
    assert linalg.nullspace(nearly, tol=1e-6).shape == (1, 2)
    assert linalg.row_reduce(nearly, tol=1e-6)[1] == (0,)
    assert linalg.rank(np.array([[1, 2], [2, 4]])) == 1
    assert linalg.rank(np.array([[True, False], [False, True]])) == 2


def test_empty_matrices() -> None:
    """Matrices without rows or columns have rank zero and full null spaces."""

    field = fcd.FiniteField(char=3)
    extension = fcd.FiniteField(char=3, deg=2)
    assert linalg.rank(np.zeros((0, 3))) == 0 and linalg.rank(np.zeros((3, 0))) == 0
    assert np.array_equal(linalg.nullspace(np.zeros((0, 3))), np.eye(3))
    assert linalg.nullspace(np.zeros((3, 0))).shape == (0, 0)
    assert linalg.row_reduce(np.zeros((2, 3)))[1] == ()
    empty = np.zeros((0, 3), dtype=np.uint32)
    assert linalg.rank(empty, field=field) == 0
    assert linalg.nullspace(empty, field=field).tolist() == np.eye(3).tolist()
    assert linalg.nullspace(np.zeros((3, 0), dtype=np.uint32), field=field).shape == (
        0,
        0,
    )
    wide = np.zeros((0, 2, 2), dtype=np.uint32)
    assert linalg.nullspace(wide, field=extension).tolist() == [
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]],
    ]


def test_linalg_rejects_invalid_arguments() -> None:
    """Shape, type and field errors raise the documented exceptions."""

    field = fcd.FiniteField(char=3)
    extension = fcd.FiniteField(char=3, deg=2)
    for function in (linalg.rank, linalg.row_reduce, linalg.nullspace):
        with pytest.raises(ValueError):
            function([1, 2, 3])
        with pytest.raises(ValueError):
            function(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError):
            function(np.zeros((2, 2, 2), dtype=np.uint32), field=field)
        with pytest.raises(ValueError):
            function(np.zeros((2, 2), dtype=np.uint32), field=extension)
        with pytest.raises(TypeError):
            function([["a", "b"]])
        with pytest.raises(TypeError):
            function([[1.5, 2.0]], field=field)
        with pytest.raises(TypeError):
            function([[1, 2]], field=3)


ALGEBRAS = [
    (cdl.CDLoop(2), {}),
    (cdl.CDLoop(3), {"char": 7}),
    (cdl.CDLoop(3), {"char": 0, "deg": 2}),
    (cdl.CentralProduct(1, 2), {"char": 3}),
    (cdl.CDLoop(2), {"char": 7, "deg": 2}),
    (cdl.CentralProduct(2, 2), {"char": 3, "deg": 2}),
]


def _element(algebra: Any, rng: Any) -> Any:
    """Return a random element with exactly representable coefficients."""

    shape = algebra.coefficient_shape
    if algebra.field is not None:
        return algebra(rng.integers(0, algebra.char, shape).astype(np.uint32))
    values = rng.integers(-3, 4, shape).astype(algebra.dtype)
    if algebra.deg == 2:
        values = values + 1j * rng.integers(-3, 4, shape)
    return algebra(values)


def _row(element: Any) -> np.ndarray:
    """Return the loop-basis coefficients of an element as one row."""

    algebra = element.algebra
    tail = algebra.coefficient_shape[2:]
    return element.coefficients.reshape((algebra.dimension,) + tail)


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS)
def test_translate_matrix_rows_are_products_with_basis_elements(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """Row g is [g] x on the left and x [g] on the right."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    x = _element(algebra, np.random.default_rng(loop.order))
    left = codes.translate_matrix(x)
    right = codes.translate_matrix(x, side="right")
    tail = algebra.coefficient_shape[2:]
    assert left.shape == right.shape == (algebra.dimension, algebra.dimension) + tail
    assert left.dtype == algebra.dtype
    for position, g in enumerate(loop):
        assert np.array_equal(left[position], _row(algebra.basis(g) * x))
        assert np.array_equal(right[position], _row(x * algebra.basis(g)))
    assert np.array_equal(left[0], _row(x))
    assert x.weight() == int((left[0] != 0).reshape(algebra.dimension, -1).any(1).sum())


def _span_contains(algebra: Any, generator: np.ndarray, rows: np.ndarray) -> bool:
    """Return whether every row lies in the row space of a generator matrix."""

    rank = linalg.rank(generator, field=algebra.field)
    stacked = np.concatenate([generator, rows])
    return linalg.rank(stacked, field=algebra.field) == rank == generator.shape[0]


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_generator_matrix_spans_a_one_sided_ideal(
    loop: Any, selectors: dict[str, Any], side: str
) -> None:
    """The code contains its generators and is closed under translation."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + len(side))
    e_plus, e_minus = algebra.central_idempotents()
    generators = [e_plus * _element(algebra, rng), e_minus * _element(algebra, rng)]
    tail = algebra.coefficient_shape[2:]
    for items in ([generators[0]], [generators[1]], generators, iter(generators)):
        items = list(items)
        matrix = codes.generator_matrix(items, side=side)
        assert matrix.ndim == 2 + len(tail) and matrix.shape[1] == algebra.dimension
        reduced, pivots = linalg.row_reduce(matrix, field=algebra.field)
        assert len(pivots) == matrix.shape[0]
        if algebra.field is not None:
            assert np.array_equal(reduced, matrix)
        else:
            assert np.allclose(reduced, matrix)
        assert _span_contains(algebra, matrix, np.stack([_row(x) for x in items]))
        for row in matrix:
            codeword = algebra(row.reshape(algebra.coefficient_shape))
            translates = codes.translate_matrix(codeword, side=side)
            assert _span_contains(algebra, matrix, translates)
    single = codes.generator_matrix(generators[0], side=side)
    assert np.array_equal(single, codes.generator_matrix([generators[0]], side=side))


def test_generator_matrix_closure() -> None:
    """Translates of a generator need not span the ideal it generates."""

    loop = cdl.CDLoop(3)
    algebra = cdl.LoopAlgebra(loop, char=3)
    x = algebra.one() + algebra.basis(loop.basis(1)) + algebra.basis(loop.basis(2))
    translates = codes.generator_matrix(x, closure=False)
    ideal = codes.generator_matrix(x)
    assert translates.shape == (10, 16) and ideal.shape == (14, 16)
    assert linalg.rank(codes.translate_matrix(x), field=algebra.field) == 10
    stacked = np.concatenate([translates, codes.translate_matrix(x)])
    assert linalg.rank(stacked, field=algebra.field) == 10
    assert linalg.rank(np.concatenate([ideal, translates]), field=algebra.field) == 14
    # In the associative quaternion case the translates already span the ideal
    quaternion = cdl.LoopAlgebra(cdl.CDLoop(2), char=3)
    y = quaternion.one() + quaternion.basis(cdl.CDLoop(2).basis(1))
    assert np.array_equal(
        codes.generator_matrix(y, closure=False), codes.generator_matrix(y)
    )
    assert codes.generator_matrix(y).shape == (6, 8)


def test_generator_matrices_of_known_ideals() -> None:
    """The unit, the sum of all basis elements and the idempotents give known codes."""

    loop = cdl.CDLoop(3)
    for selectors in ({"char": 5}, {}):
        algebra = cdl.LoopAlgebra(loop, **selectors)
        whole = codes.generator_matrix(algebra.one())
        assert np.array_equal(whole, np.eye(16, dtype=algebra.dtype))
        everything = algebra(np.ones(algebra.coefficient_shape, dtype=algebra.dtype))
        repetition = codes.generator_matrix(everything)
        assert repetition.shape == (1, 16) and (repetition == 1).all()
        assert everything.weight() == 16 and everything.distance(algebra.zero()) == 16
        e_plus, e_minus = algebra.central_idempotents()
        assert codes.generator_matrix(e_plus).shape == (8, 16)
        assert codes.generator_matrix(e_minus, side="right").shape == (8, 16)
        assert codes.generator_matrix([e_plus, e_minus]).shape == (16, 16)
        assert codes.generator_matrix(algebra.zero()).shape == (0, 16)


def test_codes_reject_invalid_arguments() -> None:
    """Generators must be elements of one algebra and sides must be named."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=3)
    other = cdl.LoopAlgebra(cdl.CDLoop(2), char=5)
    x = algebra.one()
    with pytest.raises(ValueError):
        codes.generator_matrix([])
    with pytest.raises(ValueError):
        codes.generator_matrix([x, other.one()])
    with pytest.raises(ValueError):
        codes.generator_matrix(x, side="both")
    with pytest.raises(TypeError):
        codes.generator_matrix([x, 1])
    with pytest.raises(TypeError):
        codes.generator_matrix(x, side=0)
    with pytest.raises(TypeError):
        codes.translate_matrix(algebra.asarray(x.coefficients[None].copy()))
    with pytest.raises(TypeError):
        codes.translate_matrix(x.coefficients)
    with pytest.raises(ValueError):
        codes.translate_matrix(x, side="middle")


def test_export_to_gap_and_sage() -> None:
    """Exports list the residues row by row and exist over prime fields only."""

    field = fcd.FiniteField(char=3)
    loop = cdl.CDLoop(2)
    algebra = cdl.LoopAlgebra(loop, field=field)
    matrix = codes.generator_matrix(algebra.one() + algebra.basis(loop.basis(1)))
    gap = codes.to_gap(matrix, field=field)
    assert gap.startswith("GeneratorMatCode([\n") and gap.endswith("]*Z(3)^0, GF(3));")
    rows = [line.strip().rstrip(",") for line in gap.splitlines()[1:-1]]
    assert np.array_equal(np.array([eval(row) for row in rows]), matrix)
    assert " " not in "".join(rows)
    sage = codes.to_sage(matrix, field=field)
    assert sage.startswith("LinearCode(matrix(GF(3), [\n") and sage.endswith("]))")
    rows = [line.strip().rstrip(",") for line in sage.splitlines()[1:-1]]
    assert np.array_equal(np.array([eval(row) for row in rows]), matrix)
    assert codes.to_gap([[4, -1]], field=field).splitlines()[1].strip() == "[1,2]"
    extension = fcd.FiniteField(char=3, deg=2)
    for export in (codes.to_gap, codes.to_sage):
        with pytest.raises(ValueError):
            export(np.zeros((1, 2, 2), dtype=np.uint32), field=extension)
        with pytest.raises(ValueError):
            export(matrix, field=None)
        with pytest.raises(ValueError):
            export(matrix[0], field=field)
        with pytest.raises(ValueError):
            export(matrix.astype(np.float64), field=field)
        with pytest.raises(TypeError):
            export(matrix)
