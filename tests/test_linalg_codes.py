"""Tests of linear algebra over the coefficient fields and of codes in loop algebras."""

from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
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
    with pytest.raises(ValueError):
        codes.translate_matrix(x, side="two-sided")
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
    """Exports list the residues row by row, with the dimensions of the matrix."""

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
    assert sage.startswith("LinearCode(matrix(GF(3), 6, 8, [\n") and sage.endswith(
        "]))"
    )
    rows = [line.strip().rstrip(",") for line in sage.splitlines()[1:-1]]
    assert np.array_equal(np.array([eval(row) for row in rows]), matrix)
    assert codes.to_gap([[4, -1]], field=field).splitlines()[1].strip() == "[1,2]"
    extension = fcd.FiniteField(char=3, deg=2)
    for export in (codes.to_gap, codes.to_sage):
        with pytest.raises(ValueError):
            export(np.zeros((1, 2), dtype=np.uint32), field=extension)
        with pytest.raises(ValueError):
            export(np.zeros((1, 2, 3), dtype=np.uint32), field=extension)
        with pytest.raises(ValueError):
            export(np.zeros((1, 2, 2), dtype=np.uint32), field=field)
        with pytest.raises(ValueError):
            export(matrix, field=None)
        with pytest.raises(ValueError):
            export(matrix[0], field=field)
        with pytest.raises(ValueError):
            export(matrix.astype(np.float64), field=field)
        with pytest.raises(TypeError):
            export(matrix)


EXPORT_FIELDS = [
    fcd.FiniteField(char=3),
    cdl.BinaryField(),
    cdl.BinaryField(deg=2),
    cdl.BinaryField(deg=3),
    cdl.BinaryField.from_modulus(modulus=(1, 0, 1, 1)),
    cdl.BinaryField(deg=12),
    fcd.FiniteField(char=3, deg=2),
    fcd.FiniteField(char=13, deg=3),
]
EXPORT_IDS = ["F3", "F2", "F4", "F8", "F8-other-modulus", "F4096", "F9", "F2197"]


def _export_matrices(field: Any) -> list[np.ndarray]:
    """Return an ideal, a matrix that is not reduced and the zero code of length 8."""

    loop = cdl.CDLoop(2)
    algebra = cdl.LoopAlgebra(loop, field=field)
    rng = np.random.default_rng(field.char * field.deg)
    x = _element(algebra, rng) * (algebra.one() + algebra.basis(loop.minus_one()))
    return [
        codes.generator_matrix(x),
        codes.translate_matrix(_element(algebra, rng))[:3],
        codes.generator_matrix(algebra.zero()),
    ]


class _SageField:
    """A stand-in for a SageMath finite field, which records what it is given."""

    def __init__(self, order: int, name: Any = None, modulus: Any = None) -> None:
        self.order, self.name, self.modulus = order, name, modulus

    def __call__(self, coefficients: Any) -> tuple[int, ...]:
        return tuple(coefficients)


def _evaluate_sage(expression: str) -> tuple[_SageField, int, int, list[Any]]:
    """Evaluate a SageMath expression for a code with stand-ins for its functions."""

    names = {
        "GF": _SageField,
        "matrix": lambda field, rows, columns, entries: (field, rows, columns, entries),
        "LinearCode": lambda generator: generator,
    }
    result: tuple[_SageField, int, int, list[Any]] = eval(expression, names)
    return result


@pytest.mark.parametrize("field", EXPORT_FIELDS, ids=EXPORT_IDS)
def test_sage_export_carries_the_field_the_dimensions_and_the_entries(
    field: Any,
) -> None:
    """The expression rebuilds the matrix over the field with the same modulus."""

    for matrix in _export_matrices(field):
        made, rows, columns, entries = _evaluate_sage(
            codes.to_sage(matrix, field=field)
        )
        assert (rows, columns) == matrix.shape[:2] and columns == 8
        assert made.order == field.char**field.deg
        if field.deg == 1:
            assert made.name is None and made.modulus is None
        else:
            assert made.name == "t" and tuple(made.modulus) == field.modulus
        assert np.array_equal(
            np.array(entries, dtype=np.uint32).reshape(matrix.shape), matrix
        )


@pytest.mark.parametrize("field", EXPORT_FIELDS, ids=EXPORT_IDS)
def test_gap_export_carries_the_field_the_dimensions_and_the_entries(
    field: Any,
) -> None:
    """The expression lists the coefficients and names the field and its modulus."""

    order = field.char**field.deg
    ideal, unreduced, zero = _export_matrices(field)
    assert codes.to_gap(zero, field=field) == f"NullCode(8, GF({order}));"
    for matrix in (ideal, unreduced):
        gap = codes.to_gap(matrix, field=field)
        listing = gap[gap.index("[\n") : gap.index("\n]") + 2]
        assert " " not in listing.replace("\n  ", "")
        assert np.array_equal(np.array(eval(listing), dtype=np.uint32), matrix)
        assert gap.count("GeneratorMatCode(") == 1 and gap.endswith(";")
        if field.deg == 1:
            assert gap.endswith(f"]*Z({field.char})^0, GF({field.char}));")
            continue
        modulus = str(list(field.modulus)).replace(" ", "")
        assert gap.startswith("CallFuncList(function()\n") and gap.endswith("end, []);")
        assert (
            f"f := UnivariatePolynomial(GF({field.char}), {modulus}*Z({field.char})^0);"
            in gap
        )
        assert f"First(Concatenation([Z({order})], RootsOfUPol(GF({order}), f))" in gap
        assert f"powers := List([0..{field.deg - 1}], i -> t^i);" in gap
        assert f"row -> List(row, c -> c*powers)), GF({order}));" in gap


def test_export_of_the_zero_code_keeps_its_length() -> None:
    """A matrix without rows exports as the zero code of the right length."""

    for loop, length in ((cdl.CDLoop(2), 8), (cdl.CDLoop(3), 16)):
        algebra = cdl.LoopAlgebra(loop, char=3)
        zero = codes.generator_matrix(algebra.zero())
        assert zero.shape == (0, length)
        assert codes.to_sage(zero, field=algebra.field) == (
            f"LinearCode(matrix(GF(3), 0, {length}, []))"
        )
        assert codes.to_gap(zero, field=algebra.field) == f"NullCode({length}, GF(3));"
    quaternary = cdl.BinaryField(deg=2)
    empty = np.zeros((0, 8, 2), dtype=np.uint32)
    made, rows, columns, entries = _evaluate_sage(
        codes.to_sage(empty, field=quaternary)
    )
    assert (made.order, rows, columns, entries) == (4, 0, 8, [])
    assert codes.to_gap(empty, field=quaternary) == "NullCode(8, GF(4));"


def _field_from(char: int, deg: int, modulus: list[int]) -> Any:
    """Return the field with the given characteristic and modulus."""

    if char == 2:
        return (
            cdl.BinaryField()
            if deg == 1
            else cdl.BinaryField.from_modulus(modulus=tuple(modulus))
        )
    if deg == 1:
        return fcd.FiniteField(char=char)
    return fcd.FiniteField.from_modulus(char=char, modulus=tuple(modulus))


@pytest.mark.skipif(shutil.which("sage") is None, reason="SageMath is not installed")
def test_sage_export_builds_the_same_codes_in_sage(tmp_path: Any) -> None:
    """SageMath reads each expression as the code with the same reduced basis."""

    cases = [
        (field, matrix) for field in EXPORT_FIELDS for matrix in _export_matrices(field)
    ]
    lines = [
        "import json",
        "def coefficients(e, deg):",
        "    if deg == 1:",
        "        return int(e)",
        "    found = [int(c) for c in e.polynomial().list()]",
        "    return found + [int(0)] * int(deg - len(found))",
        "results = []",
    ]
    for field, matrix in cases:
        lines += [
            f"C = {codes.to_sage(matrix, field=field)}",
            "F = C.base_field()",
            "results.append([int(C.length()), int(C.dimension()), int(F.order()),",
            "    [int(c) for c in F.modulus().list()],",
            "    [[coefficients(e, F.degree()) for e in row]",
            "     for row in C.generator_matrix().rows()]])",
        ]
    lines.append("print('RESULTS ' + json.dumps(results))")
    script = tmp_path / "codes.sage"
    script.write_text("\n".join(lines) + "\n")
    run = subprocess.run(
        ["sage", str(script)], capture_output=True, text=True, timeout=600, cwd=tmp_path
    )
    assert run.returncode == 0, run.stderr
    found = json.loads(run.stdout[run.stdout.index("RESULTS ") + 8 :])
    assert len(found) == len(cases)
    for (field, matrix), (length, dimension, order, modulus, entries) in zip(
        cases, found, strict=True
    ):
        reduced, pivots = linalg.row_reduce(matrix, field=field)
        assert (length, dimension) == (matrix.shape[1], len(pivots))
        assert order == field.char**field.deg
        if field.deg > 1:
            assert tuple(modulus) == field.modulus
        # SageMath keeps the reduced row echelon form of the generator matrix
        expected = reduced[: len(pivots)]
        assert np.array_equal(
            np.array(entries, dtype=np.uint32).reshape(expected.shape), expected
        )


@pytest.mark.skipif(shutil.which("gap") is None, reason="GAP is not installed")
def test_gap_export_builds_the_same_matrices_in_gap(tmp_path: Any) -> None:
    """GAP reads each expression as the matrix with the same coefficients."""

    cases = [
        (field, matrix) for field in EXPORT_FIELDS for matrix in _export_matrices(field)
    ]
    # Without the GUAVA package the two functions it would supply only record
    # their arguments, which is what is compared
    lines = [
        'if LoadPackage("guava") = fail then',
        "  GeneratorMatCode := function(m, F)",
        "    return rec(mat := m, field := F, n := Length(m[1]));",
        "  end;",
        "  NullCode := function(n, F) return rec(mat := [], field := F, n := n); end;",
        "  CodeLength := C -> C.n;",
        "  CodeMatrix := C -> C.mat;",
        "  CodeField := C -> C.field;",
        "else",
        "  CodeLength := WordLength;",
        "  CodeMatrix := GeneratorMat;",
        "  CodeField := LeftActingDomain;",
        "fi;",
    ]
    for number, (field, matrix) in enumerate(cases):
        order = field.char**field.deg
        modulus = str(list(field.modulus)).replace(" ", "")
        root = f"First(Concatenation([Z({order})], RootsOfUPol(GF({order}), f)),"
        one = f"Z({field.char})^0"
        lines += [
            # A second semicolon keeps GAP from printing the value
            f"C := {codes.to_gap(matrix, field=field)};",
            f"f := UnivariatePolynomial(GF({field.char}), {modulus}*{one});;",
            f"t := {root}",
            "  r -> IsZero(Value(f, r)));;",
            f"B := Basis(GF({order}), List([0..{field.deg - 1}], i -> t^i));;",
            f'Print("CASE {number} ", CodeLength(C), " ", Size(CodeField(C)), " ");',
            f'Print(t = Z({order}), " ");',
            "Print(List(CodeMatrix(C), row -> List(row,",
            '  e -> List(Coefficients(B, e), IntFFE))), "\\n");',
        ]
    script = tmp_path / "codes.g"
    script.write_text("\n".join(lines) + "\n")
    with script.open() as stream:
        run = subprocess.run(
            ["gap", "-q"], stdin=stream, capture_output=True, text=True, timeout=600
        )
    output = run.stdout.replace("\\\n", "")
    chunks = [
        "".join(chunk.split("\n"))
        for chunk in re.split(r"(?=CASE \d+ )", output)
        if chunk.startswith("CASE ")
    ]
    assert len(chunks) == len(cases), run.stdout + run.stderr
    for (field, matrix), chunk in zip(cases, chunks, strict=True):
        match = re.fullmatch(r"CASE \d+ (\d+) (\d+) (true|false) (.*)", chunk)
        assert match is not None, chunk
        assert int(match.group(1)) == matrix.shape[1]
        assert int(match.group(2)) == field.char**field.deg
        entries = eval(match.group(4).replace(" ", ""))
        got = np.array(entries, dtype=np.uint32).reshape((-1,) + matrix.shape[1:])
        if field.deg == 1:
            got = got.reshape((-1, matrix.shape[1]))
        # GUAVA may hold another basis of the same code
        assert linalg.rank(got, field=field) == linalg.rank(matrix, field=field)
        stacked = np.concatenate([got, matrix])
        assert linalg.rank(stacked, field=field) == linalg.rank(matrix, field=field)
        if (
            isinstance(field, cdl.BinaryField)
            and field.deg > 1
            and field.modulus == cdl.BinaryField(deg=field.deg).modulus
        ):
            # The default moduli are Conway polynomials, whose root in GAP is Z(q)
            assert match.group(3) == "true"


IDEAL_SIDES = ["left", "right", "two-sided"]


def _closure(algebra: Any, generators: list[Any], sides: tuple[str, ...]) -> np.ndarray:
    """Return the reduced basis of an ideal over a finite field, from the definition."""

    rows = np.stack([_row(x) for x in generators])
    rank = -1
    while True:
        stacked = [rows]
        for row in rows:
            element = algebra(row.reshape(algebra.coefficient_shape))
            stacked += [codes.translate_matrix(element, side=side) for side in sides]
        reduced, pivots = linalg.row_reduce(
            np.concatenate(stacked), field=algebra.field
        )
        rows = reduced[: len(pivots)]
        if len(pivots) == rank:
            return rows
        rank = len(pivots)


EXACT_ALGEBRAS = [
    (cdl.CDLoop(2), {"char": 3}),
    (cdl.CDLoop(3), {"char": 7}),
    (cdl.CDLoop(3), {"char": 2}),
    (cdl.CDLoop(4), {"char": 2, "deg": 2}),
    (cdl.CentralProduct(1, 2), {"char": 3}),
    (cdl.CentralProduct(2, 2), {"char": 3, "deg": 2}),
    (cdl.CentralProduct(2, 3), {"char": 2}),
]


def _sparse(algebra: Any, rng: Any) -> Any:
    """Return a random element with few nonzero coefficients."""

    x = _element(algebra, rng)
    mask = rng.random(algebra.coefficient_shape[:2]) < 0.2
    keep = mask if algebra.field is None or algebra.deg == 1 else mask[..., None]
    return algebra(x.coefficients * keep.astype(x.coefficients.dtype))


@pytest.mark.parametrize(("loop", "selectors"), EXACT_ALGEBRAS)
def test_ideals_on_each_side_agree_with_the_definition(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """Each kind of ideal is the smallest space closed under its translations."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order)
    for count in (1, 1, 2):
        generators = [_sparse(algebra, rng) for _ in range(count)]
        found = {
            side: codes.generator_matrix(generators, side=side) for side in IDEAL_SIDES
        }
        assert np.array_equal(found["left"], _closure(algebra, generators, ("left",)))
        assert np.array_equal(found["right"], _closure(algebra, generators, ("right",)))
        both = _closure(algebra, generators, ("left", "right"))
        assert np.array_equal(found["two-sided"], both)
        for side in ("left", "right"):
            # A one-sided ideal lies in the two-sided one, and is two-sided
            # exactly when the two coincide
            assert _span_contains(algebra, both, found[side])
            same = found[side].shape == both.shape
            assert codes.is_ideal(found[side], algebra, side="two-sided") is same
        for side in IDEAL_SIDES:
            assert codes.is_ideal(found[side], algebra, side=side) is True
            assert codes.is_ideal(both, algebra, side=side) is True


def test_two_sided_ideals_of_known_dimension() -> None:
    """Small ideals whose dimensions follow from the structure of the algebra."""

    loop = cdl.CDLoop(2)
    algebra = cdl.LoopAlgebra(loop, char=3)
    x = algebra.one() + algebra.basis(loop.basis(1)) + algebra.basis(loop.basis(2))
    assert codes.generator_matrix(x).shape == (5, 8)
    assert codes.generator_matrix(x, side="right").shape == (5, 8)
    assert codes.generator_matrix(x, side="two-sided").shape == (7, 8)
    # The translates on both sides, without the closure
    both = codes.generator_matrix(x, side="two-sided", closure=False)
    stacked = np.concatenate(
        [codes.translate_matrix(x), codes.translate_matrix(x, side="right")]
    )
    reduced, pivots = linalg.row_reduce(stacked, field=algebra.field)
    assert np.array_equal(both, reduced[: len(pivots)])
    # A central element generates the same ideal on every side
    central = algebra.one() + algebra.basis(loop.minus_one())
    matrices = [codes.generator_matrix(central, side=side) for side in IDEAL_SIDES]
    assert matrices[0].shape == (4, 8)
    assert all(np.array_equal(matrix, matrices[0]) for matrix in matrices)
    # The augmentation ideal has codimension 1 and is two-sided
    binary = cdl.LoopAlgebra(loop, char=2)
    generators = [binary.basis(g) + binary.one() for g in loop if g != loop.one()]
    for side in IDEAL_SIDES:
        code = codes.generator_matrix(generators, side=side)
        assert code.shape == (7, 8) and not (code.sum(axis=1) % 2).any()
    # The zero ideal and the whole algebra
    for side in IDEAL_SIDES:
        assert codes.generator_matrix(algebra.zero(), side=side).shape == (0, 8)
        whole = codes.generator_matrix(algebra.one(), side=side)
        assert np.array_equal(whole, np.eye(8, dtype=np.uint32))


def test_is_ideal() -> None:
    """A row space is an ideal when translation does not lead out of it."""

    loop = cdl.CDLoop(3)
    algebra = cdl.LoopAlgebra(loop, char=3)
    x = algebra.one() + algebra.basis(loop.basis(1)) + algebra.basis(loop.basis(2))
    # The translates of x span 10 dimensions and the left ideal 14
    translates = codes.generator_matrix(x, closure=False)
    ideal = codes.generator_matrix(x)
    assert codes.is_ideal(translates, algebra) is False
    assert codes.is_ideal(ideal, algebra) is True
    assert codes.is_ideal(codes.translate_matrix(x), algebra) is False
    assert codes.is_ideal(_row(x)[None], algebra) is False
    # The rows need not be independent or reduced
    doubled = np.concatenate([ideal[::-1], ideal, (2 * ideal) % 3])
    assert codes.is_ideal(doubled, algebra) is True
    assert codes.is_ideal(doubled.tolist(), algebra) is True
    # A left ideal that is not a right ideal
    quaternion = cdl.LoopAlgebra(cdl.CDLoop(2), char=3)
    q = cdl.CDLoop(2)
    y = quaternion.one() + quaternion.basis(q.basis(1)) + quaternion.basis(q.basis(2))
    left = codes.generator_matrix(y)
    assert codes.is_ideal(left, quaternion) is True
    assert codes.is_ideal(left, quaternion, side="right") is False
    assert codes.is_ideal(left, quaternion, side="two-sided") is False
    # The zero ideal and the whole algebra
    for side in IDEAL_SIDES:
        assert codes.is_ideal(np.zeros((0, 16), dtype=np.uint32), algebra, side=side)
        assert codes.is_ideal(np.zeros((3, 16), dtype=np.uint32), algebra, side=side)
        assert codes.is_ideal(np.eye(16, dtype=np.uint32), algebra, side=side)
    # Extension fields, and R and C
    for selectors in (
        {"char": 2, "deg": 2},
        {"char": 3, "deg": 2},
        {},
        {"char": 0, "deg": 2},
    ):
        other = cdl.LoopAlgebra(loop, **selectors)
        z = other.one() + other.basis(loop.basis(1)) + other.basis(loop.basis(2))
        closed = codes.generator_matrix(z)
        assert codes.is_ideal(closed, other) is True
        assert codes.is_ideal(codes.generator_matrix(z, closure=False), other) is (
            closed.shape[0] == linalg.rank(codes.translate_matrix(z), field=other.field)
        )
        assert codes.is_ideal(_row(z)[None], other) is False


def test_is_ideal_rejects_invalid_arguments() -> None:
    """The matrix must fit the algebra and the side must be named."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=3)
    matrix = np.eye(8, dtype=np.uint32)
    with pytest.raises(TypeError):
        codes.is_ideal(matrix, algebra.field)
    with pytest.raises(TypeError):
        codes.is_ideal(matrix, cdl.CDLoop(2))
    with pytest.raises(TypeError):
        codes.is_ideal(matrix, algebra, side=1)
    with pytest.raises(ValueError):
        codes.is_ideal(matrix, algebra, side="both")
    with pytest.raises(ValueError):
        codes.is_ideal(matrix[:, :4], algebra)
    with pytest.raises(ValueError):
        codes.is_ideal(matrix[0], algebra)
    with pytest.raises(ValueError):
        codes.is_ideal(np.zeros((2, 8, 2), dtype=np.uint32), algebra)
    with pytest.raises(TypeError):
        codes.is_ideal(matrix.astype(np.float64), algebra)
    extension = cdl.LoopAlgebra(cdl.CDLoop(2), char=2, deg=2)
    with pytest.raises(ValueError):
        codes.is_ideal(matrix, extension)
    real = cdl.LoopAlgebra(cdl.CDLoop(2))
    with pytest.raises(TypeError):
        codes.is_ideal([["a"] * 8], real)


@pytest.mark.parametrize(
    ("loop", "selectors"),
    [
        (cdl.CDLoop(3), {"char": 3}),
        (cdl.CDLoop(4), {"char": 2}),
        (cdl.CentralProduct(2, 2), {"char": 2, "deg": 2}),
        (cdl.CDLoop(3), {}),
        (cdl.CentralProduct(2, 2), {"char": 0, "deg": 2}),
    ],
)
def test_closure_does_not_depend_on_the_block_size(
    loop: Any, selectors: dict[str, Any], monkeypatch: Any
) -> None:
    """Translating a few rows at a time gives the same code within a memory bound."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 1)
    generators = [_sparse(algebra, rng) for _ in range(3)]
    expected = {
        (side, closure): codes.generator_matrix(generators, side=side, closure=closure)
        for side in IDEAL_SIDES
        for closure in (True, False)
    }
    # The coefficients in the translates of one row by every loop element
    per_row = algebra.dimension * int(np.prod(algebra.coefficient_shape))
    original = codes._translates
    largest = 0

    def recording(rows: np.ndarray, tables: list[np.ndarray]) -> np.ndarray:
        nonlocal largest
        moved = original(rows, tables)
        largest = max(largest, moved.size // len(tables))
        return moved

    monkeypatch.setattr(codes, "_translates", recording)
    for block in (1, 3 * per_row, 1 << 12):
        monkeypatch.setattr(codes, "_BLOCK", block)
        for (side, closure), matrix in expected.items():
            largest = 0
            found = codes.generator_matrix(generators, side=side, closure=closure)
            # No step translates more coefficients than the block allows,
            # except that one row is always translated whole
            assert largest <= max(block, per_row)
            if algebra.field is not None:
                assert np.array_equal(found, matrix)
            else:
                assert found.shape == matrix.shape
                assert np.allclose(found, matrix, atol=1e-9)
            if closure:
                assert codes.is_ideal(found, algebra, side=side) is True


def _over_a_prime(algebra: Any, x: Any, prime: int) -> Any:
    """Return an element with integer coefficients over a large finite field."""

    values = x.coefficients
    if algebra.deg == 1:
        exact = cdl.LoopAlgebra(algebra.loop, char=prime)
        return exact((np.rint(values).astype(np.int64) % prime).astype(np.uint32))
    # C becomes F_p[t] / (t^2 + 1), a field when p is 3 modulo 4
    field = fcd.FiniteField.from_modulus(char=prime, modulus=(1, 0, 1))
    exact = cdl.LoopAlgebra(algebra.loop, field=field)
    parts = np.stack([np.rint(values.real), np.rint(values.imag)], axis=-1)
    return exact((parts.astype(np.int64) % prime).astype(np.uint32))


def _pivots(matrix: np.ndarray) -> list[int]:
    """Return the column of the first nonzero coefficient of each row."""

    flat = np.abs(matrix.reshape(matrix.shape[0], matrix.shape[1], -1)).max(axis=2)
    return [int(np.flatnonzero(row > 0.5)[0]) for row in flat]


@pytest.mark.parametrize("selectors", [{}, {"char": 0, "deg": 2}], ids=["R", "C"])
@pytest.mark.parametrize(
    "loop", [cdl.CDLoop(3), cdl.CentralProduct(2, 2), cdl.CentralProduct(2, 3)]
)
def test_dimensions_over_r_and_c_agree_with_exact_arithmetic(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """Integer generators give the dimension and pivots found over a large prime."""

    prime = 1000003
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + algebra.deg)
    for trial in range(4):
        x = _sparse(algebra, rng) if trial % 2 else _element(algebra, rng)
        exact = _over_a_prime(algebra, x, prime)
        for side in IDEAL_SIDES:
            for closure in (True, False):
                want = codes.generator_matrix(exact, side=side, closure=closure)
                for scale in (1.0, 1e-9, 1e7):
                    found = codes.generator_matrix(
                        x * scale, side=side, closure=closure
                    )
                    assert found.shape == want.shape[:2]
                    assert _pivots(found) == _pivots(want)
                    if closure:
                        assert codes.is_ideal(found, algebra, side=side) is True


def test_dimension_over_c_is_not_raised_by_rounding() -> None:
    """Keep rounding noise from counting as a dimension of an ideal."""

    # The translates of this element span 63 of the 64 dimensions, and they
    # span its left ideal. Elimination with the tolerance of row_reduce
    # finds a 64th pivot of size 1e-13 among them
    real = [
        [-2, 1, 0, 1, -2, -2, 1, 2, 0, 2, -1, -2, -1, -1, 0, 1]
        + [1, -1, 0, 1, 1, 0, 0, 2, 2, 2, 1, 1, -1, -2, 0, 2],
        [-2, 0, -2, 1, 0, 0, 0, -1, -1, 0, 1, 2, -2, 2, -1, 2]
        + [0, -2, 0, 0, 0, -1, 2, 0, 0, 0, -1, 0, -1, 1, -2, -1],
    ]
    imaginary = [
        [2, -1, 1, -2, 1, 0, 1, -2, 0, 1, 0, 2, -1, -1, 1, -1]
        + [2, 0, 0, 0, 0, 0, 0, -1, 2, -1, 2, 0, 2, -1, 2, -1],
        [2, 0, -1, -1, 0, 0, 1, 0, 2, 0, -1, -2, 2, 0, -2, 0]
        + [0, 2, 0, 2, -2, 1, -1, 0, 0, 1, 0, 0, -1, 1, 1, -1],
    ]
    loop = cdl.CentralProduct(2, 3)
    algebra = cdl.LoopAlgebra(loop, char=0, deg=2)
    x = algebra(np.array(real) + 1j * np.array(imaginary))
    exact = _over_a_prime(algebra, x, 1000003)
    assert codes.generator_matrix(exact).shape[0] == 63
    assert linalg.rank(codes.translate_matrix(x)) == 63
    for closure in (True, False):
        found = codes.generator_matrix(x, closure=closure)
        assert found.shape == (63, 64)
        assert _pivots(found) == _pivots(codes.generator_matrix(exact, closure=closure))
    assert codes.is_ideal(codes.generator_matrix(x), algebra) is True
    assert codes.is_ideal(codes.translate_matrix(x), algebra) is True
    # A tolerance that takes every direction for zero leaves nothing
    assert codes.generator_matrix(x, tol=2.0).shape == (0, 64)
