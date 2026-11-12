"""Tests of loop algebras over fields of characteristic 2."""

from __future__ import annotations

import itertools
import pickle

import numpy as np
import pytest

import pycdloop as cdl
from pycdloop import _algebra, _kernel, _lift, codes, linalg, testing

try:  # Support both unittest discovery and package-style test collection.
    from . import reference
except ImportError:  # pragma: no cover - depends on the collection mechanism
    import reference

LOOPS = [
    cdl.CDLoop(0),
    cdl.CDLoop(1),
    cdl.CDLoop(2),
    cdl.CDLoop(3),
    cdl.CDLoop(4),
    cdl.CentralProduct(1, 1),
    cdl.CentralProduct(1, 2),
    cdl.CentralProduct(2, 2),
    cdl.CentralProduct(2, 1, 1),
]
FIELDS = [cdl.BinaryField(deg=k) for k in (1, 2, 3, 6, 7, 8, 11, 12)] + [
    cdl.BinaryField.from_modulus(modulus=(1, 0, 1, 1))
]
SMALL_FIELDS = [cdl.BinaryField(), cdl.BinaryField(deg=2), cdl.BinaryField(deg=7)]
SMALL_LOOPS = [cdl.CDLoop(1), cdl.CDLoop(2), cdl.CentralProduct(1, 1)]
POWER_LOOPS = [cdl.CDLoop(level) for level in range(5)] + [
    cdl.CentralProduct(1, 1),
    cdl.CentralProduct(1, 2),
    cdl.CentralProduct(2, 2),
    cdl.CentralProduct(2, 3),
    cdl.CentralProduct(2, 1, 1),
]
LARGER_LOOPS = [cdl.CDLoop(3), cdl.CDLoop(4), cdl.CentralProduct(2, 3)]


def _ids(value: object) -> str:
    """Name a loop or field in test identifiers."""

    return repr(value).replace(" ", "")


def _oracle_product(
    algebra: cdl.LoopAlgebra, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Multiply loop-basis coefficients by the definition, in exact arithmetic."""

    field = algebra.field
    ring = reference.FiniteField(2, field.modulus)
    scalar = field.deg == 1
    left = reference.to_ring(ring, x[..., None] if scalar else x, 2)
    right = reference.to_ring(ring, y[..., None] if scalar else y, 2)
    product = reference.loop_algebra_multiply(ring, algebra.loop.levels, left, right)
    values = np.array([[list(c) for c in row] for row in product], dtype=np.uint32)
    return values[..., 0] if scalar else values


def _is_central_by_definition(algebra: cdl.LoopAlgebra, x: cdl.Element) -> bool:
    """Check that an element commutes and associates with all basis elements."""

    basis = [algebra.basis(g) for g in algebra.loop]
    for g in basis:
        if x * g != g * x:
            return False
    for g, h in itertools.product(basis, repeat=2):
        if (x * g) * h != x * (g * h):
            return False
        if (g * x) * h != g * (x * h):
            return False
        if (g * h) * x != g * (h * x):
            return False
    return True


# Parents and selectors


def test_char_two_selects_a_binary_field() -> None:
    """Accept ``char=2`` and a ``BinaryField`` and return the same algebra."""

    loop = cdl.CDLoop(3)
    for degree in (1, 2, 12):
        field = cdl.BinaryField(deg=degree)
        algebra = cdl.LoopAlgebra(loop, char=2, deg=degree)
        assert algebra is cdl.LoopAlgebra(loop, field=field)
        assert algebra.field is field
        assert (algebra.char, algebra.deg) == (2, degree)
        assert algebra.dtype == np.dtype(np.uint32)
        expected = (2, 8) + ((degree,) if degree > 1 else ())
        assert algebra.coefficient_shape == expected
        assert algebra.cd_algebra is None
        assert pickle.loads(pickle.dumps(algebra)) is algebra
        assert repr(algebra) == f"LoopAlgebra(CDLoop(3), char=2, deg={degree})"
    other = cdl.BinaryField.from_modulus(modulus=(1, 0, 1, 1))
    assert cdl.LoopAlgebra(loop, field=other) is not cdl.LoopAlgebra(
        loop, char=2, deg=3
    )
    assert pickle.loads(pickle.dumps(cdl.LoopAlgebra(loop, field=other))).field is other


def test_a_binary_field_is_passed_on_its_own() -> None:
    """Follow the selector rules of pyfastcd for a field object."""

    loop = cdl.CDLoop(1)
    field = cdl.BinaryField(deg=3)
    assert cdl.LoopAlgebra(loop, char=0, deg=1, field=field).field is field
    for selectors in ({"char": 2}, {"deg": 3}, {"char": 2, "deg": 3}, {"char": 3}):
        with pytest.raises(TypeError, match="cannot be combined"):
            cdl.LoopAlgebra(loop, field=field, **selectors)
    for wrong in (2, "F_8", (1, 0, 1, 1)):
        with pytest.raises(TypeError, match="BinaryField"):
            cdl.LoopAlgebra(loop, field=wrong)
        with pytest.raises(TypeError, match="BinaryField"):
            codes.to_gap(np.eye(2, dtype=np.uint32), field=wrong)
        with pytest.raises(TypeError, match="BinaryField"):
            linalg.rank(np.eye(2, dtype=np.uint32), field=wrong)


def test_algorithm_names_and_automatic_choices() -> None:
    """List ``lift`` in place of ``split`` and pin the automatic thresholds."""

    assert _algebra.LIFT_THRESHOLDS == ((2, 8192), (8, 512), (16, 4096))
    real = cdl.LoopAlgebra(cdl.CDLoop(3))
    assert real.algorithms == ("auto", "split", "table")
    assert real.default_algorithm == real.default_batch_algorithm == "split"
    # For each degree: the least level at which one product is lifted, and
    # the least level at which a large batch is
    choices = {1: (7, 1), 2: (5, 3), 6: (5, 3), 7: (6, 4), 12: (6, 4)}
    for degree, (single, batch) in choices.items():
        for level in range(9):
            algebra = cdl.LoopAlgebra(cdl.CDLoop(level), char=2, deg=degree)
            assert algebra.algorithms == ("auto", "lift", "table")
            expected = "lift" if level >= single else "table"
            assert algebra.default_algorithm == expected, (degree, level)
            expected = "lift" if level >= batch else "table"
            assert algebra.default_batch_algorithm == expected, (degree, level)
    # Between the two, the choice follows the number of records times D^2
    algebra = cdl.LoopAlgebra(cdl.CDLoop(4), char=2)
    assert [algebra._binary_choice(count) for count in (1, 31, 32, 1000)] == [
        "table",
        "table",
        "lift",
        "lift",
    ]
    central = cdl.LoopAlgebra(cdl.CentralProduct(2, 3), char=2, deg=4)
    assert [central._binary_choice(count) for count in (1, 2)] == ["lift", "lift"]
    assert (
        cdl.LoopAlgebra(cdl.CDLoop(2), char=2, deg=4)._binary_choice(10**6) == "table"
    )


def test_automatic_products_use_the_selected_algorithm(monkeypatch) -> None:
    """Call the lifted product exactly when the rule selects it."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(4), char=2)
    calls = []
    original = _lift._Lift.product

    def product(self, left, right, signed_algorithm):
        calls.append(left[0])
        return original(self, left, right, signed_algorithm)

    monkeypatch.setattr(_lift._Lift, "product", product)
    for count, lifted in ((1, False), (31, False), (32, True), (64, True)):
        x = algebra.random(1, batch_shape=(count,))
        y = algebra.random(2, batch_shape=(count,))
        calls.clear()
        result = x * y
        assert bool(calls) == lifted, count
        assert (result == algebra.multiply(x, y, algorithm="table")).all()
        calls.clear()
        assert (algebra.multiply(x, y) == result).all()
        assert bool(calls) == lifted, count
        calls.clear()
        assert (x.square() == algebra.multiply(x, x, algorithm="table")).all()
        assert bool(calls) == lifted, count
    u, v = algebra.random(3), algebra.random(4)
    calls.clear()
    assert u * v == algebra.multiply(u, v, algorithm="table")
    assert not calls
    large = cdl.LoopAlgebra(cdl.CDLoop(7), char=2)
    u, v = large.random(3), large.random(4)
    calls.clear()
    assert u * v == large.multiply(u, v, algorithm="table")
    assert calls == ["lift"]


def test_signed_methods_of_lifted_products() -> None:
    """Offer the signed-component methods of the algebra that products lift to."""

    real = cdl.LoopAlgebra(cdl.CDLoop(3))
    algebra = cdl.LoopAlgebra(cdl.CDLoop(3), char=2)
    assert algebra.signed_algorithms == ("auto", "direct", "quasilinear")
    central = cdl.LoopAlgebra(cdl.CentralProduct(1, 3), char=2)
    assert central.signed_algorithms == ("auto", "native", "rows", "schedule")
    assert central.default_signed_algorithm == "native"
    x, y = algebra.random(1), algebra.random(2)
    with pytest.raises(ValueError):
        algebra.multiply(x, y, algorithm="split")
    with pytest.raises(ValueError):
        real.multiply(real.one(), real.one(), algorithm="lift")
    with pytest.raises(ValueError):
        algebra.multiply(x, y, signed_algorithm="rows")


# Products


@pytest.mark.parametrize("field", FIELDS, ids=_ids)
@pytest.mark.parametrize("loop", LOOPS, ids=_ids)
def test_products_agree_with_the_definition(loop, field) -> None:
    """Compare both algorithms with the exact oracle, for elements and arrays."""

    algebra = cdl.LoopAlgebra(loop, field=field)
    rng = np.random.default_rng(loop.order + field.deg)
    x, y = algebra.random(rng), algebra.random(rng)
    expected = _oracle_product(algebra, x.coefficients, y.coefficients)
    for name in algebra.algorithms:
        product = algebra.multiply(x, y, algorithm=name)
        assert np.array_equal(product.coefficients, expected), name
    assert np.array_equal((x * y).coefficients, expected)
    batch = algebra.random(rng, batch_shape=(3,))
    other = algebra.random(rng, batch_shape=(1,))
    for name in ("lift", "table"):
        products = algebra.multiply(batch, other, algorithm=name)
        assert products.batch_shape == (3,)
        for i in range(3):
            want = _oracle_product(
                algebra, batch.coefficients[i], other.coefficients[0]
            )
            assert np.array_equal(products.coefficients[i], want), (name, i)
    assert ((batch * y) == algebra.multiply(batch, y, algorithm="table")).all()
    assert ((y * batch) == algebra.multiply(y, batch, algorithm="table")).all()


@pytest.mark.parametrize("degree", [1, 4, 6, 7, 9, 12])
@pytest.mark.parametrize("level", [5, 6])
def test_larger_products_agree_between_the_two_algorithms(
    level: int, degree: int
) -> None:
    """Compare the lifted product with the table product at larger levels."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(level), char=2, deg=degree)
    rng = np.random.default_rng(level * 100 + degree)
    x = algebra.random(rng, batch_shape=(2,))
    y = algebra.random(rng, batch_shape=(2,))
    lifted = algebra.multiply(x, y, algorithm="lift")
    assert (lifted == algebra.multiply(x, y, algorithm="table")).all()
    for name in algebra.signed_algorithms:
        again = algebra.multiply(x, y, algorithm="lift", signed_algorithm=name)
        assert (again == lifted).all(), name


def test_central_products_lift_with_every_signed_method() -> None:
    """Agree with the table product for each method of the lifted algebra."""

    for levels in ((1, 3), (2, 3), (3, 3)):
        for degree in (1, 2):
            algebra = cdl.LoopAlgebra(cdl.CentralProduct(*levels), char=2, deg=degree)
            x, y = algebra.random(1), algebra.random(2)
            expected = algebra.multiply(x, y, algorithm="table")
            for name in algebra.signed_algorithms:
                got = algebra.multiply(x, y, algorithm="lift", signed_algorithm=name)
                assert got == expected, (levels, degree, name)


def test_elements_keep_their_lifts_without_changing_results() -> None:
    """Reuse transformed lifts and return the same product as fresh elements."""

    for degree in (1, 3, 9):
        algebra = cdl.LoopAlgebra(cdl.CDLoop(4), char=2, deg=degree)
        x, y = algebra.random(3), algebra.random(4)
        assert x._lift is None
        first = algebra.multiply(x, y, algorithm="lift")
        assert x._lift is not None and len(x._lift) == (1 if degree <= 6 else 3)
        saved = [(a.copy(), b.copy()) for a, b in x._lift]
        second = algebra.multiply(x, y, algorithm="lift")
        fresh = algebra.multiply(
            algebra(x.coefficients.copy()),
            algebra(y.coefficients.copy()),
            algorithm="lift",
        )
        assert first == second == fresh
        for (a, b), (c, d) in zip(saved, x._lift, strict=True):
            assert np.array_equal(a, c) and np.array_equal(b, d)
            assert not c.flags.writeable and not d.flags.writeable
        assert pickle.loads(pickle.dumps(x)) == x
        array = algebra.random(5, batch_shape=(2,))
        for i in range(2):
            assert (x * array)[i] == x * array[i]
            assert (array * x)[i] == array[i] * x


def test_lifting_refuses_loops_beyond_the_coefficient_bound(monkeypatch) -> None:
    """Raise when a lifted coefficient could reach the prime."""

    monkeypatch.setattr(_lift, "PRIME", 61)
    algebra = cdl.LoopAlgebra(cdl.CentralProduct(2, 4), char=2)
    x = algebra.random(1)
    try:
        with pytest.raises(ValueError):
            algebra.multiply(x, x, algorithm="lift")
        assert algebra.multiply(x, x, algorithm="table") == x.left_power(2)
    finally:
        # The lifting data built with the small prime must not outlive the test
        algebra._lazy.pop("lift", None)
    monkeypatch.undo()
    assert algebra.multiply(x, x, algorithm="lift") == x.left_power(2)


# Arithmetic and derived operations


@pytest.mark.parametrize("field", SMALL_FIELDS, ids=_ids)
@pytest.mark.parametrize("loop", LOOPS, ids=_ids)
def test_operations_follow_their_definitions(loop, field) -> None:
    """Check the ring laws and every derived operation against products."""

    algebra = cdl.LoopAlgebra(loop, field=field)
    rng = np.random.default_rng(7 * loop.order + field.deg)
    x, y, z = algebra.random(rng), algebra.random(rng), algebra.random(rng)
    assert (x + y) * z == x * z + y * z and z * (x + y) == z * x + z * y
    assert x + x == 0 and x - y == x + y and -x == x and x * 1 == x and x * 2 == 0
    assert (x + 1) + 1 == x and 1 - x == x + 1 and x / 1 == x
    assert x.square() == x * x
    assert x.commutator(y) == x * y - y * x
    assert x.anticommutator(y) == x * y + y * x
    assert x.associator(y, z) == (x * y) * z - x * (y * z)
    assert algebra.left_operator_commutator(x, y, z) == x * (y * z) - y * (x * z)
    assert algebra.right_operator_commutator(x, y, z) == (z * y) * x - (z * x) * y
    assert x.left_power(3) == x * (x * x) and x.right_power(3) == (x * x) * x
    assert x.left_power(0) == 1 and x.right_power(1) == x
    assert x.conjugate().conjugate() == x
    assert (x * y).conjugate() == y.conjugate() * x.conjugate()
    if field.deg > 1:
        c = field([0, 1] + [0] * (field.deg - 2))
        assert (x * c) / c == x and c * x == x * c
        assert (x * c) * y == (x * y) * c
    with pytest.raises(ZeroDivisionError):
        x / 0
    with pytest.raises(TypeError):
        x.jordan_product(y)
    batch = algebra.random(rng, batch_shape=(2,))
    assert (batch.square() == batch * batch).all()
    assert (batch.commutator(x) == batch * x - x * batch).all()
    assert (batch.associator(x, y) == (batch * x) * y - batch * (x * y)).all()
    assert ((batch + x) - x == batch).all() and ((batch * 1) == batch).all()


@pytest.mark.parametrize("field", SMALL_FIELDS, ids=_ids)
@pytest.mark.parametrize("loop", LOOPS, ids=_ids)
def test_the_square_of_a_square_is_the_fourth_power_of_the_augmentation(
    loop, field
) -> None:
    """Check ``(x x)(x x) = aug(x)^4`` and that the augmentation is multiplicative."""

    algebra = cdl.LoopAlgebra(loop, field=field)
    rng = np.random.default_rng(loop.order + 11 * field.deg)
    for _ in range(4):
        x, y = algebra.random(rng), algebra.random(rng)
        alpha = x.augmentation()
        assert alpha.algebra is field
        assert (x * x) * (x * x) == algebra.scalar(alpha**4)
        assert (x * y).augmentation() == alpha * y.augmentation()
        assert (x + y).augmentation() == alpha + y.augmentation()


@pytest.mark.parametrize("field", SMALL_FIELDS, ids=_ids)
@pytest.mark.parametrize("loop", LOOPS + LARGER_LOOPS, ids=_ids)
def test_squares_are_central_and_powers_do_not_depend_on_bracketing(
    loop, field
) -> None:
    """Check that ``x x`` is central and that every bracketing of a power agrees."""

    algebra = cdl.LoopAlgebra(loop, field=field)
    rng = np.random.default_rng(3 * loop.order + field.deg)
    for _ in range(3):
        x, y, z = algebra.random(rng), algebra.random(rng), algebra.random(rng)
        square = x * x
        assert square.is_central()
        assert square * y == y * square
        assert (square * y) * z == square * (y * z)
        assert (y * square) * z == y * (square * z)
        assert (y * z) * square == y * (z * square)
        assert (x * square) * x == square * square == x * (x * square)
        for exponent in range(7):
            assert x.left_power(exponent) == x.right_power(exponent) == x**exponent
    batch = algebra.random(rng, batch_shape=(3,))
    assert (batch * batch).is_central().all()


@pytest.mark.parametrize("field", SMALL_FIELDS, ids=_ids)
@pytest.mark.parametrize("loop", POWER_LOOPS, ids=_ids)
def test_powers_and_inverses_agree_with_products(loop, field) -> None:
    """Check powers against repeated products and inverses by multiplying back."""

    algebra = cdl.LoopAlgebra(loop, field=field)
    rng = np.random.default_rng(loop.order + 5 * field.deg)
    for _ in range(4):
        x = algebra.random(rng)
        alpha = x.augmentation()
        power = algebra.one()
        for exponent in range(0, 10):
            assert x**exponent == power, exponent
            power = power * x
        assert x**4 == algebra.scalar(alpha**4)
        assert x ** (4 * 10**20 + 3) == x**3 * alpha ** (4 * 10**20)
        assert x.has_inverse() == bool(alpha)
        assert x.is_nilpotent() == (not bool(alpha))
        if alpha:
            inverse = x.inverse()
            assert x * inverse == 1 and inverse * x == 1
            assert x**-1 == inverse and x**-3 == inverse * (inverse * inverse)
        else:
            assert x**4 == 0
            with pytest.raises(ZeroDivisionError):
                x.inverse()
            with pytest.raises(ZeroDivisionError):
                x**-1
    batch = algebra.random(rng, batch_shape=(6,))
    marks = batch.has_inverse()
    assert (marks == ~batch.is_nilpotent()).all()
    for i in range(6):
        assert (batch**3)[i] == batch[i] ** 3
    units = algebra.asarray(batch.coefficients[marks]) if marks.any() else None
    if units is not None:
        assert ((units * units.inverse()) == 1).all()
    if not marks.all():
        with pytest.raises(ZeroDivisionError):
            batch.inverse()


def test_minimal_polynomials_vanish_and_are_minimal() -> None:
    """Evaluate the polynomial at the element and check no smaller power vanishes."""

    loops = [cdl.CDLoop(1), cdl.CDLoop(2), cdl.CDLoop(3)]
    loops += [cdl.CentralProduct(1, 2), cdl.CentralProduct(2, 2)]
    for field in SMALL_FIELDS[:2]:
        for loop in loops:
            algebra = cdl.LoopAlgebra(loop, field=field)
            rng = np.random.default_rng(loop.order + field.deg)
            for _ in range(6):
                x = algebra.random(rng)
                alpha = x.augmentation()
                polynomial = x.minimal_polynomial()
                degree = polynomial.shape[0] - 1
                assert 1 <= degree <= 4
                shifted = x + alpha
                assert shifted**degree == 0
                assert degree == 1 or shifted ** (degree - 1) != 0
                total, power = algebra.zero(), algebra.one()
                for coefficient in polynomial:
                    total = total + power * algebra._field.python(coefficient)
                    power = power * x
                assert total == 0
                description = algebra.subalgebra_generated_by(x)
                assert description.dimension == degree
                assert description.spectral_values is None
                assert description.signed_polynomial is None
            scalar = algebra.scalar(1)
            assert scalar.minimal_polynomial().shape[0] == 2


def test_the_only_idempotents_are_zero_and_one() -> None:
    """Enumerate every element of three small algebras over F_2."""

    for loop in SMALL_LOOPS:
        algebra = cdl.LoopAlgebra(loop, char=2)
        count = algebra.dimension
        bits = (np.arange(1 << count)[:, None] >> np.arange(count)) & 1
        everything = algebra.array(bits.reshape((-1,) + algebra.coefficient_shape))
        marks = everything.is_idempotent()
        assert int(marks.sum()) == 2
        assert (everything[marks] == everything[marks] * everything[marks]).all()
        found = {int(n) for n in np.flatnonzero(marks)}
        assert found == {0, 1}


# Structure


@pytest.mark.parametrize("loop", SMALL_LOOPS, ids=_ids)
def test_central_elements_are_exactly_those_of_the_definition(loop) -> None:
    """Compare ``is_central`` with a brute-force check on every element over F_2."""

    algebra = cdl.LoopAlgebra(loop, char=2)
    count = algebra.dimension
    bits = (np.arange(1 << count)[:, None] >> np.arange(count)) & 1
    everything = algebra.array(bits.reshape((-1,) + algebra.coefficient_shape))
    marks = everything.is_central()
    for number in range(0, 1 << count, 7 if count > 4 else 1):
        element = everything[number]
        assert bool(marks[number]) == _is_central_by_definition(algebra, element)
        assert element.is_central() == bool(marks[number])
    basis = algebra.center_basis()
    assert linalg.rank(
        basis.coefficients.reshape(len(basis), -1), field=algebra.field
    ) == len(basis)
    assert 1 << len(basis) == int(marks.sum())


@pytest.mark.parametrize("field", SMALL_FIELDS[:2], ids=_ids)
@pytest.mark.parametrize("loop", LARGER_LOOPS, ids=_ids)
def test_center_basis_has_the_expected_size_and_is_central(loop, field) -> None:
    """Count the basis and check that its elements commute and associate."""

    algebra = cdl.LoopAlgebra(loop, field=field)
    basis = algebra.center_basis()
    central = len(loop.center())
    assert len(basis) == central + (loop.quotient_order - central // 2)
    assert basis.is_central().all()
    x, y = algebra.random(1), algebra.random(2)
    for element in basis:
        assert element * x == x * element
        assert (element * x) * y == element * (x * y)
        assert (x * element) * y == x * (element * y)
        assert (x * y) * element == x * (y * element)
    assert not algebra.basis(loop.basis(1)).is_central() or loop.is_commutative()


def test_zero_divisors_and_annihilators_are_consistent() -> None:
    """Multiply annihilators back and match them with the zero-divisor tests."""

    for field in SMALL_FIELDS[:2]:
        for loop in (
            cdl.CDLoop(1),
            cdl.CDLoop(2),
            cdl.CDLoop(3),
            cdl.CentralProduct(1, 2),
        ):
            algebra = cdl.LoopAlgebra(loop, field=field)
            rng = np.random.default_rng(loop.order + field.deg)
            for _ in range(4):
                x = algebra.random(rng)
                left, right = x.left_annihilator(), x.right_annihilator()
                assert all(y * x == 0 for y in left)
                assert all(x * y == 0 for y in right)
                assert x.is_zero_divisor("left") == (len(right) > 0)
                assert x.is_zero_divisor("right") == (len(left) > 0)
                assert x.is_zero_divisor() == (len(left) + len(right) > 0)
                matrix = algebra.right_matrix(x)
                nullity = algebra.dimension - linalg.rank(matrix, field=field)
                assert len(left) == nullity
            assert algebra.zero().is_zero_divisor()
            assert not algebra.one().is_zero_divisor()
            batch = algebra.random(rng, batch_shape=(3,))
            marks = batch.is_zero_divisor()
            assert [bool(m) for m in marks] == [
                batch[i].is_zero_divisor() for i in range(3)
            ]


def test_annihilator_counts_match_an_exhaustive_search() -> None:
    """Count the annihilating elements of every element of a small algebra."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1), char=2)
    bits = (np.arange(16)[:, None] >> np.arange(4)) & 1
    everything = algebra.array(bits.reshape(16, 2, 2))
    for number in range(16):
        x = everything[number]
        killed = int((everything * x == 0).sum())
        assert killed == 1 << len(x.left_annihilator())
        assert int((x * everything == 0).sum()) == 1 << len(x.right_annihilator())


def test_multiplication_matrices_reproduce_products() -> None:
    """Apply the loop-basis matrices to coefficients over F_2 and F_4."""

    for field in SMALL_FIELDS[:2]:
        for loop in (cdl.CDLoop(2), cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            algebra = cdl.LoopAlgebra(loop, field=field)
            x, y = algebra.random(1), algebra.random(2)
            domain = algebra._field
            flat = y.coefficients.reshape((algebra.dimension,) + domain.shape)
            cases = [(algebra.left_matrix(x), x * y), (algebra.right_matrix(x), y * x)]
            for matrix, product in cases:
                terms = domain.multiply(matrix, flat[None])
                image = domain.sum(terms, 1)
                assert np.array_equal(
                    image.reshape(product.shape), product.coefficients
                )
            with pytest.raises(TypeError):
                algebra.left_matrix(x, basis="split")


# Decomposition and unavailable operations


def test_kappa_is_a_homomorphism_onto_the_group_algebra() -> None:
    """Map products to dyadic convolutions of group components."""

    for field in SMALL_FIELDS:
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            algebra = cdl.LoopAlgebra(loop, field=field)
            x, y = algebra.random(1), algebra.random(2)
            group = cdl.dyadic_convolve(x.kappa(), y.kappa(), field=field)
            assert isinstance(x.kappa(), np.ndarray)
            assert np.array_equal((x * y).kappa(), group)
            batch = algebra.random(3, batch_shape=(2,))
            assert np.array_equal(batch.kappa()[1], batch[1].kappa())


def test_dyadic_convolution_agrees_with_the_definition() -> None:
    """Compare the lifted convolution with a direct evaluation."""

    rng = np.random.default_rng(4)
    for degree in (1, 2, 5, 7, 12):
        field = cdl.BinaryField(deg=degree)
        ring = reference.FiniteField(2, field.modulus)
        for level in (0, 1, 3, 5):
            shape = (1 << level,) + field.coefficient_shape
            a = rng.integers(0, 2, shape, dtype=np.uint32)
            b = rng.integers(0, 2, shape, dtype=np.uint32)
            scalar = degree == 1
            left = reference.to_ring(ring, a[:, None] if scalar else a, 1)
            right = reference.to_ring(ring, b[:, None] if scalar else b, 1)
            expected = np.array(
                [list(c) for c in reference.dyadic_convolve(ring, left, right)],
                dtype=np.uint32,
            )
            got = cdl.dyadic_convolve(a, b, field=field)
            assert np.array_equal(got, expected[:, 0] if scalar else expected)
    with pytest.raises(TypeError):
        cdl.wht([1, 0, 1, 1], field=cdl.BinaryField())
    with pytest.raises(ValueError):
        cdl.dyadic_convolve([1, 0, 1], [1, 1, 0], field=cdl.BinaryField())


def _decomposition_calls(algebra, x, batch) -> list:
    """Return the calls that need the decomposition of the algebra."""

    half = np.zeros((algebra.loop.quotient_order, 2), dtype=np.uint32)
    return [
        x.spectrum,
        x.epsilon,
        batch.split,
        batch.spectrum,
        batch.epsilon,
        algebra.central_idempotents,
        algebra.primitive_idempotents,
        lambda: algebra.primitive_idempotent(0),
        lambda: algebra.from_components(half, half),
        lambda: algebra.lift_group(half),
        lambda: algebra.lift_signed(half),
        lambda: algebra.split_array(half, half),
        lambda: x.jordan_product(x),
        lambda: batch.jordan_product(x),
        lambda: algebra.right_matrix(x, basis="split"),
    ]


def test_operations_that_need_the_decomposition_raise() -> None:
    """Raise ``TypeError`` with the shared message for every such operation."""

    for loop in (cdl.CDLoop(2), cdl.CentralProduct(1, 2)):
        algebra = cdl.LoopAlgebra(loop, char=2, deg=2)
        x = algebra.random(1)
        batch = algebra.random(2, batch_shape=(2,))
        for call in _decomposition_calls(algebra, x, batch):
            with pytest.raises(TypeError, match="characteristic 2"):
                call()


def test_central_products_have_powers_only_in_characteristic_two() -> None:
    """Offer powers and inverses for a central product over F_2 but not over F_3."""

    loop = cdl.CentralProduct(2, 3)
    binary = cdl.LoopAlgebra(loop, char=2)
    x = binary.random(1)
    assert x**4 == binary.scalar(x.augmentation() ** 4)
    assert x.has_inverse() != x.is_nilpotent()
    assert binary.subalgebra_generated_by(x).dimension <= 4
    odd = cdl.LoopAlgebra(loop, char=3)
    y = odd.random(1)
    calls = [
        lambda: y**2,
        y.inverse,
        y.has_inverse,
        y.is_nilpotent,
        y.minimal_polynomial,
        lambda: odd.subalgebra_generated_by(y),
    ]
    for call in calls:
        with pytest.raises(TypeError, match="characteristic 2"):
            call()


# Construction, comparison and maps


def test_construction_reduces_integers_and_rejects_other_values() -> None:
    """Build elements from integers, scalars and coordinates."""

    loop = cdl.CDLoop(1)
    prime = cdl.LoopAlgebra(loop, char=2)
    assert prime([[3, 2], [5, -1]]).coefficients.tolist() == [[1, 0], [1, 1]]
    assert prime(1) == prime.one() and prime(2) == prime.zero()
    with pytest.raises(TypeError):
        prime([[1.0, 0], [0, 0]])
    with pytest.raises(TypeError):
        prime(True)
    field = cdl.BinaryField(deg=2)
    algebra = cdl.LoopAlgebra(loop, field=field)
    t = field([0, 1])
    x = algebra.from_terms({loop.basis(1): t, -loop.basis(0): 1})
    assert x.coefficient(loop.basis(1)) == t and x.coefficient(-loop.basis(0)) == 1
    assert x.terms() == {loop.basis(1): t, -loop.basis(0): field(1)}
    assert x.weight() == 2 and x.distance(algebra.zero()) == 2
    assert algebra.scalar(t).coefficients.tolist() == [
        [[0, 1], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    assert algebra([[t, 0], [1, t]]) == algebra([[[0, 1], [0, 0]], [[1, 0], [0, 1]]])
    assert x + t == x + algebra.scalar(t) and t * x == x * t
    with pytest.raises(ValueError):
        algebra([[1, 0], [0, 1]])
    with pytest.raises(TypeError):
        x + cdl.BinaryField(deg=3)(1)


def test_equality_hashing_bytes_and_random_elements() -> None:
    """Compare with scalars, hash consistently, and round-trip encodings."""

    for field in SMALL_FIELDS:
        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), field=field)
        x = algebra.random(9)
        assert algebra.from_bytes(x.to_bytes()) == x
        assert len(x.to_bytes()) == int(np.prod(algebra.coefficient_shape))
        assert hash(algebra(x.coefficients.copy())) == hash(x)
        assert algebra.one() == 1 and algebra.one() != 3 and algebra.zero() == 0
        assert hash(algebra.one()) == hash(1) and algebra.one() == field(1)
        assert x != cdl.LoopAlgebra(cdl.CDLoop(2), char=3)(1)
        secure = algebra.random(secure=True)
        assert int(secure.coefficients.max()) <= 1
        shared = np.zeros((2,) + algebra.coefficient_shape, dtype=np.uint32)
        assert np.shares_memory(algebra.asarray(shared).coefficients, shared)
        with pytest.raises(ValueError):
            algebra.asarray(shared + 2, copy=False)
        with pytest.raises(ValueError):
            algebra.from_bytes(b"\x02" * len(x.to_bytes()))


def test_automorphisms_and_embeddings_are_homomorphisms() -> None:
    """Preserve products under the maps induced by loop maps."""

    field = cdl.BinaryField(deg=2)
    algebra = cdl.LoopAlgebra(cdl.CDLoop(3), field=field)
    x, y = algebra.random(1), algebra.random(2)
    for index in range(8):
        images = [
            algebra.character_automorphism(index, value) for value in (x, y, x * y)
        ]
        assert images[0] * images[1] == images[2]
    small = cdl.LoopAlgebra(cdl.CDLoop(2), field=field)
    u, v = small.random(3), small.random(4)
    assert algebra.embed(u) * algebra.embed(v) == algebra.embed(u * v)
    assert algebra.embed(small.one()) == 1
    product = cdl.LoopAlgebra(cdl.CentralProduct(2, 1), field=field)
    assert product.factor_embedding(0, u) * product.factor_embedding(0, v) == (
        product.factor_embedding(0, u * v)
    )
    with pytest.raises(ValueError):
        algebra.embed(cdl.LoopAlgebra(cdl.CDLoop(2), char=3).one())


def test_identities_of_loop_algebras_in_characteristic_two() -> None:
    """Find every algebra flexible and power-associative, and the rest by level."""

    loops = [cdl.CDLoop(level) for level in (1, 2, 3, 4)]
    loops += [cdl.CentralProduct(*levels) for levels in ((1, 1), (1, 2), (2, 2))]
    loops += [cdl.CentralProduct(*levels) for levels in ((2, 3), (3, 3), (1, 4))]
    for loop in loops:
        for degree in (1, 2):
            algebra = cdl.LoopAlgebra(loop, char=2, deg=degree)
            highest = max(loop.levels)
            assert testing.is_flexible(algebra, 24, 1)
            assert testing.is_power_associative(algebra, 24, 2)
            assert testing.is_associative(algebra, 24, 3) == (highest <= 2)
            assert testing.is_alternative(algebra, 24, 4) == (highest <= 3)
            assert testing.is_commutative(algebra, 24, 5) == (highest <= 1)


# Linear algebra and codes


def test_linear_algebra_over_binary_fields() -> None:
    """Check ranks, row reduction and null spaces against exhaustive counts."""

    rng = np.random.default_rng(12)
    for field in (cdl.BinaryField(), cdl.BinaryField(deg=2)):
        domain = cdl.LoopAlgebra(cdl.CDLoop(0), field=field)._field
        for _ in range(6):
            matrix = rng.integers(
                0, 2, (4, 5) + field.coefficient_shape, dtype=np.uint32
            )
            rank = linalg.rank(matrix, field=field)
            reduced, pivots = linalg.row_reduce(matrix, field=field)
            assert len(pivots) == rank
            again, same = linalg.row_reduce(reduced, field=field)
            assert np.array_equal(again, reduced) and same == pivots
            kernel = linalg.nullspace(matrix, field=field)
            assert kernel.shape[0] == 5 - rank
            for vector in kernel:
                image = domain.sum(domain.multiply(matrix, vector[None]), 1)
                assert not image.any()
        if field.deg == 1:
            matrix = rng.integers(0, 2, (3, 4), dtype=np.uint32)
            vectors = (np.arange(16)[:, None] >> np.arange(4)) & 1
            solutions = int(((vectors @ matrix.T) % 2 == 0).all(axis=1).sum())
            assert solutions == 1 << linalg.nullspace(matrix, field=field).shape[0]


def test_ideals_of_the_loop_algebra_as_codes() -> None:
    """Build the augmentation ideal and the kernel of the group component."""

    loop = cdl.CDLoop(2)
    algebra = cdl.LoopAlgebra(loop, char=2)
    one = algebra.one()
    # The augmentation ideal has codimension one
    generators = [algebra.basis(g) + one for g in loop if g != loop.one()]
    code = codes.generator_matrix(generators)
    assert code.shape == (algebra.dimension - 1, algebra.dimension)
    assert not (code.sum(axis=1) % 2).any()
    # The kernel of the group component is generated by 1 + [-1]
    kernel = codes.generator_matrix(one + algebra.basis(loop.minus_one()))
    assert kernel.shape == (loop.quotient_order, algebra.dimension)
    for row in kernel:
        element = algebra(row.reshape(algebra.coefficient_shape))
        assert not element.kappa().any()
    # A left ideal is closed under left translation
    x = algebra.basis(loop.basis(0)) + algebra.basis(loop.basis(1))
    ideal = codes.generator_matrix(x)
    for row in ideal:
        translates = codes.translate_matrix(algebra(row.reshape(2, 4)))
        stacked = np.concatenate([ideal, translates])
        assert linalg.rank(stacked, field=algebra.field) == ideal.shape[0]
    assert codes.to_gap(ideal, field=algebra.field).endswith("]*Z(2)^0, GF(2));")
    assert codes.to_sage(ideal, field=algebra.field).startswith(
        "LinearCode(matrix(GF(2)"
    )
    with pytest.raises(ValueError):
        codes.to_gap(ideal, field=cdl.BinaryField(deg=2))


def test_powers_use_at_most_two_products(monkeypatch) -> None:
    """Count the products of positive and negative powers."""

    algebra = cdl.LoopAlgebra(cdl.CentralProduct(2, 1), char=2, deg=3)
    x = next(y for y in map(algebra.random, range(20)) if y.has_inverse())
    calls = []
    original = _kernel.multiply

    def multiply(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_kernel, "multiply", multiply)
    expected = {0: 0, 1: 0, 2: 1, 3: 2, 4: 0, 7: 2, -1: 2, -2: 1, -3: 0, -4: 0, -7: 0}
    for exponent, count in expected.items():
        calls.clear()
        power = x**exponent
        assert len(calls) == count, exponent
        calls.clear()
        if exponent < 0:
            assert power * x ** (-exponent) == 1, exponent
    batch = algebra.random(1, batch_shape=(4,))
    if not batch.has_inverse().all():
        with pytest.raises(ZeroDivisionError):
            batch**-2


def test_results_never_share_the_storage_of_an_operand() -> None:
    """Return independent arrays from powers with exponent one."""

    for selectors in ({"char": 2}, {"char": 2, "deg": 3}, {}, {"char": 7}):
        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
        # Small integer coefficients, so that powers are exact over R
        batch = testing.random_elements(algebra, 3, 0)
        results = [batch**1, batch.left_power(1), batch.right_power(1), +batch]
        for result in results:
            assert not np.shares_memory(result.coefficients, batch.coefficients)
            assert (result == batch).all()
        if algebra.char != 2:
            split = batch.split()
            for result in (split**1, split.left_power(1), split.right_power(1)):
                assert not np.shares_memory(result.spectral, split.spectral)
                assert not np.shares_memory(
                    np.asarray(result.signed), np.asarray(split.signed)
                )
                assert (result == split).all()


def test_the_automatic_choice_counts_broadcast_products(monkeypatch) -> None:
    """Choose the algorithm from the number of products after broadcasting."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(3), char=2, deg=3)
    seen = []
    original = cdl.LoopAlgebra._binary_choice

    def choice(self, records):
        seen.append(records)
        return original(self, records)

    monkeypatch.setattr(cdl.LoopAlgebra, "_binary_choice", choice)
    x = algebra.random(1, batch_shape=(4, 1))
    y = algebra.random(2, batch_shape=(1, 4))
    product = x * y
    assert seen == [16] and product.batch_shape == (4, 4)
    seen.clear()
    assert (algebra.multiply(x, y) == product).all() and seen == [16]
    seen.clear()
    element = algebra.random(3)
    assert (x * element).batch_shape == (4, 1) and seen == [4]
    seen.clear()
    assert element * element == algebra.multiply(element, element, algorithm="table")
    assert seen == [1]
    assert element.dimension == algebra.dimension == x.dimension == 16
