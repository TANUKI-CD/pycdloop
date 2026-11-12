"""Tests of the quasilinear schedule with coefficients in an algebra."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyfastcd as fcd
import pytest

from pycdloop._field import _coefficient_field
from pycdloop._schedule import schedule_multiply, schedule_product_count

try:
    from . import reference
except ImportError:  # Support collection from inside the tests directory.
    import reference

FIELDS = [
    ({}, reference.Rationals()),
    ({"char": 0, "deg": 2}, reference.GaussianRationals()),
    ({"char": 13}, reference.FiniteField(13)),
    (
        {"field": fcd.FiniteField.from_modulus(char=7, modulus=(1, 0, 1))},
        reference.FiniteField(7, (1, 0, 1)),
    ),
]
FIELD_IDS = ["R", "C", "F13", "F7^2"]


def _random(field: Any, shape: tuple[int, ...], rng: Any) -> np.ndarray:
    """Return exactly representable random coefficients of a field."""

    if field.kind == "finite":
        return rng.integers(0, field.char, shape + field.shape).astype(np.uint32)
    values = rng.integers(-4, 5, shape).astype(field.dtype)
    if field.kind == "complex":
        values = values + 1j * rng.integers(-4, 5, shape)
    return values


class _Counter:
    """A product of the inner algebra that records how many products it forms."""

    def __init__(self, multiply: Any) -> None:
        """Wrap a product on arrays of shape ``(calls, count, ...)``."""

        self.multiply = multiply
        self.products = 0
        self.batches = 0

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Count one product for each row of each call and delegate."""

        assert left.shape == right.shape
        self.products += left.shape[0] * left.shape[1]
        self.batches += 1
        return self.multiply(left, right)


@pytest.mark.parametrize(("selectors", "ring"), FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("level", range(8))
def test_scalar_coefficients_match_the_doubling_formula(
    selectors: dict[str, Any], ring: Any, level: int
) -> None:
    """With a one-dimensional inner algebra the schedule multiplies in A_n."""

    field = _coefficient_field(**selectors)
    rng = np.random.default_rng(100 + level)
    size = 1 << level
    u = _random(field, (2, size, 1), rng)
    v = _random(field, (2, size, 1), rng)
    before = u.copy(), v.copy()
    result = schedule_multiply(field, field.multiply, level, u, v)
    assert result.shape == u.shape and result.dtype == field.dtype
    assert np.array_equal(u, before[0]) and np.array_equal(v, before[1])
    for call in range(2):
        expected = reference.cd_multiply(
            ring,
            reference.to_ring(ring, u[call, :, 0], 1),
            reference.to_ring(ring, v[call, :, 0], 1),
        )
        stored = reference.from_ring(ring, expected, 1, field.dtype)
        assert np.array_equal(result[call, :, 0], stored)


@pytest.mark.parametrize("level", range(11))
def test_scalar_coefficients_match_pyfastcd(level: int) -> None:
    """Over R the schedule agrees with every pyfastcd algorithm on integers."""

    field = _coefficient_field()
    algebra = fcd.Algebra(level=level)
    rng = np.random.default_rng(level)
    u = rng.integers(-4, 5, (3, 1 << level, 1)).astype(np.float64)
    v = rng.integers(-4, 5, (3, 1 << level, 1)).astype(np.float64)
    result = schedule_multiply(field, field.multiply, level, u, v)[..., 0]
    left = algebra.asarray(np.ascontiguousarray(u[..., 0]))
    right = algebra.asarray(np.ascontiguousarray(v[..., 0]))
    expected = algebra.multiply(left, right, algorithm="direct")
    assert np.array_equal(result, expected.coefficients)


@pytest.mark.parametrize("level", range(13))
def test_product_count_is_the_closed_form(level: int) -> None:
    """The number of inner products is (9n - 15) 2^(n-1) + 10, and 1 for n = 0."""

    expected = 1 if level == 0 else (9 * level - 15) * (1 << (level - 1)) + 10
    assert schedule_product_count(level) == expected
    field = _coefficient_field()
    rng = np.random.default_rng(level)
    for calls in (1, 3):
        counter = _Counter(field.multiply)
        u = rng.integers(-2, 3, (calls, 1 << level, 1)).astype(np.float64)
        v = rng.integers(-2, 3, (calls, 1 << level, 1)).astype(np.float64)
        schedule_multiply(field, counter, level, u, v)
        assert counter.products == calls * expected
        # One batch at the root and one for each level of the recursion below it
        assert counter.batches == max(1, level - 1)


def test_product_count_beats_the_direct_formula_from_level_three() -> None:
    """The schedule needs fewer inner products than N^2 once N is at least 8."""

    assert [schedule_product_count(n) for n in range(5)] == [1, 4, 16, 58, 178]
    for level in range(3, 20):
        assert schedule_product_count(level) < 4**level


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply rows of quaternion coefficients through pyfastcd."""

    algebra = fcd.Algebra(dimension=4)
    product = algebra.multiply(algebra.asarray(left), algebra.asarray(right))
    return np.asarray(product.coefficients)


@pytest.mark.parametrize("level", range(6))
def test_quaternion_coefficients_keep_operand_order(level: int) -> None:
    """Rows in the quaternions multiply as the tensor product A_n (x) H."""

    field = _coefficient_field()
    ring = reference.Rationals()
    rng = np.random.default_rng(40 + level)
    size = 1 << level
    u = rng.integers(-3, 4, (2, size, 4)).astype(np.float64)
    v = rng.integers(-3, 4, (2, size, 4)).astype(np.float64)
    result = schedule_multiply(field, _quaternion_product, level, u, v)
    for call in range(2):
        expected = reference.tensor_multiply(
            ring,
            (level, 2),
            reference.to_ring(ring, u[call].ravel(), 1),
            reference.to_ring(ring, v[call].ravel(), 1),
        )
        stored = reference.from_ring(ring, expected, 1, np.float64)
        assert np.array_equal(result[call].ravel(), stored)
    if level >= 1:
        # Reversing every inner product gives a different element
        reversed_products = schedule_multiply(
            field, lambda a, b: _quaternion_product(b, a), level, u, v
        )
        assert not np.array_equal(reversed_products, result)


def _matrix_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply rows of four coefficients as 2 x 2 matrices."""

    shape = left.shape[:-1] + (2, 2)
    return np.matmul(left.reshape(shape), right.reshape(shape)).reshape(left.shape)


@pytest.mark.parametrize("level", range(6))
def test_matrix_coefficients_keep_operand_order(level: int) -> None:
    """With rows in the 2 x 2 matrices the schedule matches the defining sum."""

    field = _coefficient_field()
    rng = np.random.default_rng(70 + level)
    size = 1 << level
    u = rng.integers(-3, 4, (3, size, 4)).astype(np.float64)
    v = rng.integers(-3, 4, (3, size, 4)).astype(np.float64)
    result = schedule_multiply(field, _matrix_product, level, u, v)
    signs = reference.sign_table(level)
    expected = np.zeros(u.shape)
    for p in range(size):
        for r in range(size):
            expected[:, p ^ r] += signs[p][r] * _matrix_product(u[:, p], v[:, r])
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(("selectors", "ring"), FIELDS, ids=FIELD_IDS)
def test_calls_are_independent(selectors: dict[str, Any], ring: Any) -> None:
    """A batch of calls gives the results of the calls made one at a time."""

    field = _coefficient_field(**selectors)
    rng = np.random.default_rng(5)
    for level in (0, 1, 2, 4, 5):
        u = _random(field, (4, 1 << level, 3), rng)
        v = _random(field, (4, 1 << level, 3), rng)
        together = schedule_multiply(field, field.multiply, level, u, v)
        for call in range(4):
            alone = schedule_multiply(
                field, field.multiply, level, u[call : call + 1], v[call : call + 1]
            )
            assert np.array_equal(together[call : call + 1], alone)
        empty = schedule_multiply(field, field.multiply, level, u[:0], v[:0])
        assert empty.shape == u[:0].shape


@pytest.mark.parametrize(("selectors", "ring"), FIELDS, ids=FIELD_IDS)
def test_identity_and_zero(selectors: dict[str, Any], ring: Any) -> None:
    """The row e_0 (x) 1 is a two-sided identity and zero annihilates."""

    field = _coefficient_field(**selectors)
    rng = np.random.default_rng(9)
    for level in (0, 1, 3, 6):
        size = 1 << level
        v = _random(field, (2, size, 1), rng)
        one = field.zeros((2, size, 1))
        one[:, 0, 0] = field.scalar(1)
        zero = field.zeros((2, size, 1))
        assert np.array_equal(
            schedule_multiply(field, field.multiply, level, one, v), v
        )
        assert np.array_equal(
            schedule_multiply(field, field.multiply, level, v, one), v
        )
        product = schedule_multiply(field, field.multiply, level, zero, v)
        assert not product.any()


def test_basis_products_reproduce_the_sign_table() -> None:
    """Products of basis rows are signed basis rows at the XOR of the indices."""

    field = _coefficient_field()
    level = 4
    size = 1 << level
    signs = reference.sign_table(level)
    basis = np.eye(size)[:, :, None]
    for p in range(size):
        left = np.broadcast_to(basis[p], (size, size, 1)).copy()
        products = schedule_multiply(field, field.multiply, level, left, basis.copy())
        for q in range(size):
            expected = np.zeros(size)
            expected[p ^ q] = signs[p][q]
            assert np.array_equal(products[q, :, 0], expected)
