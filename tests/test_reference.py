"""Checks of the independent reference arithmetic against hand-computed facts."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

try:
    from . import reference
except ImportError:  # Support collection from inside the tests directory.
    import reference


def _basis(size: int, index: int) -> list[Fraction]:
    """Return a rational basis vector."""

    return [Fraction(int(i == index)) for i in range(size)]


def test_quaternion_and_octonion_orientation() -> None:
    """The sign tables fix the orientation documented for pyfastcd."""

    quaternions = reference.sign_table(2)
    assert quaternions[1][2] == 1 and quaternions[2][3] == 1 and quaternions[3][1] == 1
    assert quaternions[2][1] == -1
    octonions = reference.sign_table(3)
    for p, q, r in ((1, 2, 3), (1, 4, 5), (2, 4, 6), (3, 4, 7)):
        assert p ^ q == r
        assert octonions[p][q] == 1 and octonions[q][p] == -1


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_sign_table_basic_identities(level: int) -> None:
    """The identity is neutral, imaginary units square to -1 and anticommute."""

    table = reference.sign_table(level)
    size = 1 << level
    for p in range(size):
        assert table[0][p] == 1 and table[p][0] == 1
        assert table[p][p] == (1 if p == 0 else -1)
        for q in range(1, size):
            if p not in (0, q):
                assert table[p][q] == -table[q][p]


def test_doubling_product_of_complex_numbers_and_quaternions() -> None:
    """The doubling recursion reproduces two products computed by hand."""

    ring = reference.Rationals()
    x = [Fraction(1), Fraction(2)]
    y = [Fraction(3), Fraction(-1)]
    # (1 + 2i)(3 - i) = 5 + 5i
    assert reference.cd_multiply(ring, x, y) == [5, 5]
    # (1 + 2i + 3j + 4k)(2 - i + j) = 1 - i + 3j + 13k
    left = [Fraction(v) for v in (1, 2, 3, 4)]
    right = [Fraction(v) for v in (2, -1, 1, 0)]
    assert reference.cd_multiply(ring, left, right) == [1, -1, 3, 13]


def test_conjugation_and_norm() -> None:
    """Conjugation negates the imaginary part and x conj(x) is the norm."""

    ring = reference.Rationals()
    x = [Fraction(v) for v in (1, -2, 3, 4, 0, 5, -1, 2)]
    conjugate = reference.cd_conjugate(ring, x)
    assert conjugate == [x[0]] + [-v for v in x[1:]]
    product = reference.cd_multiply(ring, x, conjugate)
    assert product == [sum(v * v for v in x)] + [0] * 7


def test_gaussian_rationals() -> None:
    """Pairs of rationals multiply as complex numbers."""

    ring = reference.GaussianRationals()
    a, b = ring.convert(1 + 2j), ring.convert(3 - 1j)
    assert ring.mul(a, b) == (5, 5)
    assert ring.store(ring.add(a, ring.neg(b))) == -2 + 3j


def test_finite_field_multiplication_by_hand() -> None:
    """Reduction modulo t^2 + 1 over F_7 and t^3 + t + 1 over F_5 by hand."""

    gaussian = reference.FiniteField(7, (1, 0, 1))
    # (2 + 3t)(4 + 5t) = 8 + 22t + 15t^2 = (8 - 15) + 22t = 0 + t
    assert gaussian.mul((2, 3), (4, 5)) == (0, 1)
    cubic = reference.FiniteField(5, (1, 1, 0, 1))
    # t^2 * t^2 = t * t^3 = t(-t - 1) = -t^2 - t
    assert cubic.mul((0, 0, 1), (0, 0, 1)) == (0, 4, 4)
    prime = reference.FiniteField(13)
    assert prime.mul((7,), (8,)) == (4,)
    assert prime.neg((0,)) == (0,) and prime.add((12,), (5,)) == (4,)


def test_finite_field_agrees_with_second_implementation() -> None:
    """Polynomial multiplication matches NumPy convolution and long division."""

    modulus = (2, 0, 1, 1)
    field = reference.FiniteField(11, modulus)
    rng = np.random.default_rng(3)
    for _ in range(40):
        a = tuple(int(v) for v in rng.integers(0, 11, 3))
        b = tuple(int(v) for v in rng.integers(0, 11, 3))
        full = np.convolve(a[::-1], b[::-1])
        _, remainder = np.polydiv(full, np.array(modulus[::-1], dtype=float))
        expected = [int(round(v)) % 11 for v in remainder[::-1]]
        expected += [0] * (3 - len(expected))
        assert list(field.mul(a, b)) == expected[:3]


def test_factor_indices_and_twist() -> None:
    """Flat indices concatenate factor indices and the twist multiplies signs."""

    assert reference.factor_indices((1, 2), 0b110) == (1, 2)
    assert reference.factor_indices((2, 1, 1), 0b1101) == (3, 0, 1)
    # e_1 (x) e_1 squares to (-1)(-1) = +1
    assert reference.twist((1, 1), 3, 3) == 1
    assert reference.twist((1, 2), 0b101, 0b110) == reference.sign_table(2)[1][2] * -1
    assert reference.loop_product((2,), (-1, 1), (1, 2)) == (-1, 3)


def test_walsh_hadamard_and_dyadic_convolution() -> None:
    """Tiny transforms and convolutions match values computed by hand."""

    ring = reference.Rationals()
    values = [Fraction(v) for v in (1, 2, 3, 4)]
    assert reference.walsh_hadamard(ring, values) == [10, -2, -4, 0]
    delta = _basis(4, 1)
    # Convolution with the basis vector at 1 swaps neighbours
    assert reference.dyadic_convolve(ring, values, delta) == [2, 1, 4, 3]
    twice = reference.walsh_hadamard(ring, reference.walsh_hadamard(ring, values))
    assert twice == [4 * v for v in values]


def test_ring_round_trip() -> None:
    """Conversion to ring elements and back preserves stored coefficients."""

    array = np.array([[1.5, -2.0], [0.25, 3.0]])
    ring = reference.Rationals()
    nested = reference.to_ring(ring, array, 2)
    assert nested[0][0] == Fraction(3, 2)
    assert np.array_equal(reference.from_ring(ring, nested, 2, np.float64), array)
    field = reference.FiniteField(7, (1, 0, 1))
    residues = np.array([[1, 2], [3, 4]], dtype=np.uint32)
    nested = reference.to_ring(field, residues, 1)
    assert nested == [(1, 2), (3, 4)]
    assert np.array_equal(reference.from_ring(field, nested, 1, np.uint32), residues)
