"""Independent exact arithmetic and coefficient-field regression tests."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pyfastcd as fcd
import pytest

import pycdloop as cdl

try:  # Support both package-style collection and collection from this directory.
    from . import reference
except ImportError:  # pragma: no cover - depends on the collection mechanism
    import reference


# Characteristic and degree of every finite field exercised below
FIELDS = [
    (3, 1),
    (5, 1),
    (13, 1),
    (65521, 1),
    (2147483647, 1),
    (3, 5),
    (5, 2),
    (13, 5),
    (65521, 2),
]
LEVELS = list(range(7))

# The reference derives the signs of a level by multiplying every pair of basis
# vectors with the doubling formula. Above this level that table is costly, and
# products are checked through the decomposition instead.
DIRECT_ORACLE_LEVEL = 5


def ring_of(algebra: Any) -> Any:
    """Return the oracle ring that matches an algebra's coefficient field."""

    if algebra.field is not None:
        return reference.FiniteField(algebra.char, algebra.field.modulus)
    if algebra.deg == 2:
        return reference.GaussianRationals()
    return reference.Rationals()


def exact_values(algebra: Any, rng: Any) -> np.ndarray:
    """Draw coefficients on which the arithmetic of the tests is exact."""

    shape = algebra.coefficient_shape
    if algebra.field is not None:
        return rng.integers(0, algebra.char, size=shape, dtype=np.uint32)
    values = rng.integers(-3, 4, size=shape).astype(algebra.dtype)
    if algebra.deg == 2:
        values = values + 1j * rng.integers(-3, 4, size=shape)
    return values


def half(ring: Any) -> Any:
    """Return the ring element one half."""

    if isinstance(ring, reference.FiniteField):
        return ((ring.p + 1) // 2,) + (0,) * (ring.deg - 1)
    if isinstance(ring, reference.GaussianRationals):
        return (Fraction(1, 2), Fraction(0))
    return Fraction(1, 2)


def oracle_direct(algebra: Any, x: Any, y: Any) -> np.ndarray:
    """Multiply two coefficient arrays straight from the definition."""

    ring = ring_of(algebra)
    left = reference.to_ring(ring, x, 2)
    right = reference.to_ring(ring, y, 2)
    product = reference.loop_algebra_multiply(ring, algebra.loop.levels, left, right)
    return reference.from_ring(ring, product, 2, algebra.dtype)


def oracle_decomposed(algebra: Any, x: Any, y: Any) -> np.ndarray:
    """Multiply two coefficient arrays of a Cayley--Dickson loop algebra.

    The group components are multiplied by dyadic convolution and the signed
    components by the doubling formula, both from the reference, and the
    product is assembled from the two results.
    """

    ring = ring_of(algebra)

    def components(values: Any) -> tuple[list[Any], list[Any]]:
        """Return the group and signed components of one operand."""

        plus, minus = reference.to_ring(ring, values, 2)
        group = [ring.add(a, b) for a, b in zip(plus, minus, strict=True)]
        signed = [ring.add(a, ring.neg(b)) for a, b in zip(plus, minus, strict=True)]
        return group, signed

    (c, d), (c2, d2) = components(x), components(y)
    f = reference.dyadic_convolve(ring, c, c2)
    g = reference.cd_multiply(ring, d, d2)
    scale = half(ring)
    plus = [ring.mul(ring.add(a, b), scale) for a, b in zip(f, g, strict=True)]
    minus = [
        ring.mul(ring.add(a, ring.neg(b)), scale) for a, b in zip(f, g, strict=True)
    ]
    return reference.from_ring(ring, [plus, minus], 2, algebra.dtype)


def oracle_product(algebra: Any, x: Any, y: Any) -> np.ndarray:
    """Multiply with the definition where affordable, else by decomposition."""

    if algebra.loop.level <= DIRECT_ORACLE_LEVEL:
        return oracle_direct(algebra, x, y)
    return oracle_decomposed(algebra, x, y)


def check_products(algebra: Any, seed: int) -> None:
    """Compare every algorithm choice of an algebra with the reference."""

    rng = np.random.default_rng(seed)
    x_values, y_values = exact_values(algebra, rng), exact_values(algebra, rng)
    expected = oracle_product(algebra, x_values, y_values)
    x, y = algebra(x_values), algebra(y_values)
    np.testing.assert_array_equal((x * y).coefficients, expected)
    for name in ("split", "table"):
        product = algebra.multiply(x, y, algorithm=name)
        np.testing.assert_array_equal(product.coefficients, expected)
    for name in algebra.signed_algorithms:
        product = algebra.multiply(x, y, signed_algorithm=name)
        np.testing.assert_array_equal(product.coefficients, expected)
    batch_x = algebra.array(np.stack([x_values, y_values]))
    batch_y = algebra.array(np.stack([y_values, y_values]))
    np.testing.assert_array_equal((batch_x * batch_y)[0].coefficients, expected)
    split = (batch_x.split() * batch_y.split()).to_loop()
    np.testing.assert_array_equal(split[0].coefficients, expected)


@pytest.mark.parametrize("p,d", FIELDS)
@pytest.mark.parametrize("level", LEVELS)
def test_finite_products_match_reference(p: int, d: int, level: int) -> None:
    """Check products over every finite field against schoolbook arithmetic."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(level), char=p, deg=d)
    assert algebra.dtype == np.uint32
    assert algebra.coefficient_shape == (2, 1 << level) + ((d,) if d > 1 else ())
    check_products(algebra, p + d + level)


@pytest.mark.parametrize("level", LEVELS)
def test_complex_products_match_reference(level: int) -> None:
    """Check products over C against exact Gaussian rational arithmetic."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(level), char=0, deg=2)
    assert algebra.dtype == np.complex128
    check_products(algebra, level)


@pytest.mark.parametrize("level", LEVELS)
def test_real_products_match_reference(level: int) -> None:
    """Check products over R against exact rational arithmetic."""

    check_products(cdl.LoopAlgebra(cdl.CDLoop(level)), 100 + level)


@pytest.mark.parametrize(
    "char,modulus",
    [(7, (1, 0, 1)), (5, (1, 1, 0, 1)), (13, (2, 0, 1)), (3, (2, 1, 1))],
)
@pytest.mark.parametrize("level", [0, 2, 3, 4])
def test_explicit_presentations_match_reference(
    char: int, modulus: tuple[int, ...], level: int
) -> None:
    """Check products over fields given by an explicit irreducible polynomial."""

    field = fcd.FiniteField.from_modulus(char=char, modulus=modulus)
    assert tuple(field.modulus) == modulus
    algebra = cdl.LoopAlgebra(cdl.CDLoop(level), field=field)
    assert algebra.field is field
    assert algebra.deg == len(modulus) - 1
    check_products(algebra, char + level)


@pytest.mark.parametrize("p,d", [(3, 1), (13, 1), (5, 2), (2147483647, 1)])
@pytest.mark.parametrize("levels", [(1, 1), (1, 2), (2, 2), (2, 1, 1)])
def test_central_products_match_reference(
    p: int, d: int, levels: tuple[int, ...]
) -> None:
    """Check central-product algebras over finite fields against the definition."""

    algebra = cdl.LoopAlgebra(cdl.CentralProduct(*levels), char=p, deg=d)
    check_products(algebra, p + d + sum(levels))


@pytest.mark.parametrize(
    "selectors", [{}, {"char": 0, "deg": 2}, {"char": 13, "deg": 2}]
)
@pytest.mark.parametrize("level", [0, 1, 3, 5])
def test_reference_routes_agree(selectors: dict[str, int], level: int) -> None:
    """Check the decomposition route of the reference against the definition."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
    rng = np.random.default_rng(level)
    x, y = exact_values(algebra, rng), exact_values(algebra, rng)
    np.testing.assert_array_equal(
        oracle_decomposed(algebra, x, y), oracle_direct(algebra, x, y)
    )


@pytest.mark.parametrize("p,d", FIELDS)
def test_algebra_laws_hold_exactly(p: int, d: int) -> None:
    """Check the laws that hold in every loop algebra, without rounding."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(4), char=p, deg=d)
    rng = np.random.default_rng(p + d)
    x, y, z = (algebra(exact_values(algebra, rng)) for _ in range(3))
    assert (x + y) * z == x * z + y * z
    assert z * (x + y) == z * x + z * y
    assert (x * y) * x == x * (y * x)
    assert x.commutator(y) == x * y - y * x
    assert x.associator(y, z) == (x * y) * z - x * (y * z)
    assert x**3 == (x * x) * x
    assert x**5 == x * (x * (x * (x * x)))
    assert (x - x) == 0
    assert x * 2 == x + x
    assert (x / 2) * 2 == x
    assert algebra.from_components(x.kappa(), x.epsilon()) == x
    assert algebra.from_bytes(x.to_bytes()) == x
    plus, minus = algebra.central_idempotents()
    assert plus * plus == plus and minus * minus == minus
    assert plus + minus == 1 and not plus * minus
    assert (x * y).conjugate() == y.conjugate() * x.conjugate()


@pytest.mark.parametrize("level", [0, 3, 8])
def test_largest_residues_do_not_overflow(level: int) -> None:
    """Multiply all-``p - 1`` operands at the largest supported characteristic."""

    p = 2147483647
    algebra = cdl.LoopAlgebra(cdl.CDLoop(level), char=p)
    size = algebra.loop.quotient_order
    values = np.full((2, size), p - 1, dtype=np.uint32)
    x = algebra(values)
    # x is minus the sum s of all basis elements, and s s = len(loop) s
    expected = np.full((2, size), (2 * size) % p, dtype=np.uint32)
    for name in ("split", "table"):
        product = algebra.multiply(x, x, algorithm=name)
        np.testing.assert_array_equal(product.coefficients, expected)
    for name in algebra.signed_algorithms:
        product = algebra.multiply(x, x, signed_algorithm=name)
        np.testing.assert_array_equal(product.coefficients, expected)
    np.testing.assert_array_equal((x + x).coefficients, np.full((2, size), p - 2))
    np.testing.assert_array_equal((x * (p - 1)).coefficients, np.ones((2, size)))
    np.testing.assert_array_equal((-x).coefficients, np.ones((2, size)))
    np.testing.assert_array_equal((x - x).coefficients, np.zeros((2, size)))
    spectrum = np.asarray(x.spectrum().coefficients)
    assert int(spectrum[0]) == (-2 * size) % p
    assert not spectrum[1:].any()
    assert not np.asarray(x.epsilon().coefficients).any()
    batch = algebra.array(np.stack([values, values]))
    np.testing.assert_array_equal((batch * batch).coefficients[1], expected)
    np.testing.assert_array_equal(
        (batch.split() * batch.split()).to_loop().coefficients[0], expected
    )


@pytest.mark.parametrize("p,d", [(65521, 2), (2147483647, 1), (13, 5)])
def test_alternating_extreme_residues(p: int, d: int) -> None:
    """Multiply operands whose residues alternate between 1 and ``p - 1``."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(5), char=p, deg=d)
    shape = algebra.coefficient_shape
    pattern = np.where(np.arange(int(np.prod(shape))) % 2 == 0, 1, p - 1)
    values = pattern.reshape(shape).astype(np.uint32)
    reversed_values = np.ascontiguousarray(values[::-1, ::-1])
    expected = oracle_direct(algebra, values, reversed_values)
    x, y = algebra(values), algebra(reversed_values)
    for name in ("split", "table"):
        product = algebra.multiply(x, y, algorithm=name)
        np.testing.assert_array_equal(product.coefficients, expected)
    assert int((x * y).coefficients.max()) < p


def test_invalid_selectors_are_rejected_as_in_pyfastcd() -> None:
    """Raise the exception types that pyfastcd documents for selectors."""

    loop = cdl.CDLoop(1)
    for selectors in (
        {"char": 9},
        {"char": 15},
        {"char": -3},
        {"char": 1},
        {"char": 2**31 + 11},
        {"char": 3, "deg": 0},
        {"char": 3, "deg": 13},
        {"char": 0, "deg": 3},
        {"char": 0, "deg": 0},
    ):
        with pytest.raises(ValueError):
            cdl.LoopAlgebra(loop, **selectors)
        with pytest.raises(ValueError):
            fcd.Algebra(level=1, **selectors)
    for selectors in (
        {"char": True},
        {"char": np.bool_(True)},
        {"char": 3, "deg": True},
        {"char": 3.0},
        {"char": "3"},
        {"field": "F_3"},
        {"field": fcd.Algebra(level=1)},
        {"field": fcd.FiniteField(char=3), "deg": 2},
        {"field": fcd.FiniteField(char=3), "char": 3},
    ):
        with pytest.raises(TypeError):
            cdl.LoopAlgebra(loop, **selectors)


def test_selectors_identify_the_pyfastcd_field() -> None:
    """Share the coefficient field object with pyfastcd."""

    loop = cdl.CDLoop(2)
    for p, d in FIELDS:
        algebra = cdl.LoopAlgebra(loop, char=p, deg=d)
        field = fcd.FiniteField(char=p, deg=d)
        assert algebra.field is field
        assert algebra is cdl.LoopAlgebra(loop, field=field)
        assert algebra.cd_algebra is fcd.Algebra(level=2, field=field)
        assert (algebra.char, algebra.deg) == (p, d)
    assert cdl.LoopAlgebra(loop, char=np.int64(7)) is cdl.LoopAlgebra(loop, char=7)
    # Every monic linear modulus presents the same prime field
    linear = fcd.FiniteField.from_modulus(char=7, modulus=(3, 1))
    assert cdl.LoopAlgebra(loop, field=linear) is cdl.LoopAlgebra(loop, char=7)


def test_real_scalar_types() -> None:
    """Accept real Python and NumPy scalars over R and reject complex ones."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
    x = algebra(np.arange(8.0).reshape(2, 4))
    for scalar in (
        2,
        2.0,
        Fraction(2),
        Decimal(2),
        np.float64(2),
        np.float32(2),
        np.int16(2),
        np.uint8(2),
        np.array(2),
    ):
        assert x * scalar == x + x
        assert scalar * x == x + x
        assert x + scalar == x + algebra.scalar(2)
        assert algebra(scalar) == 2
    assert x * Fraction(1, 4) == x / 4
    assert x * Decimal("0.25") == x / 4
    for scalar in (1j, np.complex64(1), "2", b"2", None):
        with pytest.raises(TypeError):
            x * scalar
        with pytest.raises(TypeError):
            algebra.scalar(scalar)


def test_complex_scalar_types() -> None:
    """Accept real and complex scalars over C."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=0, deg=2)
    values = np.arange(8.0).reshape(2, 4) * (1 - 2j)
    x = algebra(values)
    for scalar in (2, 0.5, 1j, 2 - 3j, np.complex128(1 + 1j), np.complex64(1j)):
        np.testing.assert_array_equal((x * scalar).coefficients, values * scalar)
        np.testing.assert_array_equal((scalar * x).coefficients, values * scalar)
        assert algebra.scalar(scalar) == complex(scalar)
    assert x * Fraction(1, 2) == x / 2
    assert algebra(np.ones((2, 4))).coefficients.dtype == np.complex128
    assert algebra.scalar(1j).coefficient(algebra.loop.one()) == 1j
    # The involution moves coefficients and does not conjugate them
    loop = algebra.loop
    moved = (1j * algebra.basis(loop.basis(1))).conjugate()
    assert moved == 1j * algebra.basis(-loop.basis(1))
    for scalar in ("2", None, [1j]):
        with pytest.raises(TypeError):
            x * scalar


@pytest.mark.parametrize("p,d", FIELDS)
def test_finite_scalar_types(p: int, d: int) -> None:
    """Accept integers and scalars of the same field over a finite field."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=p, deg=d)
    field = algebra.field
    x = algebra.random(p + d)
    two = x + x
    for scalar in (2, p + 2, 2 - p, np.int64(2), np.uint32(2), field(2), field(p + 2)):
        assert x * scalar == two
        assert scalar * x == two
        assert (x + scalar) - scalar == x
        assert algebra.scalar(scalar) == 2
    assert x * 0 == 0 and x * p == 0
    assert (x * 2) / 2 == x
    assert x / field(2) * 2 == x
    with pytest.raises(ZeroDivisionError):
        x / p
    with pytest.raises(ZeroDivisionError):
        x / field(0)
    other = fcd.FiniteField(char=7 if p != 7 else 11)
    for scalar in (2.0, 1j, Fraction(2), Decimal(2), "2", None, other(2), True):
        with pytest.raises(TypeError):
            x * scalar
    if d > 1:
        generator = field([0, 1] + [0] * (d - 2))
        ring = reference.FiniteField(p, field.modulus)
        factor = ring.convert(np.asarray(generator.coefficients))
        expected = [
            [ring.mul(entry, factor) for entry in row]
            for row in reference.to_ring(ring, x.coefficients, 2)
        ]
        np.testing.assert_array_equal(
            (x * generator).coefficients,
            reference.from_ring(ring, expected, 2, np.uint32),
        )
        assert (x * generator) / generator == x
        assert algebra.scalar(generator).coefficient(algebra.loop.one()) == generator


@pytest.mark.parametrize("p,d", FIELDS)
def test_equality_and_hashing_with_field_scalars(p: int, d: int) -> None:
    """Keep ``==`` and ``hash`` consistent with scalars of the field."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=p, deg=d)
    field = algebra.field
    two = algebra.scalar(2)
    assert two == 2 and 2 == two
    assert two == field(2) and field(2) == two
    assert hash(two) == hash(2) == hash(field(2))
    # Integers outside the residue range are not reduced for comparison
    assert not two == p + 2
    assert not two == 2 - p
    assert two == field(p + 2)
    assert two != 3 and two != field(3)
    assert len({two, algebra.scalar(p + 2), algebra(2)}) == 1
    assert not algebra.basis(algebra.loop.basis(1)) == 1
    assert not algebra.basis(algebra.loop.minus_one()) == 1
    other = fcd.FiniteField(char=7 if p != 7 else 11)
    assert not two == other(2)
    if d > 1:
        scalar = field(list(range(1, d + 1)))
        element = algebra.scalar(scalar)
        assert element == scalar
        assert hash(element) == hash(scalar)
        assert not element == 1
        assert {element: "found"}[algebra.scalar(scalar)] == "found"


def test_elements_from_field_scalars_and_nested_integers() -> None:
    """Build coefficients from field scalars and from polynomial coordinates."""

    loop = cdl.CDLoop(1)
    prime = cdl.LoopAlgebra(loop, char=13)
    scalar = prime.field
    element = prime([[scalar(1), scalar(15)], [scalar(0), scalar(4)]])
    np.testing.assert_array_equal(element.coefficients, [[1, 2], [0, 4]])
    mixed = prime([[scalar(1), 15], [0, scalar(4)]])
    assert mixed == element
    field = fcd.FiniteField(char=13, deg=2)
    extension = cdl.LoopAlgebra(loop, field=field)
    by_scalars = extension([[field([1, 2]), field(3)], [field([0, 5]), field(4)]])
    by_coordinates = extension([[[1, 2], [3, 0]], [[0, 5], [4, 0]]])
    assert by_scalars == by_coordinates
    assert extension([[[14, -11], [16, 13]], [[0, 18], [4, 0]]]) == by_coordinates
    batch = extension.array([[[field([1, 2]), field(3)], [field([0, 5]), field(4)]]])
    assert batch.batch_shape == (1,)
    assert batch[0] == by_scalars
    with pytest.raises(ValueError):
        extension([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        extension(np.ones((2, 2, 2)))


@pytest.mark.parametrize(
    "p,width",
    [
        (3, 1),
        (251, 1),
        (257, 2),
        (65521, 2),
        (65537, 3),
        (16777259, 4),
        (2147483647, 4),
    ],
)
def test_byte_width_follows_the_characteristic(p: int, width: int) -> None:
    """Encode each residue in the fewest whole bytes that hold ``p - 1``."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1), char=p)
    largest = algebra(np.full((2, 2), p - 1, dtype=np.uint32))
    data = largest.to_bytes()
    assert len(data) == 4 * width
    assert data == (p - 1).to_bytes(width, "big") * 4
    assert algebra.from_bytes(data) == largest
    element = algebra([[1, 0], [p - 2, 2]])
    encoded = element.to_bytes()
    expected = b"".join(v.to_bytes(width, "big") for v in (1, 0, p - 2, 2))
    assert encoded == expected
    assert algebra.from_bytes(encoded) == element
    with pytest.raises(ValueError):
        algebra.from_bytes(encoded[:-1])
    if p < 256**width - 1:
        with pytest.raises(ValueError):
            algebra.from_bytes(p.to_bytes(width, "big") * 4)


def test_extension_bytes_follow_ravel_order() -> None:
    """Write the polynomial coordinates of each coefficient consecutively."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1), char=65521, deg=2)
    values = np.arange(8, dtype=np.uint32).reshape(2, 2, 2) + 65000
    element = algebra(values)
    expected = b"".join(int(v).to_bytes(2, "big") for v in values.ravel())
    assert element.to_bytes() == expected
    assert algebra.from_bytes(expected) == element
    complex_algebra = cdl.LoopAlgebra(cdl.CDLoop(1), char=0, deg=2)
    z = complex_algebra([[1 + 2j, 0], [3j, -1]])
    assert z.to_bytes() == np.array([1 + 2j, 0, 3j, -1], dtype="<c16").tobytes()
    assert complex_algebra.from_bytes(z.to_bytes()) == z


@pytest.mark.parametrize("p,d", FIELDS)
def test_random_residues_are_canonical(p: int, d: int) -> None:
    """Draw residues below the characteristic, reproducibly."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(3), char=p, deg=d)
    batch = algebra.random(7, batch_shape=(64,))
    coefficients = batch.coefficients
    assert coefficients.dtype == np.uint32
    assert coefficients.shape == (64,) + algebra.coefficient_shape
    assert int(coefficients.max()) < p
    np.testing.assert_array_equal(
        coefficients, algebra.random(7, batch_shape=(64,)).coefficients
    )
    assert not np.array_equal(
        coefficients, algebra.random(8, batch_shape=(64,)).coefficients
    )
    secure = algebra.random(secure=True)
    assert int(secure.coefficients.max()) < p
    if p <= 13:
        # Every residue occurs among this many draws
        assert set(np.unique(coefficients).tolist()) == set(range(p))
    else:
        assert int(coefficients.max()) > p // 2


@pytest.mark.parametrize("p,d", FIELDS)
def test_inverse_and_powers_are_exact(p: int, d: int) -> None:
    """Invert and exponentiate without rounding over every finite field."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(3), char=p, deg=d)
    rng = np.random.default_rng(p * d)
    while True:
        x = algebra(exact_values(algebra, rng))
        if x.has_inverse():
            break
    inverse = x.inverse()
    assert x * inverse == 1 and inverse * x == 1
    assert x**-1 == inverse
    assert x**-3 * x**3 == 1
    exponent = 10**30 + 57
    assert x**exponent * x == x ** (exponent + 1)
    assert x ** (2 * exponent) == (x**exponent) ** 2
    polynomial = x.minimal_polynomial()
    assert polynomial.dtype == np.uint32
    value = algebra.zero()
    for coefficient in polynomial[::-1]:
        scalar = algebra.field(coefficient.tolist()) if d > 1 else int(coefficient)
        value = value * x + scalar
    assert not value


def test_isotropic_elements_exist_only_where_expected() -> None:
    """Find nonzero signed components of quadratic norm zero over C and F_p."""

    loop = cdl.CDLoop(1)
    real = cdl.LoopAlgebra(loop)
    generic = real.central_idempotents()[0] + real.lift_signed([1.0, 1.0])
    assert generic.is_zero_divisor() is False
    assert generic.has_inverse() is True
    complex_algebra = cdl.LoopAlgebra(loop, char=0, deg=2)
    isotropic = complex_algebra.lift_signed([1, 1j])
    assert isotropic and isotropic * isotropic.conjugate() == 0
    five = cdl.LoopAlgebra(loop, char=5)
    isotropic = five.lift_signed([1, 2])
    assert isotropic and isotropic * isotropic.conjugate() == 0
    plus = five.central_idempotents()[0]
    assert (plus + isotropic).is_zero_divisor()
    assert not (plus + isotropic).has_inverse()
    # -1 is not a square modulo 7, so the norm form of level 1 is anisotropic
    seven = cdl.LoopAlgebra(loop, char=7)
    for a in range(7):
        for b in range(7):
            element = seven.central_idempotents()[0] + seven.lift_signed([a, b])
            assert element.has_inverse() == bool(a or b)
