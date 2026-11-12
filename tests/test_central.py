"""Tests of the loop algebras of central products of Cayley--Dickson loops."""

from __future__ import annotations

import itertools
import pickle
import unittest
from typing import Any

import numpy as np
import pyfastcd as fcd
import pytest

import pycdloop as cdl
from pycdloop import linalg, testing

try:
    from . import reference
except ImportError:  # Support collection from inside the tests directory.
    import reference

SELECTORS: dict[str, dict[str, Any]] = {
    "R": {},
    "C": {"char": 0, "deg": 2},
    "F7": {"char": 7},
    "F13": {"char": 13},
    "F7^2": {"char": 7, "deg": 2},
    "F13^2": {"char": 13, "deg": 2},
}
LEVELS = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 3),
    (1, 1, 2),
    (2, 1, 3),
]
SMALL_LEVELS = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (1, 1, 2)]


def _algebra(levels: tuple[int, ...], name: str = "R") -> Any:
    """Return the loop algebra of a central product over a named field."""

    return cdl.LoopAlgebra(cdl.CentralProduct(*levels), **SELECTORS[name])


def _ring(algebra: Any) -> Any:
    """Return the reference ring of an algebra's coefficient field."""

    if algebra.field is not None:
        return reference.FiniteField(algebra.char, algebra.field.modulus)
    if algebra.deg == 2:
        return reference.GaussianRationals()
    return reference.Rationals()


def _coefficients(algebra: Any, rng: Any, lead: tuple[int, ...] = ()) -> np.ndarray:
    """Return exactly representable random loop-basis coefficients."""

    shape = lead + algebra.coefficient_shape
    if algebra.field is not None:
        return rng.integers(0, algebra.char, shape).astype(np.uint32)
    values = rng.integers(-3, 4, shape).astype(algebra.dtype)
    if algebra.deg == 2:
        values = values + 1j * rng.integers(-3, 4, shape)
    return values


def _expected(algebra: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the product of two coefficient arrays computed by the oracle."""

    ring = _ring(algebra)
    product = reference.loop_algebra_multiply(
        ring,
        algebra.loop.levels,
        reference.to_ring(ring, x, 2),
        reference.to_ring(ring, y, 2),
    )
    return reference.from_ring(ring, product, 2, algebra.dtype)


def _is_zero(values: Any) -> bool:
    """Return whether coefficients vanish up to rounding."""

    return bool(np.allclose(np.asarray(values), 0, atol=1e-9))


@pytest.mark.parametrize("name", list(SELECTORS))
@pytest.mark.parametrize("levels", LEVELS)
def test_every_algorithm_matches_the_definition(
    levels: tuple[int, ...], name: str
) -> None:
    """All signed-component methods and the table product agree with the oracle."""

    algebra = _algebra(levels, name)
    rng = np.random.default_rng(sum(levels) + len(name))
    xs, ys = _coefficients(algebra, rng, (2,)), _coefficients(algebra, rng, (2,))
    expected = np.stack([_expected(algebra, xs[i], ys[i]) for i in range(2)])
    x, y = algebra(xs[0]), algebra(ys[0])
    assert np.array_equal((x * y).coefficients, expected[0])
    table = algebra.multiply(x, y, algorithm="table")
    assert np.array_equal(table.coefficients, expected[0])
    arrays = algebra.asarray(xs), algebra.asarray(ys)
    splits = arrays[0].split(), arrays[1].split()
    assert "auto" in algebra.signed_algorithms
    for method in algebra.signed_algorithms:
        single = algebra.multiply(x, y, signed_algorithm=method)
        assert np.array_equal(single.coefficients, expected[0]), method
        batch = algebra.multiply(*arrays, signed_algorithm=method)
        assert np.array_equal(batch.coefficients, expected), method
        split = algebra.multiply(*splits, signed_algorithm=method)
        assert np.array_equal(split.to_loop().coefficients, expected), method
    batch = algebra.multiply(*arrays, algorithm="table")
    assert np.array_equal(batch.coefficients, expected)


@pytest.mark.parametrize("levels", LEVELS)
def test_native_method_is_offered_exactly_when_it_applies(
    levels: tuple[int, ...],
) -> None:
    """The extension-field method needs a factor of level 1 and a suitable field."""

    fields: dict[str, dict[str, Any]] = dict(SELECTORS)
    fields.update({"F3": {"char": 3}, "F5": {"char": 5}, "F11": {"char": 11}})
    suitable = {"R", "F3", "F7", "F11"}
    for name, selectors in fields.items():
        algebra = cdl.LoopAlgebra(cdl.CentralProduct(*levels), **selectors)
        offered = "native" in algebra.signed_algorithms
        assert offered == (min(levels) == 1 and name in suitable), name
        assert {"auto", "rows", "schedule"} <= set(algebra.signed_algorithms)
        default = algebra.default_signed_algorithm
        assert default in algebra.signed_algorithms and default != "auto"
        if offered:
            assert default == "native"
        assert algebra.cd_algebra is None
        assert algebra.default_algorithm in ("split", "table")


class CentralProductProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create one nonassociative algebra and two exact elements."""

        self.algebra = _algebra((2, 3))
        rng = np.random.default_rng(23)
        self.x = self.algebra(_coefficients(self.algebra, rng))
        self.y = self.algebra(_coefficients(self.algebra, rng))

    def test_algorithm_names_are_validated(self) -> None:
        """Unknown names raise ValueError and non-strings raise TypeError."""

        algebra, x, y = self.algebra, self.x, self.y
        for name in ("direct", "quasilinear", "native", "ROWS", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                algebra.multiply(x, y, signed_algorithm=name)
        with self.assertRaises(TypeError):
            algebra.multiply(x, y, signed_algorithm=None)
        with self.assertRaises(ValueError):
            algebra.multiply(x, y, algorithm="rows")
        with self.assertRaises(TypeError):
            algebra.multiply(x, y, algorithm=1)
        cayley_dickson = cdl.LoopAlgebra(cdl.CDLoop(3))
        with self.assertRaises(ValueError):
            cayley_dickson.multiply(
                cayley_dickson.one(), cayley_dickson.one(), signed_algorithm="rows"
            )

    def test_power_operator_raises_and_bracketed_powers_work(self) -> None:
        """Powers need an explicit bracketing in a central-product algebra."""

        x = self.x
        with self.assertRaisesRegex(TypeError, "left_power"):
            x**2
        with self.assertRaisesRegex(TypeError, "right_power"):
            x**-1
        array = self.algebra.asarray(x.coefficients[None].copy())
        with self.assertRaises(TypeError):
            array**3
        with self.assertRaises(TypeError):
            array.split() ** 3
        self.assertEqual(x.left_power(0), 1)
        self.assertEqual(x.right_power(0), 1)
        self.assertEqual(x.left_power(1), x)
        self.assertEqual(x.left_power(2), x * x)
        self.assertEqual(x.left_power(3), x * (x * x))
        self.assertEqual(x.right_power(3), (x * x) * x)
        self.assertEqual(x.left_power(4), x * (x * (x * x)))
        self.assertEqual(x.right_power(4), ((x * x) * x) * x)
        self.assertNotEqual(x.left_power(3), x.right_power(3))
        self.assertTrue((array.left_power(3) == x.left_power(3)).all())
        self.assertTrue(
            (array.split().right_power(3).to_loop() == x.right_power(3)).all()
        )
        with self.assertRaises(ValueError):
            x.left_power(-1)
        with self.assertRaises(TypeError):
            x.right_power(1.5)
        with self.assertRaises(TypeError):
            x.left_power(True)

    def test_bracketed_powers_agree_when_powers_are_well_defined(self) -> None:
        """Over complexified octonions and for H (x) H both bracketings agree."""

        for levels in ((1, 3), (3, 1), (2, 2), (1, 1)):
            algebra = _algebra(levels)
            x = algebra(_coefficients(algebra, np.random.default_rng(4)))
            with self.subTest(levels=levels):
                for exponent in (2, 3, 4, 5):
                    self.assertEqual(x.left_power(exponent), x.right_power(exponent))

    def test_operations_that_need_well_defined_powers_raise(self) -> None:
        """Operations that need powers free of bracketing are refused."""

        algebra, x = self.algebra, self.x
        array = algebra.asarray(x.coefficients[None].copy())
        for value in (x, array, array.split()):
            with self.subTest(kind=type(value).__name__):
                for name in ("inverse", "has_inverse", "is_nilpotent"):
                    with self.assertRaisesRegex(TypeError, "central product"):
                        getattr(value, name)()
        with self.assertRaisesRegex(TypeError, "central product"):
            x.minimal_polynomial()
        with self.assertRaises(TypeError):
            algebra.subalgebra_generated_by(x)
        with self.assertRaises(TypeError):
            algebra.embed(x)

    def test_mixing_array_kinds_raises(self) -> None:
        """Loop-basis and split arrays do not combine."""

        array = self.algebra.asarray(self.x.coefficients[None].copy())
        split = array.split()
        for operation in (
            lambda: array * split,
            lambda: split + array,
            lambda: array == split,
            lambda: self.algebra.multiply(array, split),
            lambda: array.commutator(split),
        ):
            with self.assertRaises(TypeError):
                operation()
        self.assertTrue(((split * self.x).to_loop() == array * self.x).all())
        self.assertIsInstance(split.signed, np.ndarray)
        self.assertEqual(split.signed.shape, (1, 32))
        self.assertTrue(np.shares_memory(split.signed, split._signed))

    def test_pickling(self) -> None:
        """Algebras unpickle to the cached object and elements to equal ones."""

        algebra, x = self.algebra, self.x
        self.assertIs(pickle.loads(pickle.dumps(algebra)), algebra)
        product = x * self.y
        for value in (x, product):
            restored = pickle.loads(pickle.dumps(value))
            self.assertEqual(restored, value)
            self.assertIs(restored.algebra, algebra)
        self.assertEqual(algebra.from_bytes(x.to_bytes()), x)
        self.assertEqual(repr(algebra), "LoopAlgebra(CentralProduct(2, 3))")


@pytest.mark.parametrize("name", ["R", "C", "F7", "F13^2"])
@pytest.mark.parametrize("levels", SMALL_LEVELS)
def test_derived_products_follow_their_definitions(
    levels: tuple[int, ...], name: str
) -> None:
    """Commutators, associators and their relatives expand to plain products."""

    algebra = _algebra(levels, name)
    rng = np.random.default_rng(3)
    x, y, z = (algebra(_coefficients(algebra, rng)) for _ in range(3))
    assert x.square() == x * x
    assert x.commutator(y) == x * y - y * x
    assert x.anticommutator(y) == x * y + y * x
    assert x.jordan_product(y) * 2 == x * y + y * x
    assert x.associator(y, z) == (x * y) * z - x * (y * z)
    assert algebra.left_operator_commutator(x, y, z) == x * (y * z) - y * (x * z)
    assert algebra.right_operator_commutator(x, y, z) == (z * y) * x - (z * x) * y
    assert (x + y) * z == x * z + y * z and z * (x + y) == z * x + z * y
    arrays = [algebra.asarray(v.coefficients[None].copy()) for v in (x, y, z)]
    assert (arrays[0].associator(arrays[1], z) == x.associator(y, z)).all()
    splits = [array.split() for array in arrays]
    assert (splits[0].commutator(splits[1]).to_loop() == x.commutator(y)).all()


@pytest.mark.parametrize("name", ["R", "F7", "F13^2"])
@pytest.mark.parametrize("levels", SMALL_LEVELS)
def test_decomposition_and_components(levels: tuple[int, ...], name: str) -> None:
    """The components have one axis per factor and invert through from_components."""

    algebra = _algebra(levels, name)
    loop = algebra.loop
    sizes = tuple(1 << level for level in levels)
    tail = algebra.coefficient_shape[2:]
    rng = np.random.default_rng(6)
    values = _coefficients(algebra, rng)
    x, y = algebra(values), algebra(_coefficients(algebra, rng))
    epsilon = x.epsilon()
    assert isinstance(epsilon, np.ndarray) and epsilon.shape == sizes + tail
    assert not epsilon.flags.writeable
    plus, minus = values[0].astype(np.int64), values[1].astype(np.int64)
    if algebra.field is None:
        assert np.array_equal(epsilon.reshape(plus.shape), values[0] - values[1])
    else:
        expected = (plus - minus) % algebra.char
        assert np.array_equal(epsilon.reshape(plus.shape), expected)
    kappa = x.kappa()
    if algebra.field is not None:
        assert isinstance(kappa, fcd.ElementArray) and kappa.algebra is algebra.field
        kappa = kappa.coefficients
    assert kappa.shape == (loop.quotient_order,) + tail
    assert algebra.from_components(x.kappa(), epsilon) == x
    assert algebra.from_components(x.kappa(), epsilon.reshape(plus.shape)) == x
    e_plus, e_minus = algebra.central_idempotents()
    assert e_plus * e_plus == e_plus and e_minus * e_minus == e_minus
    assert e_plus * e_minus == 0 and e_plus + e_minus == 1
    assert algebra.lift_group(x.kappa()) == e_plus * x
    assert algebra.lift_signed(epsilon) == e_minus * x
    assert (e_minus * x) * (e_minus * y) == e_minus * (x * y)
    array = algebra.asarray(values[None].copy())
    assert np.asarray(array.epsilon()).shape == (1, loop.quotient_order) + tail
    idempotents = algebra.primitive_idempotents()
    assert idempotents.batch_shape == (loop.quotient_order,)
    assert idempotents.is_idempotent().all()
    total = idempotents[0]
    for index in range(1, loop.quotient_order):
        assert idempotents[0] * idempotents[index] == 0
        total = total + idempotents[index]
    assert total == e_plus


@pytest.mark.parametrize("name", ["R", "C", "F7", "F7^2"])
@pytest.mark.parametrize("levels", SMALL_LEVELS)
def test_conjugation_is_an_anti_automorphism(
    levels: tuple[int, ...], name: str
) -> None:
    """The involution of loop inversion reverses products."""

    algebra = _algebra(levels, name)
    loop = algebra.loop
    rng = np.random.default_rng(12)
    x, y = (algebra(_coefficients(algebra, rng)) for _ in range(2))
    assert x.conjugate().conjugate() == x
    assert (x * y).conjugate() == y.conjugate() * x.conjugate()
    assert (x + y).conjugate() == x.conjugate() + y.conjugate()
    for element in list(loop)[:: max(1, loop.order // 16)]:
        assert algebra.basis(element).conjugate() == algebra.basis(element.inverse())
    product = x * y
    assert product.conjugate() == algebra(product.coefficients).conjugate()
    array = algebra.asarray(x.coefficients[None].copy())
    assert (array.conjugate() == x.conjugate()).all()
    assert (array.split().conjugate().to_loop() == x.conjugate()).all()


@pytest.mark.parametrize("name", ["R", "F7", "F13^2"])
@pytest.mark.parametrize("levels", [(1, 2), (2, 2), (2, 3), (2, 1, 2)])
def test_factor_embedding_is_a_unital_homomorphism(
    levels: tuple[int, ...], name: str
) -> None:
    """Each factor's loop algebra embeds compatibly with all operations."""

    algebra = _algebra(levels, name)
    loop = algebra.loop
    rng = np.random.default_rng(31)
    images = []
    for number, level in enumerate(levels):
        factor = cdl.LoopAlgebra(cdl.CDLoop(level), **SELECTORS[name])
        x = factor(_coefficients(factor, rng))
        y = factor(_coefficients(factor, rng))
        fx, fy = (
            algebra.factor_embedding(number, x),
            algebra.factor_embedding(number, y),
        )
        assert algebra.factor_embedding(number, x * y) == fx * fy
        assert algebra.factor_embedding(number, x + y) == fx + fy
        assert algebra.factor_embedding(number, factor.one()) == algebra.one()
        for element in cdl.CDLoop(level):
            image = algebra.factor_embedding(number, factor.basis(element))
            assert image == algebra.basis(loop.factor_embedding(number, element))
        batch = factor.asarray(np.stack([x.coefficients, y.coefficients]))
        embedded = algebra.factor_embedding(number, batch)
        assert isinstance(embedded, cdl.ElementArray)
        assert embedded[0] == fx and embedded[1] == fy
        images.append(fx)
    # Images of different factors commute and associate with each other
    for fx, fy in itertools.combinations(images, 2):
        assert fx * fy == fy * fx
        assert (fx * fy) * fx == fx * (fy * fx)
    wrong = cdl.LoopAlgebra(cdl.CDLoop(5), **SELECTORS[name])
    with pytest.raises(ValueError):
        algebra.factor_embedding(0, wrong.one())
    with pytest.raises(IndexError):
        algebra.factor_embedding(len(levels), wrong.one())
    with pytest.raises(TypeError):
        algebra.factor_embedding(0, 1.0)
    with pytest.raises(TypeError):
        wrong.factor_embedding(0, wrong.one())
    if name != "R":
        real = cdl.LoopAlgebra(cdl.CDLoop(levels[0]))
        with pytest.raises(ValueError):
            algebra.factor_embedding(0, real.one())


def test_square_of_the_mixed_unit_in_the_smallest_central_product() -> None:
    """In Q_1 * Q_1 the element e_1 (x) e_1 squares to +1, unlike e_1 in Q_1."""

    loop = cdl.CentralProduct(1, 1)
    algebra = cdl.LoopAlgebra(loop)
    mixed = algebra.basis(loop.basis(1, 1))
    assert mixed * mixed == algebra.one() and mixed * mixed == 1
    assert (loop.basis(1, 1) * loop.basis(1, 1)) == loop.one()
    single = cdl.LoopAlgebra(cdl.CDLoop(1))
    unit = single.basis(cdl.CDLoop(1).basis(1))
    assert unit * unit == single.basis(cdl.CDLoop(1).minus_one())
    first = algebra.basis(loop.basis(1, 0))
    assert first * first == algebra.basis(loop.minus_one())
    assert first.left_power(4) == 1 and mixed.left_power(2) == 1


@pytest.mark.parametrize("name", ["R", "F7"])
@pytest.mark.parametrize("levels", SMALL_LEVELS)
def test_center_basis_elements_are_central(levels: tuple[int, ...], name: str) -> None:
    """Every basis element of the center commutes and associates with everything."""

    algebra = _algebra(levels, name)
    loop = algebra.loop
    rng = np.random.default_rng(2)
    x, y = (algebra(_coefficients(algebra, rng)) for _ in range(2))
    basis = algebra.center_basis()
    assert basis.batch_shape == (loop.quotient_order + len(loop.center()) // 2,)
    assert basis.is_central().all()
    rows = basis.coefficients.reshape(basis.batch_shape[0], -1)
    assert linalg.rank(rows, field=algebra.field) == basis.batch_shape[0]
    for c in basis:
        assert c.is_central()
        assert c * x == x * c
        assert (c * x) * y == c * (x * y)
        assert (x * c) * y == x * (c * y)
        assert (x * y) * c == x * (y * c)
    assert x.is_central() == loop.is_commutative()
    assert algebra.one().is_central() and algebra.zero().is_central()


def test_zero_divisors_in_the_smallest_central_product() -> None:
    """(1 + xi)(1 - xi) = 0 for the mixed unit xi of Q_1 * Q_1, which squares to 1."""

    for name in ("R", "F7", "F13"):
        algebra = _algebra((1, 1), name)
        loop = algebra.loop
        one = algebra.lift_signed([1, 0, 0, 0])
        xi = algebra.lift_signed([0, 0, 0, 1])
        x, y = one + xi, one - xi
        assert x * y == 0 and y * x == 0
        assert x.is_zero_divisor() and x.is_zero_divisor("left")
        assert x.is_zero_divisor("right")
        assert algebra.zero().is_zero_divisor()
        # A unit of the signed component still has zero group component
        assert one.is_zero_divisor()
        assert not algebra.one().is_zero_divisor()
        generic = algebra.one() + algebra.basis(loop.basis(1, 0)) * 2
        assert not generic.is_zero_divisor()
        assert generic.right_annihilator().batch_shape == (0,)
        array = algebra.array(np.stack([x.coefficients, generic.coefficients]))
        assert array.is_zero_divisor().tolist() == [True, False]
        assert array.split().is_zero_divisor("left").tolist() == [True, False]
        with pytest.raises(ValueError):
            x.is_zero_divisor("middle")


@pytest.mark.parametrize("name", ["R", "C", "F7", "F13"])
@pytest.mark.parametrize("levels", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_annihilators_multiply_back_to_zero(levels: tuple[int, ...], name: str) -> None:
    """Annihilator bases have the dimension of a kernel and are annihilated."""

    algebra = _algebra(levels, name)
    rng = np.random.default_rng(len(levels) + 50)
    size = algebra.loop.quotient_order
    # A zero spectral coordinate and a singular signed component
    spectral = _coefficients(algebra, rng)[0]
    spectral[0] = 0
    signed = _coefficients(algebra, rng)[0]
    signed[1:] = 0
    signed[0] = 1
    signed[size - 1] = 1
    x = algebra.split_array(spectral, signed)[()]
    assert isinstance(x, cdl.Element)
    for side, matrix in (
        ("right", algebra.left_matrix(x)),
        ("left", algebra.right_matrix(x)),
    ):
        basis = getattr(x, f"{side}_annihilator")()
        nullity = algebra.dimension - linalg.rank(matrix, field=algebra.field)
        assert basis.batch_shape == (nullity,) and nullity >= 1
        rows = basis.coefficients.reshape(nullity, -1)
        assert linalg.rank(rows, field=algebra.field) == nullity
        products = x * basis if side == "right" else basis * x
        assert _is_zero(products.coefficients)
    assert x.is_zero_divisor()


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_level_one_factor_is_the_complexification(level: int) -> None:
    """A_1 (x) A_m is the Cayley--Dickson algebra with complex coefficients."""

    complexified = fcd.Algebra(level=level, char=0, deg=2)
    rng = np.random.default_rng(level)
    for levels, axis in (((1, level), 0), ((level, 1), 1)):
        algebra = _algebra(levels)
        x, y = (algebra(_coefficients(algebra, rng)) for _ in range(2))

        def pack(element: Any, axis: int = axis) -> Any:
            """Read a signed component as an element with complex coefficients."""

            parts = np.moveaxis(element.epsilon(), axis, 0)
            return complexified(parts[0] + 1j * parts[1])

        expected = complexified.multiply(pack(x), pack(y), algorithm="direct")
        for method in algebra.signed_algorithms:
            product = algebra.multiply(x, y, signed_algorithm=method)
            assert pack(product) == expected, method


def test_identities_of_small_tensor_products() -> None:
    """Associativity, alternativity and flexibility of the first central products."""

    quaternions = _algebra((2, 2))
    assert testing.is_associative(quaternions)
    assert not testing.is_commutative(quaternions)
    abelian = _algebra((1, 1))
    assert testing.is_commutative(abelian) and testing.is_associative(abelian)
    complex_octonions = _algebra((1, 3))
    assert testing.is_alternative(complex_octonions)
    assert not testing.is_associative(complex_octonions)
    mixed = _algebra((2, 3))
    assert not testing.is_flexible(mixed)
    assert not testing.is_power_associative(mixed)
    assert not testing.is_alternative(mixed)
    assert cdl.CentralProduct(2, 3).is_moufang()
    finite = _algebra((2, 3), "F7")
    assert not testing.is_flexible(finite)
    assert testing.is_associative(_algebra((2, 2), "F13^2"))


@pytest.mark.parametrize("name", ["R", "F7"])
@pytest.mark.parametrize("levels", [(1, 2), (2, 3), (1, 1, 2)])
def test_character_automorphisms_and_matrices(
    levels: tuple[int, ...], name: str
) -> None:
    """Characters act as automorphisms and matrices reproduce products."""

    algebra = _algebra(levels, name)
    rng = np.random.default_rng(77)
    xs, ys = _coefficients(algebra, rng), _coefficients(algebra, rng)
    x, y = algebra(xs), algebra(ys)
    product = x * y
    for a in range(0, algebra.loop.quotient_order, 3):
        images = [algebra.character_automorphism(a, value) for value in (x, y, product)]
        assert images[0] * images[1] == images[2]
        assert algebra.character_automorphism(a, images[0]) == x
    modulus = algebra.char or None
    for matrix, expected in (
        (algebra.left_matrix(x), product),
        (algebra.right_matrix(x), y * x),
    ):
        image = matrix.astype(object) @ ys.ravel().astype(object)
        if modulus:
            image = image % modulus
        assert np.array_equal(
            image.astype(algebra.dtype), expected.coefficients.ravel()
        )
    split = algebra.left_matrix(x, basis="split")
    size = algebra.loop.quotient_order
    assert not split[:size, size:].any() and not split[size:, :size].any()
    assert np.array_equal(np.diag(split[:size, :size]), np.asarray(x._data("split")[0]))
    with pytest.raises(ValueError):
        algebra.left_matrix(x, basis="other")
    with pytest.raises(TypeError):
        algebra.left_matrix(algebra.asarray(xs[None].copy()))


def test_out_parameter_and_scalars() -> None:
    """Products write into caller storage and scalars scale either operand."""

    algebra = _algebra((2, 2))
    rng = np.random.default_rng(1)
    xs, ys = _coefficients(algebra, rng, (3,)), _coefficients(algebra, rng, (3,))
    x, y = algebra.asarray(xs), algebra.asarray(ys)
    out = np.empty(xs.shape)
    result = algebra.multiply(x, y, out=out, signed_algorithm="schedule")
    assert isinstance(result, cdl.ElementArray) and result.coefficients is out
    assert (result == x * y).all()
    target = algebra.asarray(np.empty(xs.shape))
    assert algebra.multiply(x, y, out=target, algorithm="table") is target
    assert (target == x * y).all()
    assert (algebra.multiply(2, x) == x + x).all()
    assert (algebra.multiply(x, 0.5) * 2 == x).all()
    with pytest.raises(ValueError):
        algebra.multiply(x, y, out=np.empty((2,) + algebra.coefficient_shape))
    with pytest.raises(TypeError):
        algebra.multiply(x, y, out=np.empty(xs.shape, dtype=np.float32))
    with pytest.raises(TypeError):
        algebra.multiply(x.split(), y.split(), out=out)
    with pytest.raises(TypeError):
        algebra.multiply(x.split(), y.split(), algorithm="table")
    with pytest.raises(TypeError):
        algebra.multiply(2, 3)
    other = _algebra((2, 2), "C")
    with pytest.raises(ValueError):
        algebra.multiply(x, other.one())
