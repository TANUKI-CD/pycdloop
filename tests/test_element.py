"""Deterministic tests for :class:`pycdloop.Element`."""

from __future__ import annotations

import copy
import pickle
import unittest
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pyfastcd as fcd

import pycdloop as cdl

try:  # Support both package-style collection and collection from this directory.
    from . import reference
except ImportError:  # pragma: no cover - depends on the collection mechanism
    import reference


RTOL = 1.0e-11
ATOL = 1.0e-11

# Coefficient fields exercised by most tests: R, C, two prime fields and two
# extension fields
SELECTORS: tuple[dict[str, int], ...] = (
    {},
    {"char": 0, "deg": 2},
    {"char": 7},
    {"char": 13},
    {"char": 3, "deg": 2},
    {"char": 13, "deg": 3},
)
FINITE_SELECTORS = tuple(selectors for selectors in SELECTORS if selectors.get("char"))


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


def exact_element(algebra: Any, rng: Any) -> Any:
    """Draw one element with exactly representable coefficients."""

    return algebra(exact_values(algebra, rng))


def oracle_product(algebra: Any, x: Any, y: Any) -> np.ndarray:
    """Multiply two coefficient arrays with the independent reference."""

    ring = ring_of(algebra)
    left = reference.to_ring(ring, x, 2)
    right = reference.to_ring(ring, y, 2)
    product = reference.loop_algebra_multiply(ring, algebra.loop.levels, left, right)
    return reference.from_ring(ring, product, 2, algebra.dtype)


def plain(value: Any) -> np.ndarray:
    """Return a component given as an ndarray or a pyfastcd object as an ndarray."""

    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value.coefficients)


def field_scalar(algebra: Any, coefficient: Any) -> Any:
    """Return one stored coefficient as a scalar accepted by operators."""

    if algebra.field is None:
        return coefficient.item()
    if algebra.deg > 1:
        return algebra.field(np.asarray(coefficient).tolist())
    return int(coefficient)


def evaluate(algebra: Any, polynomial: np.ndarray, x: Any) -> Any:
    """Evaluate a polynomial, constant term first, at an element by Horner's rule."""

    result = algebra.zero()
    for coefficient in polynomial[::-1]:
        result = result * x + field_scalar(algebra, coefficient)
    return result


def invertible_element(algebra: Any, seed: int) -> Any:
    """Return a random element that has an inverse."""

    rng = np.random.default_rng(seed)
    while True:
        candidate = exact_element(algebra, rng)
        if candidate.has_inverse():
            return candidate


class ElementTestCase(unittest.TestCase):
    """Common assertions with useful failure diagnostics."""

    def assertCoefficientsEqual(self, actual: object, expected: object) -> None:
        """Assert exact equality after exposing both coefficient arrays."""

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def assertCoefficientsClose(self, actual: object, expected: object) -> None:
        """Assert coefficient closeness with the test tolerances."""

        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=RTOL, atol=ATOL
        )

    def assertSameElement(self, actual: Any, expected: Any) -> None:
        """Assert that two elements share a parent and all coefficients."""

        self.assertIsInstance(actual, cdl.Element)
        self.assertIs(actual.algebra, expected.algebra)
        self.assertCoefficientsEqual(actual.coefficients, expected.coefficients)

    def assertSameBytes(self, actual: Any, expected: Any) -> None:
        """Assert that two elements have bitwise identical coefficients."""

        self.assertEqual(actual.coefficients.tobytes(), expected.coefficients.tobytes())


class ProtocolTests(ElementTestCase):
    def test_direct_construction_is_rejected(self) -> None:
        """Require a parent algebra to create elements."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        for arguments in ((), (algebra,), (algebra, np.zeros((2, 4)))):
            with self.subTest(arguments=len(arguments)), self.assertRaises(TypeError):
                cdl.Element(*arguments)

    def test_elements_are_immutable(self) -> None:
        """Reject attribute changes and writes to the coefficients."""

        x = cdl.LoopAlgebra(cdl.CDLoop(2))(np.arange(8.0).reshape(2, 4))
        with self.assertRaises(AttributeError):
            x.algebra = None  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            x.extra = 1  # type: ignore[attr-defined]
        with self.assertRaises(AttributeError):
            del x._algebra
        self.assertFalse(x.coefficients.flags.writeable)
        with self.assertRaises(ValueError):
            x.coefficients[0, 0] = 1.0
        view = x.to_numpy(copy=False)
        self.assertFalse(view.flags.writeable)
        product = x * x
        self.assertFalse(product.coefficients.flags.writeable)
        with self.assertRaises(ValueError):
            product.coefficients[0, 0] = 1.0

    def test_properties_follow_the_parent(self) -> None:
        """Report the parent, the coefficient shape and the storage type."""

        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x = algebra.random(0)
            with self.subTest(algebra=algebra):
                self.assertIs(x.algebra, algebra)
                self.assertEqual(x.shape, algebra.coefficient_shape)
                self.assertEqual(x.dtype, algebra.dtype)
                self.assertEqual(x.coefficients.shape, algebra.coefficient_shape)
                self.assertEqual(x.coefficients.dtype, algebra.dtype)

    def test_numpy_conversion(self) -> None:
        """Expose coefficients to NumPy as a view or as a copy."""

        values = np.arange(8.0).reshape(2, 4)
        x = cdl.LoopAlgebra(cdl.CDLoop(2))(values)
        self.assertCoefficientsEqual(np.asarray(x), values)
        self.assertFalse(np.asarray(x).flags.writeable)
        copied = np.array(x)
        copied[0, 0] = 99.0
        self.assertEqual(x.coefficients[0, 0], 0.0)
        self.assertEqual(np.asarray(x, dtype=np.float32).dtype, np.float32)
        owned = x.to_numpy()
        self.assertTrue(owned.flags.writeable)
        owned[0, 0] = 99.0
        self.assertEqual(x.coefficients[0, 0], 0.0)
        self.assertTrue(np.shares_memory(x.to_numpy(copy=False), x.coefficients))
        for flag in (1, "yes", None):
            with self.subTest(flag=flag), self.assertRaises(TypeError):
                x.to_numpy(copy=flag)

    def test_copy_deepcopy_and_pickle(self) -> None:
        """Preserve values and the form an element was created with."""

        rng = np.random.default_rng(11)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x = exact_element(algebra, rng)
                product = x * x
                with self.subTest(algebra=algebra):
                    for value in (x, product):
                        clone = value.copy()
                        self.assertIsNot(clone, value)
                        self.assertSameElement(clone, value)
                        self.assertFalse(
                            np.shares_memory(clone.coefficients, value.coefficients)
                        )
                        self.assertIs(copy.copy(value), value)
                        self.assertSameElement(copy.deepcopy(value), value)
                        restored = pickle.loads(pickle.dumps(value))
                        self.assertSameElement(restored, value)
                        self.assertEqual(restored._born, value._born)
                        self.assertEqual(hash(restored), hash(value))
                        self.assertIs(restored.algebra, algebra)
                    self.assertIs(+x, x)

    def test_repr_shows_coefficients_and_parent(self) -> None:
        """Name the class, the coefficients and the algebra."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(1))
        text = repr(algebra([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(
            text, "Element([[1., 2.], [3., 4.]], algebra=LoopAlgebra(CDLoop(1)))"
        )
        large = repr(cdl.LoopAlgebra(cdl.CDLoop(6)).one())
        self.assertIn("...", large)
        self.assertTrue(large.endswith("algebra=LoopAlgebra(CDLoop(6)))"))


class OperatorTests(ElementTestCase):
    def test_addition_subtraction_and_negation_are_coordinatewise(self) -> None:
        """Compare linear operators with arithmetic on the coefficients."""

        rng = np.random.default_rng(21)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                a, b = exact_values(algebra, rng), exact_values(algebra, rng)
                x, y = algebra(a), algebra(b)
                with self.subTest(algebra=algebra):
                    if algebra.field is None:
                        total, difference, negative = a + b, a - b, -a
                    else:
                        wide_a, wide_b = a.astype(np.int64), b.astype(np.int64)
                        total = (wide_a + wide_b) % algebra.char
                        difference = (wide_a - wide_b) % algebra.char
                        negative = -wide_a % algebra.char
                    self.assertCoefficientsEqual(x + y, total)
                    self.assertCoefficientsEqual(x - y, difference)
                    self.assertCoefficientsEqual(-x, negative)
                    self.assertSameElement(x - x, algebra.zero())
                    self.assertSameElement(x + algebra.zero(), x)
                    self.assertSameElement(-x + x, algebra.zero())
                    self.assertIs(+x, x)

    def test_scalars_embed_in_sums_and_scale_in_products(self) -> None:
        """Add ``r`` times the identity in sums and scale in products."""

        rng = np.random.default_rng(22)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x = exact_element(algebra, rng)
                with self.subTest(algebra=algebra):
                    three = algebra.scalar(3)
                    self.assertSameElement(x + 3, x + three)
                    self.assertSameElement(3 + x, x + three)
                    self.assertSameElement(x - 3, x - three)
                    self.assertSameElement(3 - x, three - x)
                    self.assertSameElement(3 - x, -(x - 3))
                    self.assertSameElement(x * 3, x + x + x)
                    self.assertSameElement(3 * x, x + x + x)
                    self.assertSameElement(x * 1, x)
                    self.assertSameElement(x * 0, algebra.zero())
                    self.assertSameElement(x * three, 3 * x)
                    self.assertSameElement(three * x, 3 * x)
                    changed = (x + 1).coefficients != x.coefficients
                    if changed.ndim == 3:
                        changed = changed.any(axis=-1)
                    self.assertEqual(int(changed.sum()), 1)
                    self.assertTrue(changed[0, 0])

    def test_scalar_types_of_each_field(self) -> None:
        """Accept the scalar types of the coefficient field and no others."""

        loop = cdl.CDLoop(2)
        real = cdl.LoopAlgebra(loop)
        x = real(np.arange(8.0).reshape(2, 4))
        doubled = x + x
        for scalar in (
            2,
            2.0,
            True + 1,
            Fraction(2),
            Decimal(2),
            np.float64(2),
            np.int64(2),
            np.float32(2),
            np.array(2.0),
        ):
            with self.subTest(scalar=repr(scalar)):
                self.assertSameElement(x * scalar, doubled)
                self.assertSameElement(scalar * x, doubled)
                self.assertSameElement((x + scalar) - scalar, x)
        self.assertSameElement(x * Fraction(1, 2), x / 2)
        for scalar in (1j, np.complex128(1), "2", None, [2.0], np.ones(2)):
            with self.subTest(scalar=repr(scalar)):
                with self.assertRaises(TypeError):
                    x * scalar
                with self.assertRaises(TypeError):
                    x + scalar

        complex_algebra = cdl.LoopAlgebra(loop, char=0, deg=2)
        z = complex_algebra(np.arange(8.0).reshape(2, 4) * (1 + 1j))
        self.assertCoefficientsEqual(z * 1j, z.coefficients * 1j)
        self.assertCoefficientsEqual(1j * z, z.coefficients * 1j)
        self.assertCoefficientsEqual(
            z * np.complex128(2 - 1j), z.coefficients * (2 - 1j)
        )
        self.assertEqual((z + 1j).coefficients[0, 0], z.coefficients[0, 0] + 1j)
        self.assertSameElement(z * 2.0, z + z)

        finite = cdl.LoopAlgebra(loop, char=7)
        w = finite(np.arange(8).reshape(2, 4))
        self.assertSameElement(w * 9, w + w)
        self.assertSameElement(w * -5, w + w)
        self.assertSameElement(w * finite.field(2), w + w)
        self.assertSameElement(finite.field(2) * w, w + w)
        self.assertSameElement(np.int64(2) * w, w + w)
        self.assertSameElement(w + 7, w)
        for scalar in (2.0, 1j, Fraction(2), fcd.FiniteField(char=13)(2), "2"):
            with self.subTest(scalar=repr(scalar)):
                with self.assertRaises(TypeError):
                    w * scalar
                with self.assertRaises(TypeError):
                    w + scalar

    def test_extension_field_scalars_multiply_in_the_field(self) -> None:
        """Scale by a scalar of an extension field, coefficient by coefficient."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=13, deg=3)
        ring = ring_of(algebra)
        rng = np.random.default_rng(23)
        values = exact_values(algebra, rng)
        x = algebra(values)
        scalar = algebra.field([5, 0, 7])
        factor = ring.convert(np.asarray(scalar.coefficients))
        expected = [
            [ring.mul(entry, factor) for entry in row]
            for row in reference.to_ring(ring, values, 2)
        ]
        self.assertCoefficientsEqual(
            x * scalar, reference.from_ring(ring, expected, 2, algebra.dtype)
        )
        self.assertSameElement(scalar * x, x * scalar)
        self.assertSameElement((x * scalar) / scalar, x)
        self.assertSameElement(x + scalar, x + algebra.scalar(scalar))

    def test_division_by_field_scalars(self) -> None:
        """Divide by nonzero scalars and raise at zero."""

        rng = np.random.default_rng(24)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x = exact_element(algebra, rng)
            with self.subTest(algebra=algebra):
                self.assertSameElement((x * 2) / 2, x)
                with self.assertRaises(ZeroDivisionError):
                    x / 0
                with self.assertRaises(TypeError):
                    x / x
                with self.assertRaises(TypeError):
                    2 / x
                if algebra.field is not None:
                    self.assertSameElement((x / 2) * 2, x)
                    with self.assertRaises(ZeroDivisionError):
                        x / algebra.char
                else:
                    self.assertCoefficientsEqual(x / 4, x.coefficients / 4)

    def test_operands_of_other_parents_are_rejected(self) -> None:
        """Raise ``ValueError`` for elements of a different algebra."""

        loop = cdl.CDLoop(2)
        x = cdl.LoopAlgebra(loop).random(0)
        others = (
            cdl.LoopAlgebra(loop, char=0, deg=2).one(),
            cdl.LoopAlgebra(loop, char=7).one(),
            cdl.LoopAlgebra(cdl.CDLoop(3)).one(),
            cdl.LoopAlgebra(cdl.CentralProduct(1, 1)).one(),
        )
        for other in others:
            with self.subTest(other=other.algebra):
                operations = (
                    lambda other=other: x + other,
                    lambda other=other: x - other,
                    lambda other=other: x * other,
                    lambda other=other: other * x,
                    lambda other=other: x.commutator(other),
                    lambda other=other: x.associator(x, other),
                    lambda other=other: x.distance(other),
                )
                for operation in operations:
                    with self.assertRaises(ValueError):
                        operation()

    def test_named_methods_require_elements(self) -> None:
        """Reject scalars and raw arrays where an element is required."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        x = algebra.random(0)
        batch = algebra.random(0, batch_shape=(2,))
        for other in (3, 2.5, np.ones((2, 4)), batch, batch.split(), None):
            with self.subTest(other=type(other).__name__):
                with self.assertRaises(TypeError):
                    x.commutator(other)
                with self.assertRaises(TypeError):
                    x.anticommutator(other)
                with self.assertRaises(TypeError):
                    x.jordan_product(other)
                with self.assertRaises(TypeError):
                    x.associator(other, x)
                with self.assertRaises(TypeError):
                    x.distance(other)

    def test_products_with_arrays_return_arrays(self) -> None:
        """Defer to an array operand on either side."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        x = algebra.random(0)
        batch = algebra.random(1, batch_shape=(3,))
        for result in (x * batch, batch * x, x + batch, x - batch):
            self.assertIsInstance(result, cdl.ElementArray)
        for result in (x * batch.split(), batch.split() * x, x + batch.split()):
            self.assertIsInstance(result, cdl.SplitArray)


class ComparisonTests(ElementTestCase):
    def test_equal_elements_hash_equal(self) -> None:
        """Compare by value across separately created and derived elements."""

        rng = np.random.default_rng(31)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            values = exact_values(algebra, rng)
            x, same = algebra(values), algebra(values.copy())
            product = x * x
            rebuilt = algebra(product.coefficients)
            with self.subTest(algebra=algebra):
                self.assertTrue(x == same)
                self.assertFalse(x != same)
                self.assertEqual(hash(x), hash(same))
                self.assertTrue(product == rebuilt)
                self.assertEqual(hash(product), hash(rebuilt))
                self.assertFalse(x == x + 1)
                self.assertTrue(x != x + 1)
                self.assertEqual(len({x, same, x + 1}), 2)
                self.assertEqual({x: "value"}[same], "value")

    def test_scalar_elements_compare_and_hash_like_scalars(self) -> None:
        """Agree with real, complex and finite-field scalars."""

        loop = cdl.CDLoop(2)
        real = cdl.LoopAlgebra(loop)
        three = real.scalar(3)
        self.assertTrue(three == 3 and three == 3.0 and three == Fraction(3))
        self.assertTrue(3 == three)
        self.assertEqual(hash(three), hash(3))
        self.assertEqual(len({three, 3}), 1)
        self.assertTrue(real.scalar(0.5) == Fraction(1, 2))
        self.assertTrue(real.scalar(0.5) == Decimal("0.5"))
        self.assertEqual(hash(real.scalar(0.5)), hash(0.5))
        # Equality compares exact values, and 0.1 is not one tenth in binary64
        self.assertTrue(real.scalar(0.1) == 0.1)
        self.assertFalse(real.scalar(0.1) == Fraction(1, 10))
        self.assertFalse(real.scalar(0.1) == Decimal("0.1"))
        self.assertTrue(real.zero() == 0)
        self.assertEqual(hash(real.zero()), hash(0))
        self.assertFalse(real.basis(loop.basis(1)) == 1)
        self.assertFalse(real.basis(loop.minus_one()) == 1)
        self.assertFalse(real.basis(loop.minus_one()) == -1)
        self.assertTrue(three != 4)

        complex_algebra = cdl.LoopAlgebra(loop, char=0, deg=2)
        value = complex_algebra.scalar(1 + 2j)
        self.assertTrue(value == 1 + 2j)
        self.assertEqual(hash(value), hash(1 + 2j))
        self.assertTrue(complex_algebra.scalar(2) == 2)
        self.assertEqual(hash(complex_algebra.scalar(2)), hash(2))

    def test_finite_field_integers_are_not_reduced_for_comparison(self) -> None:
        """Keep equality with integers consistent with hashing."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=7)
        three = algebra.scalar(3)
        self.assertTrue(three == 3)
        self.assertFalse(three == 10)
        self.assertFalse(three == -4)
        self.assertTrue(algebra.scalar(10) == 3)
        self.assertTrue(three == algebra.field(3))
        self.assertTrue(three == algebra.field(10))
        self.assertEqual(hash(three), hash(3))
        self.assertEqual(hash(three), hash(algebra.field(3)))
        self.assertFalse(three == fcd.FiniteField(char=13)(3))
        extension = cdl.LoopAlgebra(cdl.CDLoop(2), char=3, deg=2)
        scalar = extension.field([1, 2])
        self.assertTrue(extension.scalar(scalar) == scalar)
        self.assertEqual(hash(extension.scalar(scalar)), hash(scalar))
        self.assertTrue(extension.scalar(2) == 2)
        self.assertFalse(extension.scalar(scalar) == 1)

    def test_signed_zero_is_ignored(self) -> None:
        """Treat ``-0.0`` and ``0.0`` as the same coefficient."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(1))
        positive = algebra([[0.0, 1.0], [0.0, 0.0]])
        negative = algebra([[-0.0, 1.0], [0.0, -0.0]])
        self.assertTrue(positive == negative)
        self.assertEqual(hash(positive), hash(negative))
        self.assertTrue(algebra([[-0.0, 0.0], [0.0, 0.0]]) == 0)
        self.assertEqual(hash(algebra([[-0.0, 0.0], [0.0, 0.0]])), hash(0))
        self.assertFalse(algebra([[-0.0, -0.0], [-0.0, -0.0]]))

    def test_elements_of_different_algebras_are_unequal(self) -> None:
        """Compare unequal across loops and fields without raising."""

        loop = cdl.CDLoop(2)
        one = cdl.LoopAlgebra(loop).one()
        for other in (
            cdl.LoopAlgebra(loop, char=0, deg=2).one(),
            cdl.LoopAlgebra(loop, char=7).one(),
            cdl.LoopAlgebra(cdl.CDLoop(3)).one(),
            cdl.LoopAlgebra(cdl.CentralProduct(1, 1)).one(),
        ):
            with self.subTest(other=other.algebra):
                self.assertFalse(one == other)
                self.assertTrue(one != other)

    def test_unrelated_objects_are_unequal(self) -> None:
        """Return ``False`` for objects that are not elements or scalars."""

        loop = cdl.CDLoop(2)
        x = cdl.LoopAlgebra(loop).one()
        for other in ("1", None, [1.0], (1.0,), np.ones((2, 4)), loop.one(), object()):
            with self.subTest(other=type(other).__name__):
                self.assertFalse(x == other)
                self.assertTrue(x != other)

    def test_truth_value_is_nonzero(self) -> None:
        """Report zero elements as false in either internal form."""

        rng = np.random.default_rng(32)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            x = exact_element(algebra, rng)
            product = x * x
            with self.subTest(algebra=algebra):
                self.assertFalse(algebra.zero())
                self.assertTrue(algebra.one())
                self.assertTrue(x)
                self.assertTrue(product)
                self.assertFalse(product - product)
                self.assertFalse(x * algebra.zero())
                self.assertTrue(algebra.central_idempotents()[1])


class CacheTests(ElementTestCase):
    def test_products_are_created_in_split_form(self) -> None:
        """Defer the inverse transform until coefficients are read."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(4))
        x, y = algebra.random(1), algebra.random(2)
        self.assertEqual(x._born, "loop")
        self.assertIsNone(x._split)
        product = x * y
        self.assertEqual(product._born, "split")
        self.assertIsNone(product._loop)
        self.assertIsNotNone(x._split)
        chained = product * x
        self.assertIsNone(product._loop)
        coefficients = product.coefficients
        derived = product._loop
        self.assertIsNotNone(derived)
        # The loop form is derived once; each request gets its own array
        self.assertTrue(np.array_equal(product.coefficients, coefficients))
        self.assertIs(product._loop, derived)
        self.assertEqual((x + y)._born, "loop")
        self.assertEqual((x * 2)._born, "loop")
        self.assertEqual((product + x)._born, "split")
        self.assertEqual(x.conjugate()._born, "loop")
        self.assertEqual(chained._born, "split")

    def test_reading_coefficients_does_not_change_later_results(self) -> None:
        """Give identical results before and after a derived form is read."""

        for selectors in ({}, {"char": 0, "deg": 2}):
            algebra = cdl.LoopAlgebra(cdl.CDLoop(5), **selectors)
            x, y = algebra.random(3), algebra.random(4)
            untouched = x * y
            before = [untouched * x, untouched + y, untouched - x, untouched * 3]
            touched = x * y
            # Derive and keep the loop-basis coefficients of this copy
            self.assertIsNotNone(touched.coefficients)
            self.assertIsNotNone(touched.kappa())
            after = [touched * x, touched + y, touched - x, touched * 3]
            for first, second in zip(before, after, strict=True):
                with self.subTest(algebra=algebra):
                    self.assertSameBytes(first, second)
                    self.assertEqual(first._born, second._born)

    def test_results_do_not_depend_on_cached_forms(self) -> None:
        """Round identically whether or not operands hold a derived form."""

        operations = (
            ("product", lambda a, b: a * b),
            ("sum", lambda a, b: a + b),
            ("difference", lambda a, b: a - b),
            ("mixed sum", lambda a, b: a * b + a),
            ("mixed product", lambda a, b: (a + b) * (a * b)),
            ("commutator", lambda a, b: a.commutator(b)),
            ("conjugate", lambda a, b: a.conjugate() + b),
            ("scaled", lambda a, b: 0.3 * a + b),
        )
        for selectors in ({}, {"char": 0, "deg": 2}):
            algebra = cdl.LoopAlgebra(cdl.CDLoop(5), **selectors)
            x, y = algebra.random(5), algebra.random(6)
            # Give x and y both forms before the operations are repeated
            warm_x, warm_y = algebra(x.coefficients), algebra(y.coefficients)
            warm_x * warm_y
            warm_x.spectrum()
            warm_y.epsilon()
            for name, operation in operations:
                with self.subTest(algebra=algebra, operation=name):
                    cold = operation(algebra(x.coefficients), algebra(y.coefficients))
                    warm = operation(warm_x, warm_y)
                    self.assertEqual(cold._born, warm._born)
                    self.assertSameBytes(cold, warm)
                    self.assertEqual(
                        plain(cold.spectrum()).tobytes(),
                        plain(warm.spectrum()).tobytes(),
                    )

    def test_sums_of_loop_born_elements_add_coefficients(self) -> None:
        """Add coefficient arrays directly even when split forms are cached."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(4))
        x, y = algebra.random(7), algebra.random(8)
        x * y
        self.assertCoefficientsEqual(x + y, x.coefficients + y.coefficients)
        self.assertCoefficientsEqual(x - y, x.coefficients - y.coefficients)
        self.assertCoefficientsEqual(x * 0.3, x.coefficients * 0.3)
        self.assertCoefficientsEqual(x.kappa(), x.coefficients[0] + x.coefficients[1])
        self.assertCoefficientsEqual(
            x.epsilon().coefficients, x.coefficients[0] - x.coefficients[1]
        )

    def test_accessors_return_independent_arrays(self) -> None:
        """Keep cached forms unchanged when returned arrays are modified."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(3))
        x = algebra.random(9)
        spectrum = x.spectrum()
        expected = spectrum.copy()
        spectrum[:] = 0.0
        self.assertCoefficientsEqual(x.spectrum(), expected)
        kappa = x.kappa()
        kept = kappa.copy()
        kappa[:] = 0.0
        self.assertCoefficientsEqual(x.kappa(), kept)
        product = x * x
        spectrum = product.spectrum()
        expected = spectrum.copy()
        spectrum[:] = 0.0
        self.assertCoefficientsEqual(product.spectrum(), expected)


class LawTests(ElementTestCase):
    def test_products_match_the_reference(self) -> None:
        """Compare products of elements with the independent oracle."""

        rng = np.random.default_rng(41)
        loops = [cdl.CDLoop(level) for level in range(5)]
        loops += [cdl.CentralProduct(1, 2), cdl.CentralProduct(2, 2)]
        for loop in loops:
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                a, b = exact_values(algebra, rng), exact_values(algebra, rng)
                with self.subTest(algebra=algebra):
                    expected = oracle_product(algebra, a, b)
                    self.assertCoefficientsEqual(algebra(a) * algebra(b), expected)

    def test_multiplication_is_bilinear_and_unital(self) -> None:
        """Check distributivity, scalar compatibility and the identity."""

        rng = np.random.default_rng(42)
        for loop in (cdl.CDLoop(4), cdl.CentralProduct(2, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x, y, z = (exact_element(algebra, rng) for _ in range(3))
                one = algebra.one()
                with self.subTest(algebra=algebra):
                    self.assertTrue((x + y) * z == x * z + y * z)
                    self.assertTrue(z * (x + y) == z * x + z * y)
                    self.assertTrue((2 * x) * y == 2 * (x * y))
                    self.assertTrue(x * (y * 2) == (x * y) * 2)
                    self.assertTrue(one * x == x and x * one == x)
                    self.assertFalse(x * algebra.zero())
                    minus = algebra.basis(loop.minus_one())
                    self.assertTrue(minus * minus == one)
                    self.assertTrue(minus * x == x * minus)

    def test_group_component_is_a_homomorphism(self) -> None:
        """Multiply group components by dyadic convolution."""

        rng = np.random.default_rng(43)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                ring = ring_of(algebra)
                x, y = exact_element(algebra, rng), exact_element(algebra, rng)
                with self.subTest(algebra=algebra):
                    left = reference.to_ring(ring, plain(x.kappa()), 1)
                    right = reference.to_ring(ring, plain(y.kappa()), 1)
                    expected = reference.dyadic_convolve(ring, left, right)
                    self.assertCoefficientsEqual(
                        plain((x * y).kappa()),
                        reference.from_ring(ring, expected, 1, algebra.dtype),
                    )
                    self.assertCoefficientsEqual(
                        plain((x + y).kappa()),
                        plain((algebra.lift_group(x.kappa()) + y).kappa()),
                    )

    def test_signed_component_is_a_homomorphism(self) -> None:
        """Multiply signed components by the doubling formula."""

        rng = np.random.default_rng(44)
        for level in range(5):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
                ring = ring_of(algebra)
                x, y = exact_element(algebra, rng), exact_element(algebra, rng)
                with self.subTest(algebra=algebra):
                    left = reference.to_ring(ring, plain(x.epsilon()), 1)
                    right = reference.to_ring(ring, plain(y.epsilon()), 1)
                    expected = reference.cd_multiply(ring, left, right)
                    self.assertCoefficientsEqual(
                        plain((x * y).epsilon()),
                        reference.from_ring(ring, expected, 1, algebra.dtype),
                    )

    def test_signed_component_of_a_central_product_is_a_tensor(self) -> None:
        """Multiply signed components in the tensor product of the factors."""

        rng = np.random.default_rng(45)
        loop = cdl.CentralProduct(2, 1)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(loop, **selectors)
            ring = ring_of(algebra)
            x, y = exact_element(algebra, rng), exact_element(algebra, rng)
            tail = algebra.coefficient_shape[2:]
            with self.subTest(algebra=algebra):
                self.assertEqual(x.epsilon().shape, (4, 2) + tail)
                self.assertFalse(x.epsilon().flags.writeable)
                left = reference.to_ring(ring, x.epsilon().reshape((8,) + tail), 1)
                right = reference.to_ring(ring, y.epsilon().reshape((8,) + tail), 1)
                expected = reference.tensor_multiply(ring, loop.levels, left, right)
                self.assertCoefficientsEqual(
                    (x * y).epsilon().reshape((8,) + tail),
                    reference.from_ring(ring, expected, 1, algebra.dtype),
                )

    def test_decomposition_follows_its_formulas(self) -> None:
        """Compute both components and the spectrum from the coefficients."""

        rng = np.random.default_rng(46)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                ring = ring_of(algebra)
                values = exact_values(algebra, rng)
                x = algebra(values)
                plus, minus = reference.to_ring(ring, values, 2)
                group = [ring.add(a, b) for a, b in zip(plus, minus, strict=True)]
                signed = [
                    ring.add(a, ring.neg(b)) for a, b in zip(plus, minus, strict=True)
                ]
                with self.subTest(algebra=algebra):
                    self.assertCoefficientsEqual(
                        plain(x.kappa()),
                        reference.from_ring(ring, group, 1, algebra.dtype),
                    )
                    self.assertCoefficientsEqual(
                        plain(x.epsilon()).reshape(plain(x.kappa()).shape),
                        reference.from_ring(ring, signed, 1, algebra.dtype),
                    )
                    transform = reference.walsh_hadamard(ring, group)
                    self.assertCoefficientsEqual(
                        plain(x.spectrum()),
                        reference.from_ring(ring, transform, 1, algebra.dtype),
                    )
                    self.assertSameElement(
                        algebra.from_components(x.kappa(), x.epsilon()), x
                    )
                    product = x * x
                    self.assertTrue(
                        algebra.from_components(product.kappa(), product.epsilon())
                        == product
                    )

    def test_component_types(self) -> None:
        """Return ndarrays over R and C and pyfastcd objects over finite fields."""

        loop = cdl.CDLoop(3)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(loop, **selectors)
            x = algebra.random(0)
            with self.subTest(algebra=algebra):
                for component in (x.kappa(), x.spectrum()):
                    if algebra.field is None:
                        self.assertIsInstance(component, np.ndarray)
                        self.assertEqual(component.shape, (8,))
                        self.assertEqual(component.dtype, algebra.dtype)
                    else:
                        self.assertIsInstance(component, fcd.ElementArray)
                        self.assertIs(component.algebra, algebra.field)
                        self.assertEqual(component.batch_shape, (8,))
                signed = x.epsilon()
                self.assertIsInstance(signed, fcd.Element)
                self.assertIs(signed.algebra, algebra.cd_algebra)
                self.assertIs(x.epsilon(), signed)

    def test_augmentation_is_a_homomorphism_to_the_field(self) -> None:
        """Sum all coefficients and multiply compatibly."""

        rng = np.random.default_rng(47)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                ring = ring_of(algebra)
                x, y = exact_element(algebra, rng), exact_element(algebra, rng)

                def value(element: Any, ring: Any = ring) -> Any:
                    """Return an augmentation as an oracle ring element."""

                    result = element.augmentation()
                    if hasattr(result, "coefficients"):
                        return ring.convert(np.asarray(result.coefficients))
                    return ring.convert(result)

                with self.subTest(algebra=algebra):
                    total = ring.zero
                    for row in reference.to_ring(ring, x.coefficients, 2):
                        for entry in row:
                            total = ring.add(total, entry)
                    self.assertEqual(value(x), total)
                    self.assertEqual(value(x * y), ring.mul(value(x), value(y)))
                    self.assertEqual(value(x + y), ring.add(value(x), value(y)))
                    self.assertEqual(value(algebra.one()), ring.one)
                    if algebra.field is None:
                        self.assertIsInstance(
                            x.augmentation(), complex if algebra.deg == 2 else float
                        )
                    else:
                        self.assertIs(x.augmentation().algebra, algebra.field)


class ProductMethodTests(ElementTestCase):
    def test_conjugate_extends_loop_inversion(self) -> None:
        """Move the coefficient of every loop element to its inverse."""

        rng = np.random.default_rng(51)
        for loop in (cdl.CDLoop(3), cdl.CDLoop(4), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x, y = exact_element(algebra, rng), exact_element(algebra, rng)
                with self.subTest(algebra=algebra):
                    moved = algebra.from_terms(
                        [(g.inverse(), c) for g, c in x.terms().items()]
                    )
                    self.assertSameElement(x.conjugate(), moved)
                    self.assertSameElement(x.conjugate().conjugate(), x)
                    self.assertTrue(
                        (x * y).conjugate() == y.conjugate() * x.conjugate()
                    )
                    self.assertTrue(
                        (x + y).conjugate() == x.conjugate() + y.conjugate()
                    )
                    self.assertCoefficientsEqual(
                        plain(x.conjugate().kappa()), plain(x.kappa())
                    )
                    product = x * y
                    self.assertTrue(
                        product.conjugate() == algebra(product.coefficients).conjugate()
                    )

    def test_conjugate_conjugates_the_signed_component(self) -> None:
        """Apply the Cayley--Dickson conjugation to the signed component."""

        rng = np.random.default_rng(52)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x = exact_element(algebra, rng)
            with self.subTest(algebra=algebra):
                self.assertTrue(x.conjugate().epsilon() == x.epsilon().conjugate())
                norm = x * x.conjugate()
                self.assertTrue(norm == x.conjugate() * x)
                self.assertTrue(norm.is_central())

    def test_derived_products_follow_their_definitions(self) -> None:
        """Compare each named product with explicit multiplications."""

        rng = np.random.default_rng(53)
        loops = (cdl.CDLoop(2), cdl.CDLoop(3), cdl.CDLoop(4), cdl.CentralProduct(2, 2))
        for loop in loops:
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x, y, z = (exact_element(algebra, rng) for _ in range(3))
                with self.subTest(algebra=algebra):
                    self.assertTrue(x.square() == x * x)
                    self.assertTrue(x.commutator(y) == x * y - y * x)
                    self.assertTrue(x.anticommutator(y) == x * y + y * x)
                    self.assertTrue(2 * x.jordan_product(y) == x * y + y * x)
                    self.assertTrue(x.associator(y, z) == (x * y) * z - x * (y * z))
                    self.assertFalse(x.commutator(x))
                    self.assertTrue(x.jordan_product(x) == x * x)
                    self.assertFalse(np.any(plain(x.commutator(y).kappa())))
                    self.assertFalse(np.any(plain(x.associator(y, z).kappa())))
                    for result in (x.square(), x.commutator(y), x.associator(y, z)):
                        self.assertIsInstance(result, cdl.Element)

    def test_identities_of_small_levels(self) -> None:
        """Observe commutativity, associativity and alternativity where they hold."""

        rng = np.random.default_rng(54)
        for level in range(5):
            algebra = cdl.LoopAlgebra(cdl.CDLoop(level))
            x, y, z = (exact_element(algebra, rng) for _ in range(3))
            with self.subTest(level=level):
                self.assertEqual(not x.commutator(y), level <= 1)
                self.assertEqual(not x.associator(y, z), level <= 2)
                self.assertEqual(not x.associator(x, y), level <= 3)
                self.assertFalse(x.associator(y, x))

    def test_bracketed_powers(self) -> None:
        """Nest products to the left and to the right."""

        rng = np.random.default_rng(55)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(2, 3)):
            for selectors in ({}, {"char": 13}):
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x = exact_element(algebra, rng)
                with self.subTest(algebra=algebra):
                    self.assertTrue(x.left_power(0) == 1)
                    self.assertTrue(x.right_power(0) == 1)
                    self.assertSameElement(algebra(x.left_power(1).coefficients), x)
                    self.assertTrue(x.left_power(2) == x * x)
                    self.assertTrue(x.right_power(2) == x * x)
                    self.assertTrue(x.left_power(3) == x * (x * x))
                    self.assertTrue(x.right_power(3) == (x * x) * x)
                    self.assertTrue(x.left_power(4) == x * (x * (x * x)))
                    self.assertTrue(x.right_power(4) == ((x * x) * x) * x)
                    self.assertTrue(x.left_power(np.int64(2)) == x * x)
                    with self.assertRaises(ValueError):
                        x.left_power(-1)
                    with self.assertRaises(ValueError):
                        x.right_power(-2)
                    for exponent in (1.5, "2", None, True):
                        with self.assertRaises(TypeError):
                            x.left_power(exponent)

    def test_central_product_powers_depend_on_bracketing(self) -> None:
        """Reject ``**`` where left and right powers differ."""

        algebra = cdl.LoopAlgebra(cdl.CentralProduct(2, 3))
        x = exact_element(algebra, np.random.default_rng(7))
        self.assertFalse(x.left_power(3) == x.right_power(3))
        for exponent in (0, 1, 2, -1):
            with self.subTest(exponent=exponent), self.assertRaises(TypeError):
                x**exponent
        for name in ("inverse", "has_inverse", "is_nilpotent", "minimal_polynomial"):
            with self.subTest(method=name), self.assertRaises(TypeError):
                getattr(x, name)()


class PowerTests(ElementTestCase):
    def test_small_powers_agree_with_repeated_products(self) -> None:
        """Compare ``**`` with both bracketings for small exponents."""

        rng = np.random.default_rng(61)
        for level in range(5):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
                x = exact_element(algebra, rng)
                with self.subTest(algebra=algebra):
                    self.assertTrue(x**0 == 1)
                    self.assertTrue(x**1 == x)
                    self.assertTrue(x**2 == x * x)
                    for exponent in range(1, 6):
                        power = x**exponent
                        self.assertIsInstance(power, cdl.Element)
                        self.assertTrue(power == x.left_power(exponent))
                        self.assertTrue(power == x.right_power(exponent))
                    self.assertTrue(x**2 * x**3 == x**5)
                    self.assertTrue(x ** np.int64(3) == x**3)
                    self.assertTrue(algebra.zero() ** 0 == 1)
                    self.assertFalse(algebra.zero() ** 3)

    def test_large_exponents_match_the_reference(self) -> None:
        """Compare huge powers over finite fields with oracle square-and-multiply."""

        rng = np.random.default_rng(62)
        exponents = (97, 10**18 + 7, 2**200 + 12345)
        for level in (1, 2, 3):
            for selectors in FINITE_SELECTORS:
                algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
                ring = ring_of(algebra)
                values = exact_values(algebra, rng)
                x = algebra(values)
                base = reference.to_ring(ring, values, 2)
                for exponent in exponents:
                    result = None
                    square = base
                    remaining = exponent
                    while remaining:
                        if remaining & 1:
                            result = (
                                square
                                if result is None
                                else reference.loop_algebra_multiply(
                                    ring, (level,), result, square
                                )
                            )
                        remaining >>= 1
                        if remaining:
                            square = reference.loop_algebra_multiply(
                                ring, (level,), square, square
                            )
                    with self.subTest(algebra=algebra, exponent=exponent):
                        self.assertCoefficientsEqual(
                            x**exponent,
                            reference.from_ring(ring, result, 2, algebra.dtype),
                        )

    def test_negative_powers_use_the_inverse(self) -> None:
        """Invert first, exactly over finite fields."""

        for selectors in FINITE_SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x = invertible_element(algebra, 63)
            with self.subTest(algebra=algebra):
                self.assertTrue(x**-1 == x.inverse())
                self.assertTrue(x**-3 == x.inverse() ** 3)
                self.assertTrue(x**-3 * x**3 == 1)
                self.assertTrue(x**-2 == (x * x).inverse())
        real = cdl.LoopAlgebra(cdl.CDLoop(3))
        y = real.random(64)
        self.assertSameBytes(y**-1, y.inverse())
        self.assertCoefficientsClose(y**-2 * (y * y), real.one())
        with self.assertRaises(ZeroDivisionError):
            real.zero() ** -1

    def test_exponents_must_be_integers(self) -> None:
        """Reject fractional, Boolean and non-numeric exponents."""

        x = cdl.LoopAlgebra(cdl.CDLoop(2)).random(0)
        for exponent in (1.5, 2.0, True, np.bool_(True), "2", None, x):
            with self.subTest(exponent=repr(exponent)), self.assertRaises(TypeError):
                x**exponent
        with self.assertRaises(TypeError):
            2**x


class InverseTests(ElementTestCase):
    def test_inverse_is_two_sided(self) -> None:
        """Multiply to the identity on both sides, exactly over finite fields."""

        for level in range(5):
            for selectors in FINITE_SELECTORS:
                algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
                x = invertible_element(algebra, 71 + level)
                with self.subTest(algebra=algebra):
                    inverse = x.inverse()
                    self.assertIsInstance(inverse, cdl.Element)
                    self.assertTrue(x.has_inverse())
                    self.assertTrue(x * inverse == 1)
                    self.assertTrue(inverse * x == 1)
                    self.assertTrue(inverse.inverse() == x)
            for selectors in ({}, {"char": 0, "deg": 2}):
                algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
                x = algebra.random(72)
                with self.subTest(algebra=algebra):
                    self.assertTrue(x.has_inverse())
                    one = algebra.one().coefficients
                    self.assertCoefficientsClose(x * x.inverse(), one)
                    self.assertCoefficientsClose(x.inverse() * x, one)

    def test_inverse_of_simple_elements(self) -> None:
        """Invert scalars and basis elements in closed form."""

        loop = cdl.CDLoop(3)
        algebra = cdl.LoopAlgebra(loop)
        self.assertTrue(algebra.one().inverse() == 1)
        self.assertTrue(algebra.scalar(4).inverse() == 0.25)
        for element in loop:
            self.assertTrue(
                algebra.basis(element).inverse() == algebra.basis(element.inverse())
            )

    def test_zero_spectral_coordinate_has_no_inverse(self) -> None:
        """Raise when a character vanishes on the element."""

        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            plus, minus = algebra.central_idempotents()
            candidates = (
                algebra.zero(),
                minus,
                algebra.one() - algebra.primitive_idempotent(2),
                algebra.lift_signed(algebra.cd_algebra.one()),
            )
            for number, x in enumerate(candidates):
                with self.subTest(algebra=algebra, candidate=number):
                    self.assertFalse(x.has_inverse())
                    with self.assertRaises(ZeroDivisionError):
                        x.inverse()
                    with self.assertRaises(ZeroDivisionError):
                        x**-1

    def test_isotropic_signed_component_has_no_inverse(self) -> None:
        """Raise when the signed component is nonzero with quadratic norm zero."""

        loop = cdl.CDLoop(2)
        cases = (
            (cdl.LoopAlgebra(loop, char=0, deg=2), [1, 1j, 0, 0]),
            (cdl.LoopAlgebra(loop, char=5), [1, 2, 0, 0]),
            (cdl.LoopAlgebra(loop, char=13), [0, 2, 3, 0]),
        )
        for algebra, signed in cases:
            plus, _ = algebra.central_idempotents()
            x = plus + algebra.lift_signed(signed)
            with self.subTest(algebra=algebra):
                self.assertTrue(np.all(plain(x.spectrum()) != 0))
                self.assertTrue(np.any(plain(x.epsilon())))
                self.assertFalse(x.has_inverse())
                with self.assertRaises(ZeroDivisionError):
                    x.inverse()
                with self.assertRaises(ZeroDivisionError):
                    x**-2
        self.assertTrue(plus.has_inverse() is False)


class StructureTests(ElementTestCase):
    def test_idempotents(self) -> None:
        """Recognize idempotents of both summands."""

        loop = cdl.CDLoop(2)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(loop, **selectors)
            plus, minus = algebra.central_idempotents()
            with self.subTest(algebra=algebra):
                for x in (algebra.zero(), algebra.one(), plus, minus):
                    self.assertTrue(x.is_idempotent())
                self.assertTrue(algebra.primitive_idempotent(3).is_idempotent())
                self.assertTrue(
                    (algebra.primitive_idempotent(0) + minus).is_idempotent()
                )
                self.assertFalse(algebra.scalar(2).is_idempotent())
                self.assertFalse(algebra.basis(loop.basis(1)).is_idempotent())
                self.assertFalse(algebra.basis(loop.minus_one()).is_idempotent())
        # A signed component with d_0 = 1/2 and Q(d) = 0 is idempotent
        cases = (
            (cdl.LoopAlgebra(loop, char=0, deg=2), [0.5, 0.5j, 0, 0]),
            (cdl.LoopAlgebra(loop, char=5), [3, 1, 0, 0]),
        )
        for algebra, signed in cases:
            x = algebra.lift_signed(signed)
            with self.subTest(algebra=algebra):
                self.assertTrue(x.is_idempotent())
                self.assertTrue(x * x == x)
                self.assertTrue((x + algebra.primitive_idempotent(1)).is_idempotent())

    def test_nilpotents(self) -> None:
        """Recognize signed components with scalar part and quadratic norm zero."""

        loop = cdl.CDLoop(2)
        cases = (
            (cdl.LoopAlgebra(loop, char=0, deg=2), [0, 1, 1j, 0]),
            (cdl.LoopAlgebra(loop, char=5), [0, 1, 2, 0]),
            (cdl.LoopAlgebra(loop, char=13), [0, 2, 3, 0]),
        )
        for algebra, signed in cases:
            x = algebra.lift_signed(signed)
            with self.subTest(algebra=algebra):
                self.assertTrue(x)
                self.assertTrue(x.is_nilpotent())
                self.assertFalse(x * x)
                self.assertTrue(algebra.zero().is_nilpotent())
                self.assertFalse(algebra.one().is_nilpotent())
                self.assertFalse((x + algebra.primitive_idempotent(0)).is_nilpotent())
                self.assertFalse((x + algebra.central_idempotents()[1]).is_nilpotent())
        real = cdl.LoopAlgebra(loop)
        self.assertTrue(real.zero().is_nilpotent())
        self.assertFalse(real.lift_signed([0.0, 1.0, 2.0, 0.0]).is_nilpotent())

    def test_central_elements(self) -> None:
        """Recognize elements whose signed component is central."""

        loop = cdl.CDLoop(3)
        algebra = cdl.LoopAlgebra(loop)
        rng = np.random.default_rng(81)
        x = exact_element(algebra, rng)
        self.assertFalse(x.is_central())
        self.assertTrue(algebra.lift_group(x.kappa()).is_central())
        self.assertTrue((algebra.lift_group(x.kappa()) + 5).is_central())
        self.assertTrue(algebra.basis(loop.minus_one()).is_central())
        self.assertFalse(algebra.basis(loop.basis(5)).is_central())
        product = cdl.LoopAlgebra(cdl.CentralProduct(1, 2))
        central = product.basis(product.loop.basis(1, 0))
        self.assertTrue(central.is_central())
        self.assertFalse(product.basis(product.loop.basis(0, 1)).is_central())

    def test_zero_divisors_of_composition_levels(self) -> None:
        """Decide zero divisors from the spectrum and the quadratic norm."""

        for level in range(4):
            loop = cdl.CDLoop(level)
            real = cdl.LoopAlgebra(loop)
            plus, minus = real.central_idempotents()
            with self.subTest(level=level):
                self.assertTrue(real.zero().is_zero_divisor())
                self.assertFalse(real.one().is_zero_divisor())
                self.assertTrue(plus.is_zero_divisor())
                self.assertTrue(minus.is_zero_divisor())
                self.assertFalse(real.basis(loop.minus_one()).is_zero_divisor())
                generic = real.random(82)
                for side in (None, "left", "right"):
                    self.assertFalse(generic.is_zero_divisor(side))
                    self.assertTrue(real.primitive_idempotent(0).is_zero_divisor(side))
                self.assertFalse(plus * minus)
        loop = cdl.CDLoop(2)
        cases = (
            (cdl.LoopAlgebra(loop, char=0, deg=2), [1, 1j, 0, 0]),
            (cdl.LoopAlgebra(loop, char=5), [1, 2, 0, 0]),
        )
        for algebra, signed in cases:
            x = algebra.central_idempotents()[0] + algebra.lift_signed(signed)
            with self.subTest(algebra=algebra):
                self.assertTrue(x.is_zero_divisor())
                partner = algebra.lift_signed(
                    algebra.lift_signed(signed).conjugate().epsilon()
                )
                self.assertTrue(partner)
                self.assertFalse(x * partner)
        with self.assertRaises(ValueError):
            cdl.LoopAlgebra(loop).one().is_zero_divisor("up")
        with self.assertRaises(TypeError):
            cdl.LoopAlgebra(loop).one().is_zero_divisor(1)

    def test_sedenion_zero_divisor(self) -> None:
        """Detect the zero divisor ``(e1 + e10)(e4 - e15) = 0`` at level 4."""

        for selectors in ({}, {"char": 13}):
            algebra = cdl.LoopAlgebra(cdl.CDLoop(4), **selectors)
            first = np.zeros(16, dtype=np.int64)
            first[[1, 10]] = 1
            second = np.zeros(16, dtype=np.int64)
            second[4], second[15] = 1, -1
            a = algebra.lift_signed(first.astype(float) if not selectors else first)
            b = algebra.lift_signed(second.astype(float) if not selectors else second)
            plus, _ = algebra.central_idempotents()
            with self.subTest(algebra=algebra):
                self.assertTrue(a and b)
                self.assertFalse(a * b)
                self.assertFalse(b * a)
                self.assertTrue(a.epsilon().quadratic_norm() != 0)
                x, y = plus + a, plus + b
                self.assertTrue(np.all(plain(x.spectrum()) != 0))
                for side in (None, "left", "right"):
                    self.assertTrue(x.is_zero_divisor(side))
                    self.assertTrue(y.is_zero_divisor(side))
                unit = algebra.basis(algebra.loop.basis(3))
                self.assertFalse(unit.is_zero_divisor())
                self.assertFalse(algebra.one().is_zero_divisor())
        generic = cdl.LoopAlgebra(cdl.CDLoop(4)).random(83)
        for side in (None, "left", "right"):
            self.assertFalse(generic.is_zero_divisor(side))

    def test_annihilators_of_simple_elements(self) -> None:
        """Compute exact annihilators of the identity, zero and idempotents."""

        loop = cdl.CDLoop(2)
        for selectors in (
            {},
            {"char": 0, "deg": 2},
            {"char": 13},
            {"char": 3, "deg": 2},
        ):
            algebra = cdl.LoopAlgebra(loop, **selectors)
            plus, minus = algebra.central_idempotents()
            cases = (
                (algebra.one(), 0),
                (algebra.zero(), 8),
                (plus, 4),
                (minus, 4),
                (algebra.primitive_idempotent(1), 7),
                (algebra.one() - algebra.primitive_idempotent(1), 1),
            )
            for number, (x, dimension) in enumerate(cases):
                for side in ("left", "right"):
                    basis = getattr(x, f"{side}_annihilator")()
                    with self.subTest(algebra=algebra, case=number, side=side):
                        self.assertIsInstance(basis, cdl.ElementArray)
                        self.assertEqual(basis.batch_shape, (dimension,))
                        products = basis * x if side == "left" else x * basis
                        if algebra.field is not None:
                            self.assertFalse(np.any(products.coefficients))
                        else:
                            self.assertCoefficientsClose(products.coefficients, 0.0)
                        flat = basis.coefficients.reshape(
                            (dimension, 8) + algebra.coefficient_shape[2:]
                        )
                        if dimension:
                            self.assertEqual(
                                cdl.linalg.rank(flat, field=algebra.field), dimension
                            )

    def test_annihilators_of_a_sedenion_zero_divisor(self) -> None:
        """Multiply the annihilator back and compare its dimension with the nullity."""

        for selectors in ({}, {"char": 13}):
            algebra = cdl.LoopAlgebra(cdl.CDLoop(4), **selectors)
            signed = np.zeros(16, dtype=np.int64)
            signed[[1, 10]] = 1
            values = signed if selectors else signed.astype(float)
            x = algebra.central_idempotents()[0] + algebra.lift_signed(values)
            for side, matrix in (
                ("right", algebra.left_matrix(x)),
                ("left", algebra.right_matrix(x)),
            ):
                basis = getattr(x, f"{side}_annihilator")()
                nullity = 32 - cdl.linalg.rank(matrix, field=algebra.field)
                with self.subTest(algebra=algebra, side=side):
                    self.assertEqual(basis.batch_shape, (nullity,))
                    self.assertGreater(nullity, 0)
                    products = basis * x if side == "left" else x * basis
                    if selectors:
                        self.assertFalse(np.any(products.coefficients))
                    else:
                        self.assertCoefficientsClose(products.coefficients, 0.0)
                    self.assertFalse(np.any(plain(basis.kappa())) and selectors)

    def test_minimal_polynomial_vanishes_and_is_minimal(self) -> None:
        """Evaluate the polynomial at the element and check its degree."""

        rng = np.random.default_rng(84)
        for level in range(4):
            for selectors in FINITE_SELECTORS:
                algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
                x = exact_element(algebra, rng)
                polynomial = x.minimal_polynomial()
                degree = polynomial.shape[0] - 1
                with self.subTest(algebra=algebra):
                    self.assertEqual(polynomial.dtype, algebra.dtype)
                    self.assertEqual(
                        polynomial.shape[1:], algebra.coefficient_shape[2:]
                    )
                    self.assertTrue(
                        algebra.scalar(field_scalar(algebra, polynomial[-1])) == 1
                    )
                    self.assertFalse(evaluate(algebra, polynomial, x))
                    # The powers below the degree are linearly independent
                    powers = np.stack(
                        [(x**k).coefficients for k in range(degree)]
                    ).reshape((degree, -1) + algebra.coefficient_shape[2:])
                    self.assertEqual(
                        cdl.linalg.rank(powers, field=algebra.field), degree
                    )

    def test_minimal_polynomial_of_small_real_elements(self) -> None:
        """Recover known polynomials over R."""

        loop = cdl.CDLoop(1)
        algebra = cdl.LoopAlgebra(loop)
        self.assertCoefficientsEqual(algebra.zero().minimal_polynomial(), [0, 1])
        self.assertCoefficientsEqual(algebra.one().minimal_polynomial(), [-1, 1])
        self.assertCoefficientsEqual(algebra.scalar(3).minimal_polynomial(), [-3, 1])
        minus = algebra.basis(loop.minus_one())
        self.assertCoefficientsEqual(minus.minimal_polynomial(), [-1, 0, 1])
        i = algebra.basis(loop.basis(1))
        polynomial = i.minimal_polynomial()
        self.assertCoefficientsEqual(polynomial, [-1, 0, 0, 0, 1])
        self.assertFalse(evaluate(algebra, polynomial, i))
        plus, _ = algebra.central_idempotents()
        self.assertCoefficientsEqual(plus.minimal_polynomial(), [0, -1, 1])
        x = algebra([[1.0, 2.0], [0.0, -1.0]])
        self.assertFalse(evaluate(algebra, x.minimal_polynomial(), x))

    def test_minimal_polynomial_with_coinciding_roots(self) -> None:
        """Drop spectral values that are roots of the signed polynomial."""

        loop = cdl.CDLoop(2)
        real = cdl.LoopAlgebra(loop)
        # The signed component is the scalar 2, a value the spectrum also takes
        x = real.split_array([2.0, 5.0, 5.0, 7.0], [2.0, 0.0, 0.0, 0.0])[()]
        polynomial = x.minimal_polynomial()
        self.assertCoefficientsEqual(polynomial, [-70, 59, -14, 1])
        self.assertFalse(evaluate(real, polynomial, x))

        finite = cdl.LoopAlgebra(loop, char=13)
        # t^2 - 2 t + 5 = (t - 4)(t - 11) over F_13
        y = finite.split_array([4, 11, 4, 1], [1, 2, 0, 0])[()]
        polynomial = y.minimal_polynomial()
        self.assertEqual(polynomial.shape, (4,))
        self.assertFalse(evaluate(finite, polynomial, y))
        # (t - 5)^2 is the minimal polynomial of a nonscalar signed component
        z = finite.split_array([5, 1, 5, 2], [5, 2, 3, 0])[()]
        polynomial = z.minimal_polynomial()
        self.assertCoefficientsEqual(polynomial, [11, 9, 5, 0, 1])
        self.assertFalse(evaluate(finite, polynomial, z))
        description = finite.subalgebra_generated_by(z)
        self.assertEqual(description.dimension, 4)
        self.assertCoefficientsEqual(description.signed_polynomial, [12, 3, 1])
        self.assertCoefficientsEqual(description.spectral_values, [1, 2, 5])


class DataTests(ElementTestCase):
    def test_coefficient_returns_field_scalars(self) -> None:
        """Read single coefficients as scalars of the field."""

        loop = cdl.CDLoop(2)
        real = cdl.LoopAlgebra(loop)
        values = np.arange(8.0).reshape(2, 4)
        x = real(values)
        for element in loop:
            row = 0 if element.sign > 0 else 1
            self.assertEqual(x.coefficient(element), values[row, element.index])
        self.assertIsInstance(x.coefficient(loop.basis(1)), float)
        z = cdl.LoopAlgebra(loop, char=0, deg=2)(values * 1j)
        self.assertIsInstance(z.coefficient(loop.basis(1)), complex)
        self.assertEqual(z.coefficient(-loop.basis(1)), 5j)
        finite = cdl.LoopAlgebra(loop, char=7)
        w = finite(np.arange(8).reshape(2, 4))
        scalar = w.coefficient(-loop.basis(3))
        self.assertIsInstance(scalar, fcd.Element)
        self.assertIs(scalar.algebra, finite.field)
        self.assertTrue(scalar == 0)
        self.assertTrue(w.coefficient(loop.basis(3)) == 3)
        extension = cdl.LoopAlgebra(loop, char=3, deg=2)
        v = extension.scalar(extension.field([1, 2]))
        self.assertTrue(v.coefficient(loop.one()) == extension.field([1, 2]))
        with self.assertRaises(ValueError):
            x.coefficient(cdl.CDLoop(3).basis(1))
        with self.assertRaises(TypeError):
            x.coefficient(1)

    def test_terms_and_support(self) -> None:
        """List the nonzero coefficients by loop element."""

        loop = cdl.CDLoop(2)
        algebra = cdl.LoopAlgebra(loop)
        i, j = loop.basis(1), loop.basis(2)
        x = algebra.from_terms({i: 2.0, -j: -1.5, loop.one(): 0.0})
        self.assertEqual(x.terms(), {i: 2.0, -j: -1.5})
        self.assertEqual(x.support(), frozenset({i, -j}))
        self.assertEqual(algebra.zero().terms(), {})
        self.assertEqual(algebra.zero().support(), frozenset())
        product = x * x
        self.assertEqual(product.support(), frozenset(product.terms()))
        self.assertEqual(
            (algebra.basis(i) * algebra.basis(j)).terms(), {loop.basis(3): 1.0}
        )
        for selectors in SELECTORS:
            other = cdl.LoopAlgebra(loop, **selectors)
            y = other.random(1)
            with self.subTest(algebra=other):
                self.assertSameElement(other.from_terms(y.terms()), y)
                self.assertEqual(len(y.terms()), y.weight())

    def test_weight_and_distance(self) -> None:
        """Count nonzero coefficients and differing coefficients."""

        loop = cdl.CDLoop(2)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(loop, **selectors)
            one = algebra.one()
            x = algebra.basis(loop.basis(1)) + algebra.basis(-loop.basis(2))
            with self.subTest(algebra=algebra):
                self.assertEqual(algebra.zero().weight(), 0)
                self.assertEqual(one.weight(), 1)
                self.assertEqual(x.weight(), 2)
                self.assertIsInstance(x.weight(), int)
                self.assertEqual(x.distance(x), 0)
                self.assertEqual(x.distance(algebra.zero()), 2)
                self.assertEqual(x.distance(one), 3)
                self.assertEqual(one.distance(x), 3)
                self.assertEqual(x.distance(x + one), (x + one).distance(x))
                self.assertIsInstance(x.distance(one), int)
                self.assertEqual(
                    (x * x).weight(), algebra((x * x).coefficients).weight()
                )

    def test_extension_field_weight_counts_field_elements(self) -> None:
        """Count a coefficient with any nonzero polynomial coordinate once."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(1), char=3, deg=2)
        x = algebra([[[1, 2], [0, 1]], [[0, 0], [2, 0]]])
        self.assertEqual(x.weight(), 3)
        self.assertEqual(x.distance(algebra.zero()), 3)
        self.assertEqual(len(x.terms()), 3)


if __name__ == "__main__":
    unittest.main()
