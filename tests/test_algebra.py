"""Deterministic tests for :class:`pycdloop.LoopAlgebra`."""

from __future__ import annotations

import pickle
import unittest
from typing import Any

import numpy as np
import pyfastcd as fcd

import pycdloop as cdl

try:  # Support both package-style collection and collection from this directory.
    from . import reference
except ImportError:  # pragma: no cover - depends on the collection mechanism
    import reference


# Coefficient fields exercised by most tests: R, C, two prime fields and two
# extension fields
SELECTORS: tuple[dict[str, int], ...] = (
    {},
    {"char": 0, "deg": 2},
    {"char": 3},
    {"char": 13},
    {"char": 3, "deg": 2},
    {"char": 13, "deg": 3},
)


def ring_of(algebra: Any) -> Any:
    """Return the oracle ring that matches an algebra's coefficient field."""

    if algebra.field is not None:
        return reference.FiniteField(algebra.char, algebra.field.modulus)
    if algebra.deg == 2:
        return reference.GaussianRationals()
    return reference.Rationals()


def exact_values(algebra: Any, rng: Any, lead: tuple[int, ...] = ()) -> np.ndarray:
    """Draw coefficients on which the arithmetic of the tests is exact."""

    shape = lead + algebra.coefficient_shape
    if algebra.field is not None:
        return rng.integers(0, algebra.char, size=shape, dtype=np.uint32)
    values = rng.integers(-3, 4, size=shape).astype(algebra.dtype)
    if algebra.deg == 2:
        values = values + 1j * rng.integers(-3, 4, size=shape)
    return values


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


def basis_batch(algebra: Any) -> Any:
    """Return every basis element of an algebra as one array."""

    size = algebra.dimension
    eye = np.eye(size, dtype=np.int64).reshape(size, 2, size // 2)
    if algebra.field is not None and algebra.deg > 1:
        padded = np.zeros(eye.shape + (algebra.deg,), dtype=np.int64)
        padded[..., 0] = eye
        eye = padded
    return algebra.array(eye.astype(algebra.dtype))


class LoopAlgebraTestCase(unittest.TestCase):
    """Common assertions with useful failure diagnostics."""

    def assertCoefficientsEqual(self, actual: object, expected: object) -> None:
        """Assert exact equality after exposing both coefficient arrays."""

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def assertSameElement(self, actual: Any, expected: Any) -> None:
        """Assert that two elements share a parent and all coefficients."""

        self.assertIs(actual.algebra, expected.algebra)
        self.assertCoefficientsEqual(actual.coefficients, expected.coefficients)


class ParentTests(LoopAlgebraTestCase):
    def test_algebras_are_cached_by_loop_and_field(self) -> None:
        """Return one object for equivalent loops and selectors."""

        loop = cdl.CDLoop(3)
        algebra = cdl.LoopAlgebra(loop)
        self.assertIs(algebra, cdl.LoopAlgebra(cdl.CDLoop(3)))
        self.assertIs(algebra, cdl.LoopAlgebra(loop, char=0, deg=1))
        self.assertIsNot(algebra, cdl.LoopAlgebra(loop, char=0, deg=2))
        self.assertIsNot(algebra, cdl.LoopAlgebra(cdl.CDLoop(2)))
        self.assertIsNot(algebra, cdl.LoopAlgebra(cdl.CentralProduct(1, 2)))
        prime = cdl.LoopAlgebra(loop, char=7)
        self.assertIs(prime, cdl.LoopAlgebra(loop, field=fcd.FiniteField(char=7)))
        self.assertIs(prime, cdl.LoopAlgebra(loop, char=7, deg=1))
        self.assertIsNot(prime, cdl.LoopAlgebra(loop, char=11))
        field = fcd.FiniteField(char=13, deg=2)
        self.assertIs(
            cdl.LoopAlgebra(loop, char=13, deg=2), cdl.LoopAlgebra(loop, field=field)
        )

    def test_field_presentations_give_different_parents(self) -> None:
        """Distinguish two polynomial presentations of one finite field."""

        default = fcd.FiniteField(char=13, deg=2)
        other = None
        for constant in range(1, 13):
            if (constant, 0, 1) == tuple(default.modulus):
                continue
            try:
                other = fcd.FiniteField.from_modulus(char=13, modulus=(constant, 0, 1))
            except ValueError:
                continue
            break
        self.assertIsNotNone(other)
        loop = cdl.CDLoop(2)
        first = cdl.LoopAlgebra(loop, field=default)
        second = cdl.LoopAlgebra(loop, field=other)
        self.assertIsNot(first, second)
        self.assertIs(second.field, other)
        self.assertFalse(first.one() == second.one())

    def test_invalid_parents_are_rejected(self) -> None:
        """Reject values that are not loops and conflicting selectors."""

        for value in (3, "Q_3", None, cdl.CDLoop(2).basis(1)):
            with self.subTest(value=value), self.assertRaises(TypeError):
                cdl.LoopAlgebra(value)
        with self.assertRaises(TypeError):
            cdl.LoopAlgebra(cdl.CDLoop(2), char=7, field=fcd.FiniteField(char=7))
        with self.assertRaises(ValueError):
            cdl.LoopAlgebra(cdl.CDLoop(2), char=4)
        with self.assertRaises(ValueError):
            cdl.LoopAlgebra(cdl.CDLoop(2), char=2, deg=13)
        with self.assertRaises(TypeError):
            cdl.LoopAlgebra(cdl.CDLoop(2), char=3, field=cdl.BinaryField(deg=2))

    def test_algebras_are_immutable_and_final(self) -> None:
        """Reject attribute changes and subclasses."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        with self.assertRaises(AttributeError):
            algebra.dimension = 3  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            algebra.extra = 1  # type: ignore[attr-defined]
        with self.assertRaises(AttributeError):
            del algebra.loop
        with self.assertRaises(TypeError):
            type("Derived", (cdl.LoopAlgebra,), {})

    def test_pickling_returns_the_cached_parent(self) -> None:
        """Unpickle every kind of algebra as the same object."""

        loops = (cdl.CDLoop(0), cdl.CDLoop(3), cdl.CentralProduct(1, 2))
        for loop in loops:
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                with self.subTest(algebra=algebra):
                    self.assertIs(pickle.loads(pickle.dumps(algebra)), algebra)

    def test_repr_is_the_constructor_expression(self) -> None:
        """Show the loop and the nondefault selectors."""

        loop = cdl.CDLoop(2)
        self.assertEqual(repr(cdl.LoopAlgebra(loop)), "LoopAlgebra(CDLoop(2))")
        self.assertEqual(
            repr(cdl.LoopAlgebra(loop, char=0, deg=2)),
            "LoopAlgebra(CDLoop(2), char=0, deg=2)",
        )
        self.assertEqual(
            repr(cdl.LoopAlgebra(loop, char=13, deg=2)),
            "LoopAlgebra(CDLoop(2), char=13, deg=2)",
        )
        product = cdl.LoopAlgebra(cdl.CentralProduct(1, 2), char=7)
        self.assertEqual(
            repr(product), "LoopAlgebra(CentralProduct(1, 2), char=7, deg=1)"
        )

    def test_properties_describe_loop_and_field(self) -> None:
        """Report dimension, shapes, storage type and field selectors."""

        expected = {
            (): (np.float64, 0, 1, ()),
            (("char", 0), ("deg", 2)): (np.complex128, 0, 2, ()),
            (("char", 13),): (np.uint32, 13, 1, ()),
            (("char", 13), ("deg", 3)): (np.uint32, 13, 3, (3,)),
        }
        for loop in (cdl.CDLoop(0), cdl.CDLoop(4), cdl.CentralProduct(2, 1)):
            for key, (dtype, char, deg, tail) in expected.items():
                algebra = cdl.LoopAlgebra(loop, **dict(key))
                with self.subTest(algebra=algebra):
                    self.assertIs(algebra.loop, loop)
                    self.assertEqual(algebra.dimension, loop.order)
                    self.assertEqual(
                        algebra.coefficient_shape, (2, loop.quotient_order) + tail
                    )
                    self.assertEqual(algebra.dtype, np.dtype(dtype))
                    self.assertEqual((algebra.char, algebra.deg), (char, deg))
                    if char:
                        self.assertIs(
                            algebra.field, fcd.FiniteField(char=char, deg=deg)
                        )
                    else:
                        self.assertIsNone(algebra.field)

    def test_signed_component_properties(self) -> None:
        """Expose the pyfastcd algebra and the signed-product methods."""

        real = cdl.LoopAlgebra(cdl.CDLoop(3))
        self.assertIs(real.cd_algebra, fcd.Algebra(level=3))
        self.assertEqual(real.default_algorithm, "split")
        self.assertEqual(
            real.signed_algorithms, ("auto", "direct", "cariow_cariowa", "quasilinear")
        )
        self.assertEqual(
            real.default_signed_algorithm, fcd.Algebra(level=3).default_algorithm
        )
        finite = cdl.LoopAlgebra(cdl.CDLoop(3), char=13, deg=3)
        self.assertIs(finite.cd_algebra, fcd.Algebra(level=3, char=13, deg=3))
        self.assertEqual(finite.signed_algorithms, ("auto", "direct", "quasilinear"))
        complex_algebra = cdl.LoopAlgebra(cdl.CDLoop(3), char=0, deg=2)
        self.assertIs(complex_algebra.cd_algebra, fcd.Algebra(level=3, char=0, deg=2))

        product = cdl.LoopAlgebra(cdl.CentralProduct(2, 2))
        self.assertIsNone(product.cd_algebra)
        self.assertEqual(product.signed_algorithms, ("auto", "rows", "schedule"))
        self.assertIn(product.default_signed_algorithm, ("rows", "schedule"))
        # A factor of level 1 is a field extension over R and over F_p with
        # p = 3 (mod 4), and only there
        with_native = ("auto", "native", "rows", "schedule")
        loop = cdl.CentralProduct(1, 2)
        for selectors, native in (
            ({}, True),
            ({"char": 7}, True),
            ({"char": 5}, False),
            ({"char": 0, "deg": 2}, False),
            ({"char": 7, "deg": 2}, False),
        ):
            algebra = cdl.LoopAlgebra(loop, **selectors)
            with self.subTest(algebra=algebra):
                names = with_native if native else ("auto", "rows", "schedule")
                self.assertEqual(algebra.signed_algorithms, names)
                if native:
                    self.assertEqual(algebra.default_signed_algorithm, "native")


class ConstructionTests(LoopAlgebraTestCase):
    def test_call_accepts_scalars_elements_and_coefficients(self) -> None:
        """Create elements from each documented kind of value."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        zero = np.zeros((2, 4))
        self.assertCoefficientsEqual(algebra(), zero)
        three = zero.copy()
        three[0, 0] = 3.0
        self.assertCoefficientsEqual(algebra(3), three)
        self.assertCoefficientsEqual(algebra(np.float64(3)), three)
        self.assertCoefficientsEqual(algebra(np.array(3)), three)
        values = np.arange(8.0).reshape(2, 4)
        element = algebra(values)
        self.assertCoefficientsEqual(element, values)
        self.assertIs(algebra(element), element)
        self.assertCoefficientsEqual(algebra(values.tolist()), values)
        self.assertCoefficientsEqual(algebra(values.astype(np.int32)), values)

    def test_call_copies_its_input(self) -> None:
        """Keep an element unchanged when its source array is modified."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        values = np.arange(8.0).reshape(2, 4)
        element = algebra(values)
        values[0, 0] = 99.0
        self.assertEqual(element.coefficients[0, 0], 0.0)

    def test_call_rejects_unsupported_values(self) -> None:
        """Reject loop elements, unordered collections and wrong shapes."""

        loop = cdl.CDLoop(2)
        algebra = cdl.LoopAlgebra(loop)
        for value in (loop.basis(1), {1.0, 2.0}, frozenset((1.0,)), {"a": 1.0}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                algebra(value)
        for shape in ((2, 3), (8,), (1, 2, 4), (4, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                algebra(np.ones(shape))
        with self.assertRaises(TypeError):
            algebra(np.ones((2, 4), dtype=complex))
        with self.assertRaises(TypeError):
            algebra(1j)
        with self.assertRaises(TypeError):
            algebra("element")
        with self.assertRaises(ValueError):
            algebra(cdl.LoopAlgebra(loop, char=7).one())

    def test_finite_coefficients_are_reduced_and_floats_rejected(self) -> None:
        """Reduce integers modulo the characteristic and reject floats."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=7)
        element = algebra([[8, 0, 0, -1], [7, 14, 6, 0]])
        self.assertCoefficientsEqual(element, [[1, 0, 0, 6], [0, 0, 6, 0]])
        self.assertEqual(element.coefficients.dtype, np.uint32)
        self.assertCoefficientsEqual(algebra(9), [[2, 0, 0, 0], [0, 0, 0, 0]])
        self.assertCoefficientsEqual(
            algebra(np.array([[70, 0, 0, 0], [0, 0, 0, 2**40 + 1]])),
            [[0, 0, 0, 0], [0, 0, 0, (2**40 + 1) % 7]],
        )
        with self.assertRaises(TypeError):
            algebra(np.ones((2, 4)))
        with self.assertRaises(TypeError):
            algebra(1.0)
        extension = cdl.LoopAlgebra(cdl.CDLoop(1), char=3, deg=2)
        scalar = extension(extension.field([1, 2]))
        self.assertCoefficientsEqual(scalar, [[[1, 2], [0, 0]], [[0, 0], [0, 0]]])
        self.assertCoefficientsEqual(extension(4), [[[1, 0], [0, 0]], [[0, 0], [0, 0]]])

    def test_zero_one_scalar_and_basis(self) -> None:
        """Place identities, scalars and basis elements at their coefficients."""

        loop = cdl.CDLoop(2)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(loop, **selectors)
            with self.subTest(algebra=algebra):
                self.assertFalse(algebra.zero())
                self.assertEqual(algebra.zero().weight(), 0)
                self.assertSameElement(algebra.one(), algebra.scalar(1))
                self.assertSameElement(algebra.one(), algebra.basis(loop.one()))
                self.assertSameElement(algebra.scalar(2), algebra(2))
                for element in loop:
                    basis = algebra.basis(element)
                    self.assertEqual(
                        basis.terms(), {element: algebra.one().terms()[loop.one()]}
                    )
        with self.assertRaises(ValueError):
            cdl.LoopAlgebra(loop).basis(cdl.CDLoop(3).basis(1))
        with self.assertRaises(TypeError):
            cdl.LoopAlgebra(loop).basis(1)

    def test_basis_elements_multiply_like_the_loop(self) -> None:
        """Check ``[x][y] = [xy]`` for all pairs of a small loop."""

        for loop in (cdl.CDLoop(2), cdl.CentralProduct(1, 1)):
            algebra = cdl.LoopAlgebra(loop)
            for x in loop:
                for y in loop:
                    self.assertSameElement(
                        algebra.basis(x) * algebra.basis(y), algebra.basis(x * y)
                    )

    def test_from_terms_accepts_mappings_and_pairs(self) -> None:
        """Build combinations and add coefficients of repeated elements."""

        loop = cdl.CDLoop(2)
        algebra = cdl.LoopAlgebra(loop)
        i, j = loop.basis(1), loop.basis(2)
        expected = np.zeros((2, 4))
        expected[0, 1] = 2.0
        expected[1, 2] = -1.5
        self.assertCoefficientsEqual(algebra.from_terms({i: 2.0, -j: -1.5}), expected)
        self.assertCoefficientsEqual(
            algebra.from_terms([(i, 0.5), (-j, -1.5), (i, 1.5)]), expected
        )
        self.assertSameElement(algebra.from_terms({}), algebra.zero())
        element = algebra(np.arange(8.0).reshape(2, 4))
        self.assertSameElement(algebra.from_terms(element.terms()), element)
        with self.assertRaises(ValueError):
            algebra.from_terms({cdl.CDLoop(3).basis(1): 1.0})
        with self.assertRaises(TypeError):
            algebra.from_terms({i: "one"})
        finite = cdl.LoopAlgebra(loop, char=5)
        self.assertCoefficientsEqual(
            finite.from_terms([(i, 4), (i, 3), (-i, finite.field(2))]),
            [[0, 2, 0, 0], [0, 2, 0, 0]],
        )

    def test_from_components_inverts_the_decomposition(self) -> None:
        """Rebuild elements from group and signed components of several kinds."""

        rng = np.random.default_rng(101)
        for loop in (cdl.CDLoop(0), cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                element = algebra(exact_values(algebra, rng))
                with self.subTest(algebra=algebra):
                    rebuilt = algebra.from_components(
                        element.kappa(), element.epsilon()
                    )
                    self.assertIsInstance(rebuilt, cdl.Element)
                    self.assertSameElement(rebuilt, element)
                    again = algebra.from_components(
                        plain(element.kappa()).tolist(),
                        plain(element.epsilon()).reshape(
                            (loop.quotient_order,) + algebra.coefficient_shape[2:]
                        ),
                    )
                    self.assertSameElement(again, element)

    def test_from_components_follows_the_inverse_formula(self) -> None:
        """Place ``(c + d) / 2`` and ``(c - d) / 2`` in the two rows."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        group = np.array([4.0, 2.0, 0.0, -6.0])
        signed = np.array([2.0, 0.0, 4.0, 2.0])
        element = algebra.from_components(group, algebra.cd_algebra(signed))
        self.assertCoefficientsEqual(
            element, [(group + signed) / 2, (group - signed) / 2]
        )
        self.assertCoefficientsEqual(element.kappa(), group)
        self.assertCoefficientsEqual(element.epsilon().coefficients, signed)

    def test_from_components_broadcasts_batches(self) -> None:
        """Return an array when a component has leading batch axes."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        rng = np.random.default_rng(102)
        group = rng.integers(-4, 5, size=(3, 4)).astype(float) * 2
        signed = rng.integers(-4, 5, size=(3, 4)).astype(float) * 2
        batch = algebra.from_components(group, signed)
        self.assertIsInstance(batch, cdl.ElementArray)
        self.assertEqual(batch.batch_shape, (3,))
        for index in range(3):
            single = algebra.from_components(group[index], signed[index])
            self.assertSameElement(batch[index], single)
        native = algebra.cd_algebra.asarray(signed)
        self.assertCoefficientsEqual(
            algebra.from_components(group, native).coefficients, batch.coefficients
        )
        broadcast = algebra.from_components(group[0], signed)
        self.assertEqual(broadcast.batch_shape, (3,))
        self.assertSameElement(
            broadcast[2], algebra.from_components(group[0], signed[2])
        )

    def test_from_components_accepts_factor_axes_for_central_products(self) -> None:
        """Read a signed component with one axis for each factor."""

        loop = cdl.CentralProduct(1, 2)
        algebra = cdl.LoopAlgebra(loop)
        signed = np.arange(8.0).reshape(2, 4) * 2
        group = np.zeros(8)
        by_factor = algebra.from_components(group, signed)
        flat = algebra.from_components(group, signed.reshape(8))
        self.assertSameElement(by_factor, flat)
        self.assertEqual(by_factor.epsilon().shape, (2, 4))
        self.assertCoefficientsEqual(by_factor.epsilon(), signed)

    def test_from_components_rejects_mismatched_components(self) -> None:
        """Reject wrong shapes and signed components of another algebra."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        with self.assertRaises(ValueError):
            algebra.from_components(np.zeros(3), np.zeros(4))
        with self.assertRaises(ValueError):
            algebra.from_components(np.zeros(4), np.zeros(8))
        with self.assertRaises(ValueError):
            algebra.from_components(np.zeros(4), fcd.Algebra(level=3).zero())
        with self.assertRaises(ValueError):
            algebra.from_components(
                np.zeros(4), fcd.Algebra(level=2, char=0, deg=2).zero()
            )

    def test_lifts_embed_each_summand(self) -> None:
        """Lift each component to an ideal and multiply compatibly."""

        rng = np.random.default_rng(103)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(2, 1)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x = algebra(exact_values(algebra, rng))
                y = algebra(exact_values(algebra, rng))
                with self.subTest(algebra=algebra):
                    group = algebra.lift_group(x.kappa())
                    signed = algebra.lift_signed(x.epsilon())
                    self.assertSameElement(group + signed, x)
                    self.assertFalse(np.any(plain(group.epsilon())))
                    self.assertFalse(np.any(plain(signed.kappa())))
                    self.assertFalse(group * signed)
                    self.assertFalse(signed * group)
                    self.assertTrue(
                        algebra.lift_group(x.kappa()) * algebra.lift_group(y.kappa())
                        == algebra.lift_group((x * y).kappa())
                    )
                    self.assertTrue(
                        algebra.lift_signed(x.epsilon())
                        * algebra.lift_signed(y.epsilon())
                        == algebra.lift_signed((x * y).epsilon())
                    )

    def test_lift_group_multiplies_by_dyadic_convolution(self) -> None:
        """Agree with :func:`pycdloop.dyadic_convolve` on group components."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(3))
        rng = np.random.default_rng(104)
        a = rng.integers(-3, 4, size=8).astype(float)
        b = rng.integers(-3, 4, size=8).astype(float)
        product = algebra.lift_group(a) * algebra.lift_group(b)
        self.assertCoefficientsEqual(product.kappa(), cdl.dyadic_convolve(a, b))

    def test_split_array_copies_and_broadcasts_its_parts(self) -> None:
        """Create split arrays from components without sharing the inputs."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        spectral = np.arange(12.0).reshape(3, 4)
        signed = np.arange(4.0)
        split = algebra.split_array(spectral, signed)
        self.assertIsInstance(split, cdl.SplitArray)
        self.assertEqual(split.batch_shape, (3,))
        self.assertCoefficientsEqual(split.spectral, spectral)
        self.assertCoefficientsEqual(
            split.signed.coefficients, np.broadcast_to(signed, (3, 4))
        )
        spectral[0, 0] = 99.0
        self.assertEqual(split.spectral[0, 0], 0.0)
        with self.assertRaises(ValueError):
            algebra.split_array(np.zeros((3, 5)), np.zeros(4))

    def test_random_is_reproducible_and_typed(self) -> None:
        """Honor seeds, generators and batch shapes over every field."""

        loop = cdl.CDLoop(3)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(loop, **selectors)
            with self.subTest(algebra=algebra):
                first = algebra.random(5)
                self.assertIsInstance(first, cdl.Element)
                self.assertSameElement(first, algebra.random(5))
                self.assertFalse(first == algebra.random(6))
                self.assertSameElement(first, algebra.random(np.random.default_rng(5)))
                self.assertEqual(first.coefficients.dtype, algebra.dtype)
                batch = algebra.random(7, batch_shape=(2, 3))
                self.assertIsInstance(batch, cdl.ElementArray)
                self.assertEqual(batch.batch_shape, (2, 3))
                self.assertEqual(algebra.random(7, batch_shape=4).batch_shape, (4,))
                self.assertEqual(algebra.random(7, batch_shape=()).batch_shape, ())
                if algebra.field is not None:
                    self.assertLess(int(batch.coefficients.max()), algebra.char)
        complex_values = cdl.LoopAlgebra(loop, char=0, deg=2).random(1).coefficients
        self.assertTrue(np.any(complex_values.imag))

    def test_secure_random_requires_a_finite_field(self) -> None:
        """Draw residues from the system source and reject R and C."""

        loop = cdl.CDLoop(2)
        for selectors in ({"char": 3}, {"char": 13, "deg": 3}):
            algebra = cdl.LoopAlgebra(loop, **selectors)
            element = algebra.random(secure=True)
            self.assertEqual(element.shape, algebra.coefficient_shape)
            self.assertLess(int(element.coefficients.max()), algebra.char)
            batch = algebra.random(secure=True, batch_shape=(5,))
            self.assertEqual(batch.batch_shape, (5,))
            self.assertLess(int(batch.coefficients.max()), algebra.char)
        for selectors in ({}, {"char": 0, "deg": 2}):
            with self.assertRaises(ValueError):
                cdl.LoopAlgebra(loop, **selectors).random(secure=True)

    def test_bytes_round_trip(self) -> None:
        """Encode and decode elements over every field."""

        rng = np.random.default_rng(105)
        widths = {(): 8, (0, 2): 16, (3, 1): 1, (13, 1): 1, (3, 2): 1, (13, 3): 1}
        for loop in (cdl.CDLoop(2), cdl.CentralProduct(1, 1)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                element = algebra.random(rng)
                data = element.to_bytes()
                with self.subTest(algebra=algebra):
                    key = (algebra.char, algebra.deg) if selectors else ()
                    count = int(np.prod(algebra.coefficient_shape))
                    self.assertIsInstance(data, bytes)
                    self.assertEqual(len(data), widths[key] * count)
                    self.assertSameElement(algebra.from_bytes(data), element)
                    self.assertSameElement(algebra.from_bytes(bytearray(data)), element)

    def test_bytes_layout_is_canonical(self) -> None:
        """Write little-endian doubles and big-endian residues in ravel order."""

        real = cdl.LoopAlgebra(cdl.CDLoop(1))
        element = real([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(
            element.to_bytes(), np.array([1.0, 2.0, 3.0, 4.0], dtype="<f8").tobytes()
        )
        prime = cdl.LoopAlgebra(cdl.CDLoop(1), char=65521)
        encoded = prime([[1, 258], [65520, 0]]).to_bytes()
        self.assertEqual(encoded, bytes([0, 1, 1, 2, 0xFF, 0xF0, 0, 0]))

    def test_from_bytes_rejects_bad_input(self) -> None:
        """Reject wrong lengths and residues at or above the characteristic."""

        real = cdl.LoopAlgebra(cdl.CDLoop(2))
        with self.assertRaises(ValueError):
            real.from_bytes(b"abc")
        with self.assertRaises(ValueError):
            real.from_bytes(real.one().to_bytes() + b"\0")
        finite = cdl.LoopAlgebra(cdl.CDLoop(2), char=7)
        with self.assertRaises(ValueError):
            finite.from_bytes(bytes(7))
        with self.assertRaises(ValueError):
            finite.from_bytes(bytes([7]) + bytes(7))
        self.assertSameElement(finite.from_bytes(bytes(8)), finite.zero())

    def test_array_copies_and_asarray_shares(self) -> None:
        """Follow the copy rules of ``array`` and ``asarray``."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        values = np.arange(24.0).reshape(3, 2, 4)
        copied = algebra.array(values)
        self.assertFalse(np.shares_memory(copied.coefficients, values))
        shared = algebra.asarray(values)
        self.assertTrue(np.shares_memory(shared.coefficients, values))
        self.assertIs(algebra.asarray(shared), shared)
        self.assertIs(algebra.asarray(shared, copy=False), shared)
        fresh = algebra.asarray(shared, copy=True)
        self.assertFalse(np.shares_memory(fresh.coefficients, values))
        self.assertFalse(np.shares_memory(algebra.array(shared).coefficients, values))
        self.assertFalse(
            np.shares_memory(algebra.asarray(values, copy=True).coefficients, values)
        )
        converted = algebra.asarray(values.astype(np.int64))
        self.assertEqual(converted.coefficients.dtype, np.float64)
        with self.assertRaises(ValueError):
            algebra.asarray(values.astype(np.int64), copy=False)
        with self.assertRaises(ValueError):
            algebra.asarray(values.tolist(), copy=False)
        for flag in (1, 0, "yes"):
            with self.subTest(flag=flag), self.assertRaises(TypeError):
                algebra.asarray(values, copy=flag)
        single = algebra.array(algebra.one())
        self.assertEqual(single.batch_shape, ())
        self.assertCoefficientsEqual(single.coefficients, algebra.one())

    def test_asarray_requires_canonical_residues_to_share(self) -> None:
        """Share finite-field storage only when every residue is reduced."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=7)
        reduced = np.full((2, 2, 4), 6, dtype=np.uint32)
        self.assertTrue(
            np.shares_memory(algebra.asarray(reduced).coefficients, reduced)
        )
        unreduced = np.full((2, 2, 4), 9, dtype=np.uint32)
        converted = algebra.asarray(unreduced)
        self.assertFalse(np.shares_memory(converted.coefficients, unreduced))
        self.assertCoefficientsEqual(converted.coefficients, unreduced % 7)
        with self.assertRaises(ValueError):
            algebra.asarray(unreduced, copy=False)

    def test_array_constructors_reject_bad_operands(self) -> None:
        """Reject wrong shapes, split arrays and other parents."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        other = cdl.LoopAlgebra(cdl.CDLoop(2), char=7)
        for build in (algebra.array, algebra.asarray):
            with self.subTest(build=build.__name__):
                with self.assertRaises(ValueError):
                    build(np.ones((3, 2, 5)))
                with self.assertRaises(ValueError):
                    build(np.ones(4))
                with self.assertRaises(TypeError):
                    build(algebra.random(0, batch_shape=(2,)).split())
                with self.assertRaises(ValueError):
                    build(other.random(0, batch_shape=(2,)))


class MultiplyTests(LoopAlgebraTestCase):
    def test_every_algorithm_matches_the_reference(self) -> None:
        """Compare all algorithm choices with the independent oracle."""

        rng = np.random.default_rng(201)
        for level in range(6):
            loop = cdl.CDLoop(level)
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x_values = exact_values(algebra, rng)
                y_values = exact_values(algebra, rng)
                expected = oracle_product(algebra, x_values, y_values)
                x, y = algebra(x_values), algebra(y_values)
                with self.subTest(algebra=algebra):
                    self.assertCoefficientsEqual(x * y, expected)
                    for name in ("auto", "split", "table"):
                        product = algebra.multiply(x, y, algorithm=name)
                        self.assertIsInstance(product, cdl.Element)
                        self.assertCoefficientsEqual(product, expected)
                    for name in algebra.signed_algorithms:
                        product = algebra.multiply(x, y, signed_algorithm=name)
                        self.assertCoefficientsEqual(product, expected)

    def test_array_products_match_the_reference(self) -> None:
        """Compare batched products of both array kinds with the oracle."""

        rng = np.random.default_rng(202)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x_values = exact_values(algebra, rng, (3,))
            y_values = exact_values(algebra, rng, (3,))
            x, y = algebra.asarray(x_values), algebra.asarray(y_values)
            with self.subTest(algebra=algebra):
                for name in ("split", "table"):
                    product = algebra.multiply(x, y, algorithm=name)
                    self.assertIsInstance(product, cdl.ElementArray)
                    for index in range(3):
                        expected = oracle_product(
                            algebra, x_values[index], y_values[index]
                        )
                        self.assertCoefficientsEqual(product[index], expected)
                split = algebra.multiply(x.split(), y.split())
                self.assertIsInstance(split, cdl.SplitArray)
                self.assertTrue((split.to_loop() == x * y).all())

    def test_scalar_operands_scale(self) -> None:
        """Scale by a field scalar on either side and reject two scalars."""

        rng = np.random.default_rng(203)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            x = algebra(exact_values(algebra, rng))
            with self.subTest(algebra=algebra):
                self.assertSameElement(algebra.multiply(2, x), x + x)
                self.assertSameElement(algebra.multiply(x, 2), x + x)
                with self.assertRaises(TypeError):
                    algebra.multiply(2, 3)

    def test_out_receives_loop_basis_results(self) -> None:
        """Write products into ndarray and ElementArray storage."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(3))
        rng = np.random.default_rng(204)
        x = algebra.asarray(exact_values(algebra, rng, (4,)))
        y = algebra.asarray(exact_values(algebra, rng, (4,)))
        expected = (x * y).coefficients
        for name in ("split", "table"):
            with self.subTest(algorithm=name):
                storage = np.empty((4, 2, 8))
                result = algebra.multiply(x, y, out=storage, algorithm=name)
                self.assertIsInstance(result, cdl.ElementArray)
                self.assertTrue(np.shares_memory(result.coefficients, storage))
                self.assertCoefficientsEqual(storage, expected)
                target = algebra.array(np.zeros((4, 2, 8)))
                self.assertIs(algebra.multiply(x, y, out=target), target)
                self.assertCoefficientsEqual(target.coefficients, expected)
        single = np.empty((2, 8))
        result = algebra.multiply(x[0], y[0], out=single)
        self.assertIsInstance(result, cdl.ElementArray)
        self.assertEqual(result.batch_shape, ())
        self.assertCoefficientsEqual(single, expected[0])
        # Operands may share storage with the output
        aliased = algebra.array(x.coefficients)
        algebra.multiply(aliased, y, out=aliased)
        self.assertCoefficientsEqual(aliased.coefficients, expected)

    def test_out_is_validated(self) -> None:
        """Reject mismatched, read-only and split-form output storage."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        x = algebra.random(0, batch_shape=(3,))
        with self.assertRaises(ValueError):
            algebra.multiply(x, x, out=np.empty((2, 2, 4)))
        with self.assertRaises(ValueError):
            algebra.multiply(x, x, out=np.empty((3, 8)))
        with self.assertRaises(TypeError):
            algebra.multiply(x, x, out=np.empty((3, 2, 4), dtype=np.float32))
        with self.assertRaises(TypeError):
            algebra.multiply(x, x, out=[[0.0] * 4] * 2)
        frozen = np.empty((3, 2, 4))
        frozen.setflags(write=False)
        with self.assertRaises(ValueError):
            algebra.multiply(x, x, out=frozen)
        with self.assertRaises(TypeError):
            algebra.multiply(x, x, out=x.split())
        with self.assertRaises(TypeError):
            algebra.multiply(x.split(), x.split(), out=np.empty((3, 2, 4)))
        other = cdl.LoopAlgebra(cdl.CDLoop(2), char=0, deg=2)
        with self.assertRaises(ValueError):
            algebra.multiply(x, x, out=other.array(np.zeros((3, 2, 4))))

    def test_invalid_names_and_operands_are_rejected(self) -> None:
        """Raise the documented exception for each misuse."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        other = cdl.LoopAlgebra(cdl.CDLoop(2), char=7)
        x = algebra.random(0)
        batch = algebra.random(0, batch_shape=(2,))
        for name in ("Split", "direct", "", "quasilinear"):
            with self.subTest(algorithm=name), self.assertRaises(ValueError):
                algebra.multiply(x, x, algorithm=name)
        for name in ("rows", "table", "Direct"):
            with self.subTest(signed=name), self.assertRaises(ValueError):
                algebra.multiply(x, x, signed_algorithm=name)
        for name in (None, 3, b"split"):
            with self.subTest(name=name):
                with self.assertRaises(TypeError):
                    algebra.multiply(x, x, algorithm=name)
                with self.assertRaises(TypeError):
                    algebra.multiply(x, x, signed_algorithm=name)
        with self.assertRaises(ValueError):
            algebra.multiply(x, other.one())
        with self.assertRaises(ValueError):
            algebra.multiply(other.one(), batch)
        with self.assertRaises(TypeError):
            algebra.multiply(x, np.ones((2, 4)))
        with self.assertRaises(TypeError):
            algebra.multiply(batch, batch.split())
        with self.assertRaises(TypeError):
            algebra.multiply(batch.split(), batch)
        with self.assertRaises(TypeError):
            algebra.multiply(batch.split(), batch.split(), algorithm="table")
        with self.assertRaises(ValueError):
            cdl.LoopAlgebra(cdl.CentralProduct(2, 2)).multiply(
                cdl.LoopAlgebra(cdl.CentralProduct(2, 2)).one(),
                cdl.LoopAlgebra(cdl.CentralProduct(2, 2)).one(),
                signed_algorithm="native",
            )

    def test_operator_commutators_follow_their_definitions(self) -> None:
        """Compare both operator commutators with explicit products."""

        rng = np.random.default_rng(205)
        for loop in (cdl.CDLoop(3), cdl.CDLoop(4), cdl.CentralProduct(2, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                a, b, x = (algebra(exact_values(algebra, rng)) for _ in range(3))
                with self.subTest(algebra=algebra):
                    left = algebra.left_operator_commutator(a, b, x)
                    self.assertTrue(left == a * (b * x) - b * (a * x))
                    right = algebra.right_operator_commutator(a, b, x)
                    self.assertTrue(right == (x * b) * a - (x * a) * b)
                    self.assertIsInstance(left, cdl.Element)

    def test_operator_commutators_broadcast_over_arrays(self) -> None:
        """Accept arrays of one kind together with elements."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(3))
        rng = np.random.default_rng(206)
        a = algebra(exact_values(algebra, rng))
        b = algebra.asarray(exact_values(algebra, rng, (3,)))
        x = algebra.asarray(exact_values(algebra, rng, (3,)))
        left = algebra.left_operator_commutator(a, b, x)
        self.assertIsInstance(left, cdl.ElementArray)
        self.assertTrue((left == a * (b * x) - b * (a * x)).all())
        split = algebra.right_operator_commutator(a, b.split(), x.split())
        self.assertIsInstance(split, cdl.SplitArray)
        self.assertTrue((split.to_loop() == (x * b) * a - (x * a) * b).all())
        with self.assertRaises(TypeError):
            algebra.left_operator_commutator(a, b, x.split())
        with self.assertRaises(TypeError):
            algebra.left_operator_commutator(a, b, 3)


class IdempotentTests(LoopAlgebraTestCase):
    def test_central_idempotents_project_onto_the_summands(self) -> None:
        """Check orthogonality, completeness and the two projections."""

        rng = np.random.default_rng(301)
        for loop in (cdl.CDLoop(0), cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                plus, minus = algebra.central_idempotents()
                x = algebra(exact_values(algebra, rng))
                with self.subTest(algebra=algebra):
                    self.assertTrue(plus.is_idempotent())
                    self.assertTrue(minus.is_idempotent())
                    self.assertTrue(plus.is_central() and minus.is_central())
                    self.assertFalse(plus * minus)
                    self.assertFalse(minus * plus)
                    self.assertSameElement(plus + minus, algebra.one())
                    self.assertSameElement(
                        plus - minus, algebra.basis(loop.minus_one())
                    )
                    self.assertTrue(plus * x == algebra.lift_group(x.kappa()))
                    self.assertTrue(x * minus == algebra.lift_signed(x.epsilon()))

    def test_primitive_idempotents_are_orthogonal_and_complete(self) -> None:
        """Check the idempotents of the characters of the quotient group."""

        for loop in (
            cdl.CDLoop(0),
            cdl.CDLoop(1),
            cdl.CDLoop(3),
            cdl.CentralProduct(1, 1),
        ):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                size = loop.quotient_order
                idempotents = algebra.primitive_idempotents()
                with self.subTest(algebra=algebra):
                    self.assertIsInstance(idempotents, cdl.ElementArray)
                    self.assertEqual(idempotents.batch_shape, (size,))
                    self.assertTrue(idempotents.is_idempotent().all())
                    self.assertTrue(idempotents.is_central().all())
                    products = idempotents[:, None] * idempotents[None, :]
                    zero = algebra.zero()
                    for a in range(size):
                        single = algebra.primitive_idempotent(a)
                        self.assertSameElement(idempotents[a], single)
                        delta = np.zeros(
                            (size,) + algebra.coefficient_shape[2:], dtype=algebra.dtype
                        )
                        delta[a] = plain(algebra.one().spectrum())[0]
                        self.assertCoefficientsEqual(plain(single.spectrum()), delta)
                        self.assertFalse(np.any(plain(single.epsilon())))
                        for b in range(size):
                            expected = single if a == b else zero
                            self.assertTrue(products[a, b] == expected)
                    total = idempotents.coefficients.sum(axis=0, dtype=algebra.dtype)
                    if algebra.field is not None:
                        total = idempotents.coefficients.astype(np.int64).sum(axis=0)
                    self.assertSameElement(
                        algebra(total), algebra.central_idempotents()[0]
                    )

    def test_primitive_idempotent_coefficients_are_character_values(self) -> None:
        """Compare with the defining sum over the loop."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        for a in range(4):
            signs = [(-1.0) ** bin(a & t).count("1") / 8 for t in range(4)]
            self.assertCoefficientsEqual(
                algebra.primitive_idempotent(a), [signs, signs]
            )
        with self.assertRaises(IndexError):
            algebra.primitive_idempotent(4)
        with self.assertRaises(TypeError):
            algebra.primitive_idempotent(1.0)

    def test_center_basis_is_central_and_independent(self) -> None:
        """Check the size, independence and centrality of the basis."""

        cases = (
            (cdl.CDLoop(0), 2),
            (cdl.CDLoop(1), 4),
            (cdl.CDLoop(2), 5),
            (cdl.CDLoop(3), 9),
            (cdl.CentralProduct(1, 2), 10),
            (cdl.CentralProduct(1, 1), 8),
        )
        for loop, dimension in cases:
            for selectors in ({}, {"char": 13}, {"char": 3, "deg": 2}):
                algebra = cdl.LoopAlgebra(loop, **selectors)
                center = algebra.center_basis()
                basis = basis_batch(algebra)
                with self.subTest(algebra=algebra):
                    self.assertEqual(center.batch_shape, (dimension,))
                    self.assertTrue(center.is_central().all())
                    flat = center.coefficients.reshape(
                        (dimension, algebra.dimension) + algebra.coefficient_shape[2:]
                    )
                    self.assertEqual(
                        cdl.linalg.rank(flat, field=algebra.field), dimension
                    )
                    c = center[:, None, None]
                    g = basis[None, :, None]
                    h = basis[None, None, :]
                    self.assertTrue((c * g == g * c).all())
                    self.assertTrue(((c * g) * h == c * (g * h)).all())
                    self.assertTrue(((g * c) * h == g * (c * h)).all())
                    self.assertTrue(((g * h) * c == g * (h * c)).all())

    def test_noncentral_elements_are_detected(self) -> None:
        """Report elements outside the center of a noncommutative algebra."""

        loop = cdl.CDLoop(2)
        algebra = cdl.LoopAlgebra(loop)
        self.assertFalse(algebra.basis(loop.basis(1)).is_central())
        self.assertFalse(algebra.lift_signed([0.0, 1.0, 0.0, 0.0]).is_central())
        self.assertTrue(algebra.lift_signed([2.0, 0.0, 0.0, 0.0]).is_central())
        self.assertTrue(algebra.lift_group([1.0, 2.0, 3.0, 4.0]).is_central())
        commutative = cdl.LoopAlgebra(cdl.CDLoop(1))
        self.assertTrue(commutative.random(0).is_central())


class MapTests(LoopAlgebraTestCase):
    def test_character_automorphisms_preserve_products(self) -> None:
        """Check every character automorphism on elements and arrays."""

        rng = np.random.default_rng(401)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x = algebra(exact_values(algebra, rng))
                y = algebra(exact_values(algebra, rng))
                batch = algebra.asarray(exact_values(algebra, rng, (2,)))
                with self.subTest(algebra=algebra):
                    for a in range(loop.quotient_order):
                        fx = algebra.character_automorphism(a, x)
                        fy = algebra.character_automorphism(a, y)
                        self.assertTrue(
                            fx * fy == algebra.character_automorphism(a, x * y)
                        )
                        self.assertSameElement(algebra.character_automorphism(a, fx), x)
                        moved = algebra.character_automorphism(a, batch)
                        self.assertIsInstance(moved, cdl.ElementArray)
                        self.assertSameElement(
                            moved[1], algebra.character_automorphism(a, batch[1])
                        )
                    self.assertSameElement(algebra.character_automorphism(0, x), x)
                    self.assertSameElement(
                        algebra.character_automorphism(1, algebra.one()),
                        algebra.one(),
                    )

    def test_character_automorphism_swaps_signed_coefficients(self) -> None:
        """Exchange the two coefficients of the indices of odd parity."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        values = np.arange(8.0).reshape(2, 4)
        moved = algebra.character_automorphism(1, algebra(values))
        self.assertCoefficientsEqual(moved, [[0, 5, 2, 7], [4, 1, 6, 3]])
        with self.assertRaises(IndexError):
            algebra.character_automorphism(4, algebra.one())

    def test_apply_automorphism_extends_loop_automorphisms(self) -> None:
        """Extend a loop automorphism given as a callable or as positions."""

        loop = cdl.CDLoop(2)
        cycle = {1: 2, 2: 3, 3: 1, 0: 0}

        def rotate(element: Any) -> Any:
            """Permute the three imaginary units cyclically."""

            return loop(cycle[element.index], element.sign)

        self.assertTrue(loop.is_automorphism(rotate))
        positions = np.array([rotate(element).position for element in loop])
        rng = np.random.default_rng(402)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(loop, **selectors)
            x = algebra(exact_values(algebra, rng))
            y = algebra(exact_values(algebra, rng))
            with self.subTest(algebra=algebra):
                fx = algebra.apply_automorphism(rotate, x)
                self.assertSameElement(fx, algebra.apply_automorphism(positions, x))
                fy = algebra.apply_automorphism(positions, y)
                self.assertTrue(fx * fy == algebra.apply_automorphism(rotate, x * y))
                for element in loop:
                    self.assertSameElement(
                        algebra.apply_automorphism(rotate, algebra.basis(element)),
                        algebra.basis(rotate(element)),
                    )

    def test_apply_automorphism_rejects_bad_maps_and_operands(self) -> None:
        """Reject maps that are not bijections and split-form operands."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        x = algebra.random(0)
        with self.assertRaises(ValueError):
            algebra.apply_automorphism(np.zeros(8, dtype=np.int64), x)
        with self.assertRaises(ValueError):
            algebra.apply_automorphism(np.arange(7), x)
        with self.assertRaises(ValueError):
            algebra.apply_automorphism(np.arange(8) + 1, x)
        batch = algebra.random(0, batch_shape=(2,))
        with self.assertRaises(TypeError):
            algebra.apply_automorphism(np.arange(8), batch.split())
        with self.assertRaises(TypeError):
            algebra.apply_automorphism(np.arange(8), np.ones((2, 4)))
        self.assertSameElement(algebra.apply_automorphism(np.arange(8), x), x)

    def test_embed_is_a_unital_homomorphism(self) -> None:
        """Embed the loop algebra of a smaller Cayley--Dickson loop."""

        rng = np.random.default_rng(403)
        for small, large in ((0, 2), (1, 3), (2, 4), (3, 3)):
            for selectors in SELECTORS:
                source = cdl.LoopAlgebra(cdl.CDLoop(small), **selectors)
                target = cdl.LoopAlgebra(cdl.CDLoop(large), **selectors)
                x = source(exact_values(source, rng))
                y = source(exact_values(source, rng))
                with self.subTest(source=source, target=target):
                    ex, ey = target.embed(x), target.embed(y)
                    self.assertIs(ex.algebra, target)
                    self.assertSameElement(target.embed(source.one()), target.one())
                    self.assertTrue(ex * ey == target.embed(x * y))
                    self.assertSameElement(ex + ey, target.embed(x + y))
                    count = 1 << small
                    self.assertCoefficientsEqual(
                        ex.coefficients[:, :count], x.coefficients
                    )
                    self.assertFalse(np.any(ex.coefficients[:, count:]))
                    batch = source.asarray(exact_values(source, rng, (2,)))
                    embedded = target.embed(batch)
                    self.assertIsInstance(embedded, cdl.ElementArray)
                    self.assertSameElement(embedded[1], target.embed(batch[1]))

    def test_embed_rejects_wrong_sources(self) -> None:
        """Reject larger loops, other fields, central products and raw values."""

        target = cdl.LoopAlgebra(cdl.CDLoop(2))
        with self.assertRaises(ValueError):
            target.embed(cdl.LoopAlgebra(cdl.CDLoop(3)).one())
        with self.assertRaises(ValueError):
            target.embed(cdl.LoopAlgebra(cdl.CDLoop(1), char=7).one())
        with self.assertRaises(ValueError):
            target.embed(cdl.LoopAlgebra(cdl.CentralProduct(1, 1)).one())
        with self.assertRaises(TypeError):
            target.embed(np.ones((2, 2)))
        with self.assertRaises(TypeError):
            target.embed(cdl.CDLoop(1).basis(1))
        product = cdl.LoopAlgebra(cdl.CentralProduct(1, 2))
        with self.assertRaises(TypeError):
            product.embed(cdl.LoopAlgebra(cdl.CDLoop(1)).one())
        with self.assertRaises(TypeError):
            target.factor_embedding(0, cdl.LoopAlgebra(cdl.CDLoop(1)).one())


class MatrixTests(LoopAlgebraTestCase):
    def _apply(self, algebra: Any, matrix: np.ndarray, vector: np.ndarray) -> Any:
        """Apply a matrix over R, C or a prime field to a coefficient vector."""

        if algebra.field is None:
            return matrix @ vector
        product = matrix.astype(object) @ vector.astype(object)
        return np.array([int(value) % algebra.char for value in product])

    def test_loop_basis_matrices_reproduce_products(self) -> None:
        """Multiply flattened coefficients by the left and right matrices."""

        rng = np.random.default_rng(501)
        for loop in (cdl.CDLoop(1), cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            for selectors in ({}, {"char": 0, "deg": 2}, {"char": 3}, {"char": 13}):
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x = algebra(exact_values(algebra, rng))
                y = algebra(exact_values(algebra, rng))
                size = algebra.dimension
                with self.subTest(algebra=algebra):
                    left = algebra.left_matrix(x)
                    right = algebra.right_matrix(x)
                    self.assertEqual(left.shape, (size, size))
                    self.assertEqual(left.dtype, algebra.dtype)
                    self.assertCoefficientsEqual(
                        self._apply(algebra, left, y.coefficients.ravel()),
                        (x * y).coefficients.ravel(),
                    )
                    self.assertCoefficientsEqual(
                        self._apply(algebra, right, y.coefficients.ravel()),
                        (y * x).coefficients.ravel(),
                    )
                    self.assertCoefficientsEqual(
                        algebra.left_matrix(x, basis="loop"), left
                    )

    def test_matrix_columns_are_images_of_basis_elements(self) -> None:
        """Check the columns over an extension field, entry by entry."""

        loop = cdl.CDLoop(2)
        algebra = cdl.LoopAlgebra(loop, char=13, deg=3)
        x = algebra.random(0)
        left = algebra.left_matrix(x)
        right = algebra.right_matrix(x)
        self.assertEqual(left.shape, (8, 8, 3))
        for column, element in enumerate(loop):
            basis = algebra.basis(element)
            self.assertCoefficientsEqual(
                left[:, column], (x * basis).coefficients.reshape(8, 3)
            )
            self.assertCoefficientsEqual(
                right[:, column], (basis * x).coefficients.reshape(8, 3)
            )

    def test_split_basis_matrices_are_block_diagonal(self) -> None:
        """Act on spectral and signed coordinates through separate blocks."""

        rng = np.random.default_rng(502)
        for loop in (cdl.CDLoop(3), cdl.CentralProduct(2, 1)):
            for selectors in ({}, {"char": 13}):
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x = algebra(exact_values(algebra, rng))
                y = algebra(exact_values(algebra, rng))
                size = loop.quotient_order
                with self.subTest(algebra=algebra):
                    for side, product in (("left", x * y), ("right", y * x)):
                        build = getattr(algebra, f"{side}_matrix")
                        matrix = build(x, basis="split")
                        self.assertEqual(matrix.shape, (2 * size, 2 * size))
                        self.assertFalse(np.any(matrix[:size, size:]))
                        self.assertFalse(np.any(matrix[size:, :size]))
                        self.assertCoefficientsEqual(
                            np.diag(matrix[:size, :size]), plain(x.spectrum())
                        )
                        diagonal = matrix[:size, :size].copy()
                        np.fill_diagonal(diagonal, 0)
                        self.assertFalse(np.any(diagonal))
                        vector = np.concatenate(
                            [plain(y.spectrum()), plain(y.epsilon()).ravel()]
                        )
                        expected = np.concatenate(
                            [
                                plain(product.spectrum()),
                                plain(product.epsilon()).ravel(),
                            ]
                        )
                        self.assertCoefficientsEqual(
                            self._apply(algebra, matrix, vector), expected
                        )

    def test_matrix_arguments_are_validated(self) -> None:
        """Reject arrays, other parents and unknown basis names."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        x = algebra.random(0)
        with self.assertRaises(ValueError):
            algebra.left_matrix(x, basis="spectral")
        with self.assertRaises(TypeError):
            algebra.left_matrix(x, basis=0)
        with self.assertRaises(TypeError):
            algebra.right_matrix(algebra.random(0, batch_shape=(2,)))
        with self.assertRaises(ValueError):
            algebra.left_matrix(cdl.LoopAlgebra(cdl.CDLoop(3)).one())
        identity = algebra.left_matrix(algebra.one())
        self.assertCoefficientsEqual(identity, np.eye(8))

    def test_structure_constants_encode_the_cayley_table(self) -> None:
        """Place a single one in each row of the constants."""

        for loop in (cdl.CDLoop(0), cdl.CDLoop(3), cdl.CentralProduct(1, 2)):
            algebra = cdl.LoopAlgebra(loop, char=7)
            constants = algebra.structure_constants()
            table = loop.cayley_table()
            size = algebra.dimension
            with self.subTest(loop=loop):
                self.assertEqual(constants.shape, (size, size, size))
                self.assertEqual(constants.dtype, np.int8)
                self.assertCoefficientsEqual(
                    constants.sum(axis=2), np.ones((size, size))
                )
                self.assertCoefficientsEqual(constants.argmax(axis=2), table)
        with self.assertRaises(ValueError):
            cdl.LoopAlgebra(cdl.CDLoop(8)).structure_constants()

    def test_structure_constants_reproduce_products(self) -> None:
        """Contract the constants with two coefficient vectors."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(3))
        rng = np.random.default_rng(503)
        x = algebra(exact_values(algebra, rng))
        y = algebra(exact_values(algebra, rng))
        constants = algebra.structure_constants().astype(float)
        product = np.einsum(
            "i,j,ijk->k", x.coefficients.ravel(), y.coefficients.ravel(), constants
        )
        self.assertCoefficientsEqual(product, (x * y).coefficients.ravel())

    def test_generated_subalgebra_is_described(self) -> None:
        """Report the minimal polynomial and its two sources."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=13)
        x = algebra.random(3)
        description = algebra.subalgebra_generated_by(x)
        self.assertIsInstance(description, cdl.GeneratedSubalgebra)
        polynomial = x.minimal_polynomial()
        self.assertCoefficientsEqual(description.minimal_polynomial, polynomial)
        self.assertEqual(description.dimension, polynomial.shape[0] - 1)
        distinct = np.unique(plain(x.spectrum()))
        self.assertCoefficientsEqual(description.spectral_values, distinct)
        self.assertIn(description.signed_polynomial.shape[0], (2, 3))
        self.assertEqual(int(description.signed_polynomial[-1]), 1)

        scalar = algebra.subalgebra_generated_by(algebra.scalar(5))
        self.assertEqual(scalar.dimension, 1)
        self.assertCoefficientsEqual(scalar.minimal_polynomial, [8, 1])
        self.assertCoefficientsEqual(scalar.spectral_values, [5])
        self.assertCoefficientsEqual(scalar.signed_polynomial, [8, 1])
        with self.assertRaises(TypeError):
            algebra.subalgebra_generated_by(algebra.random(0, batch_shape=(2,)))
        product = cdl.LoopAlgebra(cdl.CentralProduct(1, 1))
        with self.assertRaises(TypeError):
            product.subalgebra_generated_by(product.one())


if __name__ == "__main__":
    unittest.main()
