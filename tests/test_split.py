"""Deterministic tests for :class:`pycdloop.SplitArray`."""

from __future__ import annotations

import copy
import pickle
import unittest
from typing import Any

import numpy as np
import pyfastcd as fcd

import pycdloop as cdl
from pycdloop._validation import MIXED_BASIS_MESSAGE

# Loops and coefficient fields exercised by most tests
LOOPS = (cdl.CDLoop(3), cdl.CentralProduct(1, 2))
SELECTORS: tuple[dict[str, int], ...] = (
    {},
    {"char": 0, "deg": 2},
    {"char": 13},
    {"char": 3, "deg": 2},
)


def algebras() -> list[Any]:
    """Return every loop algebra exercised by the tests."""

    return [cdl.LoopAlgebra(loop, **s) for loop in LOOPS for s in SELECTORS]


def exact_values(algebra: Any, rng: Any, lead: tuple[int, ...] = ()) -> np.ndarray:
    """Draw coefficients on which the arithmetic of the tests is exact."""

    shape = lead + algebra.coefficient_shape
    if algebra.field is not None:
        return rng.integers(0, algebra.char, size=shape, dtype=np.uint32)
    values = rng.integers(-3, 4, size=shape).astype(algebra.dtype)
    if algebra.deg == 2:
        values = values + 1j * rng.integers(-3, 4, size=shape)
    return values


def exact_array(algebra: Any, rng: Any, lead: tuple[int, ...]) -> Any:
    """Draw a loop-basis array with exactly representable coefficients."""

    return algebra.asarray(exact_values(algebra, rng, lead))


def plain(value: Any) -> np.ndarray:
    """Return a component given as an ndarray or a pyfastcd object as an ndarray."""

    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value.coefficients)


class SplitArrayTestCase(unittest.TestCase):
    """Common assertions with useful failure diagnostics."""

    def assertCoefficientsEqual(self, actual: object, expected: object) -> None:
        """Assert exact equality after exposing both coefficient arrays."""

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def assertSplitEqualsLoop(self, split: Any, loop: Any) -> None:
        """Assert that a split array converts exactly to a loop-basis array."""

        self.assertIsInstance(split, cdl.SplitArray)
        self.assertIsInstance(loop, cdl.ElementArray)
        converted = split.to_loop()
        self.assertEqual(split.batch_shape, loop.batch_shape)
        self.assertCoefficientsEqual(converted.coefficients, loop.coefficients)

    def assertMixingRejected(self, operation: Any) -> None:
        """Assert that an operation raises the shared mixing error."""

        with self.assertRaises(TypeError) as raised:
            operation()
        self.assertEqual(str(raised.exception), MIXED_BASIS_MESSAGE)


class ConstructionTests(SplitArrayTestCase):
    def test_direct_construction_is_rejected(self) -> None:
        """Require ``split()`` or a parent algebra to create split arrays."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        for arguments in ((), (algebra,), (algebra, np.zeros(4), np.zeros(4))):
            with self.subTest(arguments=len(arguments)), self.assertRaises(TypeError):
                cdl.SplitArray(*arguments)

    def test_split_computes_both_parts(self) -> None:
        """Hold the transformed group components and the signed components."""

        rng = np.random.default_rng(1)
        for algebra in algebras():
            x = exact_array(algebra, rng, (2, 3))
            split = x.split()
            with self.subTest(algebra=algebra):
                self.assertIsInstance(split, cdl.SplitArray)
                self.assertIs(split.algebra, algebra)
                self.assertEqual(split.batch_shape, (2, 3))
                self.assertEqual(split.dtype, algebra.dtype)
                self.assertEqual(split.dimension, algebra.dimension)
                self.assertEqual(len(split), 2)
                self.assertCoefficientsEqual(plain(split.spectral), plain(x.spectrum()))
                self.assertCoefficientsEqual(plain(split.signed), plain(x.epsilon()))
                self.assertCoefficientsEqual(
                    plain(split.spectrum()), plain(x.spectrum())
                )
                self.assertCoefficientsEqual(plain(split.epsilon()), plain(x.epsilon()))
                self.assertCoefficientsEqual(plain(split.kappa()), plain(x.kappa()))
                self.assertCoefficientsEqual(
                    plain(split.spectral),
                    cdl.wht(plain(x.kappa()), field=algebra.field).astype(
                        algebra.dtype
                    ),
                )

    def test_part_types_follow_loop_and_field(self) -> None:
        """Expose ndarrays or pyfastcd arrays according to the parent."""

        for algebra in algebras():
            split = algebra.random(0, batch_shape=(3,)).split()
            size = algebra.loop.quotient_order
            with self.subTest(algebra=algebra):
                if algebra.field is None:
                    self.assertIsInstance(split.spectral, np.ndarray)
                    self.assertEqual(split.spectral.shape, (3, size))
                else:
                    self.assertIsInstance(split.spectral, fcd.ElementArray)
                    self.assertIs(split.spectral.algebra, algebra.field)
                    self.assertEqual(split.spectral.batch_shape, (3, size))
                if algebra.cd_algebra is not None:
                    self.assertIsInstance(split.signed, fcd.ElementArray)
                    self.assertIs(split.signed.algebra, algebra.cd_algebra)
                    self.assertEqual(split.signed.batch_shape, (3,))
                else:
                    self.assertIsInstance(split.signed, np.ndarray)
                    self.assertEqual(
                        split.signed.shape, (3, size) + algebra.coefficient_shape[2:]
                    )
                self.assertFalse(hasattr(split, "coefficients"))
                self.assertFalse(hasattr(split, "shape"))

    def test_parts_share_storage_with_the_array(self) -> None:
        """Change the array by writing through its two parts."""

        for algebra in algebras():
            split = algebra.array(
                np.zeros((2,) + algebra.coefficient_shape, dtype=np.int64)
            ).split()
            first = (0, 1) + (0,) * (len(algebra.coefficient_shape) - 2)
            with self.subTest(algebra=algebra):
                self.assertTrue((split == 0).all())
                plain(split.spectral)[first] = 1
                self.assertCoefficientsEqual(split == 0, [False, True])
                self.assertEqual(plain(split.spectrum())[first], 1)
                plain(split.signed)[(1,) + first[1:]] = 2
                self.assertCoefficientsEqual(split == 0, [False, False])
                self.assertEqual(plain(split.epsilon())[(1,) + first[1:]], 2)
                # The accessor methods return copies instead
                plain(split.spectrum())[first] = 5
                self.assertEqual(plain(split.spectral)[first], 1)
                plain(split.epsilon())[(1,) + first[1:]] = 5
                self.assertEqual(plain(split.signed)[(1,) + first[1:]], 2)
                with self.assertRaises(AttributeError):
                    split.spectral = split.spectral
                with self.assertRaises(AttributeError):
                    split.signed = split.signed

    def test_split_array_builds_from_components(self) -> None:
        """Create split arrays from parts given in several forms."""

        rng = np.random.default_rng(2)
        for algebra in algebras():
            x = exact_array(algebra, rng, (3,))
            split = x.split()
            with self.subTest(algebra=algebra):
                rebuilt = algebra.split_array(split.spectral, split.signed)
                self.assertIsInstance(rebuilt, cdl.SplitArray)
                self.assertTrue((rebuilt == split).all())
                self.assertFalse(
                    np.shares_memory(plain(rebuilt.spectral), plain(split.spectral))
                )
                self.assertFalse(
                    np.shares_memory(plain(rebuilt.signed), plain(split.signed))
                )
                lists = algebra.split_array(
                    plain(split.spectral).tolist(), plain(split.signed).tolist()
                )
                self.assertTrue((lists == split).all())
                single = algebra.split_array(
                    plain(split.spectral)[0], plain(split.signed)[0]
                )
                self.assertEqual(single.batch_shape, ())
                self.assertTrue(single[()] == x[0])
                broadcast = algebra.split_array(
                    plain(split.spectral), plain(split.signed)[0]
                )
                self.assertEqual(broadcast.batch_shape, (3,))
                self.assertCoefficientsEqual(
                    plain(broadcast.signed)[2], plain(split.signed)[0]
                )
                with self.assertRaises(ValueError):
                    algebra.split_array(
                        plain(split.spectral)[..., :1], plain(split.signed)
                    )

    def test_to_loop_inverts_split(self) -> None:
        """Round-trip exactly over finite fields and on integer inputs."""

        rng = np.random.default_rng(3)
        for algebra in algebras():
            x = exact_array(algebra, rng, (2, 2))
            with self.subTest(algebra=algebra):
                back = x.split().to_loop()
                self.assertIsInstance(back, cdl.ElementArray)
                self.assertCoefficientsEqual(back.coefficients, x.coefficients)
                self.assertFalse(np.shares_memory(back.coefficients, x.coefficients))
                again = back.split()
                self.assertTrue((again == x.split()).all())
                self.assertTrue(back.coefficients.flags.writeable)
        real = cdl.LoopAlgebra(cdl.CDLoop(5))
        y = real.random(4, batch_shape=(3,))
        np.testing.assert_allclose(
            y.split().to_loop().coefficients, y.coefficients, rtol=1e-13, atol=1e-13
        )

    def test_copy_deepcopy_and_pickle(self) -> None:
        """Create independent storage in every copying operation."""

        rng = np.random.default_rng(5)
        for algebra in algebras():
            split = exact_array(algebra, rng, (3,)).split()
            with self.subTest(algebra=algebra):
                for clone in (
                    split.copy(),
                    +split,
                    copy.copy(split),
                    copy.deepcopy(split),
                    pickle.loads(pickle.dumps(split)),
                ):
                    self.assertIsInstance(clone, cdl.SplitArray)
                    self.assertIsNot(clone, split)
                    self.assertIs(clone.algebra, algebra)
                    self.assertTrue((clone == split).all())
                    for name in ("spectral", "signed"):
                        self.assertFalse(
                            np.shares_memory(
                                plain(getattr(clone, name)), plain(getattr(split, name))
                            )
                        )

    def test_repr_names_batch_shape_and_parent(self) -> None:
        """Show the batch shape and the algebra."""

        split = cdl.LoopAlgebra(cdl.CDLoop(2)).array(np.zeros((3, 2, 4))).split()
        self.assertEqual(
            repr(split), "SplitArray(batch_shape=(3,), algebra=LoopAlgebra(CDLoop(2)))"
        )


class ArithmeticTests(SplitArrayTestCase):
    def test_operations_match_the_loop_basis(self) -> None:
        """Agree exactly with the same operation on loop-basis arrays."""

        rng = np.random.default_rng(11)
        for algebra in algebras():
            x = exact_array(algebra, rng, (3,))
            y = exact_array(algebra, rng, (3,))
            z = exact_array(algebra, rng, (3,))
            s, t, u = x.split(), y.split(), z.split()
            with self.subTest(algebra=algebra):
                self.assertSplitEqualsLoop(s + t, x + y)
                self.assertSplitEqualsLoop(s - t, x - y)
                self.assertSplitEqualsLoop(-s, -x)
                self.assertSplitEqualsLoop(s * t, x * y)
                self.assertSplitEqualsLoop(s + 2, x + 2)
                self.assertSplitEqualsLoop(2 + s, 2 + x)
                self.assertSplitEqualsLoop(s - 2, x - 2)
                self.assertSplitEqualsLoop(2 - s, 2 - x)
                self.assertSplitEqualsLoop(s * 2, x * 2)
                self.assertSplitEqualsLoop(2 * s, 2 * x)
                self.assertSplitEqualsLoop((s * 2) / 2, x)
                self.assertSplitEqualsLoop(np.int64(2) * s, 2 * x)
                self.assertSplitEqualsLoop(s.conjugate(), x.conjugate())
                self.assertSplitEqualsLoop(s.square(), x.square())
                self.assertSplitEqualsLoop(s.commutator(t), x.commutator(y))
                self.assertSplitEqualsLoop(s.anticommutator(t), x.anticommutator(y))
                self.assertSplitEqualsLoop(s.jordan_product(t), x.jordan_product(y))
                self.assertSplitEqualsLoop(s.associator(t, u), x.associator(y, z))
                self.assertSplitEqualsLoop(s.left_power(3), x.left_power(3))
                self.assertSplitEqualsLoop(s.right_power(3), x.right_power(3))
                self.assertSplitEqualsLoop(s.left_power(0), x.left_power(0))
                self.assertSplitEqualsLoop(
                    algebra.multiply(s, t), algebra.multiply(x, y)
                )
                self.assertSplitEqualsLoop(
                    algebra.left_operator_commutator(s, t, u),
                    algebra.left_operator_commutator(x, y, z),
                )
                self.assertSplitEqualsLoop(
                    algebra.right_operator_commutator(s, t, u),
                    algebra.right_operator_commutator(x, y, z),
                )
                for name in algebra.signed_algorithms:
                    self.assertSplitEqualsLoop(
                        algebra.multiply(s, t, signed_algorithm=name), x * y
                    )

    def test_powers_and_inverses_for_cayley_dickson_loops(self) -> None:
        """Agree with loop-basis powers and inverses, exactly where exact."""

        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            rng = np.random.default_rng(12)
            records: list[np.ndarray] = []
            while len(records) < 3:
                candidate = algebra(exact_values(algebra, rng))
                if candidate.has_inverse():
                    records.append(candidate.coefficients)
            x = algebra.array(np.stack(records))
            s = x.split()
            with self.subTest(algebra=algebra):
                for exponent in (0, 1, 2, 5):
                    self.assertSplitEqualsLoop(s**exponent, x**exponent)
                self.assertCoefficientsEqual(s.has_inverse(), [True, True, True])
                inverse = s.inverse()
                self.assertIsInstance(inverse, cdl.SplitArray)
                if algebra.field is not None:
                    self.assertSplitEqualsLoop(inverse, x.inverse())
                    self.assertTrue((s * inverse == 1).all())
                    self.assertSplitEqualsLoop(s**-2, x**-2)
                    self.assertSplitEqualsLoop(s ** (10**20 + 1), x ** (10**20 + 1))
                else:
                    np.testing.assert_allclose(
                        inverse.to_loop().coefficients,
                        x.inverse().coefficients,
                        rtol=1e-11,
                        atol=1e-11,
                    )
                broken = s.copy()
                broken[1] = 0
                self.assertCoefficientsEqual(broken.has_inverse(), [True, False, True])
                with self.assertRaises(ZeroDivisionError):
                    broken.inverse()
                with self.assertRaises(ZeroDivisionError):
                    s / 0

    def test_central_product_powers_are_rejected(self) -> None:
        """Raise for ``**`` and for inverses in a central product."""

        algebra = cdl.LoopAlgebra(cdl.CentralProduct(1, 2))
        s = algebra.random(0, batch_shape=(2,)).split()
        with self.assertRaises(TypeError):
            s**2
        with self.assertRaises(TypeError):
            s.inverse()
        with self.assertRaises(TypeError):
            s.has_inverse()

    def test_chains_stay_in_split_form(self) -> None:
        """Evaluate a chain of operations and convert once at the end."""

        rng = np.random.default_rng(13)
        for algebra in algebras():
            x = exact_array(algebra, rng, (4,))
            y = exact_array(algebra, rng, (4,))
            s, t = x.split(), y.split()
            with self.subTest(algebra=algebra):
                chain = ((s * t + s) * t - 2 * s).commutator(t) + s.conjugate() * t
                expected = ((x * y + x) * y - 2 * x).commutator(y) + x.conjugate() * y
                self.assertSplitEqualsLoop(chain, expected)
                nested = (s.associator(t, s) * s).square() - (t * t) * (s * s)
                self.assertSplitEqualsLoop(
                    nested, (x.associator(y, x) * x).square() - (y * y) * (x * x)
                )

    def test_elements_combine_on_both_sides(self) -> None:
        """Use the split form of an element with every record."""

        rng = np.random.default_rng(14)
        for algebra in algebras():
            x = exact_array(algebra, rng, (3,))
            s = x.split()
            e = algebra(exact_values(algebra, rng))
            product = e * e
            with self.subTest(algebra=algebra):
                for element in (e, product):
                    self.assertSplitEqualsLoop(s * element, x * element)
                    self.assertSplitEqualsLoop(element * s, element * x)
                    self.assertSplitEqualsLoop(s + element, x + element)
                    self.assertSplitEqualsLoop(element + s, element + x)
                    self.assertSplitEqualsLoop(s - element, x - element)
                    self.assertSplitEqualsLoop(element - s, element - x)
                    self.assertSplitEqualsLoop(
                        s.commutator(element), x.commutator(element)
                    )
                    self.assertSplitEqualsLoop(
                        s.associator(element, s), x.associator(element, x)
                    )
                    self.assertSplitEqualsLoop(
                        algebra.multiply(element, s), element * x
                    )
                    self.assertSplitEqualsLoop(
                        algebra.left_operator_commutator(element, s, s),
                        algebra.left_operator_commutator(element, x, x),
                    )
                with self.assertRaises(TypeError):
                    e.commutator(s)

    def test_batch_axes_broadcast(self) -> None:
        """Broadcast leading axes in binary operations."""

        rng = np.random.default_rng(15)
        for algebra in algebras():
            x = exact_array(algebra, rng, (3, 1))
            y = exact_array(algebra, rng, (4,))
            s, t = x.split(), y.split()
            with self.subTest(algebra=algebra):
                self.assertSplitEqualsLoop(s * t, x * y)
                self.assertSplitEqualsLoop(s + t, x + y)
                self.assertSplitEqualsLoop(s.commutator(t), x.commutator(y))
                self.assertEqual((s * t).batch_shape, (3, 4))
                self.assertEqual((s == t).shape, (3, 4))
                self.assertEqual(s.distance(t).shape, (3, 4))
                with self.assertRaises(ValueError):
                    t * exact_array(algebra, rng, (5,)).split()

    def test_other_parents_and_operands_are_rejected(self) -> None:
        """Raise for split arrays of another algebra and for raw arrays."""

        loop = cdl.CDLoop(2)
        s = cdl.LoopAlgebra(loop).random(0, batch_shape=(2,)).split()
        other = cdl.LoopAlgebra(loop, char=13).random(0, batch_shape=(2,)).split()
        for operation in (
            lambda: s + other,
            lambda: s * other,
            lambda: s.commutator(other),
            lambda: s * other[0],
        ):
            with self.assertRaises(ValueError):
                operation()
        for operation in (
            lambda: s * np.ones((2, 4)),
            lambda: np.ones((2, 4)) * s,
            lambda: s + [1.0, 2.0],
            lambda: s.commutator(2),
            lambda: s / s,
            lambda: np.add(s, s),
        ):
            with self.assertRaises(TypeError):
                operation()


class MixingTests(SplitArrayTestCase):
    def setUp(self) -> None:
        """Create one array in each coordinate system."""

        self.algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        self.loop_array = self.algebra.random(0, batch_shape=(3,))
        self.split_array = self.loop_array.split()

    def test_operators_reject_mixed_arrays(self) -> None:
        """Raise the shared error from every binary operator."""

        x, s = self.loop_array, self.split_array
        for operation in (
            lambda: x + s,
            lambda: s + x,
            lambda: x - s,
            lambda: s - x,
            lambda: x * s,
            lambda: s * x,
        ):
            self.assertMixingRejected(operation)

    def test_methods_reject_mixed_arrays(self) -> None:
        """Raise the shared error from every named method."""

        x, s = self.loop_array, self.split_array
        e = x[0]
        for operation in (
            lambda: x.commutator(s),
            lambda: s.commutator(x),
            lambda: x.anticommutator(s),
            lambda: s.anticommutator(x),
            lambda: x.jordan_product(s),
            lambda: s.jordan_product(x),
            lambda: x.associator(s, x),
            lambda: x.associator(x, s),
            lambda: s.associator(x, s),
            lambda: s.associator(s, x),
            lambda: s.associator(e, x),
            lambda: x.distance(s),
            lambda: s.distance(x),
        ):
            self.assertMixingRejected(operation)

    def test_comparisons_reject_mixed_arrays(self) -> None:
        """Raise the shared error from ``==`` and ``!=`` in both orders."""

        x, s = self.loop_array, self.split_array
        for operation in (
            lambda: x.__eq__(s),
            lambda: s.__eq__(x),
            lambda: x.__ne__(s),
            lambda: s.__ne__(x),
        ):
            self.assertMixingRejected(operation)

    def test_assignment_rejects_mixed_arrays(self) -> None:
        """Raise the shared error when assigning across coordinate systems."""

        x, s = self.loop_array.copy(), self.split_array.copy()
        self.assertMixingRejected(lambda: x.__setitem__(0, s))
        self.assertMixingRejected(lambda: x.__setitem__(slice(None), s))
        self.assertMixingRejected(lambda: s.__setitem__(0, x))
        self.assertMixingRejected(lambda: s.__setitem__(slice(None), x))
        self.assertCoefficientsEqual(x.coefficients, self.loop_array.coefficients)
        self.assertTrue((s == self.split_array).all())

    def test_parent_methods_reject_mixed_arrays(self) -> None:
        """Raise the shared error from the algebra's own methods."""

        algebra = self.algebra
        x, s = self.loop_array, self.split_array
        for operation in (
            lambda: algebra.multiply(x, s),
            lambda: algebra.multiply(s, x),
            lambda: algebra.left_operator_commutator(x, s, s),
            lambda: algebra.left_operator_commutator(s, s, x),
            lambda: algebra.right_operator_commutator(x, x, s),
            lambda: algebra.array(s),
            lambda: algebra.asarray(s),
            lambda: algebra.asarray(s, copy=False),
        ):
            self.assertMixingRejected(operation)
        with self.assertRaises(TypeError):
            algebra.multiply(s, s, algorithm="table")
        with self.assertRaises(TypeError):
            algebra.multiply(s, s, out=np.empty((3, 2, 4)))
        with self.assertRaises(TypeError):
            algebra.multiply(x, x, out=s)

    def test_explicit_conversion_resolves_the_mix(self) -> None:
        """Combine the arrays after converting either one."""

        x, s = self.loop_array, self.split_array
        self.assertIsInstance(x.split() * s, cdl.SplitArray)
        self.assertIsInstance(x * s.to_loop(), cdl.ElementArray)
        np.testing.assert_allclose(
            (x.split() * s).to_loop().coefficients,
            (x * s.to_loop()).coefficients,
            rtol=1e-12,
            atol=1e-12,
        )


class IndexingTests(SplitArrayTestCase):
    def test_single_records_are_split_born_elements(self) -> None:
        """Return an element that keeps the record's split form."""

        rng = np.random.default_rng(21)
        for algebra in algebras():
            x = exact_array(algebra, rng, (2, 3))
            s = x.split()
            with self.subTest(algebra=algebra):
                element = s[1, 2]
                self.assertIsInstance(element, cdl.Element)
                self.assertEqual(element._born, "split")
                self.assertTrue(element == x[1, 2])
                self.assertCoefficientsEqual(
                    plain(element.spectrum()), plain(s.spectral)[1, 2]
                )
                s[1, 2] = 0
                self.assertTrue(element == x[1, 2])
                self.assertTrue(s[-1, -1] == 0)
                rows = list(s)
                self.assertEqual(len(rows), 2)
                self.assertIsInstance(rows[0], cdl.SplitArray)
                self.assertEqual(rows[0].batch_shape, (3,))
                self.assertTrue(all(isinstance(e, cdl.Element) for e in s[0]))

    def test_selections_follow_numpy_on_the_batch_axes(self) -> None:
        """Select with slices, ellipsis, ``None`` and arrays."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        x = exact_array(algebra, np.random.default_rng(22), (2, 3))
        s = x.split()
        mask = np.array([[True, False, True], [False, False, True]])
        keys: tuple[Any, ...] = (
            0,
            slice(1, None),
            (slice(None), 1),
            (Ellipsis, 0),
            Ellipsis,
            None,
            (slice(None), None),
            ([1, 0],),
            mask,
            (slice(None), [0, 2]),
        )
        for key in keys:
            with self.subTest(key=repr(key)):
                selected = s[key]
                self.assertIsInstance(selected, cdl.SplitArray)
                self.assertSplitEqualsLoop(selected, x[key])
        self.assertTrue(np.shares_memory(plain(s[1:].spectral), plain(s.spectral)))
        self.assertTrue(np.shares_memory(plain(s[:, 1].signed), plain(s.signed)))
        for key in ((0, 0, 0), (Ellipsis, 0, 0, 0), (Ellipsis, Ellipsis), 2):
            with self.subTest(key=repr(key)), self.assertRaises(IndexError):
                s[key]

    def test_assignment_accepts_scalars_elements_and_split_arrays(self) -> None:
        """Write values through an index in split form."""

        rng = np.random.default_rng(23)
        for algebra in algebras():
            x = exact_array(algebra, rng, (3,))
            e = algebra(exact_values(algebra, rng))
            product = e * e
            s = x.split()
            with self.subTest(algebra=algebra):
                s[0] = 2
                self.assertTrue(s[0] == 2)
                s[1] = e
                self.assertTrue(s[1] == e)
                s[2] = product
                self.assertTrue(s[2] == product)
                s[:2] = x.split()[1:]
                self.assertSplitEqualsLoop(s[:2], x[1:])
                s[...] = e
                self.assertTrue((s == e).all())
                s[np.array([True, False, True])] = 0
                self.assertCoefficientsEqual(s == 0, [True, False, True])
                with self.assertRaises(TypeError):
                    s[0] = x.coefficients[0]
                with self.assertRaises(TypeError):
                    s[0] = "zero"
        split = cdl.LoopAlgebra(cdl.CDLoop(2)).random(0, batch_shape=(2,)).split()
        with self.assertRaises(ValueError):
            split[0] = cdl.LoopAlgebra(cdl.CDLoop(2), char=13).one()
        with self.assertRaises(ValueError):
            split[0] = (
                cdl.LoopAlgebra(cdl.CDLoop(3)).random(0, batch_shape=(2,)).split()
            )


class ComparisonTests(SplitArrayTestCase):
    def test_equality_returns_boolean_arrays(self) -> None:
        """Compare records with split arrays, elements and scalars."""

        rng = np.random.default_rng(31)
        for algebra in algebras():
            x = exact_array(algebra, rng, (3,))
            y = x.copy()
            y[1] = y[1] + 1
            s, t = x.split(), y.split()
            with self.subTest(algebra=algebra):
                equal = s == t
                self.assertIsInstance(equal, np.ndarray)
                self.assertEqual(equal.dtype, np.bool_)
                self.assertCoefficientsEqual(equal, [True, False, True])
                self.assertCoefficientsEqual(s != t, [False, True, False])
                self.assertCoefficientsEqual(s == x[2], [False, False, True])
                self.assertCoefficientsEqual(x[2] == s, [False, False, True])
                self.assertCoefficientsEqual(s != x[2], [True, True, False])
                scalars = algebra.array(
                    np.stack(
                        [
                            algebra.scalar(2).coefficients,
                            algebra.one().coefficients,
                            x.coefficients[0],
                        ]
                    )
                ).split()
                self.assertCoefficientsEqual(scalars == 2, [True, False, False])
                self.assertCoefficientsEqual(scalars != 2, [False, True, True])

    def test_other_parents_and_objects_compare_unequal(self) -> None:
        """Return all-false arrays or ``False`` without raising."""

        loop = cdl.CDLoop(2)
        s = cdl.LoopAlgebra(loop).array(np.zeros((2, 3, 2, 4))).split()
        other = cdl.LoopAlgebra(loop, char=13)
        for partner in (other.zero(), other.random(0, batch_shape=(3,)).split()):
            with self.subTest(partner=repr(partner)):
                equal = s == partner
                self.assertEqual(equal.shape, (2, 3))
                self.assertFalse(equal.any())
                self.assertTrue((s != partner).all())
        for value in ("a", None, [0.0], np.zeros((2, 4))):
            with self.subTest(value=repr(value)):
                self.assertIs(s == value, False)
                self.assertIs(s != value, True)
        finite = other.array(
            np.stack([other.scalar(2).coefficients, other.one().coefficients])
        ).split()
        self.assertCoefficientsEqual(finite == 2, [True, False])
        self.assertCoefficientsEqual(finite == 15, [False, False])
        self.assertCoefficientsEqual(finite == other.field(15), [True, False])

    def test_split_arrays_have_no_truth_value_or_hash(self) -> None:
        """Raise for ``bool()`` and ``hash()``."""

        s = cdl.LoopAlgebra(cdl.CDLoop(2)).array(np.zeros((3, 2, 4))).split()
        with self.assertRaises(ValueError):
            bool(s)
        with self.assertRaises(TypeError):
            hash(s)
        single = s[0:1]
        with self.assertRaises(ValueError):
            bool(single)


class PredicateTests(SplitArrayTestCase):
    def test_predicates_and_counts_match_the_loop_basis(self) -> None:
        """Evaluate predicates, weights and distances per record."""

        loop = cdl.CDLoop(2)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(loop, **selectors)
            plus, minus = algebra.central_idempotents()
            elements = [
                algebra.zero(),
                algebra.one(),
                plus,
                minus,
                algebra.basis(loop.basis(1)),
                algebra.primitive_idempotent(2),
                algebra(exact_values(algebra, np.random.default_rng(41))),
            ]
            x = algebra.array(np.stack([e.coefficients for e in elements]))
            s = x.split()
            with self.subTest(algebra=algebra):
                for name in (
                    "has_inverse",
                    "is_idempotent",
                    "is_nilpotent",
                    "is_central",
                    "is_zero_divisor",
                ):
                    marks = getattr(s, name)()
                    self.assertEqual(marks.dtype, np.bool_)
                    self.assertCoefficientsEqual(marks, getattr(x, name)())
                    self.assertCoefficientsEqual(
                        marks, [getattr(e, name)() for e in elements]
                    )
                self.assertCoefficientsEqual(s.weight(), x.weight())
                self.assertCoefficientsEqual(s.distance(s[1]), x.distance(x[1]))
                self.assertCoefficientsEqual(s.distance(s[::-1]), x.distance(x[::-1]))
                self.assertFalse(np.any(s.distance(s)))

    def test_components_of_a_central_product(self) -> None:
        """Expose flat signed components and spectra for a central product."""

        algebra = cdl.LoopAlgebra(cdl.CentralProduct(1, 2), char=13)
        x = exact_array(algebra, np.random.default_rng(42), (3,))
        s = x.split()
        self.assertIsInstance(s.signed, np.ndarray)
        self.assertEqual(s.signed.shape, (3, 8))
        self.assertIsInstance(s.epsilon(), np.ndarray)
        self.assertIsInstance(s.spectral, fcd.ElementArray)
        self.assertCoefficientsEqual(s.signed[1], x[1].epsilon().reshape(8))
        self.assertCoefficientsEqual(plain(s.kappa()), plain(x.kappa()))
        self.assertTrue(s.is_central().shape == (3,))


if __name__ == "__main__":
    unittest.main()
