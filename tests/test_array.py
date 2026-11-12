"""Deterministic tests for :class:`pycdloop.ElementArray`."""

from __future__ import annotations

import copy
import pickle
import unittest
from typing import Any

import numpy as np
import pyfastcd as fcd

import pycdloop as cdl

RTOL = 1.0e-11
ATOL = 1.0e-11

# Coefficient fields exercised by most tests: R, C, a prime field and an
# extension field
SELECTORS: tuple[dict[str, int], ...] = (
    {},
    {"char": 0, "deg": 2},
    {"char": 13},
    {"char": 3, "deg": 2},
)


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
    """Draw an array of elements with exactly representable coefficients."""

    return algebra.asarray(exact_values(algebra, rng, lead))


def invertible_array(algebra: Any, seed: int, count: int) -> Any:
    """Return an array of *count* random invertible elements."""

    rng = np.random.default_rng(seed)
    records: list[np.ndarray] = []
    while len(records) < count:
        candidate = algebra(exact_values(algebra, rng))
        if candidate.has_inverse():
            records.append(candidate.coefficients)
    return algebra.array(np.stack(records))


def plain(value: Any) -> np.ndarray:
    """Return a component given as an ndarray or a pyfastcd object as an ndarray."""

    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value.coefficients)


class ElementArrayTestCase(unittest.TestCase):
    """Common assertions with useful failure diagnostics."""

    def assertCoefficientsEqual(self, actual: object, expected: object) -> None:
        """Assert exact equality after exposing both coefficient arrays."""

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def assertCoefficientsClose(self, actual: object, expected: object) -> None:
        """Assert coefficient closeness with the test tolerances."""

        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=RTOL, atol=ATOL
        )

    def assertRecordsEqual(self, array: Any, records: Any) -> None:
        """Assert that an array holds exactly the given elements, in order."""

        self.assertIsInstance(array, cdl.ElementArray)
        flat = array.coefficients.reshape((-1,) + array.algebra.coefficient_shape)
        records = list(records)
        self.assertEqual(flat.shape[0], len(records))
        for stored, record in zip(flat, records, strict=True):
            self.assertIsInstance(record, cdl.Element)
            self.assertCoefficientsEqual(stored, record.coefficients)


def records_of(array: Any) -> list[Any]:
    """Return the elements of an array of any batch shape in C order."""

    algebra = array.algebra
    flat = array.coefficients.reshape((-1,) + algebra.coefficient_shape)
    return [algebra(values) for values in flat]


class ConstructionTests(ElementArrayTestCase):
    def test_direct_construction_is_rejected(self) -> None:
        """Require a parent algebra to create arrays."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        for arguments in ((), (algebra,), (algebra, np.zeros((3, 2, 4)))):
            with self.subTest(arguments=len(arguments)), self.assertRaises(TypeError):
                cdl.ElementArray(*arguments)

    def test_array_reads_coefficients_elements_and_arrays(self) -> None:
        """Copy each supported kind of input into new storage."""

        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            rng = np.random.default_rng(1)
            values = exact_values(algebra, rng, (2, 3))
            with self.subTest(algebra=algebra):
                array = algebra.array(values)
                self.assertIsInstance(array, cdl.ElementArray)
                self.assertIs(array.algebra, algebra)
                self.assertCoefficientsEqual(array.coefficients, values)
                self.assertFalse(np.shares_memory(array.coefficients, values))
                self.assertEqual(array.coefficients.dtype, algebra.dtype)
                nested = algebra.array(values.tolist())
                self.assertCoefficientsEqual(nested.coefficients, values)
                again = algebra.array(array)
                self.assertCoefficientsEqual(again.coefficients, values)
                self.assertFalse(
                    np.shares_memory(again.coefficients, array.coefficients)
                )
                single = algebra.array(array[0, 0])
                self.assertEqual(single.batch_shape, ())
                self.assertCoefficientsEqual(single.coefficients, values[0, 0])

    def test_shapes_and_sizes(self) -> None:
        """Report batch shape, storage shape, length and dimension."""

        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            array = algebra.random(0, batch_shape=(4, 2))
            with self.subTest(algebra=algebra):
                self.assertEqual(array.batch_shape, (4, 2))
                self.assertEqual(array.shape, (4, 2) + algebra.coefficient_shape)
                self.assertEqual(array.dtype, algebra.dtype)
                self.assertEqual(array.dimension, algebra.dimension)
                self.assertEqual(len(array), 4)
                rows = list(array)
                self.assertEqual(len(rows), 4)
                for index, row in enumerate(rows):
                    self.assertIsInstance(row, cdl.ElementArray)
                    self.assertEqual(row.batch_shape, (2,))
                    self.assertCoefficientsEqual(
                        row.coefficients, array.coefficients[index]
                    )
                elements = list(array[0])
                self.assertTrue(all(isinstance(e, cdl.Element) for e in elements))
        single = cdl.LoopAlgebra(cdl.CDLoop(2)).array(np.zeros((2, 4)))
        self.assertEqual(single.batch_shape, ())
        with self.assertRaises(TypeError):
            len(single)
        with self.assertRaises(TypeError):
            iter(single).__next__()

    def test_invalid_coefficients_are_rejected(self) -> None:
        """Reject shapes that do not end in the coefficient shape, and bad types."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        for shape in ((3, 2, 5), (4,), (3, 4, 2), (2,), ()):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                algebra.array(np.ones(shape))
        with self.assertRaises(TypeError):
            algebra.array(np.ones((3, 2, 4), dtype=complex))
        with self.assertRaises(TypeError):
            algebra.array(np.array([["a"] * 4] * 2))
        finite = cdl.LoopAlgebra(cdl.CDLoop(2), char=13)
        with self.assertRaises(TypeError):
            finite.array(np.ones((3, 2, 4)))
        extension = cdl.LoopAlgebra(cdl.CDLoop(2), char=3, deg=2)
        with self.assertRaises(ValueError):
            extension.array(np.ones((3, 2, 4), dtype=np.int64))
        reduced = finite.array(np.full((2, 2, 4), -1))
        self.assertCoefficientsEqual(reduced.coefficients, np.full((2, 2, 4), 12))

    def test_storage_is_shared_by_asarray_and_views(self) -> None:
        """Write through shared storage in both directions."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        values = np.zeros((3, 2, 4))
        array = algebra.asarray(values)
        self.assertIs(array.coefficients, values)
        values[1, 0, 0] = 5.0
        self.assertTrue(array[1] == 5)
        array.coefficients[2, 1, 3] = 7.0
        self.assertEqual(values[2, 1, 3], 7.0)
        tail = array[1:]
        self.assertTrue(np.shares_memory(tail.coefficients, values))
        tail[0] = 2
        self.assertEqual(values[1, 0, 0], 2.0)
        self.assertTrue(np.shares_memory(np.asarray(array), values))
        self.assertIs(array.to_numpy(copy=False), values)
        self.assertFalse(np.shares_memory(array.to_numpy(), values))
        self.assertFalse(np.shares_memory(np.array(array), values))
        self.assertEqual(np.asarray(array, dtype=np.float32).dtype, np.float32)
        with self.assertRaises(TypeError):
            array.to_numpy(copy=None)
        with self.assertRaises(AttributeError):
            array.coefficients = values  # type: ignore[misc]

    def test_strided_and_read_only_storage(self) -> None:
        """Accept noncontiguous storage and keep read-only storage read-only."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        rng = np.random.default_rng(2)
        values = rng.integers(-3, 4, size=(6, 2, 4)).astype(float)
        strided = algebra.asarray(values[::2])
        self.assertTrue(np.shares_memory(strided.coefficients, values))
        expected = algebra.array(values[::2].copy())
        self.assertTrue((strided * strided == expected * expected).all())
        reverse = algebra.asarray(values[::-1])
        self.assertTrue(reverse[0] == algebra(values[-1]))
        frozen = values.copy()
        frozen.setflags(write=False)
        array = algebra.asarray(frozen)
        self.assertTrue((array + array == algebra.array(values) * 2).all())
        with self.assertRaises(ValueError):
            array[0] = 1

    def test_copy_deepcopy_and_pickle(self) -> None:
        """Create independent storage in every copying operation."""

        rng = np.random.default_rng(3)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            array = exact_array(algebra, rng, (3,))
            with self.subTest(algebra=algebra):
                for clone in (
                    array.copy(),
                    +array,
                    copy.copy(array),
                    copy.deepcopy(array),
                    pickle.loads(pickle.dumps(array)),
                ):
                    self.assertIsInstance(clone, cdl.ElementArray)
                    self.assertIsNot(clone, array)
                    self.assertIs(clone.algebra, algebra)
                    self.assertCoefficientsEqual(clone.coefficients, array.coefficients)
                    self.assertFalse(
                        np.shares_memory(clone.coefficients, array.coefficients)
                    )
                    self.assertTrue(clone.coefficients.flags.writeable)

    def test_repr_names_batch_shape_and_parent(self) -> None:
        """Show the batch shape and the algebra."""

        array = cdl.LoopAlgebra(cdl.CDLoop(2)).array(np.zeros((3, 2, 4)))
        self.assertEqual(
            repr(array),
            "ElementArray(batch_shape=(3,), algebra=LoopAlgebra(CDLoop(2)))",
        )


class IndexingTests(ElementArrayTestCase):
    def setUp(self) -> None:
        """Create a two-axis batch with distinct records."""

        self.algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        self.values = np.arange(48.0).reshape(2, 3, 2, 4)
        self.array = self.algebra.array(self.values)

    def test_single_records_are_element_copies(self) -> None:
        """Return an immutable element that does not follow later writes."""

        element = self.array[1, 2]
        self.assertIsInstance(element, cdl.Element)
        self.assertCoefficientsEqual(element, self.values[1, 2])
        self.assertFalse(
            np.shares_memory(element.coefficients, self.array.coefficients)
        )
        self.array[1, 2] = 0
        self.assertCoefficientsEqual(element, self.values[1, 2])
        self.assertTrue(self.array[-1, -1] == 0)
        self.assertIsInstance(self.array[np.int64(0), np.int64(1)], cdl.Element)
        single = self.algebra.array(self.values[0, 0])
        self.assertIsInstance(single[()], cdl.Element)
        self.assertIsInstance(single[...], cdl.Element)

    def test_selections_follow_numpy_on_the_batch_axes(self) -> None:
        """Index with integers, slices, ellipsis, ``None`` and arrays."""

        mask = np.array([[True, False, True], [False, False, True]])
        keys: tuple[Any, ...] = (
            0,
            -1,
            slice(None),
            slice(1, None),
            (slice(None), 1),
            (slice(None), slice(0, 3, 2)),
            (Ellipsis, 0),
            (0, Ellipsis),
            Ellipsis,
            (None,),
            (slice(None), None),
            (Ellipsis, None),
            ([1, 0],),
            (np.array([0, 1]), np.array([2, 0])),
            (slice(None), [0, 2]),
            mask,
            np.array([True, False]),
            (0, np.array([True, False, True])),
            (slice(None, None, -1), slice(None)),
        )
        numbers = np.arange(6).reshape(2, 3)
        flat = self.values.reshape(6, 2, 4)
        for key in keys:
            with self.subTest(key=repr(key)):
                selected = self.array[key]
                # Index the record numbers, then look the records up
                expected = flat[numbers[key]]
                self.assertIsInstance(selected, cdl.ElementArray)
                self.assertCoefficientsEqual(selected.coefficients, expected)
                self.assertEqual(selected.batch_shape, expected.shape[:-2])

    def test_basic_selections_share_storage(self) -> None:
        """Share storage for slices and copy for array indices, as NumPy does."""

        storage = self.array.coefficients
        for key in (slice(1, None), (slice(None), 1), (Ellipsis, 0), None):
            with self.subTest(key=repr(key)):
                self.assertTrue(np.shares_memory(self.array[key].coefficients, storage))
        for key in ([1, 0], np.array([True, False])):
            with self.subTest(key=repr(key)):
                self.assertFalse(
                    np.shares_memory(self.array[key].coefficients, storage)
                )

    def test_indices_cannot_reach_the_coefficient_axes(self) -> None:
        """Reject indices beyond the batch axes and out-of-range indices."""

        for key in (
            (0, 0, 0),
            (0, 0, 0, 0),
            (Ellipsis, 0, 0, 0),
            (slice(None), slice(None), slice(None)),
            (Ellipsis, Ellipsis),
            np.ones((2, 3, 2), dtype=bool),
        ):
            with self.subTest(key=repr(key)), self.assertRaises(IndexError):
                self.array[key]
        for key in (2, -3, (0, 3), (slice(None), 5)):
            with self.subTest(key=repr(key)), self.assertRaises(IndexError):
                self.array[key]
        with self.assertRaises(IndexError):
            self.algebra.array(self.values[0, 0])[0]

    def test_assignment_accepts_scalars_elements_arrays_and_coefficients(self) -> None:
        """Write each supported kind of value through an index."""

        algebra = self.algebra
        one = algebra.one().coefficients
        array = self.array.copy()
        array[0, 0] = 5
        self.assertTrue(array[0, 0] == 5)
        array[0, 1] = algebra.one()
        self.assertCoefficientsEqual(array.coefficients[0, 1], one)
        array[1] = algebra.array(self.values[0])
        self.assertCoefficientsEqual(array.coefficients[1], self.values[0])
        array[0, 2] = np.ones((2, 4))
        self.assertCoefficientsEqual(array.coefficients[0, 2], np.ones((2, 4)))
        array[:, 0] = np.full((2, 4), 2.0).tolist()
        self.assertCoefficientsEqual(array.coefficients[:, 0], np.full((2, 2, 4), 2.0))
        array[np.array([[True, False, False], [False, False, True]])] = algebra.one()
        self.assertCoefficientsEqual(array.coefficients[0, 0], one)
        self.assertCoefficientsEqual(array.coefficients[1, 2], one)
        array[...] = 0
        self.assertFalse(np.any(array.coefficients))
        array[:] = algebra.array(self.values[:, :1])
        self.assertCoefficientsEqual(
            array.coefficients, np.broadcast_to(self.values[:, :1], (2, 3, 2, 4))
        )
        product = algebra.one() * algebra.scalar(3)
        array[0, 0] = product
        self.assertTrue(array[0, 0] == 3)

    def test_assignment_over_finite_fields_reduces_integers(self) -> None:
        """Store canonical residues and reject floats."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=13)
        array = algebra.array(np.zeros((3, 2, 4), dtype=np.int64))
        array[0] = 15
        self.assertTrue(array[0] == 2)
        array[1] = np.full((2, 4), -1)
        self.assertCoefficientsEqual(array.coefficients[1], np.full((2, 4), 12))
        array[2] = algebra.field(5)
        self.assertTrue(array[2] == 5)
        self.assertEqual(array.coefficients.dtype, np.uint32)
        self.assertLess(int(array.coefficients.max()), 13)
        with self.assertRaises(TypeError):
            array[0] = 1.5
        with self.assertRaises(TypeError):
            array[0] = np.ones((2, 4))

    def test_invalid_assignments_are_rejected(self) -> None:
        """Reject split arrays, other parents and mismatched shapes."""

        algebra = self.algebra
        array = self.array.copy()
        before = array.coefficients.copy()
        with self.assertRaises(TypeError):
            array[0] = array[0].split()
        with self.assertRaises(ValueError):
            array[0, 0] = cdl.LoopAlgebra(cdl.CDLoop(2), char=13).one()
        with self.assertRaises(ValueError):
            array[0] = cdl.LoopAlgebra(cdl.CDLoop(3)).random(0, batch_shape=(3,))
        with self.assertRaises(ValueError):
            array[0, 0] = np.ones((2, 5))
        with self.assertRaises(ValueError):
            array[0] = algebra.array(np.zeros((2, 2, 4)))
        with self.assertRaises(TypeError):
            array[0, 0] = 1j
        with self.assertRaises(IndexError):
            array[0, 0, 0] = 1
        self.assertCoefficientsEqual(array.coefficients, before)


class BroadcastingTests(ElementArrayTestCase):
    def test_binary_operations_broadcast_leading_axes(self) -> None:
        """Broadcast batch axes in every binary operation and method."""

        rng = np.random.default_rng(11)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x = exact_array(algebra, rng, (3, 1))
            y = exact_array(algebra, rng, (4,))
            operations = (
                ("add", lambda a, b: a + b),
                ("subtract", lambda a, b: a - b),
                ("multiply", lambda a, b: a * b),
                ("commutator", lambda a, b: a.commutator(b)),
                ("anticommutator", lambda a, b: a.anticommutator(b)),
                ("jordan", lambda a, b: a.jordan_product(b)),
                ("associator", lambda a, b: a.associator(b, a)),
            )
            for name, operation in operations:
                with self.subTest(algebra=algebra, operation=name):
                    result = operation(x, y)
                    self.assertIsInstance(result, cdl.ElementArray)
                    self.assertEqual(result.batch_shape, (3, 4))
                    for i in range(3):
                        for j in range(4):
                            expected = operation(x[i, 0], y[j])
                            self.assertTrue(result[i, j] == expected)
            with self.subTest(algebra=algebra, operation="distance"):
                distance = x.distance(y)
                self.assertEqual(distance.shape, (3, 4))
                self.assertEqual(int(distance[1, 2]), x[1, 0].distance(y[2]))
                self.assertEqual((x == y).shape, (3, 4))

    def test_incompatible_batch_shapes_are_rejected(self) -> None:
        """Raise ``ValueError`` when batch axes do not broadcast."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        x = algebra.random(0, batch_shape=(3,))
        y = algebra.random(1, batch_shape=(4,))
        for operation in (
            lambda: x + y,
            lambda: x * y,
            lambda: x.commutator(y),
            lambda: x == y,
            lambda: x.distance(y),
        ):
            with self.assertRaises(ValueError):
                operation()

    def test_elements_broadcast_against_arrays(self) -> None:
        """Combine an element with every record on either side."""

        rng = np.random.default_rng(12)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x = exact_array(algebra, rng, (2, 2))
            e = algebra(exact_values(algebra, rng))
            product = e * e
            with self.subTest(algebra=algebra):
                self.assertRecordsEqual(x * e, [r * e for r in records_of(x)])
                self.assertRecordsEqual(e * x, [e * r for r in records_of(x)])
                self.assertRecordsEqual(x + e, [r + e for r in records_of(x)])
                self.assertRecordsEqual(e - x, [e - r for r in records_of(x)])
                self.assertRecordsEqual(x - e, [r - e for r in records_of(x)])
                self.assertRecordsEqual(
                    x.commutator(e), [r.commutator(e) for r in records_of(x)]
                )
                self.assertRecordsEqual(
                    x.associator(e, x), [r.associator(e, r) for r in records_of(x)]
                )
                # An element created by a product is used through its coefficients
                self.assertRecordsEqual(
                    x + product,
                    [r + algebra(product.coefficients) for r in records_of(x)],
                )
                self.assertRecordsEqual(
                    product * x, [product * r for r in records_of(x)]
                )
                self.assertEqual((x == e).shape, (2, 2))
                self.assertEqual(x.distance(e).shape, (2, 2))

    def test_scalars_embed_and_scale(self) -> None:
        """Add a scalar times the identity and scale by a scalar, per record."""

        rng = np.random.default_rng(13)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            x = exact_array(algebra, rng, (3,))
            records = records_of(x)
            with self.subTest(algebra=algebra):
                self.assertRecordsEqual(x + 2, [r + 2 for r in records])
                self.assertRecordsEqual(2 + x, [2 + r for r in records])
                self.assertRecordsEqual(x - 2, [r - 2 for r in records])
                self.assertRecordsEqual(2 - x, [2 - r for r in records])
                self.assertRecordsEqual(x * 2, [r * 2 for r in records])
                self.assertRecordsEqual(2 * x, [2 * r for r in records])
                self.assertRecordsEqual(-x, [-r for r in records])
                self.assertRecordsEqual((x * 2) / 2, records)
                self.assertRecordsEqual(np.int64(2) * x, [2 * r for r in records])
                with self.assertRaises(ZeroDivisionError):
                    x / 0
                with self.assertRaises(TypeError):
                    x / x
                with self.assertRaises(TypeError):
                    2 / x
                if algebra.field is not None:
                    scalar = (
                        algebra.field(5) if algebra.deg == 1 else algebra.field([1, 2])
                    )
                    self.assertRecordsEqual(x * scalar, [r * scalar for r in records])
                    self.assertRecordsEqual(scalar + x, [scalar + r for r in records])
                    self.assertRecordsEqual(x / scalar, [r / scalar for r in records])

    def test_empty_and_scalar_batches(self) -> None:
        """Support batch shapes with no records and with no axes."""

        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            empty = algebra.array(np.zeros((0,) + algebra.coefficient_shape, np.int64))
            x = algebra.random(0)
            with self.subTest(algebra=algebra):
                self.assertEqual(len(empty), 0)
                self.assertEqual(list(empty), [])
                for result in (
                    empty * empty,
                    empty + empty,
                    empty * x,
                    x * empty,
                    empty + 1,
                    2 * empty,
                    -empty,
                    empty**3,
                    empty.conjugate(),
                    empty.square(),
                    empty.commutator(x),
                    empty.inverse(),
                    empty.left_power(3),
                    empty.split().to_loop(),
                    empty[1:],
                ):
                    self.assertIsInstance(result, cdl.ElementArray)
                    self.assertEqual(result.batch_shape, (0,))
                self.assertEqual((empty == empty).shape, (0,))
                self.assertEqual(empty.weight().shape, (0,))
                self.assertEqual(empty.has_inverse().shape, (0,))

                single = algebra.array(x)
                self.assertEqual(single.batch_shape, ())
                product = single * single
                self.assertIsInstance(product, cdl.ElementArray)
                self.assertEqual(product.batch_shape, ())
                self.assertTrue(product[()] == x * x)
                self.assertTrue(bool(single == x))
                self.assertEqual(single.weight(), x.weight())
                self.assertTrue((single + x)[()] == x + x)
                self.assertTrue((single**2)[()] == x * x)


class RecordwiseTests(ElementArrayTestCase):
    def test_products_and_methods_match_elements(self) -> None:
        """Apply every operation record by record."""

        rng = np.random.default_rng(21)
        loops = (cdl.CDLoop(1), cdl.CDLoop(3), cdl.CDLoop(4), cdl.CentralProduct(1, 2))
        for loop in loops:
            for selectors in SELECTORS:
                algebra = cdl.LoopAlgebra(loop, **selectors)
                x = exact_array(algebra, rng, (3,))
                y = exact_array(algebra, rng, (3,))
                z = exact_array(algebra, rng, (3,))
                xs, ys, zs = records_of(x), records_of(y), records_of(z)
                pairs = list(zip(xs, ys, strict=True))
                with self.subTest(algebra=algebra):
                    self.assertRecordsEqual(x * y, [a * b for a, b in pairs])
                    self.assertRecordsEqual(x + y, [a + b for a, b in pairs])
                    self.assertRecordsEqual(x - y, [a - b for a, b in pairs])
                    self.assertRecordsEqual(x.conjugate(), [a.conjugate() for a in xs])
                    self.assertRecordsEqual(x.square(), [a * a for a in xs])
                    self.assertRecordsEqual(
                        x.commutator(y), [a.commutator(b) for a, b in pairs]
                    )
                    self.assertRecordsEqual(
                        x.anticommutator(y), [a.anticommutator(b) for a, b in pairs]
                    )
                    self.assertRecordsEqual(
                        x.jordan_product(y), [a.jordan_product(b) for a, b in pairs]
                    )
                    self.assertRecordsEqual(
                        x.associator(y, z),
                        [
                            a.associator(b, c)
                            for a, b, c in zip(xs, ys, zs, strict=True)
                        ],
                    )
                    self.assertRecordsEqual(
                        x.left_power(3), [a.left_power(3) for a in xs]
                    )
                    self.assertRecordsEqual(
                        x.right_power(3), [a.right_power(3) for a in xs]
                    )
                    self.assertRecordsEqual(x.left_power(0), [algebra.one()] * 3)
                    self.assertRecordsEqual(x.left_power(1), xs)
                    self.assertRecordsEqual(
                        algebra.left_operator_commutator(x, y, z),
                        [
                            algebra.left_operator_commutator(a, b, c)
                            for a, b, c in zip(xs, ys, zs, strict=True)
                        ],
                    )

    def test_powers_match_elements(self) -> None:
        """Raise every record to the same integer power."""

        rng = np.random.default_rng(22)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x = exact_array(algebra, rng, (2, 2))
            records = records_of(x)
            with self.subTest(algebra=algebra):
                for exponent in (0, 1, 2, 3, 5):
                    self.assertRecordsEqual(x**exponent, [r**exponent for r in records])
                if algebra.field is not None:
                    big = 10**20 + 3
                    self.assertRecordsEqual(x**big, [r**big for r in records])
                for exponent in (1.5, True, "2", None):
                    with self.assertRaises(TypeError):
                        x**exponent
        product = cdl.LoopAlgebra(cdl.CentralProduct(1, 2)).random(0, batch_shape=(2,))
        with self.assertRaises(TypeError):
            product**2

    def test_inverse_matches_elements(self) -> None:
        """Invert every record, exactly over finite fields."""

        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x = invertible_array(algebra, 23, 4)
            records = records_of(x)
            with self.subTest(algebra=algebra):
                self.assertTrue(x.has_inverse().all())
                inverse = x.inverse()
                self.assertIsInstance(inverse, cdl.ElementArray)
                if algebra.field is not None:
                    self.assertRecordsEqual(inverse, [r.inverse() for r in records])
                    self.assertTrue((x * inverse == 1).all())
                    self.assertTrue((inverse * x == 1).all())
                    self.assertRecordsEqual(x**-2, [r**-2 for r in records])
                else:
                    expected = np.stack([r.inverse().coefficients for r in records])
                    self.assertCoefficientsClose(inverse.coefficients, expected)
                    identity = np.broadcast_to(
                        algebra.one().coefficients, x.coefficients.shape
                    )
                    self.assertCoefficientsClose((x * inverse).coefficients, identity)

    def test_inverse_checks_every_record_first(self) -> None:
        """Raise without a partial result when one record has no inverse."""

        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            good = invertible_array(algebra, 24, 2)
            minus = algebra.central_idempotents()[1]
            for bad in (algebra.zero(), minus, algebra.primitive_idempotent(1)):
                values = np.stack(
                    [good.coefficients[0], bad.coefficients, good.coefficients[1]]
                )
                x = algebra.array(values)
                with self.subTest(algebra=algebra):
                    self.assertCoefficientsEqual(x.has_inverse(), [True, False, True])
                    with self.assertRaises(ZeroDivisionError):
                        x.inverse()
                    with self.assertRaises(ZeroDivisionError):
                        x**-1
                    self.assertCoefficientsEqual(x.coefficients, values)

    def test_predicates_return_boolean_arrays(self) -> None:
        """Evaluate each predicate per record."""

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
                algebra(exact_values(algebra, np.random.default_rng(25))),
            ]
            x = algebra.array(np.stack([e.coefficients for e in elements]))
            with self.subTest(algebra=algebra):
                for name in (
                    "has_inverse",
                    "is_idempotent",
                    "is_nilpotent",
                    "is_central",
                    "is_zero_divisor",
                ):
                    marks = getattr(x, name)()
                    self.assertIsInstance(marks, np.ndarray)
                    self.assertEqual(marks.dtype, np.bool_)
                    self.assertEqual(marks.shape, (7,))
                    expected = [getattr(e, name)() for e in elements]
                    self.assertCoefficientsEqual(marks, expected)
                for side in ("left", "right"):
                    self.assertCoefficientsEqual(
                        x.is_zero_divisor(side),
                        [e.is_zero_divisor(side) for e in elements],
                    )
                grid = x[:6].coefficients.reshape((2, 3) + algebra.coefficient_shape)
                self.assertEqual(algebra.array(grid).is_central().shape, (2, 3))

    def test_zero_divisors_by_rank_per_record(self) -> None:
        """Decide zero divisors at level 4 for each record separately."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(4))
        signed = np.zeros(16)
        signed[[1, 10]] = 1.0
        plus = algebra.central_idempotents()[0]
        elements = [
            plus + algebra.lift_signed(signed),
            algebra.random(26),
            algebra.one(),
        ]
        x = algebra.array(np.stack([e.coefficients for e in elements]))
        self.assertCoefficientsEqual(x.is_zero_divisor(), [True, False, False])
        self.assertCoefficientsEqual(x.is_zero_divisor("left"), [True, False, False])

    def test_weight_and_distance_count_per_record(self) -> None:
        """Count nonzero and differing coefficients along the batch."""

        rng = np.random.default_rng(27)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            x = exact_array(algebra, rng, (2, 3))
            y = exact_array(algebra, rng, (2, 3))
            xs, ys = records_of(x), records_of(y)
            with self.subTest(algebra=algebra):
                weight = x.weight()
                self.assertEqual(weight.shape, (2, 3))
                self.assertCoefficientsEqual(weight.ravel(), [r.weight() for r in xs])
                distance = x.distance(y)
                self.assertEqual(distance.shape, (2, 3))
                self.assertCoefficientsEqual(
                    distance.ravel(),
                    [a.distance(b) for a, b in zip(xs, ys, strict=True)],
                )
                self.assertFalse(np.any(x.distance(x)))
                self.assertCoefficientsEqual(x.distance(algebra.zero()), weight)

    def test_components_match_elements(self) -> None:
        """Return group components, spectra and signed components per record."""

        rng = np.random.default_rng(28)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
            x = exact_array(algebra, rng, (2, 2))
            records = records_of(x)
            tail = algebra.coefficient_shape[2:]
            with self.subTest(algebra=algebra):
                for name in ("kappa", "spectrum", "epsilon"):
                    component = getattr(x, name)()
                    stacked = np.stack([plain(getattr(r, name)()) for r in records])
                    self.assertCoefficientsEqual(
                        plain(component).reshape(stacked.shape), stacked
                    )
                    self.assertEqual(plain(component).shape, (2, 2, 8) + tail)
                signed = x.epsilon()
                self.assertIsInstance(signed, fcd.ElementArray)
                self.assertIs(signed.algebra, algebra.cd_algebra)
                self.assertEqual(signed.batch_shape, (2, 2))
                if algebra.field is None:
                    self.assertIsInstance(x.kappa(), np.ndarray)
                    self.assertIsInstance(x.spectrum(), np.ndarray)
                else:
                    self.assertIsInstance(x.kappa(), fcd.ElementArray)
                    self.assertIs(x.spectrum().algebra, algebra.field)
                rebuilt = algebra.from_components(x.kappa(), x.epsilon())
                self.assertCoefficientsEqual(rebuilt.coefficients, x.coefficients)

    def test_components_of_a_central_product(self) -> None:
        """Return flat signed components for a central product."""

        algebra = cdl.LoopAlgebra(cdl.CentralProduct(1, 2))
        x = exact_array(algebra, np.random.default_rng(29), (3,))
        signed = x.epsilon()
        self.assertIsInstance(signed, np.ndarray)
        self.assertEqual(signed.shape, (3, 8))
        self.assertCoefficientsEqual(signed[1], x[1].epsilon().reshape(8))
        self.assertCoefficientsEqual(
            signed, x.coefficients[:, 0] - x.coefficients[:, 1]
        )

    def test_element_only_methods_are_absent(self) -> None:
        """Offer per-element descriptions on elements only."""

        x = cdl.LoopAlgebra(cdl.CDLoop(2)).random(0, batch_shape=(2,))
        for name in (
            "terms",
            "support",
            "coefficient",
            "minimal_polynomial",
            "left_annihilator",
            "right_annihilator",
            "augmentation",
            "to_bytes",
        ):
            with self.subTest(method=name):
                self.assertFalse(hasattr(x, name))


class ComparisonTests(ElementArrayTestCase):
    def test_equality_returns_boolean_arrays(self) -> None:
        """Compare records with arrays, elements and scalars."""

        rng = np.random.default_rng(31)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            values = exact_values(algebra, rng, (3,))
            x = algebra.array(values)
            y = x.copy()
            y[1] = y[1] + 1
            with self.subTest(algebra=algebra):
                equal = x == y
                self.assertIsInstance(equal, np.ndarray)
                self.assertEqual(equal.dtype, np.bool_)
                self.assertCoefficientsEqual(equal, [True, False, True])
                self.assertCoefficientsEqual(x != y, [False, True, False])
                self.assertCoefficientsEqual(x == x[2], [False, False, True])
                self.assertCoefficientsEqual(x[2] == x, [False, False, True])
                self.assertCoefficientsEqual(x != x[2], [True, True, False])
                scalars = algebra.array(
                    np.stack(
                        [
                            algebra.scalar(2).coefficients,
                            algebra.one().coefficients,
                            values[0],
                        ]
                    )
                )
                self.assertCoefficientsEqual(scalars == 2, [True, False, False])
                self.assertCoefficientsEqual(2 == scalars, [True, False, False])
                self.assertCoefficientsEqual(scalars != 2, [False, True, True])

    def test_finite_field_integers_are_not_reduced_for_comparison(self) -> None:
        """Compare with integers outside the residue range as unequal."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=13)
        x = algebra.array(
            np.stack([algebra.scalar(2).coefficients, algebra.one().coefficients])
        )
        self.assertCoefficientsEqual(x == 2, [True, False])
        self.assertCoefficientsEqual(x == 15, [False, False])
        self.assertCoefficientsEqual(x == -11, [False, False])
        self.assertCoefficientsEqual(x == algebra.field(15), [True, False])
        self.assertCoefficientsEqual(x != 15, [True, True])

    def test_different_algebras_compare_unequal_everywhere(self) -> None:
        """Return an all-false array of the broadcast batch shape."""

        loop = cdl.CDLoop(2)
        x = cdl.LoopAlgebra(loop).array(np.zeros((2, 3, 2, 4)))
        other = cdl.LoopAlgebra(loop, char=13)
        for partner, shape in (
            (other.zero(), (2, 3)),
            (other.array(np.zeros((3, 2, 4), dtype=np.int64)), (2, 3)),
            (other.array(np.zeros((4, 1, 1, 2, 4), dtype=np.int64)), (4, 2, 3)),
            (cdl.LoopAlgebra(cdl.CDLoop(3)).zero(), (2, 3)),
        ):
            with self.subTest(partner=repr(partner)):
                equal = x == partner
                self.assertEqual(equal.shape, shape)
                self.assertFalse(equal.any())
                self.assertTrue((x != partner).all())
        mismatched = other.array(np.zeros((5, 2, 4), dtype=np.int64))
        with self.assertRaises(ValueError):
            x.__eq__(mismatched)

    def test_unrelated_objects_and_split_arrays(self) -> None:
        """Compare unequal with other objects and reject split arrays."""

        x = cdl.LoopAlgebra(cdl.CDLoop(2)).array(np.zeros((3, 2, 4)))
        for other in ("a", None, [0.0], object()):
            with self.subTest(other=repr(other)):
                self.assertIs(x == other, False)
                self.assertIs(x != other, True)
        with self.assertRaises(TypeError):
            x.__eq__(x.split())
        with self.assertRaises(TypeError):
            x.__ne__(x.split())

    def test_arrays_have_no_truth_value_or_hash(self) -> None:
        """Raise for ``bool()`` and ``hash()``."""

        x = cdl.LoopAlgebra(cdl.CDLoop(2)).array(np.zeros((3, 2, 4)))
        with self.assertRaises(ValueError):
            bool(x)
        with self.assertRaises(ValueError):
            bool(x[:1])
        with self.assertRaises(TypeError):
            hash(x)
        with self.assertRaises(TypeError):
            set((x,))


class NumPyTests(ElementArrayTestCase):
    def test_numpy_scalars_on_the_left_use_reflected_operators(self) -> None:
        """Return arrays of elements, not coefficient arrays."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        x = algebra.random(0, batch_shape=(3,))
        for scalar in (np.float64(2), np.int32(2), np.array(2.0)):
            with self.subTest(scalar=repr(scalar)):
                for result, expected in (
                    (scalar * x, x * 2),
                    (scalar + x, x + 2),
                    (scalar - x, 2 - x),
                ):
                    self.assertIsInstance(result, cdl.ElementArray)
                    self.assertCoefficientsEqual(
                        result.coefficients, expected.coefficients
                    )

    def test_ndarray_operands_and_ufuncs_are_rejected(self) -> None:
        """Raise ``TypeError`` instead of broadcasting over coefficient storage."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        x = algebra.random(0, batch_shape=(3,))
        values = np.ones((3, 2, 4))
        for operation in (
            lambda: x * values,
            lambda: values * x,
            lambda: x + values,
            lambda: values - x,
            lambda: x + np.ones(3),
            lambda: np.add(x, x),
            lambda: np.multiply(x, 2.0),
            lambda: np.negative(x),
            lambda: x + [1.0, 2.0, 3.0],
            lambda: x * "2",
            lambda: x + None,
        ):
            with self.assertRaises(TypeError):
                operation()

    def test_numpy_functions_read_the_coefficients(self) -> None:
        """Expose storage to functions that are not ufuncs."""

        algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
        values = np.arange(24.0).reshape(3, 2, 4)
        x = algebra.array(values)
        self.assertCoefficientsEqual(np.asarray(x), values)
        self.assertTrue(np.shares_memory(np.asarray(x), x.coefficients))
        stacked = np.stack([x[0], x[1]])
        self.assertIsInstance(stacked, np.ndarray)
        self.assertCoefficientsEqual(stacked, values[:2])
        self.assertEqual(np.shape(x), (3, 2, 4))

    def test_out_parameter_of_multiply(self) -> None:
        """Store a broadcast product in caller-provided storage."""

        rng = np.random.default_rng(41)
        for selectors in SELECTORS:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
            x = exact_array(algebra, rng, (3, 1))
            y = exact_array(algebra, rng, (2,))
            expected = x * y
            storage = np.empty((3, 2) + algebra.coefficient_shape, dtype=algebra.dtype)
            with self.subTest(algebra=algebra):
                result = algebra.multiply(x, y, out=storage)
                self.assertTrue(np.shares_memory(result.coefficients, storage))
                self.assertCoefficientsEqual(storage, expected.coefficients)
                target = algebra.array(np.zeros(storage.shape, dtype=np.int64))
                self.assertIs(algebra.multiply(x, y, out=target), target)
                self.assertTrue((target == expected).all())
                with self.assertRaises(ValueError):
                    algebra.multiply(x, y, out=storage[:2])


if __name__ == "__main__":
    unittest.main()
