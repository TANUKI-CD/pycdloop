"""Tests of Cayley--Dickson loops, central products and loop elements."""

from __future__ import annotations

import copy
import itertools
import pickle
import unittest
from typing import Any

import numpy as np
import pytest

import pycdloop as cdl

try:
    from . import reference
except ImportError:  # Support collection from inside the tests directory.
    import reference

SMALL_LEVELS = [
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (1, 1),
    (1, 2),
    (2, 2),
    (1, 3),
    (2, 3),
    (1, 1, 2),
]
# Loops of order at most 32, small enough for set-based definitions
TINY_LEVELS = [levels for levels in SMALL_LEVELS if sum(levels) <= 4]


def _loop(levels: tuple[int, ...]) -> Any:
    """Return the loop with the given factor levels."""

    return cdl.CDLoop(levels[0]) if len(levels) == 1 else cdl.CentralProduct(*levels)


def _reference_table(levels: tuple[int, ...]) -> np.ndarray:
    """Return the Cayley table computed from the reference loop product."""

    size = 1 << sum(levels)
    signed = [(1, t) for t in range(size)] + [(-1, t) for t in range(size)]
    table = np.empty((2 * size, 2 * size), dtype=np.int64)
    for i, x in enumerate(signed):
        for j, y in enumerate(signed):
            sign, index = reference.loop_product(levels, x, y)
            table[i, j] = index if sign > 0 else index + size
    return table


def _closure(table: np.ndarray, positions: set[int]) -> frozenset[int]:
    """Return the closure of a set of positions under the product."""

    closed = set(positions) | {0}
    while True:
        new = {int(table[i, j]) for i in closed for j in closed} - closed
        if not new:
            return frozenset(closed)
        closed |= new


def _positions(elements: Any) -> frozenset[int]:
    """Return the positions of a collection of loop elements."""

    return frozenset(element.position for element in elements)


def _brute_normal(table: np.ndarray, subloop: frozenset[int]) -> bool:
    """Check the three defining conditions of a normal subloop on sets."""

    order = table.shape[0]
    members = sorted(subloop)
    left = [frozenset(int(table[x, s]) for s in members) for x in range(order)]
    right = [frozenset(int(table[s, x]) for s in members) for x in range(order)]
    if left != right:
        return False
    for x in range(order):
        for y in range(order):
            xy = int(table[x, y])
            if frozenset(int(table[x, t]) for t in left[y]) != left[xy]:
                return False
            if frozenset(int(table[t, y]) for t in right[x]) != right[xy]:
                return False
    return True


class LoopProtocolTests(unittest.TestCase):
    def test_loops_are_cached_and_immutable(self) -> None:
        """Equal constructor arguments give the same immutable object."""

        self.assertIs(cdl.CDLoop(3), cdl.CDLoop(3))
        self.assertIs(cdl.CDLoop(np.int64(3)), cdl.CDLoop(3))
        self.assertIs(cdl.CentralProduct(1, 2), cdl.CentralProduct(1, 2))
        self.assertIsNot(cdl.CentralProduct(1, 2), cdl.CentralProduct(2, 1))
        for loop in (cdl.CDLoop(2), cdl.CentralProduct(1, 2)):
            with self.subTest(loop=loop):
                with self.assertRaises(AttributeError):
                    loop.level = 5
                with self.assertRaises(AttributeError):
                    del loop.levels
                with self.assertRaises(AttributeError):
                    loop.extra = 1

    def test_constructor_validation(self) -> None:
        """Invalid levels raise the documented exceptions."""

        for value, error in (
            (-1, ValueError),
            (True, TypeError),
            (1.0, TypeError),
            ("2", TypeError),
            (63, OverflowError),
        ):
            with self.subTest(value=value), self.assertRaises(error):
                cdl.CDLoop(value)
        with self.assertRaises(TypeError):
            cdl.CentralProduct(3)
        with self.assertRaises(TypeError):
            cdl.CentralProduct()
        with self.assertRaises(ValueError):
            cdl.CentralProduct(0, 2)
        with self.assertRaises(TypeError):
            cdl.CentralProduct(1, False)
        with self.assertRaises(OverflowError):
            cdl.CentralProduct(40, 40)

    def test_sizes_and_repr(self) -> None:
        """Levels, orders and constructor expressions agree."""

        for levels in SMALL_LEVELS:
            loop = _loop(levels)
            with self.subTest(levels=levels):
                self.assertEqual(loop.levels, levels)
                self.assertEqual(loop.level, sum(levels))
                self.assertEqual(loop.quotient_order, 1 << sum(levels))
                self.assertEqual(loop.order, 2 << sum(levels))
                self.assertEqual(len(loop), loop.order)
                self.assertIs(eval(repr(loop), vars(cdl)), loop)
        self.assertEqual(repr(cdl.CDLoop(3)), "CDLoop(3)")
        self.assertEqual(repr(cdl.CentralProduct(2, 1, 3)), "CentralProduct(2, 1, 3)")

    def test_pickling_and_copying_preserve_identity(self) -> None:
        """Loops unpickle to the cached object and elements to equal ones."""

        for levels in ((3,), (1, 2)):
            loop = _loop(levels)
            with self.subTest(levels=levels):
                self.assertIs(pickle.loads(pickle.dumps(loop)), loop)
                element = loop(3, -1)
                restored = pickle.loads(pickle.dumps(element))
                self.assertEqual(restored, element)
                self.assertIs(restored.loop, loop)
                self.assertEqual(copy.deepcopy(element), element)

    def test_iteration_and_membership(self) -> None:
        """Iteration lists positive elements first, by index."""

        loop = cdl.CentralProduct(1, 2)
        elements = list(loop)
        self.assertEqual(len(elements), 16)
        for position, element in enumerate(elements):
            self.assertEqual(element.position, position)
            self.assertEqual(element.sign, 1 if position < 8 else -1)
            self.assertEqual(element.index, position % 8)
            self.assertIn(element, loop)
        self.assertNotIn(cdl.CDLoop(3)(1), loop)
        self.assertNotIn(1, loop)
        self.assertEqual(loop.one(), loop(0))
        self.assertEqual(loop.minus_one(), loop(0, -1))
        self.assertEqual(-loop.one(), loop.minus_one())

    def test_basis_and_index_conversion(self) -> None:
        """Flat indices concatenate the factor indices, first factor first."""

        loop = cdl.CentralProduct(2, 1, 3)
        for flat in range(loop.quotient_order):
            indices = loop.factor_indices(flat)
            self.assertEqual(indices, reference.factor_indices(loop.levels, flat))
            self.assertEqual(loop.flat_index(*indices), flat)
            self.assertEqual(loop.basis(*indices), loop(flat))
            self.assertEqual(loop(flat).indices, indices)
        self.assertEqual(loop.basis(5), loop(5))
        self.assertEqual(cdl.CDLoop(3).basis(6).indices, (6,))
        with self.assertRaises(TypeError):
            loop.basis(1, 1)
        with self.assertRaises(TypeError):
            loop.flat_index(1)
        with self.assertRaises(IndexError):
            loop.flat_index(4, 0, 0)
        with self.assertRaises(IndexError):
            loop.factor_indices(64)

    def test_element_construction_and_protocol(self) -> None:
        """Elements validate their arguments and are immutable and hashable."""

        loop = cdl.CDLoop(3)
        element = loop(5, -1)
        self.assertEqual((element.loop, element.index, element.sign), (loop, 5, -1))
        self.assertEqual(element, cdl.LoopElement(loop, 5, -1))
        self.assertEqual(hash(element), hash(cdl.LoopElement(loop, 5, -1)))
        self.assertEqual(len({loop(5), loop(5), loop(5, -1)}), 2)
        self.assertNotEqual(element, cdl.CDLoop(4)(5, -1))
        self.assertNotEqual(element, 5)
        self.assertIs(+element, element)
        self.assertEqual(repr(element), "-e_5")
        self.assertEqual(repr(loop(0)), "e_0")
        self.assertEqual(repr(cdl.CentralProduct(1, 2)(6, -1)), "-e_(1,2)")
        with self.assertRaises(AttributeError):
            element.sign = 1
        with self.assertRaises(AttributeError):
            del element.index
        for arguments, error in (
            ((8,), IndexError),
            ((-1,), IndexError),
            ((True,), TypeError),
            ((1.0,), TypeError),
            ((1, 0), ValueError),
            ((1, 2), ValueError),
            ((1, 1.0), TypeError),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(error):
                loop(*arguments)
        with self.assertRaises(TypeError):
            cdl.LoopElement("loop", 0)

    def test_random_elements_are_reproducible(self) -> None:
        """A seed determines the element and a generator advances."""

        loop = cdl.CentralProduct(2, 3)
        self.assertEqual(loop.random(5), loop.random(5))
        generator = np.random.default_rng(1)
        drawn = {loop.random(generator) for _ in range(200)}
        self.assertGreater(len(drawn), 40)
        self.assertTrue(all(element in loop for element in drawn))

    def test_operands_of_different_loops_are_rejected(self) -> None:
        """Elements combine only with elements of the same loop."""

        x, y = cdl.CDLoop(2)(1), cdl.CDLoop(3)(1)
        with self.assertRaises(ValueError):
            x * y
        with self.assertRaises(TypeError):
            x * 2
        with self.assertRaises(TypeError):
            2 * x
        with self.assertRaises(ValueError):
            x.commutator(y)
        with self.assertRaises(TypeError):
            x.commutator(1)
        with self.assertRaises(ValueError):
            x.associator(x, y)
        with self.assertRaises(ValueError):
            cdl.CDLoop(2).left_translation(y)
        with self.assertRaises(TypeError):
            x**0.5
        with self.assertRaises(TypeError):
            x**True


@pytest.mark.parametrize("levels", SMALL_LEVELS)
def test_cayley_and_twist_tables_match_the_reference(levels: tuple[int, ...]) -> None:
    """Both tables agree with the doubling formula evaluated exactly."""

    loop = _loop(levels)
    size = loop.quotient_order
    expected = np.array(
        [[reference.twist(levels, s, t) for t in range(size)] for s in range(size)]
    )
    table = loop.twist_table()
    assert table.dtype == np.int8 and np.array_equal(table, expected)
    assert np.array_equal(loop.cayley_table(), _reference_table(levels))


@pytest.mark.parametrize("levels", SMALL_LEVELS)
def test_sign_functions_on_scalars_and_arrays(levels: tuple[int, ...]) -> None:
    """Sign functions broadcast and return integers for integer arguments."""

    loop = _loop(levels)
    size = loop.quotient_order
    table = loop.twist_table().astype(np.int64)
    indices = np.arange(size)
    s, t, u = indices[:, None, None], indices[None, :, None], indices[None, None, :]
    assert np.array_equal(loop.twist(indices[:, None], indices[None, :]), table)
    commutators = loop.commutator_sign(indices[:, None], indices[None, :])
    assert np.array_equal(commutators, table * table.T)
    associators = loop.associator_sign(s, t, u)
    expected = table[s, t] * table[s ^ t, u] * table[t, u] * table[s, t ^ u]
    assert associators.shape == (size,) * 3 and np.array_equal(associators, expected)
    last = size - 1
    for value in (loop.twist(last, 0), loop.commutator_sign(last, last)):
        assert type(value) is int and value in (1, -1)
    assert type(loop.associator_sign(np.int64(last), 0, last)) is int
    assert loop.twist(np.array([0, last]), last).shape == (2,)
    assert loop.twist(np.array([], dtype=np.int64), 0).shape == (0,)


def test_sign_functions_reject_invalid_indices() -> None:
    """Out-of-range and non-integer indices raise."""

    loop = cdl.CDLoop(3)
    with pytest.raises(IndexError):
        loop.twist(8, 0)
    with pytest.raises(IndexError):
        loop.twist(np.array([0, -1]), 0)
    with pytest.raises(IndexError):
        loop.associator_sign(0, 0, np.array([[9]]))
    with pytest.raises(TypeError):
        loop.twist(1.0, 0)
    with pytest.raises(TypeError):
        loop.commutator_sign(np.array([0.0, 1.0]), 0)
    with pytest.raises(TypeError):
        loop.twist(True, 0)


def test_sign_functions_work_beyond_the_table_limit() -> None:
    """Large loops refuse full tables but evaluate signs on index arrays."""

    loop = cdl.CDLoop(13)
    with pytest.raises(ValueError):
        loop.cayley_table()
    with pytest.raises(ValueError):
        loop.twist_table()
    with pytest.raises(ValueError):
        loop.to_gap()
    big = cdl.CDLoop(40)
    p, q = (1 << 39) + 5, (1 << 38) + 3
    assert big.twist(p, q) == -big.twist(q, p)
    assert big.twist(p, p) == -1 and big.twist(0, q) == 1
    small = cdl.CDLoop(5)
    assert big.twist(np.arange(32)[:, None], np.arange(32)[None, :]).tolist() == (
        small.twist_table().tolist()
    )


@pytest.mark.parametrize("levels", SMALL_LEVELS)
def test_element_arithmetic_matches_the_table(levels: tuple[int, ...]) -> None:
    """Products, inverses, orders, powers and defects follow the Cayley table."""

    loop = _loop(levels)
    table = _reference_table(levels)
    elements = list(loop)
    one = loop.one()
    rng = np.random.default_rng(sum(levels))
    for x in elements:
        inverse = x.inverse()
        assert x * inverse == one and inverse * x == one
        power, order = x, 1
        while power != one:
            power, order = power * x, order + 1
        assert x.order() == order
        expected = one
        for exponent in range(0, 9):
            assert x**exponent == expected
            assert x ** (-exponent) == expected.inverse()
            expected = expected * x
        assert x ** (10**30 + 3) == x**3 and x ** -(10**30 + 1) == inverse
        assert (-x).index == x.index and (-x).sign == -x.sign
    pairs = rng.integers(0, loop.order, size=(300, 3))
    for i, j, k in pairs.tolist():
        x, y, z = elements[i], elements[j], elements[k]
        assert (x * y).position == table[i, j]
        commutator = x.commutator(y)
        assert commutator.index == 0 and x * y == (y * x) * commutator
        associator = x.associator(y, z)
        assert associator.index == 0 and (x * y) * z == (x * (y * z)) * associator


@pytest.mark.parametrize("levels", SMALL_LEVELS)
def test_commutant_nuclei_and_center(levels: tuple[int, ...]) -> None:
    """The substructure agrees with definitions evaluated on the whole table."""

    loop = _loop(levels)
    table = loop.cayley_table()
    order = loop.order
    everything = np.arange(order)
    commutant = {i for i in range(order) if (table[i] == table[:, i]).all()}
    y, z = everything[:, None], everything[None, :]
    left, middle, right = set(), set(), set()
    for i in range(order):
        if (table[table[i, y], z] == table[i, table[y, z]]).all():
            left.add(i)
        if (table[table[y, i], z] == table[y, table[i, z]]).all():
            middle.add(i)
        if (table[table[y, z], i] == table[y, table[z, i]]).all():
            right.add(i)
    assert _positions(loop.commutant()) == commutant
    assert _positions(loop.nucleus("left")) == left
    assert _positions(loop.nucleus("middle")) == middle
    assert _positions(loop.nucleus("right")) == right
    assert _positions(loop.nucleus()) == left & middle & right
    assert _positions(loop.center()) == commutant & left & middle & right
    assert loop.is_commutative() == (len(commutant) == order)
    assert loop.is_associative() == (len(left & middle & right) == order)
    with pytest.raises(ValueError):
        loop.nucleus("inner")
    with pytest.raises(TypeError):
        loop.nucleus(1)


def test_known_centers() -> None:
    """Centers of the standard examples have the expected sizes."""

    assert len(cdl.CDLoop(1).center()) == 4
    assert cdl.CDLoop(2).center() == {cdl.CDLoop(2).one(), cdl.CDLoop(2).minus_one()}
    assert len(cdl.CDLoop(2).nucleus()) == 8
    for level in (3, 4, 5, 6):
        loop = cdl.CDLoop(level)
        assert loop.nucleus() == {loop.one(), loop.minus_one()}
    assert len(cdl.CentralProduct(1, 1).center()) == 8
    product = cdl.CentralProduct(1, 3)
    assert product.center() == set(product.generate([product.basis(1, 0)]))


@pytest.mark.parametrize("levels", TINY_LEVELS)
def test_generated_subloops_match_brute_force(levels: tuple[int, ...]) -> None:
    """Generation, the subloop test, normality and cosets follow the definitions."""

    loop = _loop(levels)
    table = loop.cayley_table()
    elements = list(loop)
    rng = np.random.default_rng(17)
    families = [[x] for x in elements]
    for _ in range(12):
        chosen = rng.integers(0, loop.order, size=int(rng.integers(2, 4)))
        families.append([elements[i] for i in chosen.tolist()])
    for generators in families:
        generated = loop.generate(generators)
        expected = _closure(table, {x.position for x in generators})
        assert _positions(generated) == expected
        assert loop.is_subloop(generated)
        assert loop.is_normal(generated) == _brute_normal(table, expected)
        for side in ("left", "right"):
            cosets = loop.cosets(generated, side=side)
            brute = {
                frozenset(
                    int(table[x, s] if side == "left" else table[s, x])
                    for s in expected
                )
                for x in range(loop.order)
            }
            assert {_positions(coset) for coset in cosets} == brute
            if loop.is_normal(generated):
                assert sum(len(coset) for coset in cosets) == loop.order
    assert loop.generate([]) == {loop.one()}
    assert set(loop.generate(elements)) == set(elements)


def test_subloops_without_minus_one() -> None:
    """An element of order two outside the center generates a two-element subloop."""

    abelian = cdl.CentralProduct(1, 1)
    x = abelian.basis(1, 1)
    assert abelian.generate([x]) == {abelian.one(), x}
    assert abelian.is_normal([abelian.one(), x])
    assert len(abelian.cosets([abelian.one(), x])) == 4
    # e_1 (x) e_1 has order two in Q_1 * Q_2 but does not commute with e_0 (x) e_2
    loop = cdl.CentralProduct(1, 2)
    y = loop.basis(1, 1)
    subloop = loop.generate([y])
    assert subloop == {loop.one(), y}
    assert y * loop.basis(0, 2) != loop.basis(0, 2) * y
    assert not loop.is_normal(subloop)
    assert loop.is_normal(loop.generate([y, loop.minus_one()]))
    left, right = loop.cosets(subloop), loop.cosets(subloop, side="right")
    assert left != right and len(left) == 8


def test_subloop_predicates_reject_other_sets() -> None:
    """Sets that are not subloops are reported or rejected."""

    loop = cdl.CDLoop(3)
    assert not loop.is_subloop([])
    assert not loop.is_subloop([loop(1)])
    assert not loop.is_subloop([loop.one(), loop(1)])
    assert loop.is_subloop([loop.one(), loop.minus_one()])
    assert loop.is_subloop(iter(loop))
    with pytest.raises(ValueError):
        loop.is_normal([loop.one(), loop(1)])
    with pytest.raises(ValueError):
        loop.cosets([loop(1)])
    with pytest.raises(ValueError):
        loop.cosets([loop.one()], side="middle")
    with pytest.raises(ValueError):
        loop.is_subloop([cdl.CDLoop(2).one()])
    with pytest.raises(TypeError):
        loop.generate([0])


@pytest.mark.parametrize("levels", SMALL_LEVELS)
def test_translations_and_character_automorphisms(levels: tuple[int, ...]) -> None:
    """Translations follow the table and characters give automorphisms."""

    loop = _loop(levels)
    table = loop.cayley_table()
    order = loop.order
    for x in list(loop)[:: max(1, order // 8)]:
        assert np.array_equal(loop.left_translation(x), table[x.position])
        assert np.array_equal(loop.right_translation(x), table[:, x.position])
        assert sorted(loop.left_translation(x).tolist()) == list(range(order))
    for a in range(0, loop.quotient_order, max(1, loop.quotient_order // 8)):
        permutation = loop.character_automorphism(a)
        assert sorted(permutation.tolist()) == list(range(order))
        images = permutation[table]
        assert np.array_equal(images, table[permutation[:, None], permutation[None, :]])
        assert loop.is_automorphism(permutation)
        for t in range(loop.quotient_order):
            flipped = bin(a & t).count("1") % 2 == 1
            assert permutation[t] == (t + loop.quotient_order if flipped else t)
    with pytest.raises(IndexError):
        loop.character_automorphism(loop.quotient_order)


def test_is_automorphism_agrees_with_brute_force() -> None:
    """Callables and position arrays are checked on every pair of elements."""

    loop = cdl.CDLoop(2)
    table = loop.cayley_table()
    rng = np.random.default_rng(8)
    elements = list(loop)
    for _ in range(40):
        permutation = rng.permutation(loop.order)
        brute = np.array_equal(
            permutation[table], table[permutation[:, None], permutation[None, :]]
        )
        assert loop.is_automorphism(permutation) == brute
    x = loop(1)
    # Q_2 is a group, so conjugation by an element is an automorphism
    assert loop.is_automorphism(lambda y: (x * y) * x.inverse())
    assert loop.is_automorphism(lambda y: y)
    assert not loop.is_automorphism(lambda y: x * y)
    assert not loop.is_automorphism(lambda y: loop.one())
    assert not loop.is_automorphism(np.zeros(loop.order, dtype=np.int64))
    assert loop.is_automorphism([e.position for e in elements])
    octonion = cdl.CDLoop(3)
    octonion_table = octonion.cayley_table()
    cycle = {0: 0, 1: 2, 2: 3, 3: 1, 4: 4, 5: 6, 6: 7, 7: 5}

    def rotate(y: Any) -> Any:
        """Permute the basis indices by a fixed map."""

        return octonion(cycle[y.index], y.sign)

    images = np.array([rotate(y).position for y in octonion])
    brute = np.array_equal(
        images[octonion_table], octonion_table[images[:, None], images[None, :]]
    )
    assert octonion.is_automorphism(rotate) == brute
    with pytest.raises(ValueError):
        loop.is_automorphism(np.arange(4))
    with pytest.raises(ValueError):
        loop.is_automorphism(np.arange(loop.order) + 1)
    with pytest.raises(ValueError):
        loop.is_automorphism(np.arange(loop.order, dtype=np.float64))
    with pytest.raises(ValueError):
        loop.is_automorphism(lambda y: cdl.CDLoop(3)(0))


def test_embeddings_are_homomorphisms() -> None:
    """Smaller loops and factors embed compatibly with products."""

    for small, large in ((0, 3), (1, 1), (2, 4), (3, 5)):
        source, target = cdl.CDLoop(small), cdl.CDLoop(large)
        for x, y in itertools.product(source, repeat=2):
            assert target.embed(x * y) == target.embed(x) * target.embed(y)
        assert target.embed(source.minus_one()) == target.minus_one()
    with pytest.raises(ValueError):
        cdl.CDLoop(2).embed(cdl.CDLoop(3)(1))
    with pytest.raises(TypeError):
        cdl.CDLoop(2).embed(cdl.CentralProduct(1, 1)(1))
    with pytest.raises(TypeError):
        cdl.CDLoop(2).embed(1)
    loop = cdl.CentralProduct(2, 1, 2)
    images = []
    for number, level in enumerate(loop.levels):
        factor = cdl.CDLoop(level)
        for x, y in itertools.product(factor, repeat=2):
            image = loop.factor_embedding(number, x * y)
            left, right = (
                loop.factor_embedding(number, x),
                loop.factor_embedding(number, y),
            )
            assert image == left * right
        assert loop.factor_embedding(number, factor.minus_one()) == loop.minus_one()
        images.append([loop.factor_embedding(number, x) for x in factor])
    # Elements of different factors commute and their indices combine
    for x, y in itertools.product(images[0], images[2]):
        assert x * y == y * x
    assert loop.factor_embedding(0, cdl.CDLoop(2)(3)) == loop.basis(3, 0, 0)
    assert loop.factor_embedding(1, cdl.CDLoop(1)(1, -1)) == -loop.basis(0, 1, 0)
    with pytest.raises(ValueError):
        loop.factor_embedding(1, cdl.CDLoop(2)(1))
    with pytest.raises(IndexError):
        loop.factor_embedding(3, cdl.CDLoop(2)(1))
    with pytest.raises(ValueError):
        loop.factor_embedding(0, 1)


@pytest.mark.parametrize("levels", TINY_LEVELS)
def test_identity_tests_match_brute_force(levels: tuple[int, ...]) -> None:
    """The Moufang and diassociative tests agree with their definitions."""

    loop = _loop(levels)
    table = loop.cayley_table()
    order = loop.order
    everything = np.arange(order)
    x, y, z = (
        everything[:, None, None],
        everything[None, :, None],
        everything[None, None, :],
    )
    moufang = (table[table[x, y], table[z, x]] == table[table[x, table[y, z]], x]).all()
    assert loop.is_moufang() == bool(moufang)
    diassociative = True
    for i, j in itertools.combinations(range(order), 2):
        members = np.array(sorted(_closure(table, {i, j})))
        a, b, c = members[:, None, None], members[None, :, None], members[None, None, :]
        if not (table[table[a, b], c] == table[a, table[b, c]]).all():
            diassociative = False
            break
    assert loop.is_diassociative() == diassociative


def test_known_identities_of_standard_loops() -> None:
    """The first Cayley--Dickson loops are the familiar groups and loops."""

    cyclic = cdl.CDLoop(1)
    assert cyclic.is_commutative() and cyclic.is_associative()
    assert sorted(x.order() for x in cyclic) == [1, 2, 4, 4]
    assert len(cyclic.generate([cyclic(1)])) == 4
    quaternion = cdl.CDLoop(2)
    assert quaternion.is_associative() and not quaternion.is_commutative()
    assert sorted(x.order() for x in quaternion) == [1, 2, 4, 4, 4, 4, 4, 4]
    octonion = cdl.CDLoop(3)
    assert octonion.is_moufang() and not octonion.is_associative()
    assert octonion.is_diassociative()
    sedenion = cdl.CDLoop(4)
    assert not sedenion.is_moufang() and sedenion.is_diassociative()
    assert not sedenion.is_moufang(samples=20000)
    abelian = cdl.CentralProduct(1, 1)
    assert abelian.is_commutative() and abelian.is_associative()
    assert sorted(x.order() for x in abelian) == [1, 2, 2, 2, 4, 4, 4, 4]
    assert cdl.CentralProduct(2, 2).is_associative()
    assert cdl.CentralProduct(2, 3).is_moufang()
    assert not cdl.CentralProduct(2, 3).is_associative()
    assert not cdl.CentralProduct(1, 4).is_moufang()


def test_sampled_identity_tests_on_large_loops() -> None:
    """Loops too large for an exhaustive test are sampled."""

    loop = cdl.CDLoop(9)
    assert not loop.is_moufang()
    assert loop.is_diassociative()
    assert loop.is_diassociative(samples=1000)
    assert not loop.is_commutative() and not loop.is_associative()
    assert len(loop.center()) == 2
    with pytest.raises(ValueError):
        loop.is_moufang(samples=0)
    with pytest.raises(TypeError):
        loop.is_moufang(samples=1.5)


@pytest.mark.parametrize("levels", [(0,), (2,), (1, 1), (1, 2)])
def test_gap_export(levels: tuple[int, ...]) -> None:
    """The GAP expression lists the Cayley table with elements numbered from 1."""

    loop = _loop(levels)
    text = loop.to_gap()
    assert text.startswith("LoopByCayleyTable([\n") and text.endswith("\n]);")
    rows = [line.strip().rstrip(",") for line in text.splitlines()[1:-1]]
    assert len(rows) == loop.order
    parsed = np.array([eval(row) for row in rows])
    assert np.array_equal(parsed, loop.cayley_table() + 1)
    assert " " not in "".join(rows)
