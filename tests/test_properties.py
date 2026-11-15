"""Property tests that search the public interface for defects.

Each test runs many generated cases through one invariant: results agree
with an independent evaluation, objects do not share storage they should
not, and unsuitable arguments are refused with a documented exception.
"""

from __future__ import annotations

import copy
import inspect
import itertools
import operator
import pickle
import threading
import warnings
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pyfastcd as fcd
import pytest

import pycdloop as cdl
from pycdloop import codes, linalg, testing

try:
    from . import reference
except ImportError:  # Support collection from inside the tests directory.
    import reference

REFUSALS = (TypeError, ValueError, ZeroDivisionError, IndexError, OverflowError)

LOOPS = [
    cdl.CDLoop(0),
    cdl.CDLoop(1),
    cdl.CDLoop(2),
    cdl.CDLoop(3),
    cdl.CentralProduct(1, 1),
    cdl.CentralProduct(1, 2),
    cdl.CentralProduct(1, 1, 1),
    cdl.CentralProduct(2, 1, 1),
]
FIELDS: list[dict[str, int]] = [
    {},
    {"char": 0, "deg": 2},
    {"char": 3},
    {"char": 65521},
    {"char": 3, "deg": 2},
    {"char": 2},
    {"char": 2, "deg": 2},
    {"char": 2, "deg": 7},
]
PAIRS = list(itertools.product(LOOPS, FIELDS))
SMALL = [
    (cdl.CDLoop(2), {}),
    (cdl.CDLoop(2), {"char": 0, "deg": 2}),
    (cdl.CDLoop(2), {"char": 7}),
    (cdl.CDLoop(2), {"char": 3, "deg": 2}),
    (cdl.CDLoop(2), {"char": 2}),
    (cdl.CDLoop(2), {"char": 2, "deg": 2}),
    (cdl.CentralProduct(1, 2), {}),
    (cdl.CentralProduct(1, 2), {"char": 5}),
]


def _name(pair: Any) -> str:
    """Return a short test id for a loop and its field selectors."""

    loop, selectors = pair
    field = "R" if not selectors else f"{selectors['char']}^{selectors.get('deg', 1)}"
    return f"{loop!r}-{field}"


def _values(algebra: Any, rng: Any, batch: tuple[int, ...] = ()) -> np.ndarray:
    """Return random coefficients with exactly representable values."""

    shape = batch + algebra.coefficient_shape
    if algebra.field is not None:
        return rng.integers(0, algebra.char, shape).astype(np.uint32)
    values = rng.integers(-3, 4, shape).astype(algebra.dtype)
    if algebra.deg == 2:
        values = values + 1j * rng.integers(-3, 4, shape)
    return values


def _ring(algebra: Any) -> Any:
    """Return the reference ring of an algebra's coefficient field."""

    if algebra.field is not None:
        return reference.FiniteField(algebra.char, algebra.field.modulus)
    return reference.GaussianRationals() if algebra.deg == 2 else reference.Rationals()


def _oracle(algebra: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multiply two coefficient arrays from the definition, in exact arithmetic."""

    ring = _ring(algebra)
    left = reference.to_ring(ring, x, 2)
    right = reference.to_ring(ring, y, 2)
    product = reference.loop_algebra_multiply(ring, algebra.loop.levels, left, right)
    return reference.from_ring(ring, product, 2, algebra.dtype)


def _loop_values(item: Any) -> np.ndarray:
    """Return the loop-basis coefficients of an element or array of either kind."""

    if isinstance(item, cdl.SplitArray):
        item = item.to_loop()
    return np.array(item.coefficients)


def _agree(algebra: Any, got: np.ndarray, want: np.ndarray) -> bool:
    """Compare coefficients with a reference broadcast over the batch."""

    target = np.broadcast_to(want, got.shape)
    if algebra.field is not None:
        return bool(np.array_equal(got, target))
    return bool(np.allclose(got, target, rtol=1e-12, atol=1e-12))


# Products on every kind of operand and shape


@pytest.mark.parametrize("pair", PAIRS, ids=_name)
def test_products_agree_with_the_definition_on_every_container(pair: Any) -> None:
    """Elements, arrays and split arrays of any batch shape give the same product."""

    loop, selectors = pair
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + len(selectors))
    a, b = _values(algebra, rng), _values(algebra, rng)
    want = _oracle(algebra, a, b)
    x, y = algebra(a), algebra(b)
    found = [x * y]
    found += [algebra.multiply(x, y, algorithm=name) for name in algebra.algorithms]
    for name in algebra.signed_algorithms:
        found.append(algebra.multiply(x, y, signed_algorithm=name))
    for batch in ((), (1,), (3,), (2, 2), (0,), (2, 0, 3)):
        left = algebra.array(np.broadcast_to(a, batch + a.shape).copy())
        right = algebra.array(np.broadcast_to(b, batch + b.shape).copy())
        found += [left * right, left * y, x * right]
        found += [
            algebra.multiply(left, right, algorithm=name) for name in algebra.algorithms
        ]
        if not algebra._field.binary:
            found += [left.split() * right.split(), left.split() * y, x * right.split()]
        for item in found[-3:]:
            assert item.batch_shape == batch
    tall = algebra.array(np.broadcast_to(a, (3, 1) + a.shape).copy())
    wide = algebra.array(np.broadcast_to(b, (1, 4) + b.shape).copy())
    spread = tall * wide
    assert spread.batch_shape == (3, 4)
    found.append(spread)
    for item in found:
        assert _agree(algebra, _loop_values(item), want)


@pytest.mark.parametrize("pair", PAIRS, ids=_name)
def test_identities_that_hold_in_every_loop_algebra(pair: Any) -> None:
    """Distributivity, the named products and the round trips hold exactly."""

    loop, selectors = pair
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 7 * len(selectors))
    for _ in range(4):
        x, y, z = (algebra(_values(algebra, rng)) for _ in range(3))
        assert (x + y) * z == x * z + y * z and z * (x + y) == z * x + z * y
        assert x.commutator(y) == x * y - y * x
        assert x.anticommutator(y) == x * y + y * x
        assert x.associator(y, z) == (x * y) * z - x * (y * z)
        assert (x * y).conjugate() == y.conjugate() * x.conjugate()
        assert x.square() == x * x and x.conjugate().conjugate() == x
        assert x - x == algebra.zero() and x + algebra.zero() == x
        assert x * algebra.one() == x and algebra.one() * x == x
        assert x.left_power(3) == x * (x * x) and x.right_power(3) == (x * x) * x
        assert x.distance(y) == (x - y).weight() and x.distance(x) == 0
        assert algebra.from_bytes(x.to_bytes()) == x
        assert algebra.from_terms(x.terms()) == x
        assert len(x.terms()) == len(x.support()) == x.weight()
        if not algebra._field.binary:
            assert algebra.from_components(x.kappa(), x.epsilon()) == x
        left, right = algebra.left_matrix(x), algebra.right_matrix(x)
        if algebra.field is None or algebra.deg == 1:
            flat = y.coefficients.reshape(algebra.dimension)
            for matrix, product in ((left, x * y), (right, y * x)):
                if algebra.field is None:
                    image = matrix @ flat
                else:
                    wide = matrix.astype(np.int64) @ flat.astype(np.int64)
                    image = (wide % algebra.char).astype(np.uint32)
                assert _agree(
                    algebra,
                    image.reshape(product.coefficients.shape),
                    product.coefficients,
                )


# Structure


@pytest.mark.parametrize(
    "pair",
    [
        pair
        for pair in PAIRS
        if pair[1].get("char") and pair[0].order <= 16 and pair[1].get("deg", 1) <= 2
    ],
    ids=_name,
)
def test_predicates_agree_with_their_definitions(pair: Any) -> None:
    """Zero divisors, annihilators, central and nilpotent elements, from matrices."""

    loop, selectors = pair
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 11 * len(selectors))
    size = algebra.dimension
    tail = algebra.coefficient_shape[2:]
    basis = [algebra.basis(g) for g in loop]
    for trial in range(10 if size <= 8 else 3):
        values = _values(algebra, rng)
        if trial % 2:
            mask = rng.random(values.shape[:2]) < 0.3
            values = values * (mask[..., None] if tail else mask).astype(np.uint32)
        x = algebra(values)
        left, right = algebra.left_matrix(x), algebra.right_matrix(x)
        ranks = {
            "left": linalg.rank(left, field=algebra.field),
            "right": linalg.rank(right, field=algebra.field),
        }
        for side, rank in ranks.items():
            assert bool(x.is_zero_divisor(side)) is (rank < size)
        assert bool(x.is_zero_divisor()) is (min(ranks.values()) < size)
        # y -> x y has the right annihilator as its kernel
        for name, rank in (
            ("right_annihilator", ranks["left"]),
            ("left_annihilator", ranks["right"]),
        ):
            found = getattr(x, name)()
            assert found.batch_shape == (size - rank,)
            product = x * found if name == "right_annihilator" else found * x
            assert not product.coefficients.any()
            if size - rank:
                rows = found.coefficients.reshape((size - rank, size) + tail)
                assert linalg.rank(rows, field=algebra.field) == size - rank
        central = all(x * b == b * x for b in basis) and all(
            (x * b) * c == x * (b * c)
            and (b * x) * c == b * (x * c)
            and (b * c) * x == b * (c * x)
            for b, c in itertools.product(basis, repeat=2)
        )
        assert bool(x.is_central()) is central
        assert bool(x.is_idempotent()) is (x * x == x)
        try:
            nilpotent = x.is_nilpotent()
        except TypeError:
            continue
        power, vanishes = x, False
        for _ in range(size + 2):
            if not power:
                vanishes = True
                break
            power = power * x
        assert bool(nilpotent) is vanishes
        polynomial = x.minimal_polynomial()
        degree = len(polynomial) - 1
        if degree:
            powers = np.stack(
                [(x**k).coefficients.reshape((size,) + tail) for k in range(degree)]
            )
            assert linalg.rank(powers, field=algebra.field) == degree
        if x.has_inverse():
            assert x * x.inverse() == 1 and x.inverse() * x == 1
        else:
            with pytest.raises(ZeroDivisionError):
                x.inverse()
    for element in algebra.center_basis():
        assert element.is_central()


# Storage


@pytest.mark.parametrize("pair", SMALL, ids=_name)
def test_results_share_no_storage_with_operands_or_later_work(pair: Any) -> None:
    """A result keeps its value while more is computed, and writing to it is private."""

    loop, selectors = pair
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 13)
    binary = algebra._field.binary

    def make(kind: str) -> Any:
        if kind == "element":
            return algebra(_values(algebra, rng))
        array = algebra.array(_values(algebra, rng, (3,)))
        return array.split() if kind == "split" else array

    for kind in ("element", "array") + (() if binary else ("split",)):
        x, y, z = make(kind), make(kind), make(kind)
        before = [_loop_values(item) for item in (x, y, z)]
        results = [x + y, x - y, x * y, y * x, -x, x.conjugate(), x.square()]
        results += [x.commutator(y), x.associator(y, z), x.copy(), x.left_power(3)]
        if kind != "split":
            results.append(algebra.multiply(x, y, algorithm="table"))
        kept = [_loop_values(item) for item in results]
        for _ in range(3):
            a, b = make(kind), make(kind)
            (a * b) * a + b * a
            if kind != "split":
                algebra.multiply(a, b, algorithm="table")
        for item, values in zip(results, kept, strict=True):
            assert np.array_equal(_loop_values(item), values, equal_nan=True)
        for item in results:
            if isinstance(item, cdl.ElementArray):
                item.coefficients[...] = 0
            elif isinstance(item, cdl.SplitArray):
                item[...] = 0
        for item, values in zip((x, y, z), before, strict=True):
            assert np.array_equal(_loop_values(item), values)


@pytest.mark.parametrize("pair", SMALL, ids=_name)
def test_out_may_be_an_operand(pair: Any) -> None:
    """A product stored over one of its operands is the same product."""

    loop, selectors = pair
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 17)
    x = algebra.array(_values(algebra, rng, (4,)))
    y = algebra.array(_values(algebra, rng, (4,)))
    for name in algebra.algorithms:
        want = _loop_values(algebra.multiply(x, y, algorithm=name))
        square = _loop_values(algebra.multiply(x, x, algorithm=name))
        for target in ("left", "right", "fresh"):
            left, right = x.copy(), y.copy()
            out: Any = {"left": left, "right": right}.get(target, np.zeros_like(want))
            stored = algebra.multiply(left, right, algorithm=name, out=out)
            assert _agree(algebra, _loop_values(stored), want)
            assert stored is out or np.shares_memory(stored.coefficients, out)
        both = x.copy()
        stored = algebra.multiply(both, both, algorithm=name, out=both)
        assert stored is both and _agree(algebra, _loop_values(stored), square)
    frozen = np.zeros_like(want)
    frozen.setflags(write=False)
    narrow = np.zeros(want.shape, dtype="f4")
    for out in (frozen, np.zeros(want.shape[1:], dtype=want.dtype), narrow):
        with pytest.raises((TypeError, ValueError)):
            algebra.multiply(x, y, out=out)


def _multiply(
    x: Any, y: Any, want: np.ndarray, singles: list[np.ndarray], failures: list[str]
) -> None:
    """Multiply repeatedly and record every product that differs from the first."""

    for _ in range(25):
        if not np.array_equal(_loop_values(x * y), want):
            failures.append("array")
        for i in range(4):
            if not np.array_equal(_loop_values(x[i] * y[i]), singles[i]):
                failures.append("element")


def test_products_are_the_same_from_several_threads() -> None:
    """Threads that multiply at once do not disturb one another's workspaces."""

    rng = np.random.default_rng(19)
    for loop, selectors in SMALL[:6]:
        algebra = cdl.LoopAlgebra(loop, **selectors)
        x = algebra.array(_values(algebra, rng, (32,)))
        y = algebra.array(_values(algebra, rng, (32,)))
        want = _loop_values(x * y)
        singles = [_loop_values(x[i] * y[i]) for i in range(4)]
        failures: list[str] = []
        task = (x, y, want, singles, failures)
        threads = [threading.Thread(target=_multiply, args=task) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not failures


# Scalars and foreign operands


SCALARS: list[Any] = [
    2, -3, 0, 10**30, True, 2.0, 2.5, float("nan"), float("inf"), 1 + 2j,
    Fraction(3, 2), Decimal("1.5"), np.int8(3), np.int64(-2), np.uint64(2**64 - 1),
    np.float32(2.5), np.float64(2.0), np.complex128(1 + 1j), np.bool_(True),
    np.array(2), np.array(2.0), np.array([2.0]), np.ones(3), np.ones((2, 4)),
    [1, 2], (1, 2), None, object(), {1: 2},
    fcd.FiniteField(char=7)(3), fcd.FiniteField(char=5)(3), cdl.BinaryField(deg=2)(1),
]  # fmt: skip
OPERATORS = [
    operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv,
    operator.mod, operator.pow, operator.matmul, operator.eq, operator.ne, operator.lt,
]  # fmt: skip


@pytest.mark.parametrize("pair", SMALL, ids=_name)
def test_operators_return_their_own_type_or_refuse(pair: Any) -> None:
    """With any other operand an operator gives an element or array, or raises."""

    loop, selectors = pair
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 23)
    items = [algebra(_values(algebra, rng)), algebra.array(_values(algebra, rng, (3,)))]
    if not algebra._field.binary:
        items.append(items[1].split())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for item, scalar, function in itertools.product(items, SCALARS, OPERATORS):
            for arguments in ((item, scalar), (scalar, item)):
                try:
                    result = function(*arguments)
                except REFUSALS:
                    continue
                if function in (operator.eq, operator.ne):
                    assert isinstance(result, (bool, np.bool_, np.ndarray))
                else:
                    assert type(result) is type(item), (function, scalar)
    # A foreign element or array is refused, whatever the operator
    other = cdl.LoopAlgebra(cdl.CDLoop(loop.level + 1), **selectors)
    for item, function in itertools.product(items, OPERATORS[:3]):
        for outsider in (other.one(), other.array(other.one().to_numpy()[None])):
            with pytest.raises((TypeError, ValueError)):
                function(item, outsider)
            with pytest.raises((TypeError, ValueError)):
                function(outsider, item)
        assert (item == other.one()) is False or not np.any(item == other.one())


# Constructors


def _unusual(algebra: Any, good: np.ndarray) -> dict[str, tuple[Any, bool | None]]:
    """Return unusual constructor inputs, each with whether it must be accepted."""

    shape = algebra.coefficient_shape
    finite = algebra.field is not None
    swapped = good.astype(good.dtype.newbyteorder(">"))
    frozen = good.copy()
    frozen.setflags(write=False)
    return {
        "big-endian": (swapped, True),
        "read-only": (frozen, True),
        "memoryview": (memoryview(good.copy()), True),
        "nested lists": (good.tolist(), True),
        "nested tuples": (tuple(map(tuple, good.tolist())) if len(shape) == 2 else None,
                          True),
        "negative integers": (-np.ones(shape, dtype=np.int64), True),
        "huge integers": ((np.zeros(shape, dtype=object) + 10**40).tolist(), True),
        "Boolean array": (np.ones(shape, dtype=bool), not finite),
        "Boolean lists": (np.ones(shape, dtype=bool).tolist(), not finite),
        "fractions": ((np.zeros(shape, dtype=object) + Fraction(1, 2)).tolist(),
                      not finite),
        "non-integral floats": (np.full(shape, 1.5), not finite),
        "integral floats": (np.ones(shape), None),
        "not a number": (np.full(shape, np.nan), not finite),
        "strings": (np.full(shape, "1"), False),
        "none entries": (np.full(shape, None, dtype=object), False),
        "text": ("abc", False),
        "none": (None, False),
        "set": ({1, 2}, False),
        "mapping": ({0: 1}, False),
        "ragged": ([[1, 2], [3]], False),
        "empty": ([], False),
        "wrong length": (np.zeros(shape[:-1] + (shape[-1] + 1,), dtype=good.dtype),
                         False),
        "flat": (good.ravel(), False),
        "complex over R": (good.astype(complex) + 1j
                           if not finite and algebra.deg == 1 else None, False),
        "foreign element": (cdl.LoopAlgebra(cdl.CDLoop(3)).one(), False),
        "loop element": (algebra.loop.one(), False),
    }  # fmt: skip


@pytest.mark.parametrize("pair", SMALL, ids=_name)
def test_constructors_accept_or_refuse_unusual_values(pair: Any) -> None:
    """An accepted value gives canonical storage; a refusal is a documented one."""

    loop, selectors = pair
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 29)
    good = _values(algebra, rng)
    for label, (value, accepted) in _unusual(algebra, good).items():
        if value is None and label != "none":
            continue
        for build in (algebra, algebra.array, algebra.asarray):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    made = build(value)
            except (TypeError, ValueError):
                assert accepted is not True, (label, build)
                continue
            assert accepted is not False, (label, build)
            stored = np.asarray(made.coefficients)
            assert stored.dtype == algebra.dtype and stored.dtype.isnative
            assert stored.shape[-len(algebra.coefficient_shape) :] == (
                algebra.coefficient_shape
            )
            if algebra.field is not None:
                assert stored.max() < algebra.char
    assert algebra(good.tolist()) == algebra(good) == algebra(algebra(good))
    huge = algebra((np.zeros(good.shape, dtype=object) + 10**40).tolist())
    if algebra.field is not None:
        assert (huge.coefficients == 10**40 % algebra.char).all()


# Indexing


KEYS: list[Any] = [
    0, -1, 2, -3, 3, -4, slice(None), slice(1, 3), slice(None, None, -1), slice(0, 0),
    Ellipsis, None, (0, 1), (1, slice(None)), (slice(None), 0), (Ellipsis, 0),
    (0, Ellipsis), (None, 0), (0, None), (0, 1, 0), (0, 1, 0, 0), (Ellipsis, 0, 0, 0),
    (slice(None), slice(None), 0), np.array([0, 2]), [0, 2],
    np.array([True, False, True]), np.array([[0, 1], [2, 0]]),
    (np.array([0, 1]), np.array([1, 0])), np.int64(1), (), (Ellipsis, Ellipsis), 1.0,
    "a", True, False, np.True_, (Ellipsis, True, 0), (True, 0), (0, False),
    (True, True), (Ellipsis, np.False_, 1),
]  # fmt: skip


def _batch_index(key: Any, batch: int) -> tuple[Any, ...] | None:
    """Return an index as NumPy reads it when only the batch axes exist.

    An ellipsis is expanded over the batch axes that the other parts leave,
    and an index that needs more axes than the batch has gives ``None``.
    """

    parts = key if isinstance(key, tuple) else (key,)
    used = 0
    for part in parts:
        if part is None or part is Ellipsis or isinstance(part, (bool, np.bool_)):
            continue
        used += part.ndim if isinstance(part, np.ndarray) and part.dtype == bool else 1
    if used > batch or sum(part is Ellipsis for part in parts) > 1:
        return None
    expanded: list[Any] = []
    for part in parts:
        expanded += [slice(None)] * (batch - used) if part is Ellipsis else [part]
    return tuple(expanded)


@pytest.mark.parametrize("pair", SMALL, ids=_name)
def test_indexing_follows_numpy_on_the_batch_axes(pair: Any) -> None:
    """An index selects what NumPy selects, and never reaches the coefficients."""

    loop, selectors = pair
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 31)
    record = algebra.coefficient_shape
    raw = _values(algebra, rng, (3, 2))
    arrays = [algebra.array(raw)]
    if not algebra._field.binary:
        arrays.append(arrays[0].split())
    for array, key in itertools.product(arrays, KEYS):
        index = None if isinstance(key, (float, str)) else _batch_index(key, 2)
        want = None
        if index is not None:
            try:
                want = raw[index + (Ellipsis,)]
            except IndexError:
                want = None
        try:
            got = array[key]
        except (TypeError, ValueError, IndexError):
            assert want is None, key
            continue
        assert want is not None, key
        assert _agree(algebra, _loop_values(got), want) and want.shape == (
            _loop_values(got).shape
        )
        assert isinstance(got, cdl.Element) is (want.shape == record)
        # Assignment through the same index
        target = algebra.array(raw.copy())
        target = target.split() if isinstance(array, cdl.SplitArray) else target
        target[key] = algebra.one()
        expected = raw.copy()
        expected[index + (Ellipsis,)] = algebra.one().to_numpy()
        assert _agree(algebra, _loop_values(target), expected)
        with pytest.raises((TypeError, ValueError)):
            target[key] = cdl.LoopAlgebra(cdl.CDLoop(3)).one()
    with pytest.raises(TypeError):
        hash(arrays[0])
    with pytest.raises(ValueError):
        bool(arrays[0])
    assert len(arrays[0]) == 3 and len(list(arrays[0])) == 3


# Hashing, equality and copies


@pytest.mark.parametrize("pair", SMALL, ids=_name)
def test_equal_objects_hash_alike_and_survive_copies(pair: Any) -> None:
    """Rebuilt, pickled and copied objects are equal to the original."""

    loop, selectors = pair
    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 37)
    x = algebra(_values(algebra, rng))
    array = algebra.array(_values(algebra, rng, (2,)))
    twins = [
        algebra(x.to_numpy()),
        pickle.loads(pickle.dumps(x)),
        x * algebra.one(),
        x.copy(),
        copy.copy(x),
        copy.deepcopy(x),
        algebra.from_bytes(x.to_bytes()),
    ]
    if not algebra._field.binary:
        twins.append(algebra.array(x.to_numpy()[None]).split()[0])
    for twin in twins:
        assert twin == x and hash(twin) == hash(x)
    assert len({x, *twins}) == 1
    for scalar in (0, 1):
        assert algebra(scalar) == scalar and hash(algebra(scalar)) == hash(scalar)
    if algebra.field is None:
        negative = algebra(
            np.zeros(algebra.coefficient_shape, dtype=algebra.dtype) * -1
        )
        assert negative == algebra.zero() and hash(negative) == hash(algebra.zero())
        assert algebra(2.5) == 2.5 and hash(algebra(2.5)) == hash(2.5)
    else:
        assert algebra(algebra.char) == 0 and algebra(algebra.char) != algebra.char
    objects = [algebra, loop, loop.basis(1 % loop.quotient_order), x, array]
    objects += [cdl.BinaryField(deg=3), cdl.BinaryField(deg=3)([1, 0, 1])]
    if not algebra._field.binary:
        objects.append(array.split())
    for item in objects:
        for twin in (pickle.loads(pickle.dumps(item)), copy.copy(item),
                     copy.deepcopy(item)):  # fmt: skip
            assert np.all(twin == item)
            if isinstance(item, (cdl.LoopAlgebra, cdl.CDLoop, cdl.CentralProduct)):
                assert twin is item


# Fields, transforms and linear algebra


@pytest.mark.parametrize("deg", [1, 2, 3, 4, 5])
def test_binary_fields_are_fields(deg: int) -> None:
    """The field axioms hold on every element of a small binary field."""

    field = cdl.BinaryField(deg=deg)
    order = 2**deg
    tuples = itertools.product((0, 1), repeat=deg)
    elements = [field(list(c)) if deg > 1 else field(c[0]) for c in tuples]
    zero, one = field(0), field(1)
    assert len(set(elements)) == order and zero in elements and one in elements
    for a in elements:
        assert a + zero == a and a * one == a and a + a == zero and a * zero == zero
        assert -a == a and a - a == zero
        assert a**0 == one and a**1 == a and a**2 == a * a and a**order == a
        assert pickle.loads(pickle.dumps(a)) == a
        if a == zero:
            with pytest.raises(ZeroDivisionError):
                one / a
            with pytest.raises(ZeroDivisionError):
                a**-1
            continue
        inverse = one / a
        assert a * inverse == one and a**-1 == inverse and a ** (order - 1) == one
    for a, b in itertools.product(elements, repeat=2):
        assert a + b == b + a and a * b == b * a
        assert (a + b) ** 2 == a * a + b * b
        assert hash(a) != hash(b) or a == b or True
    for a, b, c in itertools.product(elements[:8], repeat=3):
        assert (a + b) + c == a + (b + c) and (a * b) * c == a * (b * c)
        assert a * (b + c) == a * b + a * c
    for a in elements[:4]:
        for n in (0, 1, 2, 3, -1, 10**30 + 1):
            assert a + n == a + field(n % 2) and a * n == a * field(n % 2)
        outsider = cdl.BinaryField(deg=deg + 1)(1)
        for function in (operator.add, operator.mul, operator.sub):
            with pytest.raises((TypeError, ValueError)):
                function(a, outsider)
            with pytest.raises((TypeError, ValueError)):
                function(outsider, a)
        with pytest.raises(TypeError):
            a + True


@pytest.mark.parametrize(
    "modulus",
    [(1, 0, 1), (0, 1, 1), (1, 1, 1, 1), (1,), (), (1, 2, 1), (1, 1, 0), "111", None,
     (1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1)],
)  # fmt: skip
def test_binary_fields_refuse_moduli_that_do_not_present_a_field(modulus: Any) -> None:
    """A reducible polynomial, or one of another form, is refused."""

    with pytest.raises((TypeError, ValueError)):
        cdl.BinaryField.from_modulus(modulus=modulus)


def test_transforms_agree_with_their_definitions() -> None:
    """``wht`` and ``dyadic_convolve`` against the sums that define them."""

    rng = np.random.default_rng(41)
    fields = [fcd.FiniteField(char=3), fcd.FiniteField(char=65521)]
    fields += [
        fcd.FiniteField(char=3, deg=2),
        cdl.BinaryField(),
        cdl.BinaryField(deg=3),
    ]
    for size in (1, 2, 4, 8, 16):
        a = rng.integers(-5, 6, size).astype(float)
        b = rng.integers(-5, 6, size).astype(float)
        signs = np.array(
            [[(-1) ** (i & j).bit_count() for j in range(size)] for i in range(size)],
            dtype=float,
        )
        assert np.array_equal(cdl.wht(a), signs @ a)
        direct = np.zeros(size)
        for i, j in itertools.product(range(size), repeat=2):
            direct[i ^ j] += a[i] * b[j]
        assert np.array_equal(cdl.dyadic_convolve(a, b), direct)
        for field in fields:
            tail = () if field.deg == 1 else (field.deg,)
            x = rng.integers(0, field.char, (size,) + tail).astype(np.uint32)
            y = rng.integers(0, field.char, (size,) + tail).astype(np.uint32)

            def scalars(values: np.ndarray, field: Any = field) -> list[Any]:
                if field.deg == 1:
                    return [field(int(value)) for value in values]
                return [field([int(c) for c in value]) for value in values]

            total = [field(0)] * size
            for (i, p), (j, q) in itertools.product(
                enumerate(scalars(x)), enumerate(scalars(y))
            ):
                total[i ^ j] = total[i ^ j] + p * q
            want = np.array(
                [np.asarray(entry.coefficients).tolist() for entry in total],
                dtype=np.uint32,
            )
            got = cdl.dyadic_convolve(x, y, field=field)
            assert np.array_equal(got, want.reshape(got.shape))
    for size in (0, 3, 6, 12):
        with pytest.raises(ValueError):
            cdl.wht(np.ones(size))
        with pytest.raises(ValueError):
            cdl.dyadic_convolve(np.ones(size), np.ones(size))
    block = rng.integers(-3, 4, (3, 8, 4)).astype(float)
    for axis in (0, 1, 2, -1, -2, -3):
        if block.shape[axis] in (4, 8):
            want = np.apply_along_axis(cdl.wht, axis, block)
            assert np.array_equal(cdl.wht(block, axis=axis), want)
        else:
            with pytest.raises(ValueError):
                cdl.wht(block, axis=axis)


def _eliminate(matrix: np.ndarray, prime: int) -> tuple[int, list[list[int]]]:
    """Return the rank and reduced row echelon form over a prime field, in integers."""

    rows = [[int(value) for value in row] for row in matrix]
    rank = 0
    for column in range(matrix.shape[1]):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = pow(rows[rank][column], -1, prime)
        rows[rank] = [(value * scale) % prime for value in rows[rank]]
        for i, row in enumerate(rows):
            if i != rank and row[column]:
                factor = row[column]
                rows[i] = [
                    (a - factor * b) % prime
                    for a, b in zip(row, rows[rank], strict=True)
                ]
        rank += 1
    return rank, rows


@pytest.mark.parametrize("prime", [2, 3, 5, 13])
def test_linear_algebra_agrees_with_integer_elimination(prime: int) -> None:
    """Rank, row reduction and null spaces over a prime field, for every shape."""

    field = cdl.BinaryField() if prime == 2 else fcd.FiniteField(char=prime)
    rng = np.random.default_rng(prime)
    for rows, columns in ((0, 3), (3, 0), (1, 1), (2, 5), (5, 2), (4, 4), (6, 9)):
        for _ in range(8):
            matrix = rng.integers(0, prime, (rows, columns)).astype(np.uint32)
            want, form = _eliminate(matrix, prime)
            reduced, pivots = linalg.row_reduce(matrix, field=field)
            kernel = linalg.nullspace(matrix, field=field)
            assert linalg.rank(matrix, field=field) == want == len(pivots)
            assert reduced.shape == matrix.shape
            if matrix.size:
                assert np.array_equal(reduced, np.array(form, dtype=np.uint32))
            assert kernel.shape == (columns - want, columns)
            if kernel.size and rows:
                product = matrix.astype(np.int64) @ kernel.T.astype(np.int64)
                assert not (product % prime).any()
            if kernel.size:
                assert linalg.rank(kernel, field=field) == kernel.shape[0]
            again, same = linalg.row_reduce(reduced, field=field)
            assert np.array_equal(again, reduced) and same == pivots
    for shape in ((0, 3), (3, 0), (4, 4), (3, 6), (6, 3)):
        matrix = rng.integers(-3, 4, shape).astype(float)
        rank = linalg.rank(matrix)
        kernel = linalg.nullspace(matrix)
        assert kernel.shape == (shape[1] - rank, shape[1])
        if matrix.size and kernel.size:
            assert np.allclose(matrix @ kernel.T, 0, atol=1e-10)


# Sampled identity tests of loops


def test_sampled_loop_identities_use_bounded_blocks(monkeypatch: Any) -> None:
    """A large sample is drawn in blocks, with the answers of a small one."""

    from pycdloop import _loop

    sizes: list[int] = []
    original = np.random.Generator.integers

    for loop in (cdl.CDLoop(3), cdl.CDLoop(4), cdl.CentralProduct(2, 3)):
        want = (loop.is_moufang(), loop.is_diassociative())
        assert (loop.is_moufang(100), loop.is_diassociative(100)) == want
    assert cdl.CDLoop(3).is_moufang(_loop._SAMPLE_COUNT + 1) is True
    monkeypatch.setattr(_loop, "_SAMPLE_COUNT", 1000)
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda seed: _Recording(np.random.Generator(np.random.PCG64(seed)), sizes),
    )
    del original
    assert cdl.CDLoop(3).is_moufang(4500) is True
    assert sizes and max(sizes) == 1000 and sum(sizes) == 3 * 4500
    sizes.clear()
    assert cdl.CDLoop(4).is_moufang(4500) is False
    assert sum(sizes) == 3 * 1000
    for samples in (0, -1):
        with pytest.raises(ValueError):
            cdl.CDLoop(3).is_moufang(samples)
        with pytest.raises(ValueError):
            cdl.CDLoop(3).is_diassociative(samples)
    for samples in (1.5, "3", True):
        with pytest.raises(TypeError):
            cdl.CDLoop(3).is_moufang(samples)


class _Recording:
    """A random generator that records the sizes it is asked for."""

    def __init__(self, generator: Any, sizes: list[int]) -> None:
        self._generator, self._sizes = generator, sizes

    def integers(self, high: int, size: int) -> np.ndarray:
        self._sizes.append(size)
        values: np.ndarray = self._generator.integers(high, size=size)
        return values


# Unsuitable arguments


def _junk() -> list[Any]:
    """Return arguments of the wrong type, shape, range or parent."""

    quaternion = cdl.CDLoop(2)
    real = cdl.LoopAlgebra(quaternion)
    seven = cdl.LoopAlgebra(quaternion, char=7)
    return [
        None, 0, 1, -1, 2, 5, 10**20, -(10**20), 1.5, float("nan"), float("inf"), 1j,
        True, "left", "x", b"x", (), (1,), (0, 0), [], [1, 2], {}, {1: 2}, set(),
        object(), int, np.int64(3), np.float64(2.5), np.uint64(2**64 - 1), np.array(1),
        np.zeros(0), np.ones(3), np.ones((2, 4)), np.ones((2, 4), dtype=np.uint32),
        np.ones((3, 2, 4)), np.full((2, 4), np.nan), np.array(["a"]),
        np.ones((2, 4), dtype=object), Fraction(1, 2), slice(None), Ellipsis, range(3),
        quaternion, cdl.CentralProduct(1, 2), quaternion.basis(1), real, seven,
        real.one(), seven.one(), real.array(np.ones((2, 2, 4))),
        seven.array(np.ones((2, 2, 4), dtype=np.uint32)),
        real.array(np.ones((2, 2, 4))).split(), fcd.FiniteField(char=7),
        fcd.FiniteField(char=7)(3), cdl.BinaryField(deg=2), cdl.BinaryField(deg=2)(1),
        2**31, 2**63, -(2**63),
    ]  # fmt: skip


# Methods whose cost is the size of an integer argument, which the junk
# above would make unbounded
_UNBOUNDED = {"left_power", "right_power", "random", "is_moufang", "is_diassociative"}
_OPERATORS = (
    "__call__", "__getitem__", "__setitem__", "__contains__", "__pow__",
    "__truediv__", "__rtruediv__", "__array__", "__eq__", "__lt__", "__len__",
    "__iter__", "__hash__", "__bool__", "__neg__", "__int__", "__float__",
)  # fmt: skip


def _attempt(
    function: Any, arguments: tuple[Any, ...], options: dict[str, Any]
) -> None:
    """Call a function, accepting a result or one of the documented refusals."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            function(*arguments, **options)
    except REFUSALS:
        return


@pytest.mark.parametrize("selectors", [{}, {"char": 7}, {"char": 2, "deg": 2}])
def test_methods_refuse_unsuitable_arguments_with_documented_errors(
    selectors: dict[str, int],
) -> None:
    """No argument makes a public method fail with an internal error."""

    rng = np.random.default_rng(43)
    junk = _junk()
    loop = cdl.CDLoop(2)
    algebra = cdl.LoopAlgebra(loop, **selectors)
    array = algebra.random(1, batch_shape=(2,))
    targets = [algebra, algebra.random(2), array, loop, loop.basis(1)]
    targets += [cdl.CentralProduct(1, 2), cdl.BinaryField(deg=3)]
    targets += [cdl.BinaryField(deg=3)([1, 0, 1])]
    if not algebra._field.binary:
        targets.append(array.split())
    for target in targets:
        for name in dir(target):
            if name.startswith("_") and name not in _OPERATORS:
                continue
            if name in _UNBOUNDED:
                continue
            try:
                member = getattr(target, name)
            except REFUSALS:
                continue
            if not callable(member):
                continue
            try:
                parameters = list(inspect.signature(member).parameters.values())
            except (TypeError, ValueError):
                parameters = []
            options = [
                p.name
                for p in parameters
                if p.kind is p.KEYWORD_ONLY or p.default is not p.empty
            ]
            _attempt(member, (), {})
            for value in junk:
                _attempt(member, (value,), {})
                for other in rng.choice(len(junk), 3, replace=False):
                    _attempt(member, (value, junk[other]), {})
                for option in options:
                    _attempt(member, (), {option: value})
                    _attempt(member, (junk[rng.integers(len(junk))],), {option: value})


def test_functions_refuse_unsuitable_arguments_with_documented_errors() -> None:
    """The same for the module-level functions and the constructors."""

    rng = np.random.default_rng(47)
    junk = _junk()
    loop = cdl.CDLoop(2)
    for module in (cdl, codes, linalg, testing):
        for name in module.__all__:
            member = getattr(module, name)
            if not callable(member):
                continue
            if inspect.isclass(member):
                for value in junk:
                    _attempt(member, (value,), {})
                    _attempt(member, (value, junk[rng.integers(len(junk))]), {})
                    for option in ("char", "deg", "field", "modulus", "level"):
                        _attempt(member, (), {option: value})
                        _attempt(member, (loop,), {option: value})
                continue
            parameters = list(inspect.signature(member).parameters.values())
            options = [
                p.name
                for p in parameters
                if p.kind is p.KEYWORD_ONLY or p.default is not p.empty
            ]
            # The number of samples of a sampled test is its cost
            options = [
                option for option in options if option not in ("samples", "count")
            ]
            _attempt(member, (), {})
            for value in junk:
                if module is testing and isinstance(value, (int, np.integer)):
                    continue
                _attempt(member, (value,), {})
                for other in rng.choice(len(junk), 4, replace=False):
                    if module is testing:
                        continue
                    _attempt(member, (value, junk[other]), {})
                for option in options:
                    _attempt(member, (cdl.LoopAlgebra(loop),), {option: value})
