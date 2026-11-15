"""Tests of the boundaries of the public interface.

These cover what an element or array promises about its storage, how raw
values are accepted, and how the predicates behave on values at the edge of
the floating-point range.
"""

from __future__ import annotations

import copy
import itertools
import pickle
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyfastcd as fcd
import pytest

import pycdloop as cdl
from pycdloop import _core, codes, linalg

ALGEBRAS = [
    (cdl.CDLoop(1), {}),
    (cdl.CDLoop(3), {}),
    (cdl.CDLoop(2), {"char": 0, "deg": 2}),
    (cdl.CDLoop(2), {"char": 7}),
    (cdl.CDLoop(3), {"char": 3, "deg": 2}),
    (cdl.CDLoop(2), {"char": 2}),
    (cdl.CDLoop(3), {"char": 2, "deg": 3}),
    (cdl.CentralProduct(1, 2), {}),
    (cdl.CentralProduct(2, 2), {"char": 13}),
    (cdl.CentralProduct(1, 2), {"char": 2, "deg": 2}),
]
FINITE = [pair for pair in ALGEBRAS if pair[1].get("char")]


def _values(algebra: Any, rng: Any, batch: tuple[int, ...] = ()) -> np.ndarray:
    """Return random coefficients with exactly representable values."""

    shape = batch + algebra.coefficient_shape
    if algebra.field is not None:
        return rng.integers(0, algebra.char, shape).astype(np.uint32)
    values = rng.integers(-3, 4, shape).astype(algebra.dtype)
    if algebra.deg == 2:
        values = values + 1j * rng.integers(-3, 4, shape)
    return values


def _same(algebra: Any, a: np.ndarray, b: np.ndarray) -> bool:
    """Compare coefficients, exactly over a finite field."""

    if algebra.field is not None:
        return bool(np.array_equal(a, b))
    return bool(np.allclose(a, b, rtol=1e-12, atol=1e-12))


def _owners(array: np.ndarray) -> list[Any]:
    """Return an array and every object it borrows its memory from."""

    chain: list[Any] = [array]
    while getattr(chain[-1], "base", None) is not None:
        chain.append(chain[-1].base)
    return chain


def _assert_immutable(array: np.ndarray) -> None:
    """Check that neither an array nor an owner of its memory can be written."""

    assert not array.flags.writeable
    for owner in _owners(array):
        if isinstance(owner, np.ndarray):
            assert not owner.flags.writeable
            with pytest.raises(ValueError):
                owner.setflags(write=True)
        elif isinstance(owner, memoryview):
            assert owner.readonly
        else:
            assert isinstance(owner, bytes)
    with pytest.raises(ValueError):
        array[(0,) * array.ndim] = array[(0,) * array.ndim]


# Inverses


def _two_sided_inverses(algebra: Any, x: Any) -> tuple[np.ndarray | None, int]:
    """Solve ``x v = v x = 1``: one solution and the dimension of the solutions."""

    size = algebra.dimension
    system = np.concatenate([algebra.left_matrix(x), algebra.right_matrix(x)])
    one = algebra.one().coefficients.reshape(size)
    column = np.concatenate([one, one]).reshape(2 * size, 1)
    reduced, pivots = linalg.row_reduce(
        np.concatenate([system, column], axis=1), field=algebra.field
    )
    if size in pivots:
        return None, 0
    solution = np.zeros(size, dtype=reduced.dtype)
    for row, pivot in enumerate(pivots):
        solution[pivot] = reduced[row, size]
    return solution, size - len(pivots)


def test_inverse_is_the_one_that_is_a_polynomial_in_the_element() -> None:
    """The inverse is two-sided, a combination of powers, and gives negative powers."""

    rng = np.random.default_rng(101)
    checked = 0
    for level, char in ((1, 7), (2, 7), (3, 5), (4, 7), (5, 13)):
        algebra = cdl.LoopAlgebra(cdl.CDLoop(level), char=char)
        for _ in range(60):
            x = algebra(_values(algebra, rng))
            if not x.has_inverse():
                with pytest.raises(ZeroDivisionError):
                    x.inverse()
                continue
            checked += 1
            y = x.inverse()
            assert x * y == 1 and y * x == 1
            assert x**-1 == y and x**-2 == y * y and x**-3 * x**3 == 1
            powers = np.stack(
                [(x**k).coefficients.ravel() for k in range(algebra.dimension + 2)]
            )
            rank = linalg.rank(powers, field=algebra.field)
            stacked = np.concatenate([powers, y.coefficients.reshape(1, -1)])
            assert linalg.rank(stacked, field=algebra.field) == rank
    assert checked >= 20


def test_inverse_does_not_promise_every_two_sided_inverse() -> None:
    """From level 4 an element of norm zero can have inverses that are not computed."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(4), char=5)
    ones = np.ones(16, dtype=np.uint32)
    d = np.zeros(16, dtype=np.uint32)
    v = np.zeros(16, dtype=np.uint32)
    d[[3, 6, 10, 15]] = [1, 2, 1, 3]
    v[[3, 6, 10, 15]] = [1, 3, 1, 2]
    x = algebra.split_array(ones, d)[()]
    y = algebra.split_array(ones, v)[()]
    assert sum(int(t) ** 2 for t in d) % 5 == 0
    assert x * y == 1 and y * x == 1
    assert x.has_inverse() is False
    with pytest.raises(ZeroDivisionError, match="quadratic norm zero"):
        x.inverse()
    with pytest.raises(ZeroDivisionError):
        x**-1
    # Such inverses are not unique: the solutions form a space
    solution, dimension = _two_sided_inverses(algebra, x)
    assert solution is not None and dimension == 4
    # Nor do they behave as an inverse of the powers of x
    assert (x * x) * y != x * (x * y)


def test_norm_zero_excludes_every_inverse_through_level_three() -> None:
    """Up to the octonions an element without the inverse has none at all."""

    rng = np.random.default_rng(102)
    for level in (1, 2, 3):
        algebra = cdl.LoopAlgebra(cdl.CDLoop(level), char=5)
        found = 0
        while found < 12:
            x = algebra(_values(algebra, rng))
            if x.has_inverse() or not x.spectrum().coefficients.all():
                continue
            found += 1
            assert _two_sided_inverses(algebra, x)[0] is None


def test_zero_divisors_over_r_have_several_two_sided_inverses() -> None:
    """In the sedenions the computed inverse is one of a family."""

    loop = cdl.CDLoop(4)
    algebra = cdl.LoopAlgebra(loop)
    signed = np.zeros(16)
    signed[[1, 10]] = 1
    x = algebra.split_array(np.ones(16), signed)[()]
    assert x.has_inverse() and x.is_zero_divisor()
    y = x.inverse()
    assert np.allclose((x * y).coefficients, algebra.one().coefficients)
    system = np.concatenate([algebra.left_matrix(x), algebra.right_matrix(x)])
    kernel = linalg.nullspace(system)
    assert kernel.shape[0] == 4
    other = y + algebra(kernel[0].reshape(algebra.coefficient_shape))
    for product in (x * other, other * x):
        assert np.allclose(product.coefficients, algebra.one().coefficients)


# The build


def test_windows_build_requests_c11() -> None:
    """The extension uses ``restrict``, which the Microsoft compiler needs C11 for."""

    setup = Path(__file__).resolve().parents[1] / "setup.py"
    if not setup.is_file():
        pytest.skip("the build script is not beside the tests")
    text = setup.read_text()
    branch = text[text.index('sys.platform == "win32"') : text.index("else:")]
    assert '"/std:c11"' in branch
    source = setup.parent / "src" / "pycdloop" / "_core.c"
    if source.is_file():
        assert "restrict" in source.read_text()


# Immutable elements


def _elements(algebra: Any, rng: Any) -> list[Any]:
    """Return elements created in each of the ways the package creates them."""

    x = algebra(_values(algebra, rng))
    y = algebra(_values(algebra, rng))
    batch = algebra.array(_values(algebra, rng, (3,)))
    made = [x, x * y, x + y, 2 * x, x.copy(), batch[1], (batch * batch)[0]]
    made += [pickle.loads(pickle.dumps(x * y)), copy.deepcopy(x)]
    made += [algebra.zero(), algebra.one(), algebra.basis(algebra.loop.minus_one())]
    if not algebra._field.binary:
        made += [batch.split()[2], (batch.split() * batch.split())[0]]
    return made


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS)
def test_element_storage_cannot_be_made_writable(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """Every array an element hands out is backed by memory that stays read-only."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order)
    for x in _elements(algebra, rng):
        before = x.to_numpy()
        digest = hash(x)
        exposed = [
            x.coefficients,
            x.to_numpy(copy=False),
            np.asarray(x),
            x.__array__(),
            x.__array__(None, False),
        ]
        state = x.__reduce__()[1][2]
        exposed += list(state) if isinstance(state, tuple) else [state]
        if not algebra._field.binary:
            epsilon = x.epsilon()
            if isinstance(epsilon, np.ndarray):
                exposed.append(epsilon)
            else:
                exposed.append(np.asarray(epsilon.coefficients))
        for array in exposed:
            _assert_immutable(array)
        # Each request gets its own array, so changes to the shape of one
        # array do not reach the next, nor the pickled element
        first = x.coefficients
        assert first is not x.coefficients
        first.resize((first.size,))
        assert x.coefficients.shape == algebra.coefficient_shape
        assert np.array_equal(x.coefficients, before)
        assert pickle.loads(pickle.dumps(x)) == x
        # Copies are independent and writable
        for duplicate in (x.to_numpy(), np.array(x), x.__array__(None, True)):
            assert duplicate.flags.writeable
            duplicate[...] = 0
        assert np.array_equal(x.to_numpy(), before) and hash(x) == digest
        assert x == algebra(before)


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS)
def test_elements_do_not_alias_the_arrays_they_were_made_from(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """Writing to a source array afterwards leaves the element unchanged."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 1)
    source = _values(algebra, rng)
    batch = _values(algebra, rng, (2,))
    shared = algebra.asarray(batch)
    made = [algebra(source), shared[0], algebra.array(batch)[1]]
    expected = [source.copy(), batch[0].copy(), batch[1].copy()]
    if not algebra._field.binary:
        split = shared.split()
        made.append(split[1])
        expected.append(batch[1].copy())
    source[...] = 0
    batch[...] = 0
    if not algebra._field.binary:
        split[1] = 0
    for element, values in zip(made, expected, strict=True):
        assert _same(algebra, element.coefficients, values)


def test_loop_elements_cannot_be_initialized_again() -> None:
    """Calling ``__init__`` on an existing loop element changes nothing."""

    for loop in (cdl.CDLoop(2), cdl.CentralProduct(1, 2)):
        x = loop.basis(1)
        digest = hash(x)
        with pytest.raises(AttributeError):
            x.__init__(loop, 2)
        with pytest.raises(AttributeError):
            x.__init__(loop, 1, -1)
        assert x == loop.basis(1) and hash(x) == digest
        assert pickle.loads(pickle.dumps(x)) == x and copy.copy(x) == x
        assert -x == type(x)(loop, 1, -1)


# Integers


def _fields() -> list[Any]:
    """Return finite fields of each kind."""

    return [
        fcd.FiniteField(char=13),
        fcd.FiniteField(char=65521),
        fcd.FiniteField(char=2147483647),
        fcd.FiniteField(char=3, deg=2),
        cdl.BinaryField(),
        cdl.BinaryField(deg=2),
    ]


EXTREMES = {
    np.uint64: [0, 1, 2**63 - 1, 2**63, 2**63 + 1, 2**64 - 2, 2**64 - 1],
    np.int64: [-(2**63), -(2**63) + 1, -1, 0, 2**63 - 1],
    np.uint32: [0, 2**31, 2**32 - 1],
    np.int32: [-(2**31), -1, 2**31 - 1],
    np.uint8: [0, 13, 255],
    np.int8: [-128, -1, 127],
}


@pytest.mark.parametrize("field", _fields(), ids=repr)
@pytest.mark.parametrize("dtype", list(EXTREMES), ids=lambda t: t.__name__)
def test_integer_arrays_are_reduced_as_python_integers_are(
    field: Any, dtype: Any
) -> None:
    """An integer of any NumPy type reduces to the residue of its value."""

    loop = cdl.CDLoop(2)
    algebra = cdl.LoopAlgebra(loop, field=field)
    shape = algebra.coefficient_shape
    for value in EXTREMES[dtype]:
        raw = np.full(shape, value, dtype=dtype)
        want = value % field.char
        made = [
            algebra(raw),
            algebra(raw.tolist()),
            algebra.array(raw[None])[0],
            algebra.asarray(raw[None])[0],
        ]
        target = algebra.array(np.zeros((1,) + shape, dtype=np.uint32))
        target[0] = raw
        made.append(target[0])
        for element in made:
            assert (element.coefficients == want).all(), (value, dtype)
        # A scalar of the same type and value
        scaled = algebra.one() * dtype(value)
        assert int(np.asarray(scaled.coefficients).ravel()[0]) == want
        # Linear algebra and the exports read matrices in the same way
        matrix = raw.reshape((1, 8) + shape[2:])
        reduced = linalg.row_reduce(matrix, field=field)[0]
        assert reduced.max() < field.char
        assert linalg.rank(matrix, field=field) == (1 if want else 0)
        listing = codes.to_sage(matrix, field=field)
        assert str(want) in listing
        entries = [int(token) for token in _integers(listing.split("[", 1)[1])]
        assert set(entries[: 8 * field.deg]) == {want}


def _integers(text: str) -> list[str]:
    """Return the integer literals of a source text."""

    cleaned = "".join(c if c.isdigit() else " " for c in text.split("GF(")[0])
    return cleaned.split()


# Scalars in LoopAlgebra.multiply


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS)
def test_multiply_checks_the_operand_beside_a_scalar(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """A scalar operand does not exempt the other one from belonging to the algebra."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 2)
    x = algebra(_values(algebra, rng))
    batch = algebra.array(_values(algebra, rng, (3,)))
    operands = [x, batch]
    if not algebra._field.binary:
        operands.append(batch.split())
    for operand in operands:
        for product in (algebra.multiply(2, operand), algebra.multiply(operand, 2)):
            assert type(product) is type(operand)
            assert np.all(product == operand * 2)
    other_loop = cdl.LoopAlgebra(cdl.CDLoop(loop.level + 1), **selectors)
    other_field = cdl.LoopAlgebra(loop, char=11)
    for foreign in (other_loop, other_field):
        y = foreign.one()
        outsiders = [y, foreign.array(np.stack([y.coefficients, y.coefficients]))]
        for outsider in outsiders:
            with pytest.raises(ValueError, match="different loop algebras"):
                algebra.multiply(2, outsider)
            with pytest.raises(ValueError, match="different loop algebras"):
                algebra.multiply(outsider, 2)
    for raw in (x.to_numpy(), x.to_numpy().tolist(), "x", None):
        with pytest.raises(TypeError):
            algebra.multiply(2, raw)
        with pytest.raises(TypeError):
            algebra.multiply(raw, 2)
    with pytest.raises(TypeError):
        algebra.multiply(2, 3)
    with pytest.raises(ValueError):
        algebra.multiply(2, x, algorithm="fast")


def test_multiply_cannot_store_another_fields_residues() -> None:
    """A scalar times an element of another field does not reach ``out``."""

    small = cdl.LoopAlgebra(cdl.CDLoop(1), char=3)
    large = cdl.LoopAlgebra(cdl.CDLoop(1), char=13)
    out = small.array(np.zeros((2, 2), dtype=np.uint32))
    for arguments in ((1, large(12)), (large(12), 1)):
        with pytest.raises(ValueError, match="different loop algebras"):
            small.multiply(*arguments, out=out)
    assert not out.coefficients.any()
    stored = small.multiply(2, small(2), out=out)
    assert stored is out and out.coefficients[0, 0] == 1


# Array storage


def _operations(algebra: Any, array: Any, element: Any) -> list[Any]:
    """Return the public operations that read the storage of an array."""

    calls: list[Any] = [
        lambda: array + array,
        lambda: array - element,
        lambda: array * array,
        lambda: element * array,
        lambda: array * 2,
        lambda: -array,
        lambda: array == array,
        lambda: array == 1,
        lambda: array.copy(),
        lambda: +array,
        lambda: array.square(),
        lambda: array.conjugate(),
        lambda: array.weight(),
        lambda: array.distance(array),
        lambda: array.kappa(),
        lambda: array.is_central(),
        lambda: array.commutator(array),
        lambda: algebra.multiply(array, array, algorithm="table"),
        lambda: algebra.multiply(array, 2),
        lambda: algebra.array(array),
        lambda: pickle.loads(pickle.dumps(array)) + array,
    ]
    if not algebra._field.binary:
        calls += [
            lambda: array.split(),
            lambda: array.spectrum(),
            lambda: array.epsilon(),
            lambda: algebra.multiply(array, array, algorithm="split"),
        ]
    else:
        calls.append(lambda: algebra.multiply(array, array, algorithm="lift"))
    return calls


@pytest.mark.parametrize(("loop", "selectors"), FINITE)
def test_arrays_reject_storage_that_stopped_holding_residues(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """Values written into shared storage are checked by every operation."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 3)
    raw = _values(algebra, rng, (4,))
    element = algebra(_values(algebra, rng))
    views = [algebra.asarray(raw), algebra.array(raw)]
    for number, array in enumerate(views):
        storage = array.coefficients
        assert np.shares_memory(storage, raw) is (number == 0)
        first = (2,) + (0,) * (storage.ndim - 1)
        original = storage[first]
        for call in _operations(algebra, array, element):
            call()
        for invalid in (algebra.char, algebra.char + 3, 2**32 - 1):
            storage[first] = invalid
            for call in _operations(algebra, array, element):
                with pytest.raises(ValueError, match="not residues"):
                    call()
            # Only the record that holds the value is refused as an element
            with pytest.raises(ValueError, match="not residues"):
                array[2]
            assert array[1] == algebra(storage[1])
            assert array[:2].copy().batch_shape == (2,)
            with pytest.raises(ValueError, match="not residues"):
                array[1:].copy()
        storage[first] = original
        for call in _operations(algebra, array, element):
            call()


@pytest.mark.parametrize(("loop", "selectors"), FINITE)
def test_split_arrays_reject_storage_that_stopped_holding_residues(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """The parts of a split array are checked in the same way."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    if algebra._field.binary:
        pytest.skip("no split form in characteristic 2")
    rng = np.random.default_rng(loop.order + 4)
    array = algebra.array(_values(algebra, rng, (3,))).split()
    element = algebra(_values(algebra, rng))
    calls = [
        lambda: array * array,
        lambda: array + element,
        lambda: array.to_loop(),
        lambda: array.copy(),
        lambda: array == array,
        lambda: array.kappa(),
        lambda: array.spectrum(),
        lambda: array.epsilon(),
        lambda: array.weight(),
        lambda: algebra.multiply(array, array),
    ]
    # Reading a part hands out the storage, which is checked from then on
    assert array.spectral is not None and array.signed is not None
    for part in (array._spectral, array._signed):
        first = (1,) + (0,) * (part.ndim - 1)
        original = part[first]
        part[first] = algebra.char
        for call in calls:
            with pytest.raises(ValueError, match="not residues"):
                call()
        with pytest.raises(ValueError, match="not residues"):
            array[1]
        assert isinstance(array[0], cdl.Element)
        part[first] = original
        for call in calls:
            call()


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS)
def test_storage_is_checked_once_it_has_been_handed_out(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """Arrays over storage that was handed out are checked; the others need not be."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 8)
    raw = _values(algebra, rng, (4,))

    def fresh() -> Any:
        return algebra.array(raw)

    untouched = [fresh(), fresh() * fresh(), fresh().copy(), +fresh(), fresh()[1:]]
    untouched.append(pickle.loads(pickle.dumps(fresh())))
    for array in untouched:
        assert array._shown == [False]
    exposures = [
        lambda array: array.coefficients,
        lambda array: array.to_numpy(copy=False),
        lambda array: np.asarray(array),
        lambda array: array.__array__(None, False),
    ]
    for expose in exposures:
        array = fresh()
        view, other = array[1:], array[:2]
        assert np.shares_memory(expose(array), array._values)
        # Arrays over the same storage share the flag
        assert array._shown == view._shown == other._shown == [True]
        assert array.copy()._shown == [False] and (array + array)._shown == [False]
        child = fresh()[::2]
        parent_flag = child._shown
        expose(child)
        assert parent_flag == [True]
    copying = [lambda a: a.to_numpy(), lambda a: a.kappa(), lambda a: a.copy()]
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        # NumPy 1 asks for the storage and copies it itself
        copying.append(lambda a: np.array(a))
    for copy_ in copying:
        array = fresh()
        copy_(array)
        assert array._shown == [False]
    assert algebra.asarray(raw)._shown == [True]
    assert algebra.asarray(raw.tolist())._shown == [False]
    target = np.zeros_like(raw)
    assert algebra.multiply(fresh(), fresh(), out=target)._shown == [True]
    if not algebra._field.binary:
        split = fresh().split()
        part = split[1:]
        assert split._shown == part._shown == [False]
        assert part.signed is not None
        assert split._shown == [True] and split.copy()._shown == [False]
        assert (split * split)._shown == [False]
        assert split.to_loop()._shown == [False]


def test_arrays_reject_storage_given_another_shape_or_type() -> None:
    """Storage that no longer has the algebra's layout is refused."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
    array = algebra.array(np.ones((3, 2, 4)))
    storage = array.coefficients
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            storage.shape = (3, 4, 2)
        except (AttributeError, ValueError):
            pytest.skip("this NumPy does not reshape an array in place")
    for call in (lambda: array + array, lambda: array * array, lambda: array.copy()):
        with pytest.raises(ValueError, match="dtype and shape"):
            call()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        storage.shape = (3, 2, 4)
    assert np.all(array + array == array * 2)


# Values at the edge of the floating-point range


@pytest.mark.parametrize("selectors", [{}, {"char": 0, "deg": 2}], ids=["R", "C"])
@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_inverse_predicates_do_not_depend_on_the_scale(
    level: int, selectors: dict[str, Any]
) -> None:
    """Multiplying an element by a power of two changes none of the answers."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
    rng = np.random.default_rng(200 + level)
    elements = [algebra(_values(algebra, rng)) for _ in range(12)]
    if selectors and level:
        # A signed component with Q = 1 + i^2 = 0
        signed = np.zeros(1 << level, dtype=complex)
        signed[[0, 1]] = [1, 1j]
        elements.append(algebra.split_array(np.ones(1 << level), signed)[()])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for x in elements:
            want = (x.has_inverse(), x.is_nilpotent(), x.is_zero_divisor())
            for exponent in (-1000, -600, -300, 0, 300, 480):
                scaled = x * float(np.ldexp(1.0, exponent))
                found = (
                    scaled.has_inverse(),
                    scaled.is_nilpotent(),
                    scaled.is_zero_divisor(),
                )
                assert found == want, exponent
                if not want[0]:
                    with pytest.raises(ZeroDivisionError):
                        scaled.inverse()


@pytest.mark.parametrize("magnitude", [1e-300, 1e-200, 1e-160, 1e-100, 1e100, 1e150])
def test_inverses_of_very_small_and_very_large_elements(magnitude: float) -> None:
    """A representable inverse is found whatever the size of the coefficients."""

    for selectors in ({}, {"char": 0, "deg": 2}):
        for level in (1, 3, 4):
            algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
            values = np.ones(algebra.coefficient_shape, dtype=algebra.dtype)
            values[0, 0] = 4
            x = algebra(values * magnitude)
            assert x.has_inverse() is True
            assert x.is_nilpotent() is False
            y = x.inverse()
            unit = algebra.one().coefficients
            assert np.allclose((x * y).coefficients, unit, atol=1e-12)
            assert np.allclose((y * x).coefficients, unit, atol=1e-12)
            batch = algebra.array(np.stack([values * magnitude, values]))
            assert batch.has_inverse().tolist() == [True, True]
            products = (batch * batch.inverse()).coefficients
            assert np.allclose(products, unit, atol=1e-12)
    scalar = cdl.LoopAlgebra(cdl.CDLoop(1))(magnitude)
    assert scalar.has_inverse() and scalar.inverse() == 1 / magnitude


# The compiled kernels


def test_kernels_reject_sizes_that_overflow() -> None:
    """A record size that is not representable is an error, not an overflow."""

    real = np.empty(0, dtype=np.float64)
    words = np.empty(0, dtype=np.uint32)
    longs = np.empty(0, dtype=np.int64)
    for dimension, lanes in ((1 << 62, 4), (2, 1 << 62), (1 << 40, 1 << 40)):
        with pytest.raises(OverflowError):
            _core.split(real, real, real, dimension, lanes)
        with pytest.raises(OverflowError):
            _core.unsplit(real, real, real, dimension, lanes)
        with pytest.raises(OverflowError):
            _core.split_mod(words, words, words, dimension, lanes, 7)
        with pytest.raises(OverflowError):
            _core.unsplit_mod(words, words, words, dimension, lanes, 7, 2)
        with pytest.raises(OverflowError):
            _core.transform(real, dimension, lanes)
        with pytest.raises(OverflowError):
            _core.transform_mod(longs, dimension, lanes, 7)
    # A loop buffer of odd length, or not twice the length of the parts
    for loop_size, part in ((7, 4), (4, 4), (16, 4)):
        with pytest.raises(ValueError):
            _core.split(np.zeros(loop_size), np.zeros(part), np.zeros(part), 4, 1)
        with pytest.raises(ValueError):
            _core.unsplit(np.zeros(part), np.zeros(part), np.zeros(loop_size), 4, 1)
    with pytest.raises(ValueError):
        _core.split(np.zeros(8), np.zeros(4), np.zeros(8), 4, 1)


# Unaligned and strided storage


def _unaligned(values: np.ndarray) -> np.ndarray:
    """Return a copy of an array at an address that is not aligned."""

    buffer = bytearray(values.nbytes + 1)
    copy_ = np.frombuffer(buffer, dtype=values.dtype, count=values.size, offset=1)
    copy_ = copy_.reshape(values.shape)
    copy_[...] = values
    assert not copy_.flags.aligned
    return copy_


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS)
def test_unaligned_and_strided_storage_is_accepted(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """Storage in any NumPy layout gives the results of a plain copy."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 5)
    values = _values(algebra, rng, (4,))
    plain = algebra.array(values)
    wide = np.zeros((8,) + algebra.coefficient_shape, dtype=values.dtype)
    wide[::2] = values
    layouts = {
        "unaligned": _unaligned(values),
        "strided": wide[::2],
        "reversed": values[::-1][::-1],
        "fortran": np.asfortranarray(values),
    }
    names = ["table"] + (["lift"] if algebra._field.binary else ["split"])
    for label, storage in layouts.items():
        array = algebra.asarray(storage)
        assert np.shares_memory(array.coefficients, storage), label
        for name in names:
            product = algebra.multiply(array, array, algorithm=name)
            want = algebra.multiply(plain, plain, algorithm=name)
            assert _same(algebra, product.coefficients, want.coefficients), label
        assert np.all(array * array == plain * plain)
        assert np.all(array + plain == plain * 2)
        assert np.all(array * plain[0] == plain * plain[0])
        assert array[1] == plain[1]
        assert _same(algebra, array.copy().coefficients, values)
        if not algebra._field.binary:
            split = array.split()
            assert np.all(split.to_loop() == plain)
            assert np.all((split * split).to_loop() == plain * plain)
        out = _unaligned(np.zeros_like(values))
        stored = algebra.multiply(array, array, out=out)
        assert _same(algebra, stored.coefficients, (plain * plain).coefficients)
        element = algebra(storage[0])
        assert element == plain[0] and element * element == plain[0] * plain[0]


def test_transforms_accept_unaligned_input() -> None:
    """``wht`` and ``dyadic_convolve`` read arrays in any layout."""

    rng = np.random.default_rng(9)
    real = rng.integers(-5, 6, (3, 16)).astype(float)
    assert np.array_equal(cdl.wht(_unaligned(real)), cdl.wht(real))
    assert np.array_equal(
        cdl.dyadic_convolve(_unaligned(real), _unaligned(real)),
        cdl.dyadic_convolve(real, real),
    )
    field = fcd.FiniteField(char=7)
    words = rng.integers(0, 7, (3, 16)).astype(np.uint32)
    assert np.array_equal(
        cdl.wht(_unaligned(words), field=field), cdl.wht(words, field=field)
    )
    matrix = rng.integers(0, 7, (5, 6)).astype(np.uint32)
    assert linalg.rank(_unaligned(matrix), field=field) == linalg.rank(
        matrix, field=field
    )


# The NumPy array interface


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS)
def test_array_interface_honours_copy(loop: Any, selectors: dict[str, Any]) -> None:
    """A dtype change needs a copy, which ``copy=False`` forbids."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 6)
    element = algebra(_values(algebra, rng))
    array = algebra.array(_values(algebra, rng, (2,)))
    other = np.dtype(np.complex64 if algebra.field is None else np.int64)
    for item in (element, array):
        same = item.__array__(None, False)
        assert same.dtype == algebra.dtype
        assert item.__array__(algebra.dtype, False).dtype == algebra.dtype
        with pytest.raises(ValueError, match="without a copy"):
            item.__array__(other, False)
        for flag in (None, True):
            converted = item.__array__(other, flag)
            assert converted.dtype == other
            assert not np.shares_memory(converted, same)
            assert np.array_equal(converted, same.astype(other))
        duplicate = item.__array__(None, True)
        assert not np.shares_memory(duplicate, same) and duplicate.flags.writeable
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            with pytest.raises(ValueError):
                np.asarray(item, dtype=other, copy=False)
            assert np.asarray(item, dtype=other).dtype == other
            assert np.array_equal(np.asarray(item, copy=False), same)
    assert np.shares_memory(array.__array__(None, False), array.coefficients)
    _assert_immutable(element.__array__(None, False))


# Encodings


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS)
def test_from_bytes_reads_bytes_like_objects_only(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """An integer is not read as a length, nor a list as octets."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    x = algebra(_values(algebra, np.random.default_rng(loop.order + 7)))
    encoded = x.to_bytes()
    octets = np.frombuffer(encoded, dtype=np.uint8)
    for data in (encoded, bytearray(encoded), memoryview(encoded), octets,
                 octets[::-1][::-1]):  # fmt: skip
        assert algebra.from_bytes(data) == x
    size = len(encoded)
    for data in (size, 0, 2**31, 10**20, list(encoded), tuple(encoded), "x" * size,
                 None, 1.5, x):  # fmt: skip
        with pytest.raises(TypeError, match="bytes-like"):
            algebra.from_bytes(data)
    for data in (encoded[:-1], encoded + b"\x00", b""):
        with pytest.raises(ValueError):
            algebra.from_bytes(data)
    if algebra.field is not None:
        with pytest.raises(ValueError):
            algebra.from_bytes(b"\xff" * size)


def test_coefficients_over_a_finite_field_are_integers_not_booleans() -> None:
    """Booleans are refused and integers of any size are reduced, in every field."""

    loop = cdl.CDLoop(1)
    for selectors in (
        {"char": 7},
        {"char": 3, "deg": 2},
        {"char": 2},
        {"char": 2, "deg": 3},
    ):
        algebra = cdl.LoopAlgebra(loop, **selectors)
        shape = algebra.coefficient_shape
        flags = np.ones(shape, dtype=bool)
        for build in (algebra, algebra.array, algebra.asarray):
            for values in (flags, flags.tolist()):
                with pytest.raises(TypeError):
                    build(values)
            for size in (10**40, -(10**40) - 1, 2**64, 2**63):
                large = (np.zeros(shape, dtype=object) + size).tolist()
                made = build(large)
                assert (np.asarray(made.coefficients) == size % algebra.char).all()
            for values in (np.full(shape, 1.0), np.full(shape, "1"), [[None]]):
                with pytest.raises((TypeError, ValueError)):
                    build(values)
        with pytest.raises(TypeError):
            algebra(True)
    real = cdl.LoopAlgebra(loop)
    assert real(np.ones(real.coefficient_shape, dtype=bool)) == real(
        np.ones(real.coefficient_shape)
    )


# Parents built at the same time


def test_parents_are_unique_when_threads_construct_them_together() -> None:
    """Several threads that ask for one new parent all get the same object."""

    import threading
    from concurrent.futures import ThreadPoolExecutor

    def race(build: Any) -> list[Any]:
        barrier = threading.Barrier(8)

        def construct(_: int) -> Any:
            barrier.wait()
            return build()

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(construct, range(8)))

    cdl.LoopAlgebra(cdl.CDLoop(0), char=109)
    builders = [
        lambda: cdl.LoopAlgebra(cdl.CentralProduct(5, 5), char=109),
        lambda: cdl.LoopAlgebra(cdl.CDLoop(9), char=2, deg=11),
        lambda: cdl.LoopAlgebra(cdl.CentralProduct(4, 5, 1)),
        lambda: cdl.CentralProduct(6, 7),
        lambda: cdl.CDLoop(13),
        lambda: cdl.BinaryField.from_modulus(modulus=(1, 1, 0, 1, 1, 0, 0, 0, 1)),
    ]
    for build in builders:
        made = race(build)
        assert len({id(item) for item in made}) == 1
        assert all(item is build() for item in made)
    parents = race(lambda: cdl.LoopAlgebra(cdl.CentralProduct(5, 6), char=109))
    total = parents[0].one()
    for parent in parents[1:]:
        total = total + parent.one()
    assert total == 8


_RACE = """
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pycdloop as cdl

loop = cdl.CDLoop(1)
sys.setswitchinterval(1e-6)
split = []
for char in (211, 223, 227, 229, 233, 239, 241, 251):
    for deg in (2, 3):
        barrier = Barrier(16)

        def create(_):
            barrier.wait()
            return cdl.LoopAlgebra(loop, char=char, deg=deg)

        with ThreadPoolExecutor(max_workers=16) as pool:
            parents = list(pool.map(create, range(16)))
        total = parents[0].one()
        for parent in parents[1:]:
            total = total + parent.one()
        if len({id(parent) for parent in parents}) > 1:
            split.append((char, deg))
        if len({id(parent.field) for parent in parents}) > 1:
            split.append((char, deg, "field"))
print(split)
"""


def test_threads_that_first_use_a_field_together_share_one_parent() -> None:
    """The first construction of a field of pyfastcd happens once, under the lock."""

    import subprocess
    import sys

    # A new interpreter, in which none of these fields has been created,
    # and a short switch interval, which lets the threads interleave
    done = subprocess.run(
        [sys.executable, "-c", _RACE], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_derived_tables_are_built_once_under_threads() -> None:
    """Threads that need a table of a parent for the first time share one."""

    import threading
    from concurrent.futures import ThreadPoolExecutor

    for loop, selectors, name in (
        (cdl.CDLoop(3), {"char": 2, "deg": 5}, "lift"),
        (cdl.CDLoop(4), {}, "table"),
    ):
        algebra = cdl.LoopAlgebra(loop, **selectors)
        algebra._lazy.pop(name, None)
        x, y = algebra.random(1), algebra.random(2)
        barrier = threading.Barrier(8)
        seen: list[int] = []

        algorithm = "table" if name == "table" else "lift"

        def use(
            _: int,
            algebra: Any = algebra,
            operands: tuple[Any, Any] = (x, y),
            name: str = name,
            algorithm: str = algorithm,
            barrier: Any = barrier,
            seen: list[int] = seen,
        ) -> Any:
            barrier.wait()
            product = algebra.multiply(*operands, algorithm=algorithm)
            seen.append(id(algebra._lazy[name]))
            return product

        with ThreadPoolExecutor(max_workers=8) as pool:
            products = list(pool.map(use, range(8)))
        assert len(set(seen)) == 1
        assert all(product == products[0] for product in products)


# Boolean indices


def _mask_forms(mask: np.ndarray) -> list[Any]:
    """Return a Boolean mask as every kind of object that NumPy reads as one."""

    class Convertible:
        def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
            return mask

    nested = tuple(tuple(row) for row in mask.tolist()) if mask.ndim > 1 else None
    forms = [mask, mask.tolist(), Convertible()]
    if nested is not None:
        forms += [nested, list(nested), tuple(list(row) for row in nested)]
    else:
        forms.append(tuple(mask.tolist()))
    return forms


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS[:5])
def test_boolean_masks_of_every_form_stay_on_the_batch_axes(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """A mask with more axes than the batch is rejected, however it is written."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(73)
    for batch in ((2,), (2, 3)):
        values = _values(algebra, rng, batch)
        for split in (False, True):
            if split and algebra.char == 2:
                continue
            # One axis more than the batch has, which would reach a
            # coefficient axis
            wide = rng.random(batch + (2,)) < 0.5
            for form in _mask_forms(wide):
                array = algebra.array(values)
                target = array.split() if split else array
                for key in ((form,), form if isinstance(form, tuple) else (form,)):
                    with pytest.raises(IndexError):
                        target[key]
                    with pytest.raises(IndexError):
                        target[key] = algebra.one()
                back = target.to_loop() if split else target
                assert _same(algebra, back.to_numpy(), values)
            # A mask of the batch shape selects records, as in NumPy
            fit = rng.random(batch) < 0.5
            fit.flat[0] = True
            labels = np.arange(int(np.prod(batch))).reshape(batch)
            for form in _mask_forms(fit):
                array = algebra.array(values)
                target = array.split() if split else array
                picked = target[(form,)]
                picked = picked.to_loop() if split else picked
                assert picked.batch_shape == labels[(fit,)].shape
                assert _same(algebra, picked.to_numpy(), values[fit])
                target[(form,)] = 0
                back = target.to_loop() if split else target
                expected = values.copy()
                expected[fit] = 0
                assert _same(algebra, back.to_numpy(), expected)


@pytest.mark.parametrize(("loop", "selectors"), ALGEBRAS[:5])
def test_scalar_boolean_indices_follow_numpy(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """A Boolean scalar adds an axis and takes none, in reading and in writing."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    labels = np.arange(6).reshape(3, 2)
    one = algebra.one().to_numpy()
    raw = labels.reshape(3, 2, *([1] * one.ndim)) * one
    keys = [
        True, False, np.True_, np.False_, (True, 0), (0, True), (Ellipsis, True, 0),
        (Ellipsis, True), (True, Ellipsis, 1), (Ellipsis, False, 0), (True, True),
        (True, slice(None), 1), (None, True, 0), (0, 1, True),
    ]  # fmt: skip
    arrays = [algebra.array(raw)]
    if not algebra._field.binary:
        arrays.append(arrays[0].split())
    first = (Ellipsis,) + (0,) * one.ndim
    for array, key in itertools.product(arrays, keys):
        want = labels[key]
        got = array[key]
        values = (got.to_loop() if isinstance(got, cdl.SplitArray) else got).to_numpy()
        assert values[first].shape == want.shape
        assert np.array_equal(values[first].real, want % (algebra.char or 10**9))
        target = algebra.array(raw.copy())
        target = target.split() if isinstance(array, cdl.SplitArray) else target
        target[key] = algebra(5)
        expected = labels.copy()
        expected[key] = 5
        stored = target.to_loop() if isinstance(target, cdl.SplitArray) else target
        assert np.array_equal(
            stored.to_numpy()[first].real, expected % (algebra.char or 10**9)
        )
    for key in ((True, 0, 0, 0), (Ellipsis, True, 0, 0, 0)):
        with pytest.raises(IndexError):
            arrays[0][key]


# The ends of the floating-point range


def _scaled(values: np.ndarray, power: int) -> np.ndarray:
    """Return an array times ``2**power``, which is exact while it is representable."""

    if np.iscomplexobj(values):
        return np.ldexp(values.real, power) + 1j * np.ldexp(values.imag, power)
    return np.ldexp(values, power)


RANGE = [
    (cdl.CDLoop(0), {}),
    (cdl.CDLoop(1), {}),
    (cdl.CDLoop(3), {}),
    (cdl.CDLoop(4), {}),
    (cdl.CDLoop(1), {"char": 0, "deg": 2}),
    (cdl.CDLoop(2), {"char": 0, "deg": 2}),
    (cdl.CDLoop(4), {"char": 0, "deg": 2}),
    (cdl.CentralProduct(1, 2), {}),
    (cdl.CentralProduct(1, 2), {"char": 0, "deg": 2}),
]


@pytest.mark.parametrize(("loop", "selectors"), RANGE)
def test_operations_scale_with_their_operands(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """An operation of degree k on operands times 2**e gives 2**(k e) times its result.

    Multiplying by a power of two is exact, so with integer operands the two
    sides agree in every bit whenever the result is representable. No
    operation may fail because a square inside it leaves the range.
    """

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 60)
    central = isinstance(loop, cdl.CentralProduct)
    operations: list[tuple[int, Any]] = [
        (1, lambda a, b, c: a + b),
        (1, lambda a, b, c: -a.conjugate()),
        (2, lambda a, b, c: a * b),
        (2, lambda a, b, c: a.square()),
        (2, lambda a, b, c: a.commutator(b)),
        (2, lambda a, b, c: a.jordan_product(b)),
        (3, lambda a, b, c: a.associator(b, c)),
        (3, lambda a, b, c: a.left_power(3)),
        (1, lambda a, b, c: a.right_power(1)),
        (1, lambda a, b, c: a.kappa()),
    ]
    if not central:
        operations += [
            (0, lambda a, b, c: a**0),
            (1, lambda a, b, c: a**1),
            (2, lambda a, b, c: a**2),
            (3, lambda a, b, c: a**3),
        ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            x, y, z = (algebra(_values(algebra, rng)) for _ in range(3))
            batch = algebra.array(_values(algebra, rng, (3,)))
            other = algebra.array(_values(algebra, rng, (3,)))
            triples = [(x, y, z), (batch, other, batch)]
            triples.append((batch.split(), other.split(), batch.split()))
            for (a, b, c), (degree, operation) in itertools.product(
                triples, operations
            ):
                base = _loop(operation(a, b, c))
                for power in (-900, -600, -300, 300, 600, 900):
                    if abs(degree * power) > 1000:
                        continue
                    factor = float(np.ldexp(1.0, power))
                    got = _loop(operation(a * factor, b * factor, c * factor))
                    assert np.array_equal(got, _scaled(base, degree * power))
                # One operand large and the other small, in a product of both
                if operation in (operations[2][1], operations[4][1]):
                    for power in (600, 900):
                        up, down = (
                            float(np.ldexp(1.0, power)),
                            float(np.ldexp(1.0, -power)),
                        )
                        assert np.array_equal(
                            _loop(operation(a * up, b * down, c)), base
                        )
            for item in (x, batch, batch.split()):
                for power in (-900, 900):
                    scaled = item * float(np.ldexp(1.0, power))
                    assert np.array_equal(scaled.weight(), item.weight())
                    assert np.array_equal(scaled.is_central(), item.is_central())
                    assert np.all(scaled == scaled.copy())


def _loop(item: Any) -> np.ndarray:
    """Return loop-basis coefficients of an element, an array or a component."""

    if isinstance(item, cdl.SplitArray):
        item = item.to_loop()
    if isinstance(item, (cdl.Element, cdl.ElementArray)):
        return item.to_numpy()
    return np.asarray(getattr(item, "coefficients", item))


@pytest.mark.parametrize(
    ("loop", "selectors"), [pair for pair in RANGE if isinstance(pair[0], cdl.CDLoop)]
)
def test_inverses_and_powers_at_every_scale(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """Inverses, negative powers and predicates do not depend on the scale."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    rng = np.random.default_rng(loop.order + 61)
    unit = algebra.one().to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        found = 0
        while found < 6:
            x = algebra(_values(algebra, rng))
            if not x.has_inverse():
                continue
            found += 1
            base = x.inverse().to_numpy()
            square = (x**-2).to_numpy()
            answers = (x.is_zero_divisor(), x.is_nilpotent())
            degree = len(x.minimal_polynomial())
            for power in (-1000, -600, -300, 300, 600, 1000):
                scaled = x * float(np.ldexp(1.0, power))
                array = algebra.array(np.stack([scaled.to_numpy(), x.to_numpy()]))
                assert scaled.has_inverse() is True
                assert (scaled.is_zero_divisor(), scaled.is_nilpotent()) == answers
                assert scaled**1 == scaled and scaled**0 == 1
                assert np.array_equal((scaled**1).to_numpy(), scaled.to_numpy())
                want = _scaled(base, -power)
                tolerance = 1e-12 * np.abs(want).max()
                for inverse in (scaled.inverse(), scaled**-1):
                    assert np.allclose(
                        inverse.to_numpy(), want, rtol=1e-12, atol=tolerance
                    )
                for item in (array, array.split()):
                    assert item.has_inverse().tolist() == [True, True]
                    inverses = _loop(item.inverse())
                    assert np.allclose(inverses[0], want, rtol=1e-12, atol=tolerance)
                    assert np.allclose(inverses[1], base, rtol=1e-12, atol=1e-12)
                    assert np.array_equal(_loop(item**1), _loop(item))
                if abs(2 * power) <= 1000:
                    want = _scaled(square, -2 * power)
                    assert np.allclose(
                        (scaled**-2).to_numpy(), want, rtol=1e-11,
                        atol=1e-11 * np.abs(want).max(),
                    )  # fmt: skip
                with warnings.catch_warnings():
                    # The coefficients of the polynomial may leave the range
                    warnings.simplefilter("ignore")
                    assert len(scaled.minimal_polynomial()) == degree
                    generated = algebra.subalgebra_generated_by(scaled)
                    assert generated.dimension == degree - 1
        # A power that is too small for the range is zero, not an error
        tiny = algebra(unit * 3) * float(np.ldexp(1.0, -900))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not (tiny**2).to_numpy().any() and not (tiny**5).to_numpy().any()


def test_scalar_division_reaches_the_bottom_of_the_range() -> None:
    """Dividing by a scalar below the normal range gives the representable quotient."""

    rng = np.random.default_rng(62)
    for selectors in ({}, {"char": 0, "deg": 2}):
        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
        x = algebra(_values(algebra, rng))
        array = algebra.array(_values(algebra, rng, (3,)))
        scalars: list[Any] = [3.0, 1e-300, 1e-320, 5e-324, 2.0**-1074, 2.0**1000]
        if selectors:
            scalars += [1 + 2j, 1e-320 + 2e-320j, 3e-320j]
        for scalar in scalars:
            for item in (x, array, array.split()):
                small = scalar if abs(scalar) < 1 else 1 / scalar
                quotient = _loop((item * small) / small)
                assert np.allclose(quotient, _loop(item), rtol=1e-14, atol=1e-13)
            assert np.array_equal((x / 3).to_numpy(), x.to_numpy() / 3)
    for item in (x, array):
        with pytest.raises(ZeroDivisionError):
            item / 0


def test_codes_do_not_depend_on_the_scale_of_their_generators() -> None:
    """Generators of any size, also of different sizes, give the same code."""

    rng = np.random.default_rng(63)
    for selectors in ({}, {"char": 0, "deg": 2}):
        algebra = cdl.LoopAlgebra(cdl.CDLoop(3), **selectors)
        for _ in range(4):
            x = algebra(_values(algebra, rng) * (rng.random((2, 8)) < 0.3))
            y = algebra(_values(algebra, rng) * (rng.random((2, 8)) < 0.3))
            if not x or not y:
                continue
            row = x.to_numpy().reshape(1, -1)
            for side in ("left", "right", "two-sided"):
                want = codes.generator_matrix(x, side=side)
                pair = codes.generator_matrix([x, y], side=side)
                ideal = codes.is_ideal(row, algebra, side=side)
                for power in (-1074, -1000, -600, 600, 1000):
                    factor = float(np.ldexp(1.0, power))
                    scaled = x * factor
                    found = codes.generator_matrix(scaled, side=side)
                    assert found.shape == want.shape
                    assert np.allclose(found, want, atol=1e-9)
                    mixed = codes.generator_matrix([scaled, y], side=side)
                    assert mixed.shape == pair.shape
                    assert np.allclose(mixed, pair, atol=1e-9)
                    assert codes.is_ideal(row * factor, algebra, side=side) is ideal
                    assert codes.is_ideal(found, algebra, side=side) is True
    real = cdl.LoopAlgebra(cdl.CDLoop(1))
    for scale in (1.0, 1e-200, 1e200, 5e-324, 1e308):
        scalar = real(scale)
        assert codes.generator_matrix(scalar).shape == (4, 4)
        assert codes.is_ideal(scalar.to_numpy().reshape(1, -1), real) is False


def test_linear_algebra_does_not_depend_on_the_scale_of_a_matrix() -> None:
    """Rank, null space and row reduction are those of the matrix at unit size."""

    rng = np.random.default_rng(64)
    for kind in (float, complex):
        for rows, columns in ((2, 2), (3, 5), (5, 3), (4, 4)):
            matrix = rng.integers(-3, 4, (rows, columns)).astype(kind)
            if kind is complex:
                matrix = matrix + 1j * rng.integers(-3, 4, (rows, columns))
            matrix[-1] = matrix[0] + matrix[1 % rows]
            rank = linalg.rank(matrix)
            reduced, pivots = linalg.row_reduce(matrix)
            kernel = linalg.nullspace(matrix)
            for power in (-1070, -900, 900, 1020):
                scaled = _scaled(matrix, power)
                assert linalg.rank(scaled) == rank
                again, same = linalg.row_reduce(scaled)
                assert same == pivots and np.allclose(again, reduced, atol=1e-12)
                assert linalg.nullspace(scaled).shape == kernel.shape
    for large in (np.eye(2) * 1e308, np.eye(2) * (1e308 + 1e308j), np.eye(3) * 5e-324):
        size = large.shape[0]
        assert linalg.rank(large) == size
        assert linalg.nullspace(large).shape == (0, size)
        reduced, pivots = linalg.row_reduce(large)
        assert pivots == tuple(range(size)) and np.array_equal(reduced, np.eye(size))
    # Complex entries whose modulus is out of range although both parts are not
    for size, columns in ((4, 6), (6, 6)):
        matrix = rng.integers(-7, 8, (size, columns)) + 1j * rng.integers(
            -7, 8, (size, columns)
        )
        matrix[0, 0] = 7 + 7j
        matrix[-1] = 1j * matrix[0]
        top = _scaled(matrix, 1020)
        assert np.isfinite(top.real).all() and np.isfinite(top.imag).all()
        assert linalg.rank(top) == linalg.rank(matrix) == size - 1
        assert linalg.nullspace(top).shape == linalg.nullspace(matrix).shape
        kernel = linalg.nullspace(top)
        assert np.allclose(matrix @ kernel.T, 0, atol=1e-10)
        assert linalg.row_reduce(top)[1] == linalg.row_reduce(matrix)[1]
    # Small entries are kept beside large ones
    wide = np.diag([1e200, 1e-200])
    assert linalg.rank(wide, tol=1e-210) == 2 and linalg.rank(wide, tol=0) == 2
    assert linalg.nullspace(wide, tol=1e-210).shape == (0, 2)
    assert linalg.row_reduce(wide, tol=1e-210)[1] == (0, 1)
    # A tolerance is scaled with the matrix
    nearly = np.array([[1.0, 0.0], [0.0, 1e-9]])
    for power in (0, 900, -900):
        scaled = _scaled(nearly, power)
        cut = float(np.ldexp(1e-6, power))
        assert linalg.rank(scaled) == 2 and linalg.rank(scaled, tol=cut) == 1
        assert linalg.row_reduce(scaled, tol=cut)[1] == (0,)


# Cancellation in the quadratic norm over C


def _exact_norm(values: np.ndarray) -> tuple[Fraction, Fraction]:
    """Return the real and imaginary parts of the sum of squares, exactly."""

    real = sum(
        (Fraction(v.real) ** 2 - Fraction(v.imag) ** 2 for v in values.tolist()),
        Fraction(0),
    )
    imaginary = sum(
        (2 * Fraction(v.real) * Fraction(v.imag) for v in values.tolist()), Fraction(0)
    )
    return real, imaginary


def _exact_reciprocal(values: np.ndarray) -> np.ndarray | None:
    """Return ``conjugate(d) / Q(d)`` from rational arithmetic, if it is in range."""

    real, imaginary = _exact_norm(values)
    size = real * real + imaginary * imaginary
    result = np.empty_like(values)
    for position, value in enumerate(values.tolist()):
        sign = 1 if position == 0 else -1
        a, b = sign * Fraction(value.real), sign * Fraction(value.imag)
        try:
            result[position] = complex(
                float((a * real + b * imaginary) / size),
                float((b * real - a * imaginary) / size),
            )
        except OverflowError:
            return None
    return result


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_complex_norms_that_cancel_are_decided_exactly(level: int) -> None:
    """Whether Q(d) vanishes, and the inverse, are those of the stored coefficients."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(level), char=0, deg=2)
    rng = np.random.default_rng(65 + level)
    size = 1 << level
    spectral = np.ones(size)
    checked = isotropic = 0
    for trial in range(120):
        large = float(2.0 ** int(rng.integers(20, 700)))
        d = np.zeros(size, dtype=complex)
        where = rng.permutation(size)
        d[where[0]], d[where[1]] = large, 1j * large
        kind = trial % 4
        if size > 2:
            if kind == 0:
                d[where[2]] = int(rng.integers(1, 4))
            elif kind == 2:
                d[where[2]] = float(np.ldexp(3.0, -int(rng.integers(0, 300))))
            elif kind == 3 and size > 3:
                d[where[2]], d[where[3]] = 1, 1j
        real, imaginary = _exact_norm(d)
        nonzero = bool(real or imaginary)
        x = algebra.split_array(spectral, d)[()]
        batch = algebra.split_array(spectral, np.stack([d, np.arange(1.0, size + 1)]))
        assert x.has_inverse() is nonzero
        assert batch.has_inverse().tolist() == [nonzero, True]
        if level <= 3:
            assert x.is_zero_divisor() is (not nonzero)
        if not nonzero:
            isotropic += 1
            for call in (x.inverse, lambda x=x: x**-1, batch.inverse):
                with pytest.raises(ZeroDivisionError):
                    call()
            for power in (-700, 300):
                assert (x * float(np.ldexp(1.0, power))).has_inverse() is False
            continue
        want = _exact_reciprocal(d)
        if want is None:
            continue
        checked += 1
        for inverse in (x.inverse(), x**-1, batch.inverse()[0]):
            got = np.asarray(inverse.epsilon().coefficients)
            assert np.allclose(got, want, rtol=1e-12, atol=0)
    assert isotropic >= 20 and (checked >= 40 or size == 2)


def test_minimal_polynomial_keeps_factors_that_rounding_would_cancel() -> None:
    """The degree is that of the stored coefficients, at ordinary sizes too."""

    real = cdl.LoopAlgebra(cdl.CDLoop(2))
    spectral = np.array([1.0, 1.0, 2.0, 3.0])
    x = real.split_array(spectral, np.array([1.0, 1e-9, 0.0, 0.0]))[()]
    # Three spectral values, none a root of (t - 1)^2 + 1e-18
    assert len(x.minimal_polynomial()) - 1 == 5
    assert real.subalgebra_generated_by(x).dimension == 5
    scalar = real.split_array(spectral, np.array([1.0, 0.0, 0.0, 0.0]))[()]
    assert len(scalar.minimal_polynomial()) - 1 == 3
    complex_ = cdl.LoopAlgebra(cdl.CDLoop(2), char=0, deg=2)
    large = float(2**30)
    apart = complex_.split_array(spectral, np.array([1.0, large, 1j * large, 1e-3]))[()]
    assert len(apart.minimal_polynomial()) - 1 == 5
    # Here (t - 1)^2 + Q(d - d_0) is (t - 1)^2, of which 1 is a root
    double = complex_.split_array(spectral, np.array([1.0, large, 1j * large, 0.0]))[()]
    assert len(double.minimal_polynomial()) - 1 == 4


# Integers and algorithms


def test_binary_fields_read_integer_sequences_of_mixed_range() -> None:
    """A sequence of integers is not rejected because NumPy would read floats."""

    for deg in (1, 3):
        algebra = cdl.LoopAlgebra(cdl.CDLoop(1), char=2, deg=deg)
        tail = () if deg == 1 else (deg,)
        for entries in (
            [2**63, -1, 0, 0],
            [2**64, -3, 1, 2**70],
            [-(2**63) - 1, 5, 2**63, 7],
        ):
            values = np.array(entries, dtype=object).reshape(2, 2)
            if tail:
                values = np.stack(
                    [values] + [np.zeros_like(values)] * (deg - 1), axis=-1
                )
            nested = values.tolist()
            want = np.array(values % 2, dtype=np.uint32)
            for made in (
                algebra(nested),
                algebra.array(nested),
                algebra.asarray(nested),
            ):
                assert np.array_equal(made.to_numpy(), want)
            assert np.array_equal(algebra(values).to_numpy(), want)
            matrix = values.reshape((2, 2) + tail).tolist()
            assert linalg.rank(matrix, field=algebra.field) == linalg.rank(
                want.reshape((2, 2) + tail), field=algebra.field
            )
        for wrong in (
            [[2**63, 1.5], [0, 0]],
            [[1.0, 0], [0, 0]],
            [[2**63, "1"], [0, 0]],
        ):
            with pytest.raises((TypeError, ValueError)):
                algebra(wrong if deg == 1 else [[[v] * deg for v in r] for r in wrong])


def test_multiply_refuses_the_table_algorithm_on_split_arrays_beside_a_scalar() -> None:
    """A scalar operand does not lift the restriction on the other operand."""

    for selectors in ({}, {"char": 7}):
        algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
        array = algebra.array(_values(algebra, np.random.default_rng(66), (3,)))
        split = array.split()
        for arguments in ((2, split), (split, 2)):
            with pytest.raises(TypeError, match="loop basis"):
                algebra.multiply(*arguments, algorithm="table")
            for name in ("auto", "split"):
                product = algebra.multiply(*arguments, algorithm=name)
                assert isinstance(product, cdl.SplitArray)
                assert np.all(product == split * 2)
        for operand in (array, array[0]):
            for arguments in ((2, operand), (operand, 2)):
                product = algebra.multiply(*arguments, algorithm="table")
                assert np.all(product == operand * 2)


# Indices, checked against NumPy itself


INDEX_PARTS: list[Any] = [
    0, -1, 1, slice(None), slice(1, None), slice(None, None, -1), slice(0, 0), Ellipsis,
    None, True, False, [0, 1], [1, 0, 1], np.array([[0, 1], [1, 0]]),
    np.array([True, False]), np.array([True, False, True]), np.array(1), [],
]  # fmt: skip


@pytest.mark.parametrize("selectors", [{}, {"char": 7}], ids=["R", "F7"])
def test_every_index_selects_and_assigns_what_numpy_does(
    selectors: dict[str, Any],
) -> None:
    """Reading and writing through an index agree with a NumPy array of labels."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1), **selectors)
    modulus = algebra.char or 10**9
    one = algebra.one().to_numpy()
    keys: list[Any] = list(INDEX_PARTS)
    keys += list(itertools.product(INDEX_PARTS, repeat=2))
    keys += [
        (slice(None), [0, 1], Ellipsis, [1, 2]),
        ([0, 1], Ellipsis, [1, 2]),
        ([0, 1], slice(None), [1, 2]),
        (Ellipsis, [0, 1], [1, 2]),
        ([1, 0], [0, 1], Ellipsis),
        (slice(None), [0, 1], Ellipsis, True, [1, 2]),
        (0, Ellipsis, [0, 1], None, [2, 0]),
    ]
    checked = 0
    for shape in ((2, 2, 3), (3,)):
        labels = np.arange(int(np.prod(shape))).reshape(shape) % modulus
        raw = (labels.reshape(shape + (1, 1)) * one).astype(algebra.dtype)
        for key, split in itertools.product(keys, (False, True)):
            try:
                want = labels[key]
            except IndexError:
                want = None
            array = algebra.array(raw)
            target = array.split() if split else array
            if want is None:
                with pytest.raises((IndexError, TypeError, ValueError)):
                    target[key]
                continue
            got = target[key]
            got = got.to_loop() if isinstance(got, cdl.SplitArray) else got
            assert np.array_equal(got.to_numpy()[..., 0, 0].real, want), key
            fresh = (np.arange(want.size).reshape(want.shape) + 3) % modulus
            expected = labels.copy()
            expected[key] = fresh
            if want.shape:
                values = (fresh.reshape(want.shape + (1, 1)) * one).astype(
                    algebra.dtype
                )
                value = algebra.array(values)
                value = value.split() if split else value
            else:
                value = algebra(int(fresh))
            target[key] = value
            stored = target.to_loop() if split else target
            assert np.array_equal(stored.to_numpy()[..., 0, 0].real, expected), key
            checked += 1
    assert checked > 500


# Large exponents


def test_large_powers_of_units_stay_exact() -> None:
    """A power with a huge exponent does not lose the result in the recurrence."""

    for selectors in ({}, {"char": 0, "deg": 2}):
        loop = cdl.CDLoop(3)
        algebra = cdl.LoopAlgebra(loop, **selectors)
        one = algebra.one()
        assert one ** (2**28) == one and one ** (2**60 + 1) == one
        assert (-one) ** (2**40) == one and (-one) ** (2**40 + 1) == -one
        assert algebra(2.0) ** 1000 == 2.0**1000
        assert algebra(0.5) ** 1074 == 2.0**-1074
        for index in (1, 2, 5):
            e = algebra.basis(loop.basis(index)) - algebra.basis(-loop.basis(index))
            e = e / 2 + (one + algebra.basis(loop.minus_one())) / 2 * 0
            # e has signed part e_index, of order four, and spectrum zero
            base = algebra.split_array(np.zeros(8), np.eye(8)[index])[()]
            for exponent, k in (
                (2**28, 0),
                (2**28 + 1, 1),
                (2**40 + 2, 2),
                (2**50 + 3, 3),
            ):
                assert base**exponent == base.left_power(k) if k else True
                signed = np.asarray((base**exponent).epsilon().coefficients)
                want = np.asarray((base.left_power(k or 4)).epsilon().coefficients)
                assert np.array_equal(signed, want)
        # A general element against the same power in rational arithmetic:
        # (a + b w)^k with w^2 = -b^2 for the signed part a + b e_3
        signed = np.zeros(8)
        signed[[0, 3]] = [0.6, 0.8]
        turn = algebra.split_array(np.ones(8), signed)[()]
        a, b = Fraction(0.6), Fraction(0.8)
        for exponent in (2, 3, 100, 1023, 1024):
            real, other = Fraction(1), Fraction(0)
            for _ in range(exponent):
                real, other = real * a - other * b, real * b + other * a
            power = np.asarray((turn**exponent).epsilon().coefficients)
            assert abs(power[0] - float(real)) <= 1e-12 * abs(float(real))
            assert abs(power[3] - float(other)) <= 1e-12 * abs(float(other))
            assert not np.delete(power, [0, 3]).any()
        left = np.asarray((turn ** (1000003 + 7777)).epsilon().coefficients)
        right = np.asarray((turn**1000003 * turn**7777).epsilon().coefficients)
        assert np.allclose(left, right, rtol=1e-8, atol=0)


# Coefficients of very different sizes in one operand


def _spread(rng: Any, shape: tuple[int, ...], complex_: bool, low: int = -900) -> Any:
    """Return small integers times independent powers of two."""

    values = rng.integers(-3, 4, shape) * np.ldexp(1.0, rng.integers(low, -low, shape))
    if complex_:
        extra = rng.integers(-3, 4, shape) * np.ldexp(
            1.0, rng.integers(low, -low, shape)
        )
        values = values + 1j * extra
    return values


def _near(got: np.ndarray, want: np.ndarray, tolerance: float = 1e-12) -> bool:
    """Compare coefficients, each to a tolerance relative to its own size."""

    for a, b in zip(np.ravel(got).tolist(), np.ravel(want).tolist(), strict=True):
        a, b = complex(a), complex(b)
        if not (np.isfinite(b.real) and np.isfinite(b.imag)):
            continue
        size = max(abs(b.real), abs(b.imag))
        if max(abs(a.real - b.real), abs(a.imag - b.imag)) > tolerance * size + 2e-323:
            return False
    return True


@pytest.mark.parametrize("selectors", [{}, {"char": 0, "deg": 2}], ids=["R", "C"])
@pytest.mark.parametrize("level", [0, 1, 2, 4])
def test_small_coefficients_survive_beside_large_ones(
    level: int, selectors: dict[str, Any]
) -> None:
    """Powers, inverses and minimal polynomials of operands with a wide spread."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)
    complex_ = bool(selectors)
    rng = np.random.default_rng(70 + level)
    size = 1 << level
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for trial in range(40):
            spectral = _spread(rng, (size,), complex_)
            d = _spread(rng, (size,), complex_)
            if trial % 4 == 0 and size > 1:
                d[1:] = 0
            x = algebra.split_array(spectral, d)[()]
            first = x**1
            assert np.array_equal(np.asarray(first.epsilon().coefficients), d)
            assert np.array_equal(np.asarray(first.spectrum()), spectral)
            assert x**0 == 1
            real, imaginary = _exact_norm(d.astype(complex))
            nonzero = bool(real or imaginary) and bool(np.all(spectral != 0))
            assert x.has_inverse() is nonzero
            if nonzero:
                want = _exact_reciprocal(d.astype(complex))
                if want is not None:
                    for inverse in (x.inverse(), x**-1):
                        got = np.asarray(inverse.epsilon().coefficients)
                        assert _near(got, want)
            # The degree counts the distinct spectral values that are not
            # roots of the polynomial of d, decided in rational arithmetic
            entries = [(Fraction(v.real), Fraction(v.imag)) for v in d.astype(complex)]
            scalar = not any(a or b for a, b in entries[1:])
            values = {
                (Fraction(v.real), Fraction(v.imag)) for v in spectral.astype(complex)
            }
            if scalar:
                kept = [value for value in values if value != entries[0]]
            else:
                kept = [
                    (p, q)
                    for p, q in values
                    if _at(entries, real, imaginary, p, q) != (0, 0)
                ]
            degree = len(kept) + (1 if scalar else 2)
            assert len(x.minimal_polynomial()) - 1 == degree
            assert algebra.subalgebra_generated_by(x).dimension == degree


def _at(
    entries: Any, real: Fraction, imaginary: Fraction, p: Fraction, q: Fraction
) -> Any:
    """Evaluate ``t^2 - 2 d_0 t + Q(d)`` at ``t = p + q i`` in rational arithmetic."""

    a, b = entries[0]
    return (
        p * p - q * q - 2 * (a * p - b * q) + real,
        2 * p * q - 2 * (a * q + b * p) + imaginary,
    )


def test_wide_spreads_keep_their_small_terms() -> None:
    """The cases in which rescaling by the largest coefficient loses the answer."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1))
    large, small = float(2**600), float(2.0**-600)
    x = algebra.split_array(np.ones(2), np.array([large, small]))[()]
    assert np.asarray((x**1).epsilon().coefficients).tolist() == [large, small]
    roots = algebra.split_array(np.array([large, small]), np.zeros(2))[()]
    assert roots.minimal_polynomial().tolist() == [0.0, 1.0, -large, 1.0]
    assert algebra.subalgebra_generated_by(roots).dimension == 3
    matrix = np.diag([1e200, 1e-200])
    for tol in (1e-210, 0):
        assert linalg.rank(matrix, tol=tol) == 2
        assert linalg.nullspace(matrix, tol=tol).shape == (0, 2)
        reduced, pivots = linalg.row_reduce(matrix, tol=tol)
        assert pivots == (0, 1) and np.array_equal(reduced, np.eye(2))
    assert linalg.rank(matrix, tol=1e-190) == 1
    # Over C the norm of this signed part is 1, although its squares cancel
    complex_ = cdl.LoopAlgebra(cdl.CDLoop(2), char=0, deg=2)
    d = np.array([0, large, 1j * large, 1])
    y = complex_.split_array(np.ones(4), d)[()]
    assert y.has_inverse() is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        square = np.asarray((y**2).epsilon().coefficients).tolist()
        assert square == [-1, 0, 0, 0]
        assert y.minimal_polynomial().tolist() == [-1, 1, -1, 1]


def test_scalar_division_of_independent_subnormal_values() -> None:
    """A quotient of two values below the normal range is the rounded quotient."""

    rng = np.random.default_rng(71)
    tiny = float(np.nextafter(0.0, 1.0))
    real = cdl.LoopAlgebra(cdl.CDLoop(0))
    complex_ = cdl.LoopAlgebra(cdl.CDLoop(0), char=0, deg=2)
    assert (real(tiny) / 1e-320).to_numpy()[0, 0] == tiny / 1e-320
    quotient = complex(tiny) / complex(1e-320, 1e-320)
    assert (complex_(tiny) / complex(1e-320, 1e-320)).to_numpy()[0, 0] == quotient
    for _ in range(300):
        a, b = (int(v) * tiny for v in rng.integers(-2000, 2000, 2))
        c, e = (int(v) * tiny for v in rng.integers(-3000, 3000, 2))
        if c:
            assert (real(a) / c).to_numpy()[0, 0] == a / c
        if not (c or e):
            continue
        got = complex((complex_(complex(a, b)) / complex(c, e)).to_numpy()[0, 0])
        p, q, r, s = (Fraction(v) for v in (a, b, c, e))
        size = r * r + s * s
        want = complex(float((p * r + q * s) / size), float((q * r - p * s) / size))
        assert abs(got - want) <= 4e-16 * abs(want)


def test_binary_fields_read_numpy_integers_in_mixed_sequences() -> None:
    """NumPy integer scalars of any type count as integers beside Python ones."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1), char=2)
    cases = [
        [[2**63 + 1, -1], [0, 0]],
        [[np.uint64(2**63 + 1), -1], [0, 0]],
        [[2**63 + 1, np.int64(-1)], [0, 0]],
        [[np.uint64(3), np.int8(-1)], [np.int64(2), 0]],
        [[np.uint64(2**64 - 1), 2**70], [np.int32(-5), np.uint8(4)]],
    ]
    for values in cases:
        want = [[int(v) % 2 for v in row] for row in values]
        for build in (algebra, algebra.array, algebra.asarray):
            assert build(values).to_numpy().tolist() == want
        assert linalg.rank(values, field=algebra.field) == linalg.rank(
            np.array(want, dtype=np.uint32), field=algebra.field
        )
    for wrong in ([[np.float64(1.0), -1], [0, 0]], [[np.bool_(True), 2**70], [0, 0]]):
        with pytest.raises(TypeError):
            algebra(wrong)
