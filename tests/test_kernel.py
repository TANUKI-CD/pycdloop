"""Tests of the compiled Walsh--Hadamard kernel against the reference butterflies."""

from __future__ import annotations

import numpy as np
import pytest

from pycdloop import _core, _transform

try:  # Support both unittest discovery and package-style test collection.
    from . import reference
except ImportError:  # pragma: no cover - depends on the collection mechanism
    import reference


def _butterflies(values: np.ndarray, modulus: int | None = None) -> np.ndarray:
    """Return the reference transform of an array along its last axis."""

    work = np.array(values, order="C", copy=True)
    _transform._butterflies(work, modulus)
    return work


@pytest.mark.parametrize("level", range(0, 13))
def test_real_kernel_is_bit_identical_to_the_butterflies(level: int) -> None:
    """Agree bit for bit with the stage-by-stage evaluation on random doubles."""

    rng = np.random.default_rng(level)
    values = rng.standard_normal((3, 1 << level)) * 2.0 ** rng.integers(-40, 40)
    work = values.copy()
    _core.transform(work, 1 << level, 1)
    assert work.tobytes() == _butterflies(values).tobytes()


@pytest.mark.parametrize("level", range(0, 11))
def test_complex_kernel_is_bit_identical_to_the_butterflies(level: int) -> None:
    """Treat a complex coefficient as two lanes of doubles."""

    rng = np.random.default_rng(100 + level)
    shape = (2, 1 << level)
    values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    work = values.copy()
    _core.transform(work, 1 << level, 2)
    assert work.tobytes() == _butterflies(values).tobytes()


@pytest.mark.parametrize("lanes", [1, 2, 3, 5, 12])
@pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 7])
def test_lanes_transform_each_lane_separately(level: int, lanes: int) -> None:
    """Transform along the coefficient axis and leave the lane axis alone."""

    rng = np.random.default_rng(1000 * lanes + level)
    values = rng.standard_normal((2, 1 << level, lanes))
    work = values.copy()
    _core.transform(work, 1 << level, lanes)
    expected = np.moveaxis(_butterflies(np.moveaxis(values, -1, -2)), -2, -1)
    assert np.array_equal(work, expected)


def test_real_kernel_matches_the_definition_exactly_on_integers() -> None:
    """Agree with the transform matrix on integer inputs."""

    ring = reference.Rationals()
    rng = np.random.default_rng(7)
    for level in range(0, 7):
        values = rng.integers(-9, 10, size=1 << level).astype(np.float64)
        work = values.copy()
        _core.transform(work, 1 << level, 1)
        exact = reference.walsh_hadamard(ring, reference.to_ring(ring, values, 1))
        assert work.tolist() == [float(value) for value in exact]


@pytest.mark.parametrize("modulus", [3, 13, 65521, 2147483647])
@pytest.mark.parametrize("lanes", [1, 2, 5])
@pytest.mark.parametrize("level", [0, 1, 3, 4, 6, 9])
def test_modular_kernel_matches_the_butterflies(
    level: int, lanes: int, modulus: int
) -> None:
    """Return the canonical residues of the reference transform."""

    rng = np.random.default_rng(level + 10 * lanes)
    shape = (2, 1 << level) if lanes == 1 else (2, 1 << level, lanes)
    values = rng.integers(0, modulus, size=shape, dtype=np.int64)
    work = values.copy()
    _core.transform_mod(work, 1 << level, lanes, modulus)
    moved = values if lanes == 1 else np.moveaxis(values, -1, -2)
    expected = _butterflies(moved, modulus)
    if lanes != 1:
        expected = np.moveaxis(expected, -2, -1)
    assert np.array_equal(work, expected)
    assert work.min() >= 0 and work.max() < modulus


@pytest.mark.parametrize("interval", [1, 2, 3, 4, 7, 30])
@pytest.mark.parametrize("lanes", [1, 3])
def test_modular_result_does_not_depend_on_the_reduction_interval(
    lanes: int, interval: int
) -> None:
    """Reduce at any interval without changing the residues."""

    modulus = 2147483647
    level = 9
    shape = (1 << level,) if lanes == 1 else (1 << level, lanes)
    # The largest residue in every position maximises intermediate growth
    values = np.full(shape, modulus - 1, dtype=np.int64)
    work = values.copy()
    _core.transform_mod(work, 1 << level, lanes, modulus, interval)
    exact = values.astype(object)
    moved = exact if lanes == 1 else np.moveaxis(exact, -1, -2)
    expected = _butterflies(moved) % modulus
    if lanes != 1:
        expected = np.moveaxis(expected, -2, -1)
    assert np.array_equal(work, expected.astype(np.int64))


def test_kernel_releases_large_buffers_correctly() -> None:
    """Transform a batch large enough to run without the interpreter lock."""

    rng = np.random.default_rng(3)
    values = rng.standard_normal((64, 1024))
    work = values.copy()
    _core.transform(work, 1024, 1)
    assert work.tobytes() == _butterflies(values).tobytes()
    residues = rng.integers(0, 13, size=(64, 1024), dtype=np.int64)
    work = residues.copy()
    _core.transform_mod(work, 1024, 1, 13)
    assert np.array_equal(work, _butterflies(residues, 13))


def test_kernel_accepts_empty_and_single_coefficient_records() -> None:
    """Leave records of one coefficient unchanged and accept empty buffers."""

    one = np.array([[2.5], [-1.0]])
    _core.transform(one, 1, 1)
    assert one.tolist() == [[2.5], [-1.0]]
    empty = np.zeros((0, 8))
    _core.transform(empty, 8, 1)
    residues = np.array([[5], [12]], dtype=np.int64)
    _core.transform_mod(residues, 1, 1, 7)
    assert residues.tolist() == [[5], [5]]


def test_kernel_rejects_invalid_arguments() -> None:
    """Raise for bad dimensions, lanes, moduli, intervals and buffers."""

    data = np.zeros(8)
    residues = np.zeros(8, dtype=np.int64)
    for dimension in (0, 3, 6, -4):
        with pytest.raises(ValueError):
            _core.transform(data, dimension, 1)
    with pytest.raises(ValueError):
        _core.transform(data, 8, 0)
    with pytest.raises(ValueError):
        _core.transform(data, 16, 1)
    with pytest.raises(ValueError):
        _core.transform(data, 4, 3)
    for modulus in (0, 1, -7, 2**31, 2**40):
        with pytest.raises(ValueError):
            _core.transform_mod(residues, 8, 1, modulus)
    for interval in (0, 31, -1):
        with pytest.raises(ValueError):
            _core.transform_mod(residues, 8, 1, 13, interval)
    frozen = np.zeros(8)
    frozen.setflags(write=False)
    with pytest.raises((BufferError, TypeError, ValueError)):
        _core.transform(frozen, 8, 1)
    with pytest.raises(TypeError):
        _core.transform([0.0] * 8, 8, 1)
    with pytest.raises(TypeError):
        _core.transform(data, "8", 1)


def test_package_transforms_use_the_kernel_for_every_field() -> None:
    """Route real, complex and residue transforms through the compiled kernel."""

    rng = np.random.default_rng(11)
    real = rng.standard_normal((4, 64))
    assert _transform._hadamard(real).tobytes() == _butterflies(real).tobytes()
    values = real + 1j * rng.standard_normal((4, 64))
    assert _transform._hadamard(values).tobytes() == _butterflies(values).tobytes()
    residues = rng.integers(0, 13, size=(4, 64, 3)).astype(np.uint32)
    transformed = _transform._hadamard_mod(residues, 13, 3)
    expected = _butterflies(np.moveaxis(residues.astype(np.int64), -1, -2), 13)
    assert transformed.dtype == np.uint32
    assert np.array_equal(transformed, np.moveaxis(expected, -2, -1))
    counted = np.array([1, 2, 3, 4], dtype=object)
    assert _transform._hadamard(counted).tolist() == [10, -2, -4, 0]


def _reference_split(field, coefficients):
    """Return the split form computed with whole-array operations."""

    tail = (slice(None),) * (1 + field.tail)
    plus = coefficients[(Ellipsis, 0) + tail]
    minus = coefficients[(Ellipsis, 1) + tail]
    return field.wht(field.add(plus, minus)), field.subtract(plus, minus)


def _reference_unsplit(field, spectral, signed, size):
    """Return loop-basis coefficients computed with whole-array operations."""

    group = field.divide_int(field.wht(spectral), size)
    plus = field.halve(field.add(group, signed))
    minus = field.halve(field.subtract(group, signed))
    return np.stack([plus, minus], axis=plus.ndim - field.tail - 1)


FIELDS = [
    {},
    {"char": 0, "deg": 2},
    {"char": 3},
    {"char": 65521},
    {"char": 2147483647},
    {"char": 13, "deg": 3},
    {"char": 2147483647, "deg": 2},
]


@pytest.mark.parametrize("selectors", FIELDS)
@pytest.mark.parametrize("level", [0, 1, 2, 3, 5, 8])
def test_coordinate_change_matches_the_whole_array_formulas(
    level: int, selectors: dict[str, int]
) -> None:
    """Agree with the split and unsplit formulas evaluated by array operations."""

    from pycdloop._field import _coefficient_field

    field = _coefficient_field(**selectors)
    size = 1 << level
    rng = np.random.default_rng(level + field.char % 97 + field.deg)
    shape = (3, 2, size) + field.shape
    if field.kind == "finite":
        values = rng.integers(0, field.char, size=shape).astype(np.uint32)
        # The largest residues exercise the widest intermediate values
        values[0] = field.char - 1
    elif field.kind == "complex":
        values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    else:
        values = rng.standard_normal(shape)
    spectral, signed = field.split(values, size)
    expected = _reference_split(field, values)
    assert np.array_equal(spectral, expected[0])
    assert np.array_equal(signed, expected[1])
    assert spectral.dtype == signed.dtype == field.dtype
    loop = field.unsplit(spectral, signed, size)
    assert np.array_equal(loop, _reference_unsplit(field, spectral, signed, size))
    if field.kind == "finite":
        assert np.array_equal(loop, values)
        assert int(loop.max()) < field.char
    else:
        assert np.allclose(loop, values, rtol=1e-12, atol=1e-12)


def test_real_coordinate_change_is_bit_identical_to_the_formulas() -> None:
    """Round exactly as the whole-array formulas do over the reals."""

    from pycdloop._field import _coefficient_field

    field = _coefficient_field()
    rng = np.random.default_rng(5)
    for level in range(0, 11):
        size = 1 << level
        values = rng.standard_normal((4, 2, size)) * 2.0 ** rng.integers(-30, 30)
        spectral, signed = field.split(values, size)
        expected = _reference_split(field, values)
        assert spectral.tobytes() == expected[0].tobytes()
        assert signed.tobytes() == expected[1].tobytes()
        loop = field.unsplit(spectral, signed, size)
        reference_loop = _reference_unsplit(field, spectral, signed, size)
        assert loop.tobytes() == reference_loop.tobytes()


def test_coordinate_change_handles_empty_and_broadcast_parts() -> None:
    """Accept empty batches and parts with different leading shapes."""

    from pycdloop._field import _coefficient_field

    field = _coefficient_field()
    spectral, signed = field.split(np.zeros((0, 2, 8)), 8)
    assert spectral.shape == signed.shape == (0, 8)
    assert field.unsplit(spectral, signed, 8).shape == (0, 2, 8)
    rng = np.random.default_rng(2)
    first, second = rng.standard_normal((1, 8)), rng.standard_normal((3, 8))
    loop = field.unsplit(first, second, 8)
    expected = _reference_unsplit(field, np.broadcast_to(first, (3, 8)), second, 8)
    assert np.array_equal(loop, expected)


def test_kernel_rejects_buffers_of_other_element_types() -> None:
    """Raise instead of reinterpreting storage of another type."""

    wrong = [
        np.empty(8, dtype=object),
        np.zeros(8, dtype=np.float32),
        np.zeros(8, dtype=np.int64),
        np.zeros(8, dtype=">f8"),
    ]
    for array in wrong:
        with pytest.raises((TypeError, ValueError, BufferError)):
            _core.transform(array, 8, 1)
    with pytest.raises((TypeError, ValueError, BufferError)):
        _core.transform(np.zeros(16)[::2], 8, 1)
    with pytest.raises((TypeError, ValueError, BufferError)):
        _core.transform_mod(np.zeros(8), 8, 1, 13)
    target = np.zeros(8), np.zeros(8)
    with pytest.raises((TypeError, ValueError, BufferError)):
        _core.split(np.empty(16, dtype=object), *target, 8, 1)
    residues = np.zeros(8, dtype=np.uint32), np.zeros(8, dtype=np.uint32)
    with pytest.raises((TypeError, ValueError, BufferError)):
        _core.split_mod(np.zeros(16), *residues, 8, 1, 13)
    with pytest.raises(ValueError):
        _core.split(np.zeros(8), *target, 8, 1)
    with pytest.raises(ValueError):
        _core.split_mod(np.zeros(16, dtype=np.uint32), *residues, 8, 1, 12)
    with pytest.raises(ValueError):
        _core.unsplit_mod(*residues, np.zeros(16, dtype=np.uint32), 8, 1, 13, 0)
    frozen = np.zeros(8)
    frozen.setflags(write=False)
    with pytest.raises((TypeError, ValueError, BufferError)):
        _core.split(np.zeros(16), frozen, np.zeros(8), 8, 1)
