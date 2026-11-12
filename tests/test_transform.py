"""Tests of the Walsh--Hadamard transform and dyadic convolution."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyfastcd as fcd
import pytest

import pycdloop as cdl

try:
    from . import reference
except ImportError:  # Support collection from inside the tests directory.
    import reference

FIELDS = [
    None,
    "complex",
    fcd.FiniteField(char=3),
    fcd.FiniteField(char=13),
    fcd.FiniteField(char=2147483647),
    fcd.FiniteField(char=7, deg=2),
    fcd.FiniteField(char=13, deg=3),
]
FIELD_IDS = ["R", "C", "F3", "F13", "F2147483647", "F7^2", "F13^3"]


def _ring(field: Any) -> Any:
    """Return the reference ring that matches a coefficient field."""

    if field is None:
        return reference.Rationals()
    if field == "complex":
        return reference.GaussianRationals()
    return reference.FiniteField(field.char, field.modulus)


def _dtype(field: Any) -> Any:
    """Return the storage type of a coefficient field."""

    if field is None:
        return np.float64
    return np.complex128 if field == "complex" else np.uint32


def _random(field: Any, shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Return exactly representable random coefficients."""

    rng = np.random.default_rng(seed)
    if field is None:
        return rng.integers(-9, 10, shape).astype(np.float64)
    if field == "complex":
        return rng.integers(-9, 10, shape) + 1j * rng.integers(-9, 10, shape)
    tail = () if field.deg == 1 else (field.deg,)
    return rng.integers(0, field.char, shape + tail).astype(np.uint32)


def _keyword(field: Any) -> dict[str, Any]:
    """Return the keyword arguments that select a field."""

    return {} if field is None or field == "complex" else {"field": field}


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_wht_matches_the_matrix_definition(field: Any, level: int) -> None:
    """The butterflies agree with the transform matrix applied exactly."""

    ring = _ring(field)
    values = _random(field, (1 << level,), 10 * level + 1)
    expected = reference.walsh_hadamard(ring, reference.to_ring(ring, values, 1))
    result = cdl.wht(values, **_keyword(field))
    assert result.dtype == _dtype(field)
    assert np.array_equal(result, reference.from_ring(ring, expected, 1, _dtype(field)))


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
def test_wht_applied_twice_scales_by_the_length(field: Any) -> None:
    """The transform is an involution up to the factor N."""

    values = _random(field, (3, 32), 7)
    twice = cdl.wht(cdl.wht(values, **_keyword(field)), **_keyword(field))
    if field is None or field == "complex":
        assert np.array_equal(twice, 32 * values)
    else:
        scaled = values.astype(np.uint64) * 32 % field.char
        assert np.array_equal(twice, scaled.astype(np.uint32))


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
def test_wht_axis_selects_an_index_axis(field: Any) -> None:
    """Each index axis transforms like the last one, whatever the field."""

    values = _random(field, (4, 2, 8), 21)
    keyword = _keyword(field)
    for axis in (0, 1, 2, -1, -2, -3):
        moved = np.moveaxis(values, axis % 3, 2)
        expected = np.moveaxis(cdl.wht(moved, **keyword), 2, axis % 3)
        result = cdl.wht(values, axis=axis, **keyword)
        assert result.shape == values.shape
        assert result.flags.c_contiguous
        assert np.array_equal(result, expected)


def test_wht_does_not_transform_the_polynomial_axis() -> None:
    """Over an extension field every polynomial coordinate transforms alone."""

    field = fcd.FiniteField(char=7, deg=2)
    prime = fcd.FiniteField(char=7)
    values = _random(field, (5, 8), 4)
    result = cdl.wht(values, field=field)
    for coordinate in range(2):
        expected = cdl.wht(values[..., coordinate], field=prime)
        assert np.array_equal(result[..., coordinate], expected)


def test_wht_converts_and_reduces_inputs() -> None:
    """Integers become floats and residues are reduced before transforming."""

    assert cdl.wht([1, 0, 0, 0]).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert cdl.wht([True, False]).dtype == np.float64
    assert cdl.wht(np.array([1, 2], dtype=np.float32)).dtype == np.float64
    field = fcd.FiniteField(char=13)
    assert np.array_equal(cdl.wht([14, -1], field=field), cdl.wht([1, 12], field=field))
    values = np.arange(4.0)
    cdl.wht(values)
    assert values.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_wht_rejects_invalid_arguments() -> None:
    """Shape, axis and coefficient errors raise the documented exceptions."""

    field = fcd.FiniteField(char=7, deg=2)
    with pytest.raises(ValueError):
        cdl.wht([1, 2, 3])
    with pytest.raises(ValueError):
        cdl.wht(np.zeros((4, 0)))
    with pytest.raises(ValueError):
        cdl.wht(3.0)
    with pytest.raises(ValueError):
        cdl.wht(np.zeros((2, 4)), axis=2)
    with pytest.raises(ValueError):
        cdl.wht(np.zeros((2, 4)), axis=-3)
    with pytest.raises(ValueError):
        # The polynomial axis is not an index axis
        cdl.wht(np.zeros((4, 2), dtype=np.uint32), axis=1, field=field)
    with pytest.raises(ValueError):
        cdl.wht(np.zeros((3, 2), dtype=np.uint32), field=field)
    with pytest.raises(TypeError):
        cdl.wht(["a", "b"])
    with pytest.raises(TypeError):
        cdl.wht([1.0, 2.0], field=fcd.FiniteField(char=7))
    with pytest.raises(TypeError):
        cdl.wht([1, 2], field=7)


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
def test_wht_large_level(field: Any) -> None:
    """A transform of length 4096 inverts exactly on integer data."""

    values = _random(field, (4096,), 99)
    keyword = _keyword(field)
    twice = cdl.wht(cdl.wht(values, **keyword), **keyword)
    if field is None or field == "complex":
        assert np.array_equal(twice, 4096 * values)
    else:
        scaled = values.astype(np.uint64) * 4096 % field.char
        assert np.array_equal(twice, scaled.astype(np.uint32))


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_dyadic_convolution_matches_the_definition(field: Any, level: int) -> None:
    """Three transforms reproduce the XOR convolution computed term by term."""

    ring = _ring(field)
    a = _random(field, (1 << level,), 5 * level + 2)
    b = _random(field, (1 << level,), 5 * level + 3)
    if field == "complex":
        result = cdl.dyadic_convolve(a, b)
    else:
        result = cdl.dyadic_convolve(a, b, **_keyword(field))
    expected = reference.dyadic_convolve(
        ring, reference.to_ring(ring, a, 1), reference.to_ring(ring, b, 1)
    )
    assert result.dtype == _dtype(field)
    assert np.array_equal(result, reference.from_ring(ring, expected, 1, _dtype(field)))


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
def test_dyadic_convolution_broadcasts_leading_axes(field: Any) -> None:
    """Leading axes broadcast and each pair of records convolves alone."""

    a = _random(field, (3, 1, 8), 31)
    b = _random(field, (2, 8), 32)
    keyword = _keyword(field)
    result = cdl.dyadic_convolve(a, b, **keyword)
    assert result.shape[:2] == (3, 2)
    for i in range(3):
        for j in range(2):
            single = cdl.dyadic_convolve(a[i, 0], b[j], **keyword)
            assert np.array_equal(result[i, j], single)


def test_dyadic_convolution_is_the_group_algebra_product() -> None:
    """Basis vectors convolve to the basis vector of the XOR of their indices."""

    for p in range(8):
        for q in range(8):
            a, b = np.zeros(8), np.zeros(8)
            a[p], b[q] = 1.0, 1.0
            expected = np.zeros(8)
            expected[p ^ q] = 1.0
            assert np.array_equal(cdl.dyadic_convolve(a, b), expected)
    mixed = cdl.dyadic_convolve([1, 2], [1j, 0])
    assert mixed.dtype == np.complex128 and mixed.tolist() == [1j, 2j]


def test_dyadic_convolution_rejects_invalid_arguments() -> None:
    """Mismatched or invalid operands raise the documented exceptions."""

    field = fcd.FiniteField(char=7, deg=2)
    with pytest.raises(ValueError):
        cdl.dyadic_convolve([1, 2, 3, 4], [1, 2])
    with pytest.raises(ValueError):
        cdl.dyadic_convolve([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        cdl.dyadic_convolve(1.0, 2.0)
    with pytest.raises(ValueError):
        cdl.dyadic_convolve(np.zeros(2, dtype=np.uint32), [1, 2], field=field)
    with pytest.raises(TypeError):
        cdl.dyadic_convolve(["a", "b"], [1, 2])
    with pytest.raises(TypeError):
        cdl.dyadic_convolve([0.5, 1.5], [1, 2], field=fcd.FiniteField(char=5))
