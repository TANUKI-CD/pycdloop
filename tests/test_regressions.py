"""Regression tests for defects that have been fixed in :mod:`pycdloop`.

Each test pins the corrected behaviour on the smallest input that showed the
defect, so that it cannot return unnoticed.
"""

from __future__ import annotations

import numpy as np
import pyfastcd as fcd
import pytest

import pycdloop as cdl


def test_integer_powers_are_exact() -> None:
    """Keep powers of an element with integer coefficients free of rounding."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2))
    x = algebra([[2.0, 3.0, 3.0, 1.0], [3.0, -1.0, -2.0, 0.0]])
    cube = (x * x) * x
    assert x**3 == cube
    assert (x**3).coefficients.tobytes() == cube.coefficients.tobytes()
    assert x**5 == x * (x * (x * (x * x)))
    assert x**2 == x.square()
    # The signed component has quadratic norm 43, which the recurrence for
    # powers must use exactly
    np.testing.assert_array_equal(x.epsilon().coefficients, [-1.0, 4.0, 5.0, 1.0])
    np.testing.assert_array_equal(
        (x**3).epsilon().coefficients, [125.0, -156.0, -195.0, -39.0]
    )
    np.testing.assert_array_equal((x**3).spectrum(), [729.0, 27.0, 125.0, 27.0])
    batch = algebra.array(np.stack([x.coefficients, cube.coefficients]))
    assert (batch**3 == (batch * batch) * batch).all()
    assert ((batch.split() ** 3).to_loop() == batch**3).all()


@pytest.mark.parametrize("selectors", [{"char": 7}, {"char": 7, "deg": 2}])
def test_division_by_one_field_scalar_keeps_the_element_shape(
    selectors: dict[str, int],
) -> None:
    """Return an element of the right shape when dividing by a single scalar."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2), **selectors)
    x = algebra.random(1)
    for divisor in (3, algebra.field(3), np.int64(3)):
        quotient = x / divisor
        assert isinstance(quotient, cdl.Element)
        assert quotient.shape == algebra.coefficient_shape
        assert quotient * 3 == x
    product = x * x
    quotient = product / 3
    assert quotient.coefficients.shape == algebra.coefficient_shape
    assert quotient * 3 == product
    batch = algebra.random(2, batch_shape=(3,))
    assert (batch / 3).batch_shape == (3,)
    assert ((batch / 3) * 3 == batch).all()
    assert ((batch.split() / 3).to_loop() * 3 == batch).all()
    if algebra.deg > 1:
        scalar = algebra.field([2, 5])
        assert isinstance(scalar, fcd.Element)
        assert (x / scalar).shape == algebra.coefficient_shape
        assert (x / scalar) * scalar == x
        assert (x * scalar).shape == algebra.coefficient_shape


def test_automatic_choices_are_pinned() -> None:
    """Keep the automatic algorithm choices that the benchmark supports."""

    from pycdloop import _signed

    assert _signed.ROWS_MAX_LEVEL == 4
    for loop in (cdl.CDLoop(0), cdl.CDLoop(5), cdl.CentralProduct(2, 3)):
        assert cdl.LoopAlgebra(loop).default_algorithm == "split"
    expected = {
        (1, 3): "native",
        (2, 2): "rows",
        (4, 4): "rows",
        (4, 6): "rows",
        (5, 5): "schedule",
        (6, 5): "schedule",
    }
    for levels, name in expected.items():
        algebra = cdl.LoopAlgebra(cdl.CentralProduct(*levels))
        assert algebra.default_signed_algorithm == name, levels
    complex_algebra = cdl.LoopAlgebra(cdl.CentralProduct(1, 3), char=0, deg=2)
    assert complex_algebra.default_signed_algorithm == "rows"
