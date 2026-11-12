"""Tests of the reused temporaries behind products of loop-basis arrays."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import pycdloop as cdl
from pycdloop import _kernel, testing

SELECTORS = [{}, {"char": 0, "deg": 2}, {"char": 13}, {"char": 13, "deg": 3}]
LOOPS = [cdl.CDLoop(0), cdl.CDLoop(3), cdl.CDLoop(5), cdl.CentralProduct(1, 2)]


def _plain_product(algebra, x, y):
    """Multiply loop-basis coefficients without the workspace."""

    left = _kernel.to_split(algebra, x)
    right = _kernel.to_split(algebra, y)
    return _kernel.from_split(algebra, _kernel.multiply(algebra, "split", left, right))


@pytest.mark.parametrize("selectors", SELECTORS)
@pytest.mark.parametrize("loop", LOOPS)
def test_array_products_equal_the_products_without_a_workspace(loop, selectors):
    """Return the same bits whether or not temporaries are reused."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    for shapes in (
        ((4,), (4,)),
        ((3, 1), (2,)),
        ((1,), (5,)),
        ((), (2,)),
        ((0,), (1,)),
    ):
        x = algebra.random(1, batch_shape=shapes[0])
        y = algebra.random(2, batch_shape=shapes[1])
        expected = _plain_product(algebra, x.coefficients, y.coefficients)
        for _ in range(3):
            product = x * y
            assert product.coefficients.tobytes() == expected.tobytes()
            assert product.batch_shape == np.broadcast_shapes(*shapes)
        again = algebra.multiply(x, y)
        assert again.coefficients.tobytes() == expected.tobytes()


def test_results_do_not_alias_the_workspace():
    """Keep earlier results intact when later products reuse the temporaries."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(4))
    x = algebra.random(1, batch_shape=(6,))
    y = algebra.random(2, batch_shape=(6,))
    z = algebra.random(3, batch_shape=(6,))
    first = x * y
    saved = first.coefficients.copy()
    second = y * z
    third = x * x
    assert np.array_equal(first.coefficients, saved)
    assert not np.shares_memory(first.coefficients, second.coefficients)
    assert not np.shares_memory(second.coefficients, third.coefficients)
    for array in _kernel._WORKSPACE.arrays.values():
        assert not np.shares_memory(array, first.coefficients)
        assert not np.shares_memory(array, x.coefficients)


def test_element_operands_are_read_and_never_modified():
    """Leave the forms kept by an element untouched by products with arrays."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(4))
    # Integer coefficients keep every product exact, whichever algorithm
    # pyfastcd selects for single products and for batches
    array = testing.random_elements(algebra, 3, 5)
    element = testing.random_elements(algebra, 1, 4)[0]
    spectral, signed = (part.copy() for part in element._data("split"))
    coefficients = element.coefficients.copy()
    for _ in range(3):
        left = element * array
        right = array * element
    assert np.array_equal(element._data("split")[0], spectral)
    assert np.array_equal(element._data("split")[1], signed)
    assert np.array_equal(element.coefficients, coefficients)
    for index in range(3):
        assert left[index] == element * array[index]
        assert right[index] == array[index] * element


def test_same_operand_twice_uses_separate_temporaries():
    """Multiply an array by itself and by a view of itself."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(3), char=7)
    x = algebra.random(6, batch_shape=(4,))
    square = x * x
    shifted = x[:-1] * x[1:]
    for index in range(4):
        assert square[index] == x[index] * x[index]
    for index in range(3):
        assert shifted[index] == x[index] * x[index + 1]


def test_changing_shapes_and_fields_replace_the_temporaries():
    """Follow the shape and type of each request."""

    real = cdl.LoopAlgebra(cdl.CDLoop(2))
    finite = cdl.LoopAlgebra(cdl.CDLoop(2), char=5)
    for algebra, shape in (
        (real, (2,)),
        (finite, (2,)),
        (real, (3, 2)),
        (finite, (1,)),
    ):
        x = algebra.random(1, batch_shape=shape)
        y = algebra.random(2, batch_shape=shape)
        expected = _plain_product(algebra, x.coefficients, y.coefficients)
        assert np.array_equal((x * y).coefficients, expected)


def test_large_temporaries_are_not_kept(monkeypatch):
    """Drop temporaries above the size limit after use."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(3))
    x = algebra.random(1, batch_shape=(4,))
    y = algebra.random(2, batch_shape=(4,))
    expected = _plain_product(algebra, x.coefficients, y.coefficients)
    monkeypatch.setattr(_kernel, "_WORKSPACE_LIMIT", 16)
    _kernel._WORKSPACE.arrays.clear()
    assert np.array_equal((x * y).coefficients, expected)
    assert not _kernel._WORKSPACE.arrays
    monkeypatch.undo()
    assert np.array_equal((x * y).coefficients, expected)
    assert _kernel._WORKSPACE.arrays


def test_threads_have_separate_workspaces():
    """Multiply concurrently in several threads and get each product right."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(8))
    operands = [
        (
            algebra.random(10 + i, batch_shape=(64,)),
            algebra.random(20 + i, batch_shape=(64,)),
        )
        for i in range(4)
    ]
    expected = [
        _plain_product(algebra, x.coefficients, y.coefficients) for x, y in operands
    ]
    failures: list[int] = []

    def work(index: int) -> None:
        """Repeat one product and compare every result."""

        x, y = operands[index]
        for _ in range(25):
            if not np.array_equal((x * y).coefficients, expected[index]):
                failures.append(index)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures
