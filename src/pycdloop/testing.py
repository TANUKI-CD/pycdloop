"""Sampled identity checks and random elements for experiments and tests.

Each check draws random elements and tests an identity on them. A result of
``False`` exhibits a counterexample and is conclusive; ``True`` only reports
that none was found among the samples. Over R and C the samples have small
integer coefficients, so the arithmetic of the checks is exact.

Examples
--------
>>> import pycdloop as cdl
>>> from pycdloop import testing
>>> testing.is_associative(cdl.LoopAlgebra(cdl.CDLoop(2)))
True
>>> testing.is_associative(cdl.LoopAlgebra(cdl.CDLoop(3)))
False
>>> testing.is_flexible(cdl.LoopAlgebra(cdl.CDLoop(4)))
True
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._algebra import LoopAlgebra
from ._array import ElementArray
from ._validation import _index

__all__ = [
    "is_alternative",
    "is_associative",
    "is_commutative",
    "is_flexible",
    "is_power_associative",
    "random_elements",
]

# Sampled coefficients over R and C lie in this symmetric range
_BOUND = 3


def random_elements(
    algebra: Any, count: Any, rng: Any = None, *, exact: bool = True
) -> ElementArray:
    """Return an array of *count* random elements of a loop algebra.

    Parameters
    ----------
    algebra : LoopAlgebra
        The parent of the elements.
    count : int
        The number of elements; the result has batch shape ``(count,)``.
    rng : numpy.random.Generator or seed, optional
        The source of randomness.
    exact : bool, optional
        Over R and C, draw integer coefficients between -3 and 3, so that
        sums and products of a few elements are computed without rounding.
        With ``False`` the coefficients are those of
        :meth:`LoopAlgebra.random`. Over a finite field the coefficients are
        uniform residues either way.
    """

    if not isinstance(algebra, LoopAlgebra):
        raise TypeError("algebra must be a LoopAlgebra")
    size = _index(count, "count")
    if size < 0:
        raise ValueError("count must be nonnegative")
    generator = np.random.default_rng(rng)
    if not exact or algebra.field is not None:
        result: ElementArray = algebra.random(generator, batch_shape=(size,))
        return result
    shape = (size,) + algebra.coefficient_shape
    values = generator.integers(-_BOUND, _BOUND + 1, size=shape).astype(algebra.dtype)
    if algebra.deg == 2:
        values = values + 1j * generator.integers(-_BOUND, _BOUND + 1, size=shape)
    return algebra.asarray(values)


def _samples(algebra: Any, samples: Any, rng: Any, arity: int) -> list[ElementArray]:
    """Draw the operands of one identity check."""

    generator = np.random.default_rng(rng)
    return [random_elements(algebra, samples, generator) for _ in range(arity)]


def is_commutative(algebra: Any, samples: int = 32, rng: Any = None) -> bool:
    """Return whether ``x y = y x`` on random elements."""

    x, y = _samples(algebra, samples, rng, 2)
    return bool((x * y == y * x).all())


def is_associative(algebra: Any, samples: int = 32, rng: Any = None) -> bool:
    """Return whether ``(x y) z = x (y z)`` on random elements."""

    x, y, z = _samples(algebra, samples, rng, 3)
    return bool(((x * y) * z == x * (y * z)).all())


def is_alternative(algebra: Any, samples: int = 32, rng: Any = None) -> bool:
    """Return whether both alternative laws hold on random elements.

    The laws are ``(x x) y = x (x y)`` and ``(y x) x = y (x x)``.
    """

    x, y = _samples(algebra, samples, rng, 2)
    left = (x * x) * y == x * (x * y)
    right = (y * x) * x == y * (x * x)
    return bool((left & right).all())


def is_flexible(algebra: Any, samples: int = 32, rng: Any = None) -> bool:
    """Return whether ``(x y) x = x (y x)`` on random elements."""

    x, y = _samples(algebra, samples, rng, 2)
    return bool(((x * y) * x == x * (y * x)).all())


def is_power_associative(algebra: Any, samples: int = 32, rng: Any = None) -> bool:
    """Return whether third and fourth powers are well defined on random elements.

    The check tests ``(x x) x = x (x x)`` and ``((x x) x) x = (x x) (x x)``.
    In characteristic zero these two identities imply that every power is
    independent of bracketing.
    """

    (x,) = _samples(algebra, samples, rng, 1)
    square = x * x
    cube = square * x
    third = cube == x * square
    fourth = cube * x == square * square
    return bool((third & fourth).all())
