"""Private validation helpers for :mod:`pycdloop`."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

# Messages reported from more than one module, kept here so that every layer
# describes the same condition in the same words
DIFFERENT_ALGEBRA_MESSAGE = "operands belong to different loop algebras"
DIFFERENT_LOOP_MESSAGE = "operands belong to different loops"
MIXED_BASIS_MESSAGE = (
    "ElementArray and SplitArray operands cannot be combined; "
    "convert one of them with split() or to_loop()"
)
CD_ONLY_MESSAGE = (
    "{name} is defined in the loop algebra of a central product only in "
    "characteristic 2"
)
POWER_MESSAGE = (
    "powers depend on bracketing in the loop algebra of a central product "
    "outside characteristic 2; use left_power() or right_power()"
)
INTEGER_EXPONENT_MESSAGE = "loop algebra powers require an integer exponent"
BINARY_MESSAGE = "{name} is not defined in characteristic 2"
ARRAY_TRUTH_MESSAGE = (
    "the truth value of an array of elements is ambiguous; compare records and "
    "reduce the Boolean result"
)

# The largest total doubling level for which index arithmetic stays in int64
MAX_LEVEL = 62


def _index(value: Any, name: str) -> int:
    """Return *value* as a Python integer, rejecting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _level(value: Any, name: str = "level", *, minimum: int = 0) -> int:
    """Validate one doubling level."""

    level = _index(value, name)
    if level < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if level > MAX_LEVEL:
        raise OverflowError(f"{name} is too large")
    return level


def _basis_index(value: Any, size: int, name: str = "basis index") -> int:
    """Validate an index into a basis of *size* elements."""

    index = _index(value, name)
    if index < 0 or index >= size:
        raise IndexError(f"{name} is out of range")
    return index


def _sign(value: Any) -> int:
    """Validate a sign, which must be the integer +1 or -1."""

    sign = _index(value, "sign")
    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1")
    return sign


def _exponent(value: Any) -> int:
    """Validate an integer exponent."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(INTEGER_EXPONENT_MESSAGE)
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(INTEGER_EXPONENT_MESSAGE) from error


def _choice(value: Any, name: str, choices: tuple[str, ...]) -> str:
    """Validate a string option against its accepted values."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in choices:
        accepted = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {accepted}")
    return value
