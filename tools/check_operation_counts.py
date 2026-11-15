#!/usr/bin/env python3
"""Count the arithmetic of the loop-algebra and central-product algorithms.

The package's schedule and its reference evaluations of the transform and of
the change of coordinates are run on arrays of counting scalars. The compiled
kernels perform the operations of those reference evaluations, which the
tests check bit for bit, so the counts describe the code as shipped. The
product of signed components is evaluated by the scalar schedule of
``pycdloop._schedule`` rather than by pyfastcd, whose compiled kernels cannot
be instrumented; the counts therefore describe the algorithms with that
schedule. Input-dependent products are distinguished from multiplications by
fixed scalars; sign changes and operations with a known zero are free. These
are arithmetic counts, not machine-instruction counts.

Checked for every level ``n`` (``N = 2**n``):

- the schedule uses ``p_n = (9n - 15) 2**(n-1) + 10`` products (``p_0 = 1``);
- a loop-algebra product uses ``N + p_n`` products and exactly
  ``tau_n + 3nN + 10N`` operations, less the ``N`` trivial scalings at
  ``N = 1``, where ``tau_n`` is the operation count of the schedule;
- a central-product product for levels ``(n, m)`` uses ``D + p_n p_m``
  products and exactly ``N_m l_n + p_n tau_m + 3(n + m) D + 10 D``
  operations, where ``l_n = tau_n - p_n`` and ``D = N_n N_m``.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyfastcd as fcd

from pycdloop import CDLoop, _kernel
from pycdloop._field import _CoefficientField
from pycdloop._schedule import schedule_multiply, schedule_product_count


class Tally:
    """Running totals of the three kinds of counted operation."""

    def __init__(self) -> None:
        """Start from zero."""

        self.products = 0
        self.additions = 0
        self.scalings = 0

    @property
    def total(self) -> int:
        """The number of counted operations of every kind."""

        return self.products + self.additions + self.scalings

    def snapshot(self) -> dict[str, int]:
        """Return the totals as a dictionary."""

        return {
            "products": self.products,
            "additions": self.additions,
            "scalings": self.scalings,
            "total": self.total,
        }


TALLY = Tally()


class Counted:
    """An exact scalar that counts the arithmetic applied to it.

    A value is *fixed* when it does not depend on the operands of the
    product being counted. Arithmetic between fixed values is free, as is
    adding a fixed zero and multiplying by a fixed 0, 1, or -1.
    """

    __slots__ = ("value", "fixed")

    def __init__(self, value: Any, fixed: bool = False) -> None:
        """Wrap an exact value."""

        self.value = Fraction(value)
        self.fixed = fixed

    @staticmethod
    def _lift(other: Any) -> Counted:
        """Treat a plain number as a fixed scalar."""

        if isinstance(other, Counted):
            return other
        return Counted(Fraction(other), True)

    def __neg__(self) -> Counted:
        """Change the sign, which is free."""

        return Counted(-self.value, self.fixed)

    def _sum(self, other: Counted, value: Fraction) -> Counted:
        """Count one addition or subtraction unless it is trivial."""

        if self.fixed and other.fixed:
            return Counted(value, True)
        if (self.fixed and self.value == 0) or (other.fixed and other.value == 0):
            return Counted(value, False)
        TALLY.additions += 1
        return Counted(value, False)

    def __add__(self, other: Any) -> Counted:
        """Add."""

        partner = self._lift(other)
        return self._sum(partner, self.value + partner.value)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Counted:
        """Subtract."""

        partner = self._lift(other)
        return self._sum(partner, self.value - partner.value)

    def __rsub__(self, other: Any) -> Counted:
        """Subtract from a plain number."""

        partner = self._lift(other)
        return partner._sum(self, partner.value - self.value)

    def __mul__(self, other: Any) -> Counted:
        """Multiply, counting a product or a fixed scaling."""

        partner = self._lift(other)
        value = self.value * partner.value
        if self.fixed and partner.fixed:
            return Counted(value, True)
        for constant in (self, partner):
            if constant.fixed:
                if constant.value not in (0, 1, -1):
                    TALLY.scalings += 1
                return Counted(value, False)
        TALLY.products += 1
        return Counted(value, False)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Counted:
        """Divide by a fixed scalar, counted as a scaling."""

        partner = self._lift(other)
        if not partner.fixed:
            raise TypeError("division by an input-dependent value")
        return self * Counted(1 / partner.value, True)


def counting_field() -> _CoefficientField:
    """Return real-field arithmetic that stores counting scalars."""

    field = _CoefficientField(fcd.Algebra(level=0))
    field.dtype = np.dtype(object)
    return field


def operands(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Return an array of input-dependent counting scalars."""

    generator = np.random.default_rng(seed)
    values = generator.integers(-9, 10, size=shape)
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        result[index] = Counted(int(values[index]))
    return result


def plain(array: np.ndarray) -> np.ndarray:
    """Return the exact values of an array of counting scalars."""

    return np.vectorize(lambda item: item.value, otypes=[object])(array)


def scalar_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply rows that each hold one scalar."""

    result: np.ndarray = left * right
    return result


class ScheduleAlgebra:
    """The signed component of a Cayley--Dickson loop algebra, by the schedule."""

    cd = None

    def __init__(self, field: _CoefficientField, level: int) -> None:
        """Record the field and level."""

        self.field = field
        self.level = level

    def multiply(self, x: np.ndarray, y: np.ndarray, algorithm: str = "auto") -> Any:
        """Multiply two elements given as arrays of ``2**level`` scalars."""

        rows = (1, 1 << self.level, 1)
        product = schedule_multiply(
            self.field, scalar_product, self.level, x.reshape(rows), y.reshape(rows)
        )
        return product.reshape(x.shape)


class TensorAlgebra:
    """The signed component of a central product, by the nested schedule."""

    cd = None

    def __init__(self, field: _CoefficientField, outer: int, inner: int) -> None:
        """Record the field and the two levels."""

        self.field = field
        self.outer = outer
        self.inner = inner

    def _inner(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Multiply rows that each hold one element of the inner algebra."""

        rows = (-1, 1 << self.inner, 1)
        product = schedule_multiply(
            self.field,
            scalar_product,
            self.inner,
            left.reshape(rows),
            right.reshape(rows),
        )
        return product.reshape(left.shape)

    def multiply(self, x: np.ndarray, y: np.ndarray, algorithm: str = "auto") -> Any:
        """Multiply two elements stored flat, outer index first."""

        rows = (1, 1 << self.outer, 1 << self.inner)
        product = schedule_multiply(
            self.field, self._inner, self.outer, x.reshape(rows), y.reshape(rows)
        )
        return product.reshape(x.shape)


class Probe:
    """The attributes of a loop algebra that the kernel routines use."""

    def __init__(self, signed: Any, size: int) -> None:
        """Record the signed component and the quotient order."""

        self._field = signed.field
        self._signed = signed
        self._size = size


def count(action: Any) -> tuple[Any, dict[str, int]]:
    """Run *action* and return its result with the operations it used."""

    global TALLY
    TALLY = Tally()
    result = action()
    return result, TALLY.snapshot()


def schedule_counts(level: int) -> dict[str, int]:
    """Count one product of the scalar schedule at the given level."""

    signed = ScheduleAlgebra(counting_field(), level)
    x, y = operands((1 << level,), 1), operands((1 << level,), 2)
    return count(lambda: signed.multiply(x, y))[1]


def loop_product(probe: Probe, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multiply two loop-basis elements by the split algorithm."""

    left = _kernel.to_split(probe, x)
    right = _kernel.to_split(probe, y)
    product = _kernel.multiply(probe, "split", left, right)
    result: np.ndarray = _kernel.from_split(probe, product)
    return result


def reference(loop: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a loop-algebra product from the multiplication table."""

    table = loop.cayley_table()
    a, b = plain(x).ravel(), plain(y).ravel()
    out = np.zeros(loop.order, dtype=object)
    for i in range(loop.order):
        for j in range(loop.order):
            out[table[i, j]] += a[i] * b[j]
    return out.reshape(x.shape)


def check_loop(level: int, verify: bool) -> dict[str, Any]:
    """Count and check one loop-algebra product at the given level."""

    size = 1 << level
    field = counting_field()
    probe = Probe(ScheduleAlgebra(field, level), size)
    x, y = operands((2, size), 3), operands((2, size), 4)
    result, counts = count(lambda: loop_product(probe, x, y))
    schedule = schedule_counts(level)
    products = schedule_product_count(level)
    # At N = 1 the division by N is a multiplication by one and is free
    expected = schedule["total"] + 3 * level * size + 10 * size - (size == 1)
    ok = (
        schedule["products"] == products
        and counts["products"] == size + products
        and counts["total"] == expected
    )
    if verify:
        ok = ok and bool(np.all(plain(result) == reference(CDLoop(level), x, y)))
    return {
        "level": level,
        "dimension": 2 * size,
        "schedule": schedule,
        "loop": counts,
        "expected_products": size + products,
        "expected_total": expected,
        "ok": ok,
    }


def check_central(outer: int, inner: int) -> dict[str, Any]:
    """Count and check one central-product product for two levels."""

    size = 1 << (outer + inner)
    field = counting_field()
    probe = Probe(TensorAlgebra(field, outer, inner), size)
    x, y = operands((2, size), 5), operands((2, size), 6)
    counts = count(lambda: loop_product(probe, x, y))[1]
    first, second = schedule_counts(outer), schedule_counts(inner)
    products = size + first["products"] * second["products"]
    linear = first["total"] - first["products"]
    bound = (
        (1 << inner) * linear
        + first["products"] * second["total"]
        + 3 * (outer + inner) * size
        + 10 * size
    )
    ok = counts["products"] == products and counts["total"] == bound
    return {
        "levels": [outer, inner],
        "dimension": 2 * size,
        "central": counts,
        "expected_products": products,
        "bound": bound,
        "ok": ok,
    }


def main() -> int:
    """Run the checks, print the tables, and return the exit status."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-level", type=int, default=10)
    parser.add_argument("--central-max", type=int, default=4)
    parser.add_argument("--verify-through", type=int, default=4)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.max_level < 0 or args.central_max < 1:
        parser.error("levels must be nonnegative, and central levels positive")

    loops = [check_loop(n, n <= args.verify_through) for n in range(args.max_level + 1)]
    centrals = [
        check_central(n, m)
        for n in range(1, args.central_max + 1)
        for m in range(n, args.central_max + 1)
    ]

    print("Loop algebra of Q_n: split algorithm with the scalar schedule")
    print("  n      2N        M        A        E        T   tau_n   status")
    for row in loops:
        c, s = row["loop"], row["schedule"]
        print(
            f"{row['level']:3d} {row['dimension']:7d} {c['products']:8d} "
            f"{c['additions']:8d} {c['scalings']:8d} {c['total']:8d} "
            f"{s['total']:7d}   {'ok' if row['ok'] else 'MISMATCH'}"
        )
    print()
    print("Loop algebra of Q_n * Q_m: split algorithm with the nested schedule")
    print("  n   m      2D        M        A        E        T    bound   status")
    for row in centrals:
        c = row["central"]
        n, m = row["levels"]
        print(
            f"{n:3d} {m:3d} {row['dimension']:7d} {c['products']:8d} "
            f"{c['additions']:8d} {c['scalings']:8d} {c['total']:8d} "
            f"{row['bound']:8d}   {'ok' if row['ok'] else 'MISMATCH'}"
        )
    if args.json is not None:
        args.json.write_text(
            json.dumps({"loops": loops, "centrals": centrals}, indent=2)
        )
    return 0 if all(row["ok"] for row in loops + centrals) else 1


if __name__ == "__main__":
    raise SystemExit(main())
