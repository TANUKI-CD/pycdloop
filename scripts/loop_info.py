#!/usr/bin/env python3
"""Print the invariants of a Cayley--Dickson loop or central product.

Give one doubling level for ``Q_n`` or several for a central product:

    python scripts/loop_info.py 3
    python scripts/loop_info.py 2 3 --char 13
"""

from __future__ import annotations

import argparse
from collections import Counter

import pycdloop as cdl
from pycdloop import testing


def describe_loop(loop: cdl.CDLoop | cdl.CentralProduct) -> list[tuple[str, object]]:
    """Return the name and value of each invariant of a loop."""

    orders = Counter(element.order() for element in loop)
    return [
        ("loop", repr(loop)),
        ("order", loop.order),
        ("quotient order", loop.quotient_order),
        ("elements by order", dict(sorted(orders.items()))),
        ("center", len(loop.center())),
        ("commutant", len(loop.commutant())),
        ("nucleus", len(loop.nucleus())),
        ("commutative", loop.is_commutative()),
        ("associative", loop.is_associative()),
        ("Moufang", loop.is_moufang()),
        ("diassociative", loop.is_diassociative()),
    ]


def describe_algebra(algebra: cdl.LoopAlgebra) -> list[tuple[str, object]]:
    """Return the name and value of each invariant of a loop algebra."""

    return [
        ("algebra", repr(algebra)),
        ("dimension", algebra.dimension),
        ("coefficient shape", algebra.coefficient_shape),
        ("dimension of the center", len(algebra.center_basis())),
        ("signed algorithms", ", ".join(algebra.signed_algorithms)),
        ("default signed algorithm", algebra.default_signed_algorithm),
        ("commutative", testing.is_commutative(algebra)),
        ("associative", testing.is_associative(algebra)),
        ("alternative", testing.is_alternative(algebra)),
        ("flexible", testing.is_flexible(algebra)),
        ("power-associative", testing.is_power_associative(algebra)),
    ]


def main() -> int:
    """Parse the command line and print both tables."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("levels", type=int, nargs="+", help="doubling levels")
    parser.add_argument("--char", type=int, default=0, help="field characteristic")
    parser.add_argument("--deg", type=int, default=1, help="field degree")
    args = parser.parse_args()

    if len(args.levels) == 1:
        loop: cdl.CDLoop | cdl.CentralProduct = cdl.CDLoop(args.levels[0])
    else:
        loop = cdl.CentralProduct(*args.levels)
    algebra = cdl.LoopAlgebra(loop, char=args.char, deg=args.deg)
    rows = describe_loop(loop) + describe_algebra(algebra)
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
