r"""Independent reference arithmetic for the tests of :mod:`pycdloop`.

This module deliberately imports neither :mod:`pycdloop` nor :mod:`pyfastcd`.
Everything is derived from the defining doubling identity

.. math::

   (a,b)(c,d) = (ac - \overline{d}b,\; da + b\overline{c}),

evaluated in exact arithmetic, so an error in the production sign logic, in a
transform or in an optimized multiplication schedule cannot make the oracle
agree with the implementation by construction.

Scalars are handled through small ring objects: exact rationals, exact
Gaussian rationals for complex coefficients, and residue polynomials for
finite fields. The routines favor clarity over speed and are intended only
for tests of small dimension.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import cache
from itertools import product
from typing import Any

import numpy as np


class Rationals:
    """Exact rational arithmetic; binary64 inputs convert without rounding."""

    zero: Any = Fraction(0)
    one: Any = Fraction(1)

    def add(self, a: Any, b: Any) -> Any:
        """Return ``a + b``."""

        return a + b

    def neg(self, a: Any) -> Any:
        """Return ``-a``."""

        return -a

    def mul(self, a: Any, b: Any) -> Any:
        """Return ``a b``."""

        return a * b

    def convert(self, value: Any) -> Any:
        """Convert one stored coefficient to a ring element."""

        return Fraction(float(value))

    def store(self, value: Any) -> Any:
        """Convert a ring element to a stored coefficient."""

        return float(value)


class GaussianRationals(Rationals):
    """Exact arithmetic on pairs ``(re, im)`` of rationals."""

    zero = (Fraction(0), Fraction(0))
    one = (Fraction(1), Fraction(0))

    def add(self, a: Any, b: Any) -> Any:
        """Return ``a + b``."""

        return (a[0] + b[0], a[1] + b[1])

    def neg(self, a: Any) -> Any:
        """Return ``-a``."""

        return (-a[0], -a[1])

    def mul(self, a: Any, b: Any) -> Any:
        """Return ``a b``."""

        return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])

    def convert(self, value: Any) -> Any:
        """Convert one stored coefficient to a ring element."""

        number = complex(value)
        return (Fraction(number.real), Fraction(number.imag))

    def store(self, value: Any) -> Any:
        """Convert a ring element to a stored coefficient."""

        return complex(float(value[0]), float(value[1]))


class FiniteField:
    """Schoolbook arithmetic in ``F_p[t] / (modulus)``.

    Elements are tuples of ``deg`` residues, constant coefficient first, and
    *modulus* lists the ``deg + 1`` coefficients of a monic polynomial,
    constant coefficient first. A prime field has modulus ``(0, 1)``.
    """

    def __init__(self, p: int, modulus: Sequence[int] = (0, 1)) -> None:
        """Record the characteristic and the defining polynomial."""

        self.p = p
        self.modulus = tuple(int(c) % p for c in modulus)
        self.deg = len(self.modulus) - 1
        self.zero: Any = (0,) * self.deg
        self.one: Any = (1,) + (0,) * (self.deg - 1)

    def add(self, a: Any, b: Any) -> Any:
        """Return ``a + b``."""

        return tuple((u + v) % self.p for u, v in zip(a, b, strict=True))

    def neg(self, a: Any) -> Any:
        """Return ``-a``."""

        return tuple(-u % self.p for u in a)

    def mul(self, a: Any, b: Any) -> Any:
        """Return ``a b`` by polynomial multiplication and reduction."""

        d = self.deg
        t = [0] * (2 * d - 1)
        for i in range(d):
            for j in range(d):
                t[i + j] = (t[i + j] + a[i] * b[j]) % self.p
        for k in range(2 * d - 2, d - 1, -1):
            for j in range(d):
                t[k - d + j] = (t[k - d + j] - t[k] * self.modulus[j]) % self.p
        return tuple(t[:d])

    def convert(self, value: Any) -> Any:
        """Convert one stored coefficient to a ring element."""

        return tuple(int(c) % self.p for c in np.atleast_1d(value))

    def store(self, value: Any) -> Any:
        """Convert a ring element to a stored coefficient."""

        return value[0] if self.deg == 1 else list(value)


def to_ring(ring: Any, values: Any, depth: int) -> Any:
    """Convert the first *depth* axes of an array to nested lists of ring elements."""

    array = np.asarray(values)
    if depth == 0:
        return ring.convert(array)
    return [to_ring(ring, item, depth - 1) for item in array]


def from_ring(ring: Any, values: Any, depth: int, dtype: Any) -> np.ndarray:
    """Convert nested lists of ring elements back to a coefficient array."""

    def stored(item: Any, level: int) -> Any:
        """Convert one nesting level."""

        if level == 0:
            return ring.store(item)
        return [stored(entry, level - 1) for entry in item]

    return np.array(stored(values, depth), dtype=dtype)


def cd_conjugate(ring: Any, x: Sequence[Any]) -> list[Any]:
    """Return the Cayley--Dickson conjugate of a coefficient list."""

    return [x[0]] + [ring.neg(value) for value in x[1:]]


def cd_multiply(ring: Any, x: Sequence[Any], y: Sequence[Any]) -> list[Any]:
    """Multiply two coefficient lists of equal power-of-two length by doubling."""

    size = len(x)
    if size == 1:
        return [ring.mul(x[0], y[0])]
    half = size // 2
    a, b, c, d = x[:half], x[half:], y[:half], y[half:]
    first = cd_multiply(ring, a, c)
    correction = cd_multiply(ring, cd_conjugate(ring, d), b)
    second = cd_multiply(ring, d, a)
    tail = cd_multiply(ring, b, cd_conjugate(ring, c))
    upper = [ring.add(u, ring.neg(v)) for u, v in zip(first, correction, strict=True)]
    lower = [ring.add(u, v) for u, v in zip(second, tail, strict=True)]
    return upper + lower


class Integers(Rationals):
    """Exact integer arithmetic on Python integers."""

    zero: Any = 0
    one: Any = 1

    def convert(self, value: Any) -> Any:
        """Convert one stored coefficient to a ring element."""

        return int(value)

    def store(self, value: Any) -> Any:
        """Convert a ring element to a stored coefficient."""

        return int(value)


@cache
def sign_table(level: int) -> tuple[tuple[int, ...], ...]:
    """Return ``sigma(p, q)`` for the algebra of dimension ``2**level``.

    Row ``p`` is read off one product computed by the doubling formula over
    the integers: the basis vector ``e_p`` times the vector of all ones.
    Each basis vector ``e_q`` contributes to the single coordinate
    ``p ^ q``, so that coordinate of the product is ``sigma(p, q)``.
    """

    ring = Integers()
    size = 1 << level
    ones = [ring.one] * size
    table = []
    for p in range(size):
        left = [ring.one if i == p else ring.zero for i in range(size)]
        product_row = cd_multiply(ring, left, ones)
        row = tuple(int(product_row[p ^ q]) for q in range(size))
        assert all(value in (1, -1) for value in row)
        table.append(row)
    return tuple(table)


def factor_indices(levels: Sequence[int], index: int) -> tuple[int, ...]:
    """Split a flat index into one index for each factor, first factor first."""

    parts = []
    for level in reversed(levels):
        parts.append(index & ((1 << level) - 1))
        index >>= level
    return tuple(reversed(parts))


def twist(levels: Sequence[int], s: int, t: int) -> int:
    """Return the sign of the product of two basis elements of a central product."""

    sign = 1
    pairs = zip(factor_indices(levels, s), factor_indices(levels, t), strict=True)
    for level, (p, q) in zip(levels, pairs, strict=True):
        sign *= sign_table(level)[p][q]
    return sign


def loop_product(
    levels: Sequence[int], x: tuple[int, int], y: tuple[int, int]
) -> tuple[int, int]:
    """Multiply two loop elements given as ``(sign, flat index)``."""

    return (x[0] * y[0] * twist(levels, x[1], y[1]), x[1] ^ y[1])


def tensor_multiply(
    ring: Any, levels: Sequence[int], u: Sequence[Any], v: Sequence[Any]
) -> list[Any]:
    """Multiply in the tensor product of Cayley--Dickson algebras, term by term."""

    size = 1 << sum(levels)
    out = [ring.zero] * size
    for p, q in product(range(size), repeat=2):
        term = ring.mul(u[p], v[q])
        if twist(levels, p, q) < 0:
            term = ring.neg(term)
        out[p ^ q] = ring.add(out[p ^ q], term)
    return out


def loop_algebra_multiply(
    ring: Any,
    levels: Sequence[int],
    x: Sequence[Sequence[Any]],
    y: Sequence[Sequence[Any]],
) -> list[list[Any]]:
    """Multiply two loop-algebra elements straight from the definition.

    An element is a pair of coefficient lists: row 0 for the basis elements
    ``[e_t]`` and row 1 for ``[-e_t]``. Every pair of basis elements
    contributes its product of coefficients to the basis element of the
    loop product.
    """

    size = 1 << sum(levels)
    out = [[ring.zero] * size, [ring.zero] * size]
    signs = (1, -1)
    for row_x, row_y in product(range(2), repeat=2):
        for p, q in product(range(size), repeat=2):
            sign, index = loop_product(levels, (signs[row_x], p), (signs[row_y], q))
            row = 0 if sign > 0 else 1
            term = ring.mul(x[row_x][p], y[row_y][q])
            out[row][index] = ring.add(out[row][index], term)
    return out


def walsh_hadamard(ring: Any, values: Sequence[Any]) -> list[Any]:
    """Return the Walsh--Hadamard transform straight from its matrix."""

    size = len(values)
    out = []
    for i in range(size):
        total = ring.zero
        for j in range(size):
            term = values[j]
            if bin(i & j).count("1") % 2:
                term = ring.neg(term)
            total = ring.add(total, term)
        out.append(total)
    return out


def dyadic_convolve(ring: Any, a: Sequence[Any], b: Sequence[Any]) -> list[Any]:
    """Return the dyadic convolution straight from its definition."""

    size = len(a)
    out = [ring.zero] * size
    for i, j in product(range(size), repeat=2):
        out[i ^ j] = ring.add(out[i ^ j], ring.mul(a[i], b[j]))
    return out
