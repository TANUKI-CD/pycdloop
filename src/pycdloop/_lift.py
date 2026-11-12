"""Private products in characteristic 2, by lifting to a large prime.

The loop algebra of a loop ``L`` over F_2 is the reduction modulo 2 of the
integral loop ring Z[L], whose structure constants are 0 and 1. A product of
two elements with coefficients 0 and 1 therefore has integer coefficients
between 0 and ``len(L)``: each counts pairs of loop elements. Computing that
product modulo a prime larger than the bound recovers the integers exactly,
and reducing them modulo 2 gives the product over F_2. The product modulo
the prime is computed by the split algorithm of the package, so its cost is
quasilinear even though the loop algebra over F_2 has no decomposition.

An element of F_(2^k) is a polynomial of degree below ``k`` over F_2. Its
coordinates lift to a polynomial with coefficients 0 and 1, and a product of
two such polynomials has degree at most ``2k - 2``. A field extension of
degree ``2k - 1`` of the prime field holds that product without reduction,
so one product of loop-algebra elements over the extension gives every
coefficient polynomial, with integer coordinates between 0 and
``k * len(L)``. Reducing the coordinates modulo 2 and the polynomial modulo
the field polynomial finishes the product.

pyfastcd offers extensions up to degree 12, which covers ``k <= 6``. For
larger ``k`` each coefficient is cut into a low and a high half of
``h = ceil(k / 2)`` coordinates, ``x = x_0 + t^h x_1``, and the product is
assembled from three products of halves, each of which fits:

    x y = x_0 y_0 + t^h ((x_0 + x_1)(y_0 + y_1) + x_0 y_0 + x_1 y_1) + t^(2h) x_1 y_1.

The identity holds over F_2, where the sums are XORs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyfastcd as fcd

from . import _kernel
from ._binary import _reduce

# The largest prime that pyfastcd accepts as a characteristic
PRIME = 2147483647

# The largest number of coordinates of one limb, set by pyfastcd's degree limit
_LIMB_BITS = 6


class _Codec:
    """The conversion between coefficients over F_(2^k) and their lifts."""

    def __init__(self, field: Any) -> None:
        """Choose the limbs and the container field of one binary field."""

        degree = field.deg
        self.degree = degree
        self.half = degree if degree <= _LIMB_BITS else (degree + 1) // 2
        self.limb_count = 1 if degree <= _LIMB_BITS else 3
        self.width = 2 * self.half - 1
        self.prime = PRIME
        if self.width == 1:
            self.container = fcd.FiniteField(char=self.prime)
        else:
            self.container = fcd.FiniteField(char=self.prime, deg=self.width)
        if degree > 1:
            # Row i holds the coordinates of t^i modulo the field polynomial
            mask = field.field._modulus_mask
            rows = [_reduce(1 << i, mask, degree) for i in range(2 * degree - 1)]
            bits = [[(row >> j) & 1 for j in range(degree)] for row in rows]
            self.reduction = np.array(bits, dtype=np.uint32)

    def check(self, terms: int) -> None:
        """Check that sums of *terms* products of coefficients stay below the prime."""

        # Every integer coordinate of a lifted product is at most this bound
        if self.half * terms >= self.prime:
            raise ValueError(
                "the operands are too large for products in characteristic 2"
            )

    def limbs(self, coefficients: np.ndarray) -> list[np.ndarray]:
        """Return the lifted operands of loop-basis coefficients."""

        if self.degree == 1:
            return [np.ascontiguousarray(coefficients)]
        half = self.half
        shape = coefficients.shape[:-1] + (self.width,)
        if self.limb_count == 1:
            parts = [coefficients]
        else:
            low = coefficients[..., :half]
            high = np.zeros(low.shape, dtype=np.uint32)
            high[..., : self.degree - half] = coefficients[..., half:]
            parts = [low, high, np.bitwise_xor(low, high)]
        lifted = []
        for part in parts:
            limb = np.zeros(shape, dtype=np.uint32)
            limb[..., : part.shape[-1]] = part
            lifted.append(limb)
        return lifted

    def combine(self, products: list[np.ndarray]) -> np.ndarray:
        """Return coefficients over F_(2^k) from the lifted products."""

        bits = [np.bitwise_and(product, 1) for product in products]
        if self.degree == 1:
            return bits[0].astype(np.uint32)
        if self.limb_count == 1:
            total = bits[0]
        else:
            low, high, mixed = bits
            half, width = self.half, self.width
            total = np.zeros(low.shape[:-1] + (2 * self.degree - 1,), dtype=np.uint32)
            total[..., :width] = low
            total[..., half : half + width] ^= mixed ^ low ^ high
            # The high product has degree at most 2 (k - h) - 2, so the
            # coordinates beyond the end of the total are zero
            rest = 2 * self.degree - 1 - 2 * half
            total[..., 2 * half :] ^= high[..., :rest]
        reduced: np.ndarray = np.bitwise_and(total @ self.reduction, 1)
        return reduced.astype(np.uint32)


class _Lift:
    """The lifting data of one loop algebra of characteristic 2."""

    def __init__(self, algebra: Any, algebra_type: Any) -> None:
        """Create the loop algebra over the container field."""

        self.codec = _Codec(algebra._field)
        self.terms = algebra.dimension
        self.parent = algebra_type(algebra.loop, field=self.codec.container)

    def split_limbs(self, coefficients: np.ndarray) -> list[_kernel.Split]:
        """Return the split forms of the lifted operands, for an element to keep."""

        self.codec.check(self.terms)
        limbs = self.codec.limbs(coefficients)
        return [_kernel.to_split(self.parent, limb) for limb in limbs]

    def product(
        self, left: tuple[str, Any], right: tuple[str, Any], signed_algorithm: str
    ) -> np.ndarray:
        """Return a product in characteristic 2.

        Each operand is a pair of a mode and data: ``"loop"`` with
        loop-basis coefficients, or ``"lift"`` with the split forms that
        :meth:`split_limbs` returned for them.
        """

        self.codec.check(self.terms)
        operands = []
        for mode, data in (left, right):
            if mode == "lift":
                operands.append([("split", split) for split in data])
            else:
                operands.append([("loop", limb) for limb in self.codec.limbs(data)])
        products = [
            _kernel.loop_product(self.parent, first, second, signed_algorithm)
            for first, second in zip(operands[0], operands[1], strict=True)
        ]
        return self.codec.combine(products)
