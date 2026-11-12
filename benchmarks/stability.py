#!/usr/bin/env python3
"""Exact-reference accuracy study of the loop-algebra product (needs NumPy and gmpy2).

Every reference coefficient is an exact integer times a power of two. The
stored binary64 operands are converted to integers without rounding, the
product of the loop algebra is summed from its definition in arbitrary-precision
integers, and norms and error measures are evaluated with MPFR at 256 bits or
more. The reference derives the basis signs from the doubling formula on its
own and uses no pycdloop or pyfastcd arithmetic.

Two experiments are run on Cayley--Dickson loops Q_n over R and C. The product
experiment measures the split algorithm (S) and the table algorithm (T) on the
same operand pairs. The chain experiment measures left-nested products of
several operands, evaluated in split form with one conversion at the end (L)
and with a conversion back to the loop basis after every product (R).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gmpy2 as gp
except ImportError as error:
    raise SystemExit("Experiment only: install optional gmpy2 (python -m pip install gmpy2)") from error

DENSE = ("gaussian", "uniform")
SHARED_FAMILIES = (
    "gaussian", "uniform", "rademacher", "sparse_one", "sparse_four", "sparse_eighth",
    "repeated", "correlated_plus", "correlated_minus", "alternating_magnitude",
    "signed_cancel", "group_cancel", "group_ideal", "signed_ideal", "cross_ideal",
    "idempotent", "character", "scalar_only",
)
REAL_FAMILIES = SHARED_FAMILIES + (
    "zero_divisor", "near_zero_divisor_20", "near_zero_divisor_40", "near_zero_divisor_52",
)
COMPLEX_FAMILIES = SHARED_FAMILIES + ("isotropic", "near_isotropic")
CHAIN_LENGTHS = (2, 4, 8)
SCHEMA = "pycdloop-stability-v1"

# An exact element: integer real and imaginary parts of shape (2, D) and the
# exponent E of their common denominator 2**E; the imaginary part of a real
# element is None
Exact = tuple[np.ndarray, "np.ndarray | None", int]


# The exact reference


@cache
def sign_table(size: int) -> np.ndarray:
    """Derive the basis signs from (a,b)(c,d) = (ac - d* b, da + b c*).

    Here * is the Cayley--Dickson involution, and the index of each basis
    product is the XOR of the indices. The off-diagonal blocks reverse the
    multiplication. This routine neither imports nor calls the sign functions
    of pycdloop or pyfastcd.
    """
    if size < 1 or size & (size - 1):
        raise ValueError("dimension must be a positive power of two")
    if size == 1:
        return np.ones((1, 1), dtype=np.int8)
    s = sign_table(size // 2)
    conjugation = -np.ones(size // 2, dtype=np.int8)
    conjugation[0] = 1
    result = np.block([[s, s.T], [s * conjugation, -s.T * conjugation]]).astype(np.int8)
    result.setflags(write=False)
    return result


def encode(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert binary64 values exactly to integers times 2**(-exponent)."""
    ratios = [float(v).as_integer_ratio() for v in np.asarray(values, dtype=np.float64).ravel()]
    exponent = max((d.bit_length() - 1 for _, d in ratios), default=0)
    integers = [a << (exponent - (d.bit_length() - 1)) for a, d in ratios]
    result = np.empty(len(integers), dtype=object)
    result[:] = integers
    return result.reshape(np.shape(values)), exponent


def exact(values: np.ndarray) -> Exact:
    """Convert stored loop-basis coefficients of shape (2, D) to an exact element."""
    if not np.iscomplexobj(values):
        integers, exponent = encode(values)
        return integers, None, exponent
    both, exponent = encode(np.stack([values.real, values.imag]))
    return both[0], both[1], exponent


def bilinear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sum the loop-algebra product of two integer coefficient arrays from its definition.

    The basis elements [s e_p] and [t e_q] multiply to [s t sigma(p, q) e_(p ^ q)].
    For each p, the terms of all q are formed at once: with q = p ^ k, output
    index k receives the two products of equal signs in the row given by
    sigma(p, q) and the two products of opposite signs in the other row.
    """
    size = x.shape[1]
    signs = sign_table(size)
    index = np.arange(size)
    out = np.zeros((2, size), dtype=object)
    for p in range(size):
        a, b = x[0, p], x[1, p]
        if not a and not b:
            continue
        partner = index ^ p
        plus, minus = y[0, partner], y[1, partner]
        same = a * plus + b * minus
        opposite = a * minus + b * plus
        positive = signs[p, partner] > 0
        out[0] += np.where(positive, same, opposite)
        out[1] += np.where(positive, opposite, same)
    return out


def exact_product(x: Exact, y: Exact) -> Exact:
    """Return the exact product of two exact elements."""
    xr, xi, ex = x
    yr, yi, ey = y
    if xi is None and yi is None:
        return bilinear(xr, yr), None, ex + ey
    zero = np.zeros(xr.shape, dtype=object)
    xi = zero if xi is None else xi
    yi = zero if yi is None else yi
    real = bilinear(xr, yr) - bilinear(xi, yi)
    imag = bilinear(xr, yi) + bilinear(xi, yr)
    return real, imag, ex + ey


def exact_chain(operands: list[np.ndarray]) -> Exact:
    """Return the exact left-nested product ((x1 x2) x3) ... of stored operands."""
    result = exact(operands[0])
    for operand in operands[1:]:
        result = exact_product(result, exact(operand))
    return result


def squared_norm(element: Exact) -> tuple[int, int]:
    """Return the squared Euclidean norm as an integer over 2**(2 * exponent)."""
    real, imag, exponent = element
    total = sum(int(v) * int(v) for v in real.ravel())
    if imag is not None:
        total += sum(int(v) * int(v) for v in imag.ravel())
    return total, exponent


def squared_errors(outputs: list[np.ndarray], reference: Exact) -> tuple[list[int], int]:
    """Return the exact squared errors of several outputs on one dyadic scale.

    The integers share the denominator 2**(2E) for the returned exponent E, so
    they compare exactly even when the binary64 errors round to the same value.
    """
    real, imag, exponent_r = reference
    encoded = [exact(values) for values in outputs]
    exponent = max([exponent_r, *(e for _, _, e in encoded)])
    target_r = [int(v) << (exponent - exponent_r) for v in real.ravel()]
    target_i = None if imag is None else [int(v) << (exponent - exponent_r) for v in imag.ravel()]
    totals = []
    for out_r, out_i, e in encoded:
        shift = exponent - e
        total = sum(((int(a) << shift) - b) ** 2 for a, b in zip(out_r.ravel(), target_r, strict=True))
        if target_i is not None or out_i is not None:
            zeros = [0] * len(target_r)
            left = zeros if out_i is None else [int(a) << shift for a in out_i.ravel()]
            right = zeros if target_i is None else target_i
            total += sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
        totals.append(total)
    return totals, exponent


def root(total: int, exponent: int) -> Any:
    """Return sqrt(total) / 2**exponent in the current MPFR context."""
    return gp.mul_2exp(gp.sqrt(gp.mpfr(gp.mpz(total))), -exponent)


def measures(outputs: dict[str, np.ndarray], reference: Exact, operand_norms: list[tuple[int, int]],
             precision: int) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Return the error measures of several outputs and their exact squared errors."""
    names = list(outputs)
    totals, exponent = squared_errors([outputs[name] for name in names], reference)
    result = {}
    with gp.context(precision=precision):
        reference_norm = root(*squared_norm(reference))
        scale = gp.mpfr(1)
        for total, e in operand_norms:
            scale *= root(total, e)
        unit = gp.mul_2exp(gp.mpfr(1), -53)
        for name, total in zip(names, totals, strict=True):
            error = root(total, exponent)
            result[name] = {
                "absolute": float(error),
                "relative": float(error / reference_norm) if reference_norm else None,
                "normalized": float(error / (unit * scale)) if scale else None,
                "chi": float(scale / reference_norm) if reference_norm else None,
                "reference_norm": float(reference_norm),
                "zero_product": not bool(reference_norm),
            }
    return result, dict(zip(names, totals, strict=True))


def defining_product(x: list[Fraction], y: list[Fraction]) -> list[Fraction]:
    """Literal rational doubling recursion, independent of the signed-XOR oracle."""
    if len(x) == 1:
        return [x[0] * y[0]]
    h = len(x) // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]

    def conjugate(v: list[Fraction]) -> list[Fraction]:
        return v[:1] + [-a for a in v[1:]]

    ac, db = defining_product(a, c), defining_product(conjugate(d), b)
    da, bc = defining_product(d, a), defining_product(b, conjugate(c))
    return [u - v for u, v in zip(ac, db, strict=True)] + [u + v for u, v in zip(da, bc, strict=True)]


def defining_loop_product(x: list[list[Fraction]], y: list[list[Fraction]]) -> list[list[Fraction]]:
    """Loop-algebra product by pairs of basis elements, with signs from the rational recursion."""
    size = len(x[0])
    out = [[Fraction(0)] * size, [Fraction(0)] * size]
    for p in range(size):
        for q in range(size):
            left, right = [Fraction(0)] * size, [Fraction(0)] * size
            left[p], right[q] = Fraction(1), Fraction(1)
            sign = defining_product(left, right)[p ^ q]
            for s in range(2):
                for t in range(2):
                    row = (s + t + (sign < 0)) % 2
                    out[row][p ^ q] += x[s][p] * y[t][q]
    return out


def validate_oracle() -> dict[str, Any]:
    """Check the sign table and the exact product against rational arithmetic."""
    # Quaternion i*j=k, j*k=i, k*i=j and standard doubling octonion triples
    known = {(1, 2): (3, 1), (2, 3): (1, 1), (3, 1): (2, 1),
             (1, 4): (5, 1), (2, 4): (6, 1), (3, 4): (7, 1),
             (1, 6): (7, -1), (2, 5): (7, 1)}
    for (j, k), (index, sign) in known.items():
        assert j ^ k == index and sign_table(8)[j, k] == sign
    count = 0
    for n in (1, 2, 4, 8, 16):
        for j in range(n):
            for k in range(n):
                x, y = [Fraction(0)] * n, [Fraction(0)] * n
                x[j], y[k] = Fraction(1), Fraction(1)
                result = defining_product(x, y)
                assert result[j ^ k] == int(sign_table(n)[j, k])
                assert sum(bool(v) for v in result) == 1
                count += 1
    levels = []
    for level in (0, 1, 2, 3):
        size = 1 << level
        x = (np.arange(2 * size, dtype=np.float64).reshape(2, size) - size + 0.5) / 8
        y = (np.arange(2 * size, dtype=np.float64)[::-1].reshape(2, size) - 1) / 16
        real, imag, exponent = exact_product(exact(x), exact(y))
        rational = defining_loop_product([[Fraction(float(v)) for v in row] for row in x],
                                         [[Fraction(float(v)) for v in row] for row in y])
        assert imag is None
        assert all(Fraction(int(v), 1 << exponent) == q
                   for row, expected in zip(real, rational, strict=True) for v, q in zip(row, expected, strict=True))
        levels.append(level)
    return {"basis_cases_against_rational_doubling": count, "loop_algebra_levels_against_rational_definition": levels,
            "known_quaternion_octonion_cases": len(known)}


def mp_product(x: np.ndarray, y: np.ndarray, precision: int) -> list[list[Any]]:
    """Separate MPFR sum over pairs of loop elements, for precision checks only."""
    size = x.shape[1]
    signs = sign_table(size)
    with gp.context(precision=precision):
        xx = [[gp.mpc(complex(v)) for v in row] for row in x]
        yy = [[gp.mpc(complex(v)) for v in row] for row in y]
        out = [[gp.mpc(0) for _ in range(size)] for _ in range(2)]
        for s in range(2):
            for t in range(2):
                for p in range(size):
                    if not xx[s][p]:
                        continue
                    for q in range(size):
                        row = (s + t + (signs[p, q] < 0)) % 2
                        out[row][p ^ q] += xx[s][p] * yy[t][q]
    return out


def as_mp(reference: Exact) -> list[list[Any]]:
    """Return an exact element as MPFR complex numbers in the current context."""
    real, imag, exponent = reference
    zero = np.zeros(real.shape, dtype=object)
    imag = zero if imag is None else imag
    return [[gp.mpc(gp.mul_2exp(gp.mpfr(gp.mpz(int(a))), -exponent), gp.mul_2exp(gp.mpfr(gp.mpz(int(b))), -exponent))
             for a, b in zip(row_r, row_i, strict=True)] for row_r, row_i in zip(real, imag, strict=True)]


# Operands


@lru_cache(maxsize=1)
def zero_divisor() -> tuple[np.ndarray, np.ndarray]:
    """Find and independently prove a sparse sedenion zero-divisor pair."""
    n, s = 16, sign_table(16)
    for a in range(1, n):
        for b in range(a + 1, n):
            for c in range(1, n):
                d = a ^ b ^ c
                if not c < d < n:
                    continue
                for t in (-1, 1):
                    if s[a, c] + t * s[b, d] == 0 and t * s[a, d] + s[b, c] == 0:
                        x, y = np.zeros(n), np.zeros(n)
                        x[a], x[b], y[c], y[d] = 1, 1, 1, t
                        rational = defining_product([Fraction(v) for v in x], [Fraction(v) for v in y])
                        assert not any(rational)
                        return x, y
    raise AssertionError("no independently verified zero divisor found")


def euclidean_norm(values: np.ndarray) -> float:
    """Correctly rounded Euclidean norm of the binary64 coefficients.

    The squares are summed with math.fsum, so the result is the same on every
    platform and does not depend on the BLAS library NumPy uses.
    """
    parts = np.concatenate([values.real.ravel(), values.imag.ravel()]) if np.iscomplexobj(values) else values.ravel()
    return math.sqrt(math.fsum((parts * parts).tolist()))


@cache
def characters(size: int) -> np.ndarray:
    """Return the matrix of the characters (-1)**popcount(a & t) of the quotient group."""
    matrix = np.ones((1, 1))
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    matrix.setflags(write=False)
    return matrix


def lift_signed(d: np.ndarray) -> np.ndarray:
    """Return the element with signed component d and group component zero."""
    return np.stack([d / 2, -d / 2])


def make_inputs(domain: str, level: int, family: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one operand pair of loop-basis coefficients of shape (2, D)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    complex_domain = domain == "complex"
    dtype = np.complex128 if complex_domain else np.float64
    size = 1 << level
    shape = (2, size)

    def normal(form: tuple[int, ...] = shape) -> np.ndarray:
        result = rng.standard_normal(form).astype(dtype)
        if complex_domain:
            result += 1j * rng.standard_normal(form)
        return result / euclidean_norm(result)

    x, y = normal(), normal()
    if family == "uniform":
        x, y = rng.uniform(-1, 1, (2, *shape)).astype(dtype)
        if complex_domain:
            x += 1j * rng.uniform(-1, 1, shape)
            y += 1j * rng.uniform(-1, 1, shape)
    elif family == "rademacher":
        scale = 2.0 ** (-math.ceil((level + 1) / 2))
        x, y = (rng.choice([-1.0, 1.0], (2, *shape)) * scale).astype(dtype)
        if complex_domain:
            x += 1j * rng.choice([-1.0, 1.0], shape) * scale
            y += 1j * rng.choice([-1.0, 1.0], shape) * scale
    elif family.startswith("sparse_"):
        support = {"sparse_one": 1, "sparse_four": min(4, 2 * size), "sparse_eighth": max(1, size // 8)}[family]
        for v in (x, y):
            keep = rng.choice(2 * size, support, replace=False)
            flat = v.reshape(-1)
            flat[np.setdiff1d(np.arange(2 * size), keep)] = 0
    elif family in ("correlated_plus", "correlated_minus", "repeated"):
        y = x.copy() if family == "repeated" else (1 if family.endswith("plus") else -1) * x + 2.0 ** -40 * y
    elif family == "alternating_magnitude":
        magnitude = np.exp2(np.linspace(-24, 24, 2 * size).astype(int)).reshape(shape)
        x = (rng.choice([-1.0, 1.0], shape) * magnitude).astype(dtype)
        y = (rng.choice([-1.0, 1.0], shape) * magnitude[::-1, ::-1]).astype(dtype)
        if complex_domain:
            x += 1j * magnitude[::-1, ::-1]
            y -= 1j * magnitude
    elif family in ("signed_cancel", "group_cancel"):
        # b = +-a + 2**-40 noise leaves one component of the decomposition tiny
        sign = 1 if family == "signed_cancel" else -1
        x[1] = sign * x[0] + 2.0 ** -40 * x[1]
        y[1] = sign * y[0] + 2.0 ** -40 * y[1]
    elif family == "group_ideal":
        x[1], y[1] = x[0], y[0]
    elif family == "signed_ideal":
        x[1], y[1] = -x[0], -y[0]
    elif family == "cross_ideal":
        x[1], y[1] = x[0], -y[0]
    elif family == "idempotent":
        # A sum of primitive idempotents: coefficients sum_a chi_a(t) / (2 D) in both rows
        chosen = rng.random(size) < 0.5
        chosen[rng.integers(size)] = True
        row = characters(size)[chosen].sum(axis=0) / (2 * size)
        x = np.stack([row, row]).astype(dtype)
    elif family == "character":
        # One spectral coordinate: a scalar multiple of a single primitive idempotent
        row = characters(size)[rng.integers(size)] / (2 * size)
        scalar = rng.standard_normal() + (1j * rng.standard_normal() if complex_domain else 0.0)
        x = (scalar * np.stack([row, row])).astype(dtype)
    elif family == "scalar_only":
        first_x, first_y = x[0, 0], y[0, 0]
        x, y = np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype)
        x[0, 0], y[0, 0] = first_x, first_y
    elif "zero_divisor" in family:
        if size < 16 or complex_domain:
            raise ValueError("zero divisors are real and require level >= 4")
        a, b = np.zeros(size), np.zeros(size)
        a[:16], b[:16] = zero_divisor()
        noise_x, noise_y = x, y
        x, y = lift_signed(a), lift_signed(b)
        if family != "zero_divisor":
            scale = 2.0 ** -int(family.rsplit("_", 1)[1])
            x = x + scale * noise_x
            y = y + scale * noise_y
    elif family in ("isotropic", "near_isotropic"):
        # The signed component e_0 + i e_1 has Q = 1 + i**2 = 0
        if not complex_domain or size < 2:
            raise ValueError("isotropic operands are complex and require level >= 1")
        d = np.zeros(size, dtype=dtype)
        d[0], d[1] = 1, 1j
        noise = x
        x = lift_signed(d)
        if family == "near_isotropic":
            x = x + 2.0 ** -40 * noise
    return np.ascontiguousarray(x, dtype=dtype), np.ascontiguousarray(y, dtype=dtype)


def make_chain(domain: str, level: int, length: int, seed: int) -> np.ndarray:
    """Return the operands of one chain: unit-norm Gaussian elements of shape (length, 2, D)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    shape = (length, 2, 1 << level)
    operands = rng.standard_normal(shape).astype(np.complex128 if domain == "complex" else np.float64)
    if domain == "complex":
        operands += 1j * rng.standard_normal(shape)
    for operand in operands:
        operand /= euclidean_norm(operand)
    return operands


def available(domain: str, level: int, family: str) -> bool:
    """Return whether a family exists at a level of a domain."""
    if "zero_divisor" in family:
        return domain == "real" and level >= 4
    if "isotropic" in family:
        return domain == "complex" and level >= 1
    return True


# The measured methods


@cache
def algebra(domain: str, level: int) -> Any:
    """Return the loop algebra of Q_level over R or C."""
    import pycdloop as cdl

    selectors = {"char": 0, "deg": 2} if domain == "complex" else {}
    return cdl.LoopAlgebra(cdl.CDLoop(level), **selectors)


def split_product(domain: str, level: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S: the split algorithm on arrays of one record."""
    parent = algebra(domain, level)
    return np.array((parent.asarray(x[None]) * parent.asarray(y[None])).coefficients[0])


def table_product(domain: str, level: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """T: the table algorithm on arrays of one record."""
    parent = algebra(domain, level)
    result = parent.multiply(parent.asarray(x[None]), parent.asarray(y[None]), algorithm="table")
    return np.array(result.coefficients[0])


def chain_split(domain: str, level: int, operands: np.ndarray) -> np.ndarray:
    """L: a left-nested chain kept in split form, converted once at the end."""
    parent = algebra(domain, level)
    result = parent(operands[0])
    for operand in operands[1:]:
        result = result * parent(operand)
    return np.array(result.coefficients)


def chain_roundtrip(domain: str, level: int, operands: np.ndarray) -> np.ndarray:
    """R: a left-nested chain that returns to the loop basis after every product."""
    parent = algebra(domain, level)
    result = parent.asarray(operands[0][None].copy())
    for operand in operands[1:]:
        result = result * parent.asarray(operand[None].copy())
    return np.array(result.coefficients[0])


# The study


def sample_count(level: int, family: str, profile: str) -> int:
    """Return the number of operand pairs of one family at one level."""
    if profile == "smoke":
        return 2 if family in DENSE else 1
    dense = family in DENSE
    if level <= 5:
        return 128 if dense else 8
    if level <= 7:
        return 32 if dense else 4
    return 8 if dense else 2


def chain_count(level: int, profile: str) -> int:
    """Return the number of chains of one length at one level."""
    if profile == "smoke":
        return 2
    return 64 if level <= 5 else (16 if level <= 7 else 4)


def case_seed(root_seed: int, *labels: int) -> int:
    """Return the seed of one case from the root seed and its labels."""
    return int(np.random.SeedSequence([root_seed, *labels]).generate_state(1, dtype=np.uint64)[0])


def payload(*arrays: np.ndarray) -> bytes:
    """Return the little-endian bytes of stored operands."""
    return b"".join(a.astype(a.dtype.newbyteorder("<")).tobytes() for a in arrays)


def run_products(domain: str, level: int, family: str, profile: str, seed: int, precision: int) -> dict[str, Any]:
    """Measure S and T on every operand pair of one family at one level."""
    families = REAL_FAMILIES if domain == "real" else COMPLEX_FAMILIES
    rows, arrays, checks = [], {}, []
    for sample in range(sample_count(level, family, profile)):
        number = case_seed(seed, 0, domain == "complex", level, families.index(family), sample)
        x, y = make_inputs(domain, level, family, number)
        key = f"{domain}_{level}_{family}_{sample}"
        arrays[key + "_x"], arrays[key + "_y"] = x, y
        ex, ey = exact(x), exact(y)
        reference = exact_product(ex, ey)
        outputs = {"S": split_product(domain, level, x, y), "T": table_product(domain, level, x, y)}
        norms = [squared_norm(ex), squared_norm(ey)]
        methods, totals = measures(outputs, reference, norms, precision)
        rows.append({"key": key, "domain": domain, "level": level, "dimension": 2 << level, "family": family,
                     "sample": sample, "seed": number, "operands_sha256": hashlib.sha256(payload(x, y)).hexdigest(),
                     "methods": methods, "s_better": totals["S"] < totals["T"], "tied": totals["S"] == totals["T"]})
        if family == "gaussian" and sample == 0 and level in (1, 3, 5):
            for bits in (256, 512):
                with gp.context(precision=bits):
                    candidate = mp_product(x, y, bits)
                    difference = max(abs(a - b) for row_a, row_b in zip(as_mp(reference), candidate, strict=True)
                                     for a, b in zip(row_a, row_b, strict=True))
                observed, _ = measures(outputs, reference, norms, bits)
                checks.append({"key": key, "precision_bits": bits, "maximum_reference_discrepancy": float(difference),
                               "normalized_error_S": observed["S"]["normalized"]})
    return {"samples": rows, "chains": [], "arrays": arrays, "checks": checks}


def run_chains(domain: str, level: int, length: int, profile: str, seed: int, precision: int) -> dict[str, Any]:
    """Measure L and R on every chain of one length at one level."""
    rows, arrays = [], {}
    for sample in range(chain_count(level, profile)):
        number = case_seed(seed, 1, domain == "complex", level, length, sample)
        operands = make_chain(domain, level, length, number)
        key = f"chain_{domain}_{level}_{length}_{sample}"
        arrays[key] = operands
        reference = exact_chain(list(operands))
        outputs = {"L": chain_split(domain, level, operands), "R": chain_roundtrip(domain, level, operands)}
        norms = [squared_norm(exact(operand)) for operand in operands]
        methods, totals = measures(outputs, reference, norms, precision)
        rows.append({"key": key, "domain": domain, "level": level, "dimension": 2 << level, "length": length,
                     "sample": sample, "seed": number, "operands_sha256": hashlib.sha256(payload(operands)).hexdigest(),
                     "methods": methods, "l_at_least_as_accurate": totals["L"] <= totals["R"],
                     "tied": totals["L"] == totals["R"]})
    return {"samples": [], "chains": rows, "arrays": arrays, "checks": []}


def run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    """Run one task of the study in a worker process."""
    kind, *arguments = task
    return run_products(*arguments) if kind == "product" else run_chains(*arguments)


def tasks(args: argparse.Namespace) -> list[tuple[Any, ...]]:
    """List the tasks of the study, largest first so that workers stay busy."""
    result: list[tuple[Any, ...]] = []
    for domain, maximum, chain_maximum in (("real", args.real_levels, args.real_chain_levels),
                                           ("complex", args.complex_levels, args.complex_chain_levels)):
        for level in range(1, maximum + 1):
            for family in REAL_FAMILIES if domain == "real" else COMPLEX_FAMILIES:
                if available(domain, level, family):
                    result.append(("product", domain, level, family, args.profile, args.seed, args.precision))
        for level in range(1, chain_maximum + 1):
            for length in CHAIN_LENGTHS:
                result.append(("chain", domain, level, length, args.profile, args.seed, args.precision))
    return sorted(result, key=lambda task: -task[2])


def cpu_model() -> str | None:
    """Return the CPU model name where the system reports one."""
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def environment() -> dict[str, Any]:
    """Return the software and host on which the study runs."""
    import pyfastcd

    import pycdloop

    return {"python": sys.version, "platform": platform.platform(), "cpu_model": cpu_model(),
            "numpy": np.__version__, "gmpy2": gp.version(), "mpfr": gp.mpfr_version(),
            "pyfastcd": pyfastcd.__version__, "pycdloop": pycdloop.__version__}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the study and write the results and the corpus."""
    if args.precision < 256:
        raise ValueError("precision must be at least 256 bits")
    if args.profile == "smoke":
        args.real_levels, args.complex_levels = min(args.real_levels, 4), min(args.complex_levels, 4)
        args.real_chain_levels, args.complex_chain_levels = min(args.real_chain_levels, 3), min(args.complex_chain_levels, 3)
    validation = validate_oracle()
    a, b = zero_divisor()
    validation["zero_divisor"] = {"x_support": np.flatnonzero(a).tolist(), "x_values": a[a != 0].tolist(),
                                 "y_support": np.flatnonzero(b).tolist(), "y_values": b[b != 0].tolist(),
                                 "verified_exactly": True}
    start = time.perf_counter()
    work = tasks(args)
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_task, work))
    else:
        results = []
        for number, task in enumerate(work, 1):
            results.append(run_task(task))
            print(f"task {number}/{len(work)}: {task[0]} {task[1]} level {task[2]} {task[3]}; {time.perf_counter() - start:.1f}s", flush=True)
    order = {family: number for number, family in enumerate(dict.fromkeys(REAL_FAMILIES + COMPLEX_FAMILIES))}
    samples = sorted((row for result in results for row in result["samples"]),
                     key=lambda r: (r["domain"], r["level"], order[r["family"]], r["sample"]))
    chains = sorted((row for result in results for row in result["chains"]),
                    key=lambda r: (r["domain"], r["level"], r["length"], r["sample"]))
    arrays = {key: value for result in results for key, value in result["arrays"].items()}
    corpus_hash = hashlib.sha256()
    for key in sorted(arrays):
        corpus_hash.update(key.encode() + payload(arrays[key]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output.with_name(args.output.stem + "-corpus.npz")
    np.savez_compressed(corpus_path, **{key: arrays[key] for key in sorted(arrays)})
    defaults = {domain: {str(level): algebra(domain, level).default_signed_algorithm for level in range(1, maximum + 1)}
                for domain, maximum in (("real", args.real_levels), ("complex", args.complex_levels))}
    document = {
        "schema": SCHEMA, "profile": args.profile, "completed": True, "command": [sys.executable, *sys.argv],
        "seed": args.seed, "precision_bits": args.precision, "jobs": args.jobs,
        "levels": {"real": args.real_levels, "complex": args.complex_levels,
                   "real_chains": args.real_chain_levels, "complex_chains": args.complex_chain_levels},
        "oracle": "exact binary64-to-integer conversion, loop-algebra product summed from its definition in arbitrary-precision integers; MPFR norms",
        "oracle_validation": validation,
        "precision_checks": [check for result in results for check in result["checks"]],
        "signed_algorithms": defaults, "environment": environment(),
        "tool_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), Path(__file__).with_name("stability_report.py"))},
        "corpus": {"path": corpus_path.name, "npz_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                   "ordered_operands_sha256": corpus_hash.hexdigest(), "pairs": len(samples), "chains": len(chains)},
        "elapsed_seconds": time.perf_counter() - start,
        "limitations": ["Finite samples describe only the measured corpus; groups below 100 samples do not support tail estimates.",
                        "Controlled finite exponent ranges; no overflow, underflow or exceptional-value study."],
        "samples": samples, "chains": chains,
    }
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON results file; the corpus is written beside it")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--real-levels", type=int, default=10, help="largest level of Q_n over R")
    parser.add_argument("--complex-levels", type=int, default=8, help="largest level of Q_n over C")
    parser.add_argument("--real-chain-levels", type=int, default=8, help="largest level of the chain experiment over R")
    parser.add_argument("--complex-chain-levels", type=int, default=6, help="largest level of the chain experiment over C")
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 2)), help="worker processes")
    args = parser.parse_args(argv)
    for name in ("real_levels", "complex_levels", "real_chain_levels", "complex_chain_levels", "jobs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    run(parse_arguments())
