"""Operations over R and C at every size of their operands.

A float holds values from about ``2**-1074`` to ``2**1024``. An operation
whose result is representable is expected to return it whatever the sizes
of the values formed on the way, and these tests compare the operations
that form squares, norms and quotients with rational arithmetic on the
stored coefficients, over that whole range.
"""

from __future__ import annotations

import copy
import pickle
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import pycdloop as cdl
from pycdloop import _kernel, _range, _structure, codes, linalg
from pycdloop._array import _rebuild_array, _rebuild_split
from pycdloop._element import _rebuild

Exact = tuple[Fraction, Fraction]

ZERO: Exact = (Fraction(0), Fraction(0))
ONE: Exact = (Fraction(1), Fraction(0))
# A value at or beyond this rounds to an infinity
TOP = Fraction(2) ** 1024 - Fraction(2) ** 970
# Within a few units in the last place of the largest float, either is right
CLEAR = Fraction(2) ** 1024 + Fraction(2) ** 990
FLOOR = Fraction(1, 2**1070)


def _exact(value: Any) -> Exact:
    """Return a float or a complex float as a pair of rational numbers."""

    number = complex(value)
    return Fraction(number.real), Fraction(number.imag)


def _plus(x: Exact, y: Exact) -> Exact:
    return x[0] + y[0], x[1] + y[1]


def _minus(x: Exact, y: Exact) -> Exact:
    return x[0] - y[0], x[1] - y[1]


def _times(x: Exact, y: Exact) -> Exact:
    return x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]


def _over(x: Exact, y: Exact) -> Exact:
    size = y[0] * y[0] + y[1] * y[1]
    return (x[0] * y[0] + x[1] * y[1]) / size, (x[1] * y[0] - x[0] * y[1]) / size


def _size2(x: Exact) -> Fraction:
    return x[0] * x[0] + x[1] * x[1]


def _norm(row: list[Exact]) -> Exact:
    """Return the sum of the squares of exact numbers."""

    total = ZERO
    for value in row:
        total = _plus(total, _times(value, value))
    return total


def _agrees(got: Any, exact: Exact, scale2: Fraction, bits: int = 36) -> bool:
    """Return whether a computed number is its exact value, relative to a size.

    A complex number with a part beyond the range is beyond the range, and
    then the computed number has to say so with a part that is not finite.
    """

    number = complex(got)
    finite = bool(np.isfinite(number.real) and np.isfinite(number.imag))
    if abs(exact[0]) >= TOP or abs(exact[1]) >= TOP:
        if max(abs(exact[0]), abs(exact[1])) < CLEAR or not finite:
            return True
        # A finite number stands for a value beyond the range only where
        # that value is a difference of terms far larger still
        return _size2(exact) <= scale2 / Fraction(4) ** bits
    if not (np.isfinite(number.real) and np.isfinite(number.imag)):
        return False
    error2 = (Fraction(number.real) - exact[0]) ** 2
    error2 += (Fraction(number.imag) - exact[1]) ** 2
    return error2 <= scale2 / Fraction(4) ** bits + FLOOR * FLOOR


REGIMES = ("full", "top", "bottom", "mid", "two", "half", "edge")
EDGES = [-1074, -1073, -1023, -1022, -513, -512, -511, 0, 510, 511, 512, 1022, 1023]


def _exponents(rng: Any, regime: str, shape: tuple[int, ...]) -> np.ndarray:
    """Return binary exponents: anywhere, at an end, at both ends, near a limit."""

    if regime == "full":
        return rng.integers(-1074, 1024, shape)
    if regime == "top":
        return rng.integers(990, 1024, shape)
    if regime == "bottom":
        return rng.integers(-1074, -1030, shape)
    if regime == "mid":
        return rng.integers(-40, 41, shape)
    if regime == "two":
        high = rng.integers(960, 1024, shape)
        return np.where(
            rng.random(shape) < 0.5, high, rng.integers(-1074, -1000, shape)
        )
    if regime == "half":
        return rng.integers(-520, 521, shape)
    return rng.choice(EDGES, shape)


def _reals(
    rng: Any, regime: str, shape: tuple[int, ...], sparse: float, simple: bool
) -> np.ndarray:
    if simple:
        mantissa = rng.choice([1.0, 1.5, 1.25, 1.75], shape)
    else:
        mantissa = rng.uniform(1.0, 2.0, shape)
    signs = rng.choice([-1.0, 1.0], shape)
    values = np.ldexp(mantissa * signs, _exponents(rng, regime, shape))
    return np.where(rng.random(shape) < sparse, 0.0, values)


def _numbers(
    rng: Any,
    kind: str,
    regime: str,
    shape: tuple[int, ...],
    sparse: float = 0.0,
    simple: bool = False,
) -> np.ndarray:
    """Return floats, or complex floats whose parts are drawn apart."""

    real = _reals(rng, regime, shape, sparse, simple)
    if kind == "R":
        return real
    return real + 1j * _reals(rng, regime, shape, max(sparse, 0.3), simple)


def _algebra(level: int, kind: str) -> Any:
    loop = cdl.CDLoop(level)
    return cdl.LoopAlgebra(loop) if kind == "R" else cdl.LoopAlgebra(loop, deg=2)


CASES = [(level, kind) for kind in ("R", "C") for level in (0, 1, 2, 3)]


# The arithmetic with exponents that cannot leave the range


def test_wide_numbers_round_as_rational_arithmetic_does() -> None:
    """Sums, products and quotients agree with fractions over the whole range."""

    rng = np.random.default_rng(71)
    wide = _range.Wide
    for _ in range(600):
        a, b, c, d = _reals(rng, "full", (4,), 0.0, False)
        fa, fb, fc, fd = (Fraction(float(v)) for v in (a, b, c, d))
        for got, want in (
            (wide.of(a) * wide.of(b), fa * fb),
            (wide.of(a) + wide.of(b), fa + fb),
            (wide.of(a) - wide.of(b), fa - fb),
            (wide.of(a) / wide.of(b), fa / fb),
            (wide.of(a) * wide.of(b) * wide.of(c) / wide.of(d), fa * fb * fc / fd),
        ):
            with np.errstate(over="ignore"):
                value = float(got.to_array())
            assert _agrees(value, (want, Fraction(0)), want * want, bits=50)
        z, w = complex(a, b), complex(c, d)
        with np.errstate(over="ignore"):
            product = complex(
                (wide.of(np.asarray(z)) * wide.of(np.asarray(w))).to_array()
            )
        # Each part is accurate relative to the larger of its two terms
        parts = (
            (product.real, fa * fc - fb * fd, max(abs(fa * fc), abs(fb * fd))),
            (product.imag, fa * fd + fb * fc, max(abs(fa * fd), abs(fb * fc))),
        )
        for value, want, largest in parts:
            assert _agrees(value, (want, Fraction(0)), largest * largest, bits=48)


def test_wide_numbers_keep_zero_infinity_and_the_far_ends() -> None:
    """Zero, infinities and exponents far beyond every float behave as limits do."""

    wide = _range.Wide
    with np.errstate(all="ignore"):
        kept = wide.of(np.array([0.0, 1.0, np.inf, np.nan, -3.5, 5e-324])).to_array()
    assert np.array_equal(
        kept, [0.0, 1.0, np.inf, np.nan, -3.5, 5e-324], equal_nan=True
    )
    two = wide.of(np.array([2.0]))
    zero = wide.of(np.array([0.0]))
    assert (two * zero).to_array()[0] == 0 and (zero + zero).is_zero()[0]
    assert (two - two).is_zero()[0] and not (two + two).is_zero()[0]
    with np.errstate(over="ignore"):
        assert wide.of(np.array([3.0]), 5000).to_array()[0] == np.inf
        assert wide.of(np.array([3.0]), -5000).to_array()[0] == 0
        far = wide.of(np.array([2.0]), 1 << 39)
        huge = far * far * far
        assert huge.to_array()[0] == np.inf
        # The reciprocal of a number that far out is zero, and stays zero
        assert (two / huge).to_array()[0] == 0
        assert ((two / huge) * two).to_array()[0] == 0
    # A product of a huge and a tiny number is what it is
    up, down = wide.of(np.array([1.5]), 100000), wide.of(np.array([0.5]), -100003)
    assert (up * down).to_array()[0] == 1.5 * 0.5 / 8
    # A sum is that of floats, without the overflow of a partial sum
    top = 2.0**1023
    total = _range.total(wide.of(np.array([[top, top, -top, 2.0**-1000]])))
    assert total.to_array()[0] == top


def test_a_rational_number_is_rounded_once() -> None:
    """The mantissa of a fraction is the nearest of 53 bits."""

    rng = np.random.default_rng(72)
    for _ in range(400):
        top = int(rng.integers(1, 1 << 62)) * int(rng.integers(1, 1 << 62)) + 1
        bottom = int(rng.integers(1, 1 << 62)) * int(rng.integers(1, 1 << 40)) + 1
        value = Fraction(top if rng.random() < 0.5 else -top, bottom)
        mantissa, exponent = _range.rational(value)
        assert 0.5 <= abs(mantissa) < 1 or abs(mantissa) == 1.0
        assert np.ldexp(mantissa, exponent) == float(value)
    assert _range.rational(Fraction(0))[0] == 0.0
    tiny = Fraction(3, 2**3000)
    mantissa, exponent = _range.rational(tiny)
    assert (mantissa, exponent) == (0.75, -2998)


def test_keeping_small_coefficients_does_not_overflow_large_ones() -> None:
    """A row is rescaled so that both its ends stay in range where they can."""

    high, low = np.ldexp(1.0, 1020), np.ldexp(1.0, -1020)
    for row in ([high, low], [high, 1.0], [low, 1.0], [low, low], [high, high]):
        scaled, powers = _range.unit(np.array([row]), always=True, keep=True)
        assert np.isfinite(scaled).all() and (scaled != 0).all(), row
        assert np.array_equal(np.ldexp(scaled, powers), [row])
    # Where the spread fits, nothing is rounded
    row = np.array([[np.ldexp(1.0, 600), np.ldexp(1.2345, -1000)]])
    scaled, powers = _range.unit(row, always=True, keep=True)
    assert np.array_equal(np.ldexp(scaled, powers), row)
    assert np.abs(scaled).min() >= np.ldexp(1.0, -1001)


# Powers


@pytest.mark.parametrize(("level", "kind"), CASES)
def test_powers_agree_with_rational_arithmetic(level: int, kind: str) -> None:
    """A coefficient of a power in range is right, whatever was formed on the way."""

    rng = np.random.default_rng(100 + level + 10 * (kind == "C"))
    algebra = _algebra(level, kind)
    size = 1 << level
    count = 12
    for regime in REGIMES:
        for n in (0, 1, 2, 3, 5, 8):
            for squeeze in (False, True):
                d = _numbers(rng, kind, regime, (count, size), 0.3, bool(n % 2))
                s = _numbers(rng, kind, regime, (count, size))
                if squeeze and n > 1:
                    # Operands whose n-th power can be in range
                    d = _rooted(d, n)
                    s = _rooted(s, n)
                with np.errstate(all="ignore"):
                    result = algebra.split_array(s, d) ** n
                signed = np.asarray(result.signed.coefficients)
                spectral = np.asarray(result.spectral)
                for r in range(count):
                    row = [_exact(v) for v in d[r]]
                    q = _norm(row[1:])
                    a, b = ONE, ZERO
                    for _ in range(n):
                        a, b = (
                            _minus(_times(a, row[0]), _times(b, q)),
                            _plus(a, _times(b, row[0])),
                        )
                    # The powers of d_0 + i sqrt(Q) bound the terms of the result
                    bound = 2 * (_size2(row[0]) + abs(q[0]) + abs(q[1]))
                    where = (regime, n, squeeze, d[r].tolist())
                    assert _agrees(signed[r, 0], a, bound**n), where
                    for j in range(1, size):
                        limit = bound ** max(n - 1, 0) * _size2(row[j]) * n * n
                        assert _agrees(signed[r, j], _times(b, row[j]), limit), where
                    for j in range(size):
                        value = _exact(s[r, j])
                        power = ONE
                        for _ in range(n):
                            power = _times(power, value)
                        assert _agrees(spectral[r, j], power, _size2(value) ** n), where


def _rooted(values: np.ndarray, n: int) -> np.ndarray:
    """Return values with their binary exponents divided by *n*."""

    def part(x: np.ndarray) -> np.ndarray:
        mantissa, exponent = np.frexp(x)
        moved: np.ndarray = np.ldexp(mantissa, exponent // n)
        return moved

    if np.iscomplexobj(values):
        return part(values.real) + 1j * part(values.imag)
    return part(values)


def test_powers_the_audit_found_lost() -> None:
    """Powers whose recurrence passes through values beyond the range."""

    real = cdl.LoopAlgebra(cdl.CDLoop(1))
    with np.errstate(all="ignore"):
        scalar = real(2) ** 1010
    assert np.array_equal(scalar.coefficients, [[np.ldexp(1.0, 1010), 0], [0, 0]])
    # d = 2 + 2**-600 e_1: d**n = 2**n + n 2**(n - 1) 2**-600 e_1 to the precision
    with np.errstate(all="ignore"):
        power = real.split_array([2.0, 2.0], [2.0, 2.0**-600]) ** 1010
    signed = np.asarray(power.signed.coefficients)
    assert signed[0] == np.ldexp(1.0, 1010)
    assert signed[1] == 1010 * np.ldexp(1.0, 1009 - 600)
    # Q(d) = 2**-1200 exactly, from squares of 2**1200 that cancel, and
    # d**3 = -Q(d) d
    complex_ = cdl.LoopAlgebra(cdl.CDLoop(2), deg=2)
    d = np.array([0, 2.0**600, 1j * 2.0**600, 2.0**-600])
    with np.errstate(all="ignore"):
        cube = complex_.split_array(np.ones(4, dtype=complex), d) ** 3
    signed = np.asarray(cube.signed.coefficients)
    assert signed[0] == 0 and signed[3] == 0
    assert signed[1] == -(2.0**-600) and signed[2] == -1j * 2.0**-600


def test_powers_with_exponents_of_any_size() -> None:
    """The exponent of a value on the way can be beyond every float."""

    real = cdl.LoopAlgebra(cdl.CDLoop(1))
    one = [[1.0, 0.0], [0.0, 0.0]]
    for exponent in (2**28, 2**45, 10**30):
        with np.errstate(all="ignore"):
            assert np.array_equal((real.one() ** exponent).coefficients, one)
            assert np.array_equal((real(-1) ** exponent).coefficients, one)
            shrunk = real(0.5) ** exponent
            grown = real.split_array([1.0, 1.0], [2.0, 0.0]) ** exponent
        assert not shrunk.coefficients.any()
        assert np.asarray(grown.signed.coefficients).tolist() == [np.inf, 0.0]
    # 1 + w with a tiny w: (1 + w)**n = 1 + n w, to the precision
    tiny = np.ldexp(1.0, -900)
    with np.errstate(all="ignore"):
        power = real.split_array([1.0, 1.0], [1.0, tiny]) ** (2**40)
    assert np.asarray(power.signed.coefficients).tolist() == [1.0, np.ldexp(1.0, -860)]


@pytest.mark.parametrize(("level", "kind"), CASES)
def test_both_power_recurrences_agree_in_the_ordinary_range(
    level: int, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recurrence with wide exponents gives what the plain one gives."""

    rng = np.random.default_rng(140 + level)
    algebra = _algebra(level, kind)
    size = 1 << level
    d = rng.integers(-5, 6, (30, size)).astype(algebra.dtype)
    s = rng.integers(-5, 6, (30, size)).astype(algebra.dtype)
    if kind == "C":
        d = d + 1j * rng.integers(-5, 6, (30, size))
        d = d * rng.uniform(0.5, 2.0, (30, 1))
    else:
        d = d * rng.uniform(0.5, 2.0, (30, size))
    array = algebra.split_array(s, d)
    for n in (2, 3, 4, 7, 12):
        assert _kernel._fits_power(algebra, d, n)
        plain = np.asarray((array**n).signed.coefficients)
        with monkeypatch.context() as patch:
            patch.setattr(_kernel, "_fits_power", lambda *args: False)
            wide = np.asarray((array**n).signed.coefficients)
        if kind == "R":
            # The same operations in the same order, on scaled values
            assert np.array_equal(plain, wide)
        else:
            assert np.allclose(plain, wide, rtol=1e-12, atol=0)


def test_powers_keep_a_small_part_of_a_complex_norm() -> None:
    """``x**2`` has the imaginary part that ``x * x`` has."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1), deg=2)
    d = np.array([[0, 2.0**500 + 1j * 2.0**-600]])
    array = algebra.split_array(np.ones((1, 2), dtype=complex), d)
    with np.errstate(all="ignore"):
        square = np.asarray((array**2).signed.coefficients)
        product = np.asarray((array * array).signed.coefficients)
    assert np.array_equal(square, product)
    assert square[0, 0] == complex(-(2.0**1000), -(2.0**-99))


def test_complex_powers_of_coordinates_near_the_top() -> None:
    """A part that is a difference of two products is not lost with one of them."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1), deg=2)
    # (a + b i)**2 has real part a**2 - b**2 with b**2 = 2**1024
    value = complex(0.4375 * 2.0**512, -(2.0**512))
    array = algebra.split_array(np.array([value, 1.0]), np.zeros(2, dtype=complex))
    with np.errstate(all="ignore"):
        spectral = np.asarray((array**2).spectral)
    want = _times(_exact(value), _exact(value))
    assert np.isfinite(spectral[0].real)
    assert _agrees(spectral[0], (want[0], Fraction(0)), want[0] ** 2, bits=50) or (
        spectral[0].real == float(want[0])
    )
    assert spectral[0].real == float(want[0]) and spectral[1] == 1


# Inverses and division


@pytest.mark.parametrize(("level", "kind"), CASES)
def test_inverses_agree_with_rational_arithmetic(level: int, kind: str) -> None:
    """``has_inverse`` is decided exactly and an inverse in range is right."""

    rng = np.random.default_rng(200 + level + 10 * (kind == "C"))
    algebra = _algebra(level, kind)
    size = 1 << level
    count = 14
    for regime in REGIMES:
        for sparse, simple in ((0.0, False), (0.4, True)):
            d = _numbers(rng, kind, regime, (count, size), sparse, simple)
            s = _numbers(rng, kind, regime, (count, size), 0.0, simple)
            flags = np.asarray(algebra.split_array(s, d).has_inverse())
            for r in range(count):
                row = [_exact(v) for v in d[r]]
                q = _norm(row)
                where = (regime, sparse, d[r].tolist())
                assert bool(flags[r]) == bool(q[0] or q[1]), where
                if not flags[r]:
                    continue
                with np.errstate(all="ignore"):
                    inverse = algebra.split_array(s[r], d[r]).inverse()
                signed = np.asarray(inverse.signed.coefficients)
                spectral = np.asarray(inverse.spectral)
                # A norm that cancels makes the inverse that much less determined
                weight = sum((_size2(v) for v in row), Fraction(0))
                condition2 = weight * weight / _size2(q)
                for j in range(size):
                    conjugate = row[j] if j == 0 else (-row[j][0], -row[j][1])
                    exact = _over(conjugate, q)
                    assert _agrees(signed[j], exact, _size2(exact) * condition2), where
                    exact = _over(ONE, _exact(s[r, j]))
                    assert _agrees(spectral[j], exact, _size2(exact)), where


@pytest.mark.parametrize(("level", "kind"), [(0, "R"), (2, "R"), (0, "C"), (2, "C")])
def test_scalar_division_agrees_with_rational_arithmetic(level: int, kind: str) -> None:
    """A quotient in range is right for every pair of sizes of its operands."""

    rng = np.random.default_rng(300 + level + 10 * (kind == "C"))
    algebra = _algebra(level, kind)
    size = 1 << level
    for regime in REGIMES:
        for other in REGIMES:
            d = _numbers(rng, kind, regime, (3, size), 0.3)
            divisors = _numbers(rng, kind, other, (3,), 0.0, bool(rng.integers(2)))
            for r in range(3):
                divisor = divisors[r].item()
                array = algebra.split_array(d[r], d[r])
                with np.errstate(all="ignore"):
                    signed = np.asarray((array / divisor).signed.coefficients)
                    spectral = np.asarray((array / divisor).spectral)
                    element = np.asarray((array[()] / divisor).epsilon().coefficients)
                where = (regime, other, d[r].tolist(), divisor)
                assert np.array_equal(signed, spectral, equal_nan=True), where
                assert np.array_equal(signed, element, equal_nan=True), where
                for j in range(size):
                    exact = _over(_exact(d[r, j]), _exact(divisor))
                    assert _agrees(signed[j], exact, _size2(exact), bits=48), where
                    if kind == "R":
                        # One division, correctly rounded
                        assert signed[j] == _nearest(exact[0]), where


def _nearest(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")


def test_division_the_audit_found_lost() -> None:
    """A numerator is not scaled beyond the range before a quotient that is in it."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(0), deg=2)
    divisor = complex(np.ldexp(1.5, -1023), 0)
    want = complex(2) / divisor
    assert np.isfinite(want)
    with np.errstate(all="ignore"):
        quotient = algebra(2) / divisor
        batch = algebra.array(algebra(2).coefficients[None]) / divisor
        split = algebra.array(algebra(2).coefficients[None]).split() / divisor
    assert np.array_equal(quotient.coefficients, [[want], [0]])
    assert np.array_equal(batch.coefficients, [[[want], [0]]])
    assert np.array_equal(np.asarray(split.spectral), [[want]])
    assert np.array_equal(np.asarray(split.signed.coefficients), [[want]])


def test_complex_division_by_a_real_or_an_imaginary_number_is_one_division() -> None:
    """Each part of the quotient is then correctly rounded, at the ends of the range."""

    rng = np.random.default_rng(31)
    for regime in ("top", "bottom", "two", "edge"):
        z = _numbers(rng, "C", regime, (40,))
        c = _reals(rng, regime, (40,), 0.0, False)
        with np.errstate(all="ignore"):
            by_real = _range.quotient(z, c + 0j)
            by_imaginary = _range.quotient(z, 1j * c)
        for j in range(40):
            a, b, f = Fraction(z[j].real), Fraction(z[j].imag), Fraction(c[j])
            assert by_real[j].real == _nearest(a / f) and by_real[j].imag == _nearest(
                b / f
            )
            assert by_imaginary[j].real == _nearest(b / f)
            assert by_imaginary[j].imag == _nearest(-a / f)


def test_both_complex_divisions_agree_in_the_ordinary_range() -> None:
    """Division with wide exponents gives what NumPy's gives, to the rounding."""

    rng = np.random.default_rng(32)
    z = rng.normal(size=200) + 1j * rng.normal(size=200)
    s = rng.normal(size=200) + 1j * rng.normal(size=200)
    plain = _range.quotient(z, s)
    assert np.array_equal(plain, z / s)
    # The same operands moved out of the ordinary range by a power of two
    moved = _range.quotient(np.ldexp(z.real, 700) + 1j * np.ldexp(z.imag, 700), s)
    back = np.ldexp(moved.real, -700) + 1j * np.ldexp(moved.imag, -700)
    assert np.allclose(back, plain, rtol=1e-14, atol=0)
    assert _range.quotient(np.array([6.0]), np.array([4.0]))[0] == 1.5


# Minimal polynomials


def _polynomial(s_row: list[Exact], d_row: list[Exact]) -> list[Exact]:
    """Return the minimal polynomial of a split form exactly, constant term first."""

    q = _norm(d_row[1:])
    first = d_row[0]
    if all(value == ZERO for value in d_row[1:]):
        factor = [(-first[0], -first[1]), ONE]
    else:
        factor = [_plus(_times(first, first), q), (-2 * first[0], -2 * first[1]), ONE]

    def at(x: Exact) -> Exact:
        total = ZERO
        for coefficient in reversed(factor):
            total = _plus(_times(total, x), coefficient)
        return total

    polynomial = factor
    seen: list[Exact] = []
    for value in s_row:
        if value in seen or at(value) == ZERO:
            continue
        seen.append(value)
        longer = [ZERO] * (len(polynomial) + 1)
        for i, coefficient in enumerate(polynomial):
            longer[i + 1] = _plus(longer[i + 1], coefficient)
            longer[i] = _minus(longer[i], _times(coefficient, value))
        polynomial = longer
    return polynomial


def _term_sizes(s_row: list[Exact], d_row: list[Exact]) -> list[Fraction]:
    """Return a bound for the terms of each coefficient, with every root kept."""

    sizes = [abs(v[0]) + abs(v[1]) for v in d_row]
    bound = [sizes[0] ** 2 + sum(x * x for x in sizes[1:]), 2 * sizes[0], Fraction(1)]
    for value in s_row:
        root = abs(value[0]) + abs(value[1])
        longer = [Fraction(0)] * (len(bound) + 1)
        for i, coefficient in enumerate(bound):
            longer[i + 1] += coefficient
            longer[i] += coefficient * root
        bound = longer
    return bound


@pytest.mark.parametrize(("level", "kind"), CASES)
def test_minimal_polynomials_agree_with_rational_arithmetic(
    level: int, kind: str
) -> None:
    """The degree is exact and each coefficient in range is right."""

    rng = np.random.default_rng(400 + level + 10 * (kind == "C"))
    algebra = _algebra(level, kind)
    size = 1 << level
    for regime in REGIMES:
        for sparse, simple in ((0.0, False), (0.5, True)):
            d = _numbers(rng, kind, regime, (6, size), sparse, simple)
            s = _numbers(rng, kind, regime, (6, size), 0.0, simple)
            if simple and size > 1:
                s[:, 1] = s[:, 0]
            for r in range(6):
                with np.errstate(all="ignore"):
                    got = algebra.split_array(s[r], d[r])[()].minimal_polynomial()
                s_row = [_exact(v) for v in s[r]]
                d_row = [_exact(v) for v in d[r]]
                exact = _polynomial(s_row, d_row)
                where = (regime, sparse, s[r].tolist(), d[r].tolist())
                assert len(got) == len(exact), where
                sizes = _term_sizes(s_row, d_row)[-len(exact) :]
                for value, want, size_ in zip(got, exact, sizes, strict=True):
                    scale2 = _size2(want) + size_ * size_ / Fraction(4) ** 10
                    assert _agrees(value, want, scale2, bits=40), where


def test_minimal_polynomial_whose_partial_products_leave_the_range() -> None:
    """The coefficients are in range although products of the roots are not."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1), deg=2)
    s = np.array([-2.0322170674846162e-97, -1.1201744628707478e111 - 4.5e88j])
    d = np.array([2485878618378662 + 4.9e-54j, -3.797279113188613e114 - 6.9e22j])
    assert not _structure._wide(s, d, 2) and _structure._wide(s, d, 4)
    with np.errstate(all="ignore"):
        got = algebra.split_array(s, d)[()].minimal_polynomial()
    exact = _polynomial([_exact(v) for v in s], [_exact(v) for v in d])
    # The constant term is in range and the next coefficient is not
    assert len(got) == 5 and np.isfinite(got[[0, 2, 3, 4]]).all()
    assert not np.isfinite(got[1])
    for value, want in zip(got, exact, strict=True):
        assert _agrees(value, want, _size2(want), bits=40)
    # A large loop with ordinary coefficients takes the same path, quickly
    large = cdl.LoopAlgebra(cdl.CDLoop(8))
    element = large.random(5)
    polynomial = element.minimal_polynomial()
    assert polynomial.shape == (259,) and np.isfinite(polynomial).all()
    assert polynomial[-1] == 1
    # So does one with a coordinate far from unit size, whose degree is
    # too large for rational arithmetic to be worth its cost
    spectral, signed = (np.array(part) for part in element._data("split"))
    spectral[3] = 2.0**700
    with np.errstate(all="ignore"):
        polynomial = large.split_array(spectral, signed)[()].minimal_polynomial()
    assert polynomial.shape == (259,) and polynomial[-1] == 1
    assert not np.isnan(polynomial).any()
    low = cdl.LoopAlgebra(cdl.CDLoop(2))
    spread = low.split_array([1.0, 2.0, 2.0**600, -(2.0**600)], [0.0, 1.0, 0, 0])
    # Rational arithmetic keeps a small term beside two large ones that cancel
    exact = spread[()].minimal_polynomial()
    assert exact[-2] == -3.0 and exact[-1] == 1


# Row reduction


def _reduced(rows: list[list[Exact]]) -> tuple[list[list[Exact]], list[int]]:
    """Return the reduced row echelon form of an exact matrix and its pivot columns."""

    work = [list(row) for row in rows]
    pivots: list[int] = []
    r = 0
    for c in range(len(work[0])):
        found = next((i for i in range(r, len(work)) if work[i][c] != ZERO), None)
        if found is None:
            continue
        work[r], work[found] = work[found], work[r]
        pivot = work[r][c]
        work[r] = [_over(v, pivot) for v in work[r]]
        for i in range(len(work)):
            if i != r and work[i][c] != ZERO:
                factor = work[i][c]
                work[i] = [
                    _minus(v, _times(factor, p))
                    for v, p in zip(work[i], work[r], strict=True)
                ]
        pivots.append(c)
        r += 1
        if r == len(work):
            break
    return work, pivots


@pytest.mark.parametrize("kind", ["R", "C"])
def test_row_reduction_of_rows_and_columns_of_any_size(kind: str) -> None:
    """Rows scaled apart over the whole range reduce as the unscaled matrix does."""

    rng = np.random.default_rng(500 + (kind == "C"))
    for trial in range(60):
        rows = int(rng.integers(1, 5))
        columns = int(rng.integers(rows, 7))
        while True:
            base = rng.integers(-3, 4, (rows, columns)).astype(float)
            if kind == "C":
                base = base + 1j * rng.integers(-3, 4, (rows, columns))
            if abs(np.linalg.det(base[:, :rows])) > 0.5:
                break
        spread = int(rng.choice([0, 0, 40, 400]))
        row_powers = _exponents(rng, REGIMES[trial % len(REGIMES)], (rows, 1))
        row_powers = np.clip(row_powers, -1070 + spread, 1019 - spread)
        powers = row_powers + rng.integers(-spread, spread + 1, (1, columns))
        matrix = np.ldexp(base.real, powers)
        if kind == "C":
            matrix = matrix + 1j * np.ldexp(base.imag, powers)
        with np.errstate(all="ignore"):
            got, pivots = linalg.row_reduce(matrix, tol=0)
        exact, exact_pivots = _reduced([[_exact(v) for v in row] for row in matrix])
        assert list(pivots) == exact_pivots == list(range(rows)), matrix.tolist()
        for i in range(rows):
            for j in range(columns):
                # Entries are accurate relative to the scale of their column
                scale2 = _size2(exact[i][j]) + Fraction(4) ** int(
                    powers[0, j] - powers[0, i]
                )
                assert _agrees(got[i, j], exact[i][j], scale2, bits=30), matrix.tolist()


def test_row_reduction_the_audit_found_lost() -> None:
    """An entry above the tolerance is a pivot, however far below the largest."""

    high, low = np.ldexp(1.0, 1000), np.ldexp(1.0, -1000)
    matrix = np.diag([high, low])
    reduced, pivots = linalg.row_reduce(matrix, tol=np.ldexp(1.0, -1010))
    assert pivots == (0, 1) and np.array_equal(reduced, np.eye(2))
    assert linalg.row_reduce(matrix, tol=0)[1] == (0, 1)
    # The default tolerance is relative to the largest entry, as documented
    assert linalg.row_reduce(matrix)[1] == (0,)
    assert linalg.row_reduce(matrix, tol=low)[1] == (0,)
    assert linalg.row_reduce(matrix, tol=np.inf)[1] == ()
    # Rows at both ends of the range, where no single rescaling fits
    top = 7 * np.ldexp(1.0, 1020)
    matrix = np.array([[top, top, 0], [top, -top, 0], [0, 0, low]])
    reduced, pivots = linalg.row_reduce(matrix, tol=0)
    assert pivots == (0, 1, 2) and np.array_equal(reduced, np.eye(3))
    both = matrix + 1j * matrix[::-1]
    reduced, pivots = linalg.row_reduce(both, tol=0)
    assert pivots == (0, 1, 2) and np.allclose(reduced, np.eye(3), atol=1e-12)
    # An entry of the reduced form beyond the range is an infinity
    with np.errstate(all="ignore"):
        reduced, pivots = linalg.row_reduce(np.array([[low, high]]), tol=0)
    assert pivots == (0,) and reduced.tolist() == [[1.0, np.inf]]


def test_row_reduction_through_values_beyond_the_range() -> None:
    """A finite reduced form is reached although a row on the way is not finite."""

    high, low = np.ldexp(1.0, 1000), np.ldexp(1.0, -1000)
    # The first row divided by its pivot holds 2**2000, which the second
    # pivot removes again
    matrix = np.array([[low, high], [0, high]])
    for tol in (np.ldexp(1.0, -1010), 0):
        reduced, pivots = linalg.row_reduce(matrix, tol=tol)
        assert pivots == (0, 1) and np.array_equal(reduced, np.eye(2))
    # A chain: each pivot is small beside the next entry of its row
    chain = np.diag(np.full(5, low)) + np.diag(np.full(4, high), 1)
    reduced, pivots = linalg.row_reduce(chain, tol=0)
    assert pivots == (0, 1, 2, 3, 4) and np.array_equal(reduced, np.eye(5))
    milder = np.diag(np.full(6, 2.0**-200)) + np.diag(np.full(5, 2.0**200), 1)
    reduced, pivots = linalg.row_reduce(milder, tol=0)
    assert pivots == tuple(range(6)) and np.array_equal(reduced, np.eye(6))
    # Products of two small entries, which underflow on the way
    small = np.array([[low, low, 0], [low, 0, low], [0, low, low]])
    reduced, pivots = linalg.row_reduce(small, tol=0)
    assert pivots == (0, 1, 2) and np.array_equal(reduced, np.eye(3))
    assert np.array_equal(linalg.row_reduce(small * 2.0**-70, tol=0)[0], np.eye(3))


def test_row_reduction_keeps_a_small_part_of_a_complex_entry() -> None:
    """Each part of a complex entry has its own size."""

    high, low = np.ldexp(1.0, 600), np.ldexp(1.0, -600)
    # The determinant is i * low: the rows differ in an imaginary part only
    matrix = np.array([[complex(high, low), 1], [high, 1]])
    reduced, pivots = linalg.row_reduce(matrix, tol=0)
    assert pivots == (0, 1) and np.array_equal(reduced, np.eye(2))
    swapped = np.array([[complex(low, high), 1], [complex(0, high), 1]])
    reduced, pivots = linalg.row_reduce(swapped, tol=0)
    assert pivots == (0, 1) and np.array_equal(reduced, np.eye(2))
    # The same holds for the rows that a code is reduced from
    scaled, powers = _range.unit(matrix, always=True, keep=True)
    assert np.array_equal(_range.shift(scaled, powers), matrix)
    assert (scaled.imag[0, 0] != 0) and np.isfinite(scaled).all()


def _bits(value: Fraction) -> int:
    """Return the number of significant bits of a dyadic rational, or a large number."""

    if value == 0:
        return 0
    bottom = value.denominator
    if bottom & (bottom - 1):
        return 10**6
    top = abs(value.numerator)
    return top.bit_length() - ((top & -top).bit_length() - 1)


def _eliminated(rows: list[list[Exact]]) -> tuple[list[list[Exact]], list[int], int]:
    """Reduce exactly as the library does, and report the longest number formed.

    This is scaled partial pivoting with a tolerance of zero. Where no
    value on the way needs more than 53 bits, elimination in floats whose
    exponents cannot leave the range rounds nothing.
    """

    work = [list(row) for row in rows]
    pivots: list[int] = []
    widest = 0
    r = 0
    for c in range(len(work[0])):
        if r == len(work):
            break
        best, chosen = None, None
        for i in range(r, len(work)):
            entry = _size2(work[i][c])
            if entry == 0:
                continue
            relative = entry / max(_size2(v) for v in work[i])
            if best is None or relative > best:
                best, chosen = relative, i
        if chosen is None:
            continue
        work[r], work[chosen] = work[chosen], work[r]
        pivot = work[r][c]
        normalized = [_over(v, pivot) for v in work[r]]
        for i in range(len(work)):
            if i == r:
                continue
            products = [_times(work[i][c], v) for v in normalized]
            work[i] = [_minus(v, p) for v, p in zip(work[i], products, strict=True)]
            widest = max([widest] + [_bits(x) for p in products for x in p])
        work[r] = normalized
        widest = max([widest] + [_bits(x) for row in work for v in row for x in v])
        pivots.append(c)
        r += 1
    return work, pivots, widest


_POWERS = [-1074, -1073, -1050, -1022, -1000, -800, -600, -520, -300, -53, -1, 0]
_POWERS += [1, 52, 300, 511, 512, 600, 800, 1000, 1010, 1022, 1023]


@pytest.mark.parametrize("kind", ["R", "C"])
def test_row_reduction_is_exact_where_elimination_rounds_nothing(kind: str) -> None:
    """Entries that are powers of two anywhere in the range reduce exactly."""

    rng = np.random.default_rng(520 + (kind == "C"))

    def part(density: float) -> float:
        if rng.random() > density:
            return 0.0
        if rng.random() < 0.5:
            power = int(rng.choice(_POWERS))
        else:
            power = int(rng.integers(-1074, 1024))
        return float(np.ldexp(rng.choice([-1.0, 1.0]), power))

    exact_cases = 0
    for _ in range(700):
        rows, columns = int(rng.integers(1, 5)), int(rng.integers(1, 6))
        density = float(rng.choice([0.5, 0.7, 0.9]))
        if kind == "R":
            entries = [[part(density) for _ in range(columns)] for _ in range(rows)]
        else:
            entries = [
                [
                    complex(part(density), part(density) if rng.random() < 0.5 else 0.0)
                    for _ in range(columns)
                ]
                for _ in range(rows)
            ]
        matrix = np.array(entries)
        with np.errstate(all="ignore"):
            got, pivots = linalg.row_reduce(matrix, tol=0)
        exact, exact_pivots, widest = _eliminated(
            [[_exact(v) for v in row] for row in matrix]
        )
        if widest > 53:
            continue
        exact_cases += 1
        assert list(pivots) == exact_pivots, matrix.tolist()
        for i in range(rows):
            for j in range(columns):
                value, want = complex(got[i, j]), exact[i][j]
                if (
                    abs(want[0]) >= Fraction(2) ** 1024
                    or abs(want[1]) >= Fraction(2) ** 1024
                ):
                    assert not np.isfinite(value), matrix.tolist()
                    continue
                assert np.isfinite(value), matrix.tolist()
                for number, fraction in ((value.real, want[0]), (value.imag, want[1])):
                    # Exact, or the nearest float below the normal range
                    assert abs(Fraction(number) - fraction) <= Fraction(1, 2**1074)
    assert exact_cases > 250


@pytest.mark.parametrize("kind", ["R", "C"])
def test_both_eliminations_agree_in_the_ordinary_range(kind: str) -> None:
    """Elimination with wide exponents gives what elimination in floats gives."""

    rng = np.random.default_rng(530 + (kind == "C"))
    for _ in range(60):
        rows, columns = int(rng.integers(1, 7)), int(rng.integers(1, 8))
        matrix = rng.normal(size=(rows, columns))
        if kind == "C":
            matrix = matrix + 1j * rng.normal(size=(rows, columns))
        dependent = rows > 1 and rng.random() < 0.5
        if dependent:
            matrix[-1] = matrix[0] * 2 - matrix[rows // 2]
        for tol in (None, 0, 1e-8):
            if dependent and tol == 0 and kind == "C":
                # The pivot of the dependent row is rounding alone, which the
                # two complex divisions leave differently
                continue
            with np.errstate(
                over="raise", under="raise", invalid="raise", divide="raise"
            ):
                plain, first = linalg._reduce_plain(matrix, tol)
            with np.errstate(all="ignore"):
                wide, second = linalg._reduce_wide(matrix, tol)
            assert first == second
            if kind == "R":
                # The same operations in the same order
                assert np.array_equal(plain, wide)
            else:
                assert np.allclose(plain, wide, rtol=1e-11, atol=1e-13)
            reduced, pivots = linalg.row_reduce(matrix, tol=tol)
            assert list(pivots) == first and np.array_equal(reduced, plain)


# Row reduction with its default tolerance


def _integer_reduced(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the exact reduced row echelon form of an integer matrix."""

    rows = [[(Fraction(int(v)), Fraction(0)) for v in row] for row in matrix]
    exact, pivots = _reduced(rows)
    return np.array([[float(v[0]) for v in row] for row in exact]), pivots


def test_rounding_of_a_dependent_row_is_not_taken_for_a_pivot() -> None:
    """A small integer matrix with a dependent row keeps its rank."""

    matrix = np.array([[3, -1, 1, -3], [-1, 3, 3, 3], [-3, 2, 0, -3], [2, 2, 4, 0]])
    assert np.array_equal(matrix[3], matrix[0] + matrix[1])
    want = np.array([[1, 0, 0, -21], [0, 1, 0, -33], [0, 0, 1, 27], [0, 0, 0, 0.0]])
    null = np.array([21, 33, -27, 1])
    assert not (matrix @ null).any()
    assert linalg.rank(matrix) == 3
    for values in (matrix, matrix.astype(float), matrix.astype(complex)):
        for tol in (None, 1e-9):
            reduced, pivots = linalg.row_reduce(values, tol=tol)
            assert pivots == (0, 1, 2)
            assert np.allclose(reduced, want, rtol=0, atol=1e-11)
            # The row space is kept: it still annihilates the null vector
            assert np.abs(reduced @ null).max() < 1e-10
            assert not reduced[3].any()
    # The same at the ends of the range, where each entry carries its exponent
    for power in (-1060, 1015):
        with np.errstate(all="ignore"):
            reduced, pivots = linalg.row_reduce(np.ldexp(matrix.astype(float), power))
        assert pivots == (0, 1, 2) and np.allclose(reduced, want, rtol=0, atol=1e-11)
    with np.errstate(all="ignore"):
        wide, pivots = linalg._reduce_wide(matrix.astype(float), None)
    assert pivots == [0, 1, 2]
    assert np.array_equal(wide, linalg.row_reduce(matrix)[0])


@pytest.mark.parametrize("complex_", [False, True])
def test_default_row_reduction_finds_the_rank_of_integer_matrices(
    complex_: bool,
) -> None:
    """Dependent rows and columns of small integer matrices give no extra pivot."""

    rng = np.random.default_rng(540 + complex_)
    for trial in range(1500):
        size = int(rng.choice([3, 4, 5, 8]) if trial % 50 else rng.choice([12, 20]))
        columns = size + int(rng.choice([0, 0, 1, 3]))
        bound = int(rng.choice([3, 9, 50]))
        matrix = rng.integers(-bound, bound + 1, (size, columns))
        kind = trial % 4
        if kind == 0:
            matrix[-1] = matrix[0] + matrix[1]
        elif kind == 1:
            matrix[-1] = 2 * matrix[0] - 3 * matrix[1] + matrix[size // 2]
            matrix[-2] = matrix[0] - matrix[1]
        elif kind == 2:
            matrix[:, -1] = matrix[:, 0] + matrix[:, 1]
        want, pivots = _integer_reduced(matrix)
        values = matrix.astype(complex if complex_ else float)
        reduced, found = linalg.row_reduce(values)
        assert list(found) == pivots, matrix.tolist()
        scale = max(1.0, float(np.abs(want).max()))
        assert np.abs(reduced - want).max() <= 1e-9 * scale, matrix.tolist()


@pytest.mark.parametrize("complex_", [False, True])
def test_default_row_reduction_finds_the_rank_of_general_matrices(
    complex_: bool,
) -> None:
    """Products of low rank, also with rows of very different sizes, keep their rank."""

    rng = np.random.default_rng(550 + complex_)

    def normal(shape: tuple[int, ...]) -> np.ndarray:
        values = rng.normal(size=shape)
        return values + 1j * rng.normal(size=shape) if complex_ else values

    for trial in range(160):
        rows, columns = int(rng.integers(1, 41)), int(rng.integers(1, 41))
        rank = int(rng.integers(0, min(rows, columns) + 1))
        matrix = normal((rows, rank)) @ normal((rank, columns))
        if trial % 2:
            # Row reduction does not depend on the size of each row
            matrix = matrix * 10.0 ** rng.integers(-6, 7, (rows, 1))
        if trial % 4 >= 2 and rows > 1:
            matrix[rows // 2 :] = matrix[: rows - rows // 2][::-1] * 3
            rank = int(
                np.linalg.matrix_rank(
                    matrix / np.abs(matrix).max(axis=1, keepdims=True).clip(1e-300)
                )
            )
        reduced, pivots = linalg.row_reduce(matrix)
        assert len(pivots) == rank, (trial, matrix.shape)
        if rank:
            # The nonzero rows of the reduced form span the rows of the matrix
            basis = reduced[:rank]
            rest = matrix - (matrix @ np.linalg.pinv(basis)) @ basis
            sizes = np.abs(matrix).max(axis=1, keepdims=True).clip(1e-300)
            assert np.abs(rest / sizes).max() < 1e-8, (trial, matrix.shape)


def test_entries_at_or_below_the_tolerance_are_zero_in_the_elimination() -> None:
    """Such an entry is not multiplied by the pivot row into a later column."""

    # The entry 1e-11 of the last row is below the tolerance. As a factor it
    # would be multiplied by the pivot row, whose last entry is 1e12, and
    # leave 10 in the last column
    matrix = np.array([[1.0, 1e-3, 1.0], [0.0, 1e-9, 1e3], [0.0, 1e-11, 0.0]])
    reduced, pivots = linalg.row_reduce(matrix, tol=1e-10)
    assert pivots == (0, 1) and not reduced[2].any()
    assert np.allclose(reduced[:2], [[1, 0, 1 - 1e9], [0, 1, 1e12]], rtol=1e-12)
    # Without a tolerance the entry is far above the rounding of its row
    reduced, pivots = linalg.row_reduce(matrix, tol=0)
    assert pivots == (0, 1, 2) and np.array_equal(reduced, np.eye(3))
    # The default tolerance of a row is relative to that row, and the pivot
    # has also to be above the rounding of the largest entry of the matrix
    high, low = np.ldexp(1.0, 300), np.ldexp(1.0, -300)
    rows = np.array(
        [[high, 2 * high, 3 * high], [low, 2 * low, 5 * low], [1.0, 2.0, 3.0]]
    )
    reduced, pivots = linalg.row_reduce(rows)
    assert pivots == (0,) and np.array_equal(reduced[0], [1.0, 2.0, 3.0])
    reduced, pivots = linalg.row_reduce(rows, tol=0)
    assert pivots == (0, 2) and np.array_equal(reduced[:2], [[1, 2, 0], [0, 0, 1]])
    reduced, pivots = linalg.row_reduce(rows[[1, 2]])
    assert pivots == (0,)
    reduced, pivots = linalg.row_reduce(
        np.array([[low, 2 * low, 5 * low], [low, 2 * low, 3 * low]])
    )
    assert pivots == (0, 2)


def test_row_reduction_of_values_that_are_not_finite_raises_nothing() -> None:
    """Infinities and NaNs in a matrix give a result, without an error."""

    for matrix in (
        np.array([[np.inf, 1.0], [1.0, np.nan]]),
        np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        np.array([[1.0, np.inf], [0.0, 1.0]]),
        np.array([[complex(np.inf, 1), 1], [1, 1]]),
    ):
        for tol in (None, 0, 1.0):
            with np.errstate(all="ignore"):
                reduced, pivots = linalg.row_reduce(matrix, tol=tol)
            assert reduced.shape == matrix.shape and len(pivots) <= 2


def test_row_reduction_handles_degenerate_matrices_and_tolerances() -> None:
    """Empty matrices, zero matrices and unusual tolerances raise nothing."""

    for shape in ((0, 3), (3, 0), (0, 0)):
        reduced, pivots = linalg.row_reduce(np.zeros(shape))
        assert reduced.shape == shape and pivots == ()
    for tol in (None, 0, -1.0, 1e-300, np.inf):
        reduced, pivots = linalg.row_reduce(np.zeros((2, 3)), tol=tol)
        assert pivots == () and not reduced.any()
    reduced, pivots = linalg.row_reduce(np.eye(3) * 5e-324, tol=-1.0)
    assert pivots == (0, 1, 2) and np.array_equal(reduced, np.eye(3))
    dependent = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    reduced, pivots = linalg.row_reduce(dependent)
    assert pivots == (0, 1)
    assert np.allclose(reduced, [[1, 0, 1], [0, 1, 1], [0, 0, 0]])
    for power in (-1060, 1015):
        again, same = linalg.row_reduce(np.ldexp(dependent, power))
        assert same == pivots and np.array_equal(again, reduced)


# Codes


def test_codes_from_coefficients_at_both_ends_of_the_range() -> None:
    """A generator with very large and very small coefficients gives a finite basis."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1))
    high, low = np.ldexp(1.0, 1020), np.ldexp(1.0, -1020)
    with np.errstate(all="ignore"):
        basis = codes.generator_matrix(algebra([[high, low], [high, low]]))
    assert np.array_equal(basis, [[1, 0, 1, 0], [0, 1, 0, 1]])
    assert codes.is_ideal(basis, algebra)
    for first, second in ((high, 1.0), (1.0, low), (low, low), (high, high)):
        with np.errstate(all="ignore"):
            basis = codes.generator_matrix(algebra([[first, second], [first, second]]))
        assert np.isfinite(basis).all() and basis.shape[1] == 4
        assert codes.is_ideal(basis, algebra)


def test_ideal_checking_takes_coefficients_of_the_algebra() -> None:
    """A complex matrix is not one of an algebra over R."""

    real = cdl.LoopAlgebra(cdl.CDLoop(1))
    complex_ = cdl.LoopAlgebra(cdl.CDLoop(1), deg=2)
    finite = cdl.LoopAlgebra(cdl.CDLoop(1), char=3)
    row = np.array([[1, 1j, -1, -1j]])
    with pytest.raises(
        TypeError, match="complex coefficients are not supported over R"
    ):
        codes.is_ideal(row, real)
    with pytest.raises(TypeError, match="complex coefficients"):
        real(row.reshape(2, 2))
    assert codes.is_ideal(row, complex_) is True
    ones = np.ones((1, 4))
    assert codes.is_ideal(ones, real) and codes.is_ideal(ones, complex_)
    assert codes.is_ideal(ones.astype(int), real)
    assert codes.is_ideal(ones.astype(int), finite)
    assert codes.is_ideal(np.array([[1, 0, 0, 0]]), real) is False
    with pytest.raises(TypeError):
        codes.is_ideal(ones, finite)
    with pytest.raises(TypeError):
        codes.is_ideal([["a", "b", "c", "d"]], real)
    with pytest.raises(ValueError, match="two index axes"):
        codes.is_ideal(np.ones(4), real)
    with pytest.raises(ValueError, match="one column for each loop element"):
        codes.is_ideal(np.ones((1, 3)), real)
    assert codes.is_ideal(np.zeros((0, 4)), real) is True


# Data from outside is validated before it is trusted


@pytest.mark.parametrize("char", [3, 7])
def test_signed_components_from_pyfastcd_are_read_as_coefficients(char: int) -> None:
    """Values that are not residues are reduced, as they are from an ndarray."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(1), char=char)
    stored = np.array([char + 1, 2 * char], dtype=np.uint32)
    signed = algebra.cd_algebra.array(np.array([1, 0], dtype=np.uint32))
    signed.coefficients[...] = stored
    ones = np.ones(2, dtype=np.uint32)
    array = algebra.split_array(ones, signed)
    assert np.array_equal(array._signed, [1, 0])
    assert bool((array == algebra.split_array(ones, stored)).all())
    element = array[()]
    assert element == 1 and element.is_idempotent()
    assert bool(array == 1) and bool(array.to_loop() == 1)
    assert algebra.from_components(ones, signed) == algebra.from_components(
        ones, [1, 0]
    )
    assert algebra.lift_signed(signed) == algebra.lift_signed([1, 0])
    batch = algebra.cd_algebra.array(np.zeros((3, 2), dtype=np.uint32))
    batch.coefficients[...] = char + 2
    lifted = algebra.lift_signed(batch)
    assert int(lifted.coefficients.max()) < char
    # The copy is the algebra's own
    signed.coefficients[...] = 0
    assert np.array_equal(array._signed, [1, 0])


@pytest.mark.parametrize(
    ("loop", "selectors"), [(cdl.CDLoop(1), {"char": 2}), (cdl.CDLoop(2), {"char": 5})]
)
def test_pickling_does_not_launder_invalid_storage(
    loop: Any, selectors: dict[str, Any]
) -> None:
    """An array whose storage was spoiled cannot be pickled, copied or rebuilt."""

    algebra = cdl.LoopAlgebra(loop, **selectors)
    array = algebra.array(np.zeros((2,) + algebra.coefficient_shape, dtype=np.uint32))
    assert (
        pickle.loads(pickle.dumps(array)).coefficients.tolist()
        == array.to_numpy().tolist()
    )
    array.coefficients[0, 0, 0] = algebra.char + 3
    for use in (
        pickle.dumps,
        copy.copy,
        copy.deepcopy,
        lambda x: x.copy(),
        lambda x: x.square(),
    ):
        with pytest.raises(ValueError, match="not residues"):
            use(array)
    spoiled = array.to_numpy()
    with pytest.raises(ValueError, match="not residues"):
        _rebuild_array(algebra, spoiled)
    with pytest.raises(ValueError, match="dtype and shape"):
        _rebuild_array(algebra, spoiled.astype(np.int64) % algebra.char)
    with pytest.raises(ValueError, match="dtype and shape"):
        _rebuild_array(algebra, spoiled[..., :1] % algebra.char)
    with pytest.raises(ValueError, match="dtype and shape"):
        _rebuild_array(algebra, spoiled.tolist())
    with pytest.raises(ValueError, match="not residues"):
        _rebuild(algebra, "loop", spoiled[0])
    rebuilt = _rebuild_array(algebra, spoiled % algebra.char)
    assert isinstance(rebuilt, cdl.ElementArray) and not rebuilt._shown[0]
    assert _rebuild(algebra, "loop", spoiled[0] % algebra.char) == algebra(
        spoiled[0] % algebra.char
    )


def test_pickled_split_arrays_and_elements_are_checked() -> None:
    """Both parts of a split form are checked, and have to match."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=5)
    array = algebra.random(3, batch_shape=(2,)).split()
    assert bool((pickle.loads(pickle.dumps(array)) == array).all())
    element = array[0]
    assert pickle.loads(pickle.dumps(element)) == element
    spectral, signed = array._spectral.copy(), array._signed.copy()
    array.signed.coefficients[0, 0] = 9
    with pytest.raises(ValueError, match="not residues"):
        pickle.dumps(array)
    with pytest.raises(ValueError, match="not residues"):
        copy.deepcopy(array)
    bad = signed.copy()
    bad[0, 0] = 9
    with pytest.raises(ValueError, match="not residues"):
        _rebuild_split(algebra, spectral, bad)
    with pytest.raises(ValueError, match="not residues"):
        _rebuild_split(algebra, bad, signed)
    with pytest.raises(ValueError, match="dtype and shape"):
        _rebuild_split(algebra, spectral, signed[:1])
    with pytest.raises(ValueError, match="dtype and shape"):
        _rebuild_split(algebra, spectral.astype(float), signed)
    with pytest.raises(ValueError, match="not residues"):
        _rebuild(algebra, "split", (spectral[0], bad[0]))
    with pytest.raises(ValueError, match="dtype and shape"):
        _rebuild(algebra, "split", (spectral[0], signed[0][:2]))
    assert _rebuild(algebra, "split", (spectral[0], signed[0])) == element
    real = cdl.LoopAlgebra(cdl.CDLoop(1))
    values = real.random(4, batch_shape=(3,))
    assert bool((pickle.loads(pickle.dumps(values)) == values).all())
    assert bool((pickle.loads(pickle.dumps(values.split())) == values.split()).all())
    with pytest.raises(ValueError, match="dtype and shape"):
        _rebuild_array(real, values.to_numpy().astype(np.float32))


def test_unpickled_split_forms_have_the_shape_of_one_element() -> None:
    """Split-form data of another shape, or for characteristic 2, is rejected."""

    algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=5)
    element = algebra.random(6)
    spectral, signed = (np.array(part) for part in element._data("split"))
    assert _rebuild(algebra, "split", (spectral, signed)) == element
    for shape in ((3, 4), (1, 4)):
        batch = np.zeros(shape, dtype=np.uint32)
        with pytest.raises(ValueError, match="dtype and shape"):
            _rebuild(algebra, "split", (batch, batch))
        with pytest.raises(ValueError, match="dtype and shape"):
            _rebuild(algebra, "split", (spectral, batch))
    binary = cdl.LoopAlgebra(cdl.CDLoop(2), char=2)
    record = np.zeros(4, dtype=np.uint32)
    with pytest.raises(ValueError, match="no split form"):
        _rebuild(binary, "split", (record, record))
    with pytest.raises(ValueError, match="no split form"):
        _rebuild_split(binary, record[None], record[None])
    assert _rebuild(binary, "loop", np.zeros((2, 4), dtype=np.uint32)) == 0


def test_any_integer_is_an_exponent() -> None:
    """An exponent beyond every float raises nothing over R and C."""

    loop = cdl.CDLoop(2)
    for selectors in ({}, {"deg": 2}):
        algebra = cdl.LoopAlgebra(loop, **selectors)
        unit = algebra.basis(loop.basis(1))
        # e_1 has order 4 in the loop, and its cube is the loop element -e_1
        assert unit ** (10**400 + 3) == algebra.basis(-loop.basis(1))
        assert algebra.one() ** 10**400 == 1
        with np.errstate(all="ignore"):
            grown = (algebra.one() + unit) ** 10**400
        assert not np.isfinite(grown.to_numpy()).all()
