"""Quasilinear Cayley--Dickson multiplication with coefficients in an algebra.

The routines here multiply in ``A_n (x) B``, where ``A_n`` is the
Cayley--Dickson algebra of dimension ``N = 2^n`` and ``B`` is any module with
a bilinear product, not necessarily commutative or associative. An element is
an array of ``N`` rows, each an element of ``B``, and the product is

    U V = sum over p, r of sigma_n(p, r) e_(p ^ r) (x) (U_p V_r).

Evaluating this sum directly takes ``N^2`` products in ``B``. The schedule
below takes ``(9n - 15) 2^(n-1) + 10`` of them for ``n >= 1``, by running the
quasilinear recursion for ``A_n`` on rows instead of scalars. Every step of
that recursion is linear in each operand, so it remains valid for
coefficients in ``B`` provided each product keeps the factor that comes from
the first operand on the left, which the code does throughout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_Multiply = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _repeat(row: np.ndarray, count: int) -> np.ndarray:
    """Broadcast one row of each call to *count* rows."""

    return np.broadcast_to(row[:, None], (row.shape[0], count) + row.shape[1:])


def _parts(plane: np.ndarray, half: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an imaginary element X = (u, gamma + s) into u, gamma, and s."""

    return plane[:, 1:half], plane[:, half], plane[:, half + 1 :]


def _nonrecursive(
    field: Any, products: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble one evaluation of the nonrecursive map from its 6m products.

    The products arrive in the order (gamma t, s delta, s v, u t, u delta,
    gamma v), each block holding m rows, and the result is

        f = gamma t - s delta,   lambda = <s, v> - <u, t>,   g = u delta - gamma v.
    """

    blocks = [products[:, i * count : (i + 1) * count] for i in range(6)]
    f = field.subtract(blocks[0], blocks[1])
    scalar = field.subtract(field.sum(blocks[2], 1), field.sum(blocks[3], 1))
    g = field.subtract(blocks[4], blocks[5])
    return f, scalar, g


def _operands(
    u: np.ndarray,
    gamma: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    delta: np.ndarray,
    t: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return the left and right factors of the 6m products of one evaluation."""

    count = u.shape[1]
    left = [_repeat(gamma, count), s, s, u, u, _repeat(gamma, count)]
    right = [t, _repeat(delta, count), v, t, _repeat(delta, count), v]
    return left, right


def _cross(
    field: Any,
    multiply: _Multiply,
    level: int,
    x_real: np.ndarray,
    x_imag: np.ndarray,
    y_real: np.ndarray,
    y_imag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the alternating product of imaginary elements with complex coefficients.

    Each operand is given by two planes of shape ``(calls, K, ...)``, the
    parts ``a`` and ``b`` of coefficients ``a + iota b`` with ``iota^2 = -1``
    a central scalar. Row 0 of every plane is ignored, and row 0 of both
    result planes is zero. ``level`` is at least 2.
    """

    # Write X = (u, gamma + s) and Y = (v, delta + t), where u, s, v and t
    # are imaginary of half the dimension and gamma and delta are single
    # rows. Then
    #
    #     X x Y = (D + f, lambda + g - H),
    #     D = u x v - s x t,      H = u x t + s x v,
    #
    # with (f, lambda, g) the nonrecursive map. Bilinearity gives
    # C(u +- iota s, v +- iota t) = D +- iota H, so two half-size calls Z_+
    # and Z_- determine D = (Z_+ + Z_-)/2 and iota H = (Z_+ - Z_-)/2.
    size = 1 << level
    half = size // 2
    count = half - 1
    calls = x_real.shape[0]
    ur, gr, sr = _parts(x_real, half)
    ui, gi, si = _parts(x_imag, half)
    vr, dr, tr = _parts(y_real, half)
    vi, di, ti = _parts(y_imag, half)

    # The nonrecursive map with complex coefficients is evaluated as a whole
    # by the three-product formula: on the real parts, on the imaginary
    # parts, and on their sums. All 18m products go to B in one batch.
    sums_x = [field.add(a, b) for a, b in ((ur, ui), (gr, gi), (sr, si))]
    sums_y = [field.add(a, b) for a, b in ((vr, vi), (dr, di), (tr, ti))]
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for x_parts, y_parts in (
        ((ur, gr, sr), (vr, dr, tr)),
        ((ui, gi, si), (vi, di, ti)),
        (sums_x, sums_y),
    ):
        new_left, new_right = _operands(*x_parts, *y_parts)
        left += new_left
        right += new_right
    products = multiply(np.concatenate(left, axis=1), np.concatenate(right, axis=1))
    width = 6 * count
    first = _nonrecursive(field, products[:, :width], count)
    second = _nonrecursive(field, products[:, width : 2 * width], count)
    mixed = _nonrecursive(field, products[:, 2 * width :], count)
    real = [field.subtract(a, b) for a, b in zip(first, second, strict=True)]
    imag = [
        field.subtract(field.subtract(w, a), b)
        for a, b, w in zip(first, second, mixed, strict=True)
    ]

    out_real = np.zeros(x_real.shape, dtype=x_real.dtype)
    out_imag = np.zeros(x_real.shape, dtype=x_real.dtype)
    if level >= 3:
        # Children u +- iota s and v +- iota t, with the + calls first.
        # Multiplying by iota maps a + iota b to -b + iota a.
        shape = (2 * calls, half) + x_real.shape[2:]
        planes = [np.zeros(shape, dtype=x_real.dtype) for _ in range(4)]
        for plane, (a, b_plus, b_minus) in zip(
            planes,
            (
                (ur, field.negate(si), si),
                (ui, sr, field.negate(sr)),
                (vr, field.negate(ti), ti),
                (vi, tr, field.negate(tr)),
            ),
            strict=True,
        ):
            plane[:calls, 1:] = field.add(a, b_plus)
            plane[calls:, 1:] = field.add(a, b_minus)
        z_real, z_imag = _cross(field, multiply, level - 1, *planes)
        plus_real, minus_real = z_real[:calls, 1:], z_real[calls:, 1:]
        plus_imag, minus_imag = z_imag[:calls, 1:], z_imag[calls:, 1:]
        mean_real = field.halve(field.add(plus_real, minus_real))
        mean_imag = field.halve(field.add(plus_imag, minus_imag))
        gap_real = field.halve(field.subtract(plus_real, minus_real))
        gap_imag = field.halve(field.subtract(plus_imag, minus_imag))
        out_real[:, 1:half] = field.add(mean_real, real[0])
        out_imag[:, 1:half] = field.add(mean_imag, imag[0])
        # iota (Z_+ - Z_-)/2 = -H, added to g in the upper half
        out_real[:, half + 1 :] = field.subtract(real[2], gap_imag)
        out_imag[:, half + 1 :] = field.add(imag[2], gap_real)
    else:
        out_real[:, 1:half] = real[0]
        out_imag[:, 1:half] = imag[0]
        out_real[:, half + 1 :] = real[2]
        out_imag[:, half + 1 :] = imag[2]
    out_real[:, half] = real[1]
    out_imag[:, half] = imag[1]
    return out_real, out_imag


def schedule_multiply(
    field: Any, multiply: _Multiply, level: int, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Return the product of two elements of ``A_level (x) B``.

    Parameters
    ----------
    field
        Linear arithmetic of the coefficients: ``add``, ``subtract``,
        ``negate``, ``halve``, and ``sum`` on arrays.
    multiply
        The product of ``B`` on arrays of shape ``(calls, count, ...)``,
        applied row by row with the first argument on the left.
    level : int
        The doubling level ``n`` of the outer algebra.
    u, v : numpy.ndarray
        Operands of shape ``(calls, 2**level, ...)``, one element of ``B``
        in each row.

    Returns
    -------
    numpy.ndarray
        The product, with the shape of the operands.
    """

    if level == 0:
        return multiply(u, v)
    # For x = alpha + X and y = beta + Y with imaginary X and Y,
    #
    #     x y = (alpha beta - <X, Y>) + alpha Y + X beta + X x Y,
    #
    # and the alternating product X x Y comes from one half-size call with
    # complex coefficients, because the operands here are real: the call
    # C(u + iota s, v + iota t) = D + iota H returns D and H as its two parts.
    size = 1 << level
    half = size // 2
    count = half - 1
    alpha, beta = u[:, 0], v[:, 0]
    x, y = u[:, 1:], v[:, 1:]
    xu, gamma, xs = _parts(u, half)
    yv, delta, yt = _parts(v, half)
    map_left, map_right = _operands(xu, gamma, xs, yv, delta, yt)
    left = [alpha[:, None], x, _repeat(alpha, size - 1), x] + map_left
    right = [beta[:, None], y, y, _repeat(beta, size - 1)] + map_right
    products = multiply(np.concatenate(left, axis=1), np.concatenate(right, axis=1))
    cuts = np.cumsum([1, size - 1, size - 1, size - 1])
    scalar = field.subtract(products[:, 0], field.sum(products[:, 1 : cuts[1]], 1))
    linear = field.add(products[:, cuts[1] : cuts[2]], products[:, cuts[2] : cuts[3]])
    f, lam, g = _nonrecursive(field, products[:, cuts[3] :], count)
    if level >= 3:
        shape = (u.shape[0], half) + u.shape[2:]
        planes = [np.zeros(shape, dtype=u.dtype) for _ in range(4)]
        for plane, rows in zip(planes, (xu, xs, yv, yt), strict=True):
            plane[:, 1:] = rows
        z_real, z_imag = _cross(field, multiply, level - 1, *planes)
        f = field.add(z_real[:, 1:], f)
        g = field.subtract(g, z_imag[:, 1:])
    out = np.empty(u.shape, dtype=u.dtype)
    out[:, 0] = scalar
    out[:, 1:half] = field.add(linear[:, :count], f)
    out[:, half] = field.add(linear[:, count], lam)
    out[:, half + 1 :] = field.add(linear[:, count + 1 :], g)
    return out


def schedule_product_count(level: int) -> int:
    """Return the number of products in ``B`` used by :func:`schedule_multiply`."""

    if level == 0:
        return 1
    return (9 * level - 15) * (1 << (level - 1)) + 10
