"""Private structural computations on elements of a loop algebra.

In split coordinates a loop algebra is ``R^D (+) S`` with coordinatewise
operations on ``R^D``, so questions about an element reduce to one question
about its spectral coordinates and one about its signed component in ``S``.
The routines for characteristic 2, where there are no split coordinates,
work on loop-basis coefficients.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _kernel, linalg
from ._validation import _choice

Split = tuple[np.ndarray, np.ndarray]

_SIDES = ("left", "right")

# The largest level at which a Cayley--Dickson algebra is a composition
# algebra, so that its zero divisors are the elements of quadratic norm zero
_COMPOSITION_LEVEL = 3


def _identity(algebra: Any) -> np.ndarray:
    """Return the basis of the signed component as a batch of ``D`` elements."""

    field = algebra._field
    size = algebra._size
    basis = field.zeros((size, size))
    diagonal = np.arange(size)
    basis[(diagonal, diagonal) + (slice(None),) * field.tail] = field.scalar(1)
    return basis


def signed_matrix(algebra: Any, d: np.ndarray, side: str) -> np.ndarray:
    """Return the matrix of ``y -> d y`` (left) or ``y -> y d`` (right) on ``S``."""

    basis = _identity(algebra)
    if side == "left":
        images = algebra._signed.multiply(d[None], basis)
    else:
        images = algebra._signed.multiply(basis, d[None])
    # Row q of the batch is the image of basis element q, which is column q
    return np.ascontiguousarray(np.swapaxes(images, 0, 1))


def _field_object(algebra: Any) -> Any:
    """Return the ``field`` argument that :mod:`pycdloop.linalg` expects."""

    return algebra._field.field


def _signed_zero_divisor(algebra: Any, d: np.ndarray, side: str | None) -> np.ndarray:
    """Mark the records whose signed component is a zero divisor or zero."""

    signed = algebra._signed
    field = algebra._field
    if signed.cd is not None and signed.level <= _COMPOSITION_LEVEL:
        # Through the octonions d conj(d) = Q(d) and the algebra is
        # alternative, so d divides zero exactly when Q(d) = 0
        marks: np.ndarray = ~field.nonzero(signed.quadratic_norm(d))
        return marks
    size = algebra._size
    lead = field.index_shape(d)[:-1]
    flat = d.reshape((-1, size) + field.shape)
    sides = _SIDES if side is None else (side,)
    marks = np.zeros(flat.shape[0], dtype=bool)
    for number, record in enumerate(flat):
        for name in sides:
            matrix = signed_matrix(algebra, record, name)
            if linalg.rank(matrix, field=_field_object(algebra)) < size:
                marks[number] = True
                break
    return marks.reshape(lead)


def zero_divisor_marks(algebra: Any, split: Split, side: str | None) -> np.ndarray:
    """Mark the records that are zero divisors on the given side."""

    if side is not None:
        _choice(side, "side", _SIDES)
    field = algebra._field
    spectral, d = split
    marks = ~field.nonzero(spectral).all(axis=-1)
    return marks | _signed_zero_divisor(algebra, d, side)


def annihilator(algebra: Any, split: Split, side: str) -> np.ndarray:
    """Return loop-basis coefficients of a basis of an annihilator of one element.

    The left annihilator is ``{y : y x = 0}`` and the right annihilator is
    ``{y : x y = 0}``.
    """

    field = algebra._field
    size = algebra._size
    spectral, d = split
    if field.index_shape(d) != (size,):
        raise ValueError("an annihilator is computed for one element")
    # y x = 0 means that the spectrum of y vanishes wherever that of x does
    # not, and that the signed component of y is killed by right
    # multiplication by d; the right annihilator uses left multiplication
    zeros = np.flatnonzero(~field.nonzero(spectral))
    matrix = signed_matrix(algebra, d, "right" if side == "left" else "left")
    kernel = linalg.nullspace(matrix, field=_field_object(algebra))
    count = zeros.size + kernel.shape[0]
    basis_spectral = field.zeros((count, size))
    basis_signed = field.zeros((count, size))
    rows = np.arange(zeros.size)
    basis_spectral[(rows, zeros) + (slice(None),) * field.tail] = field.scalar(1)
    basis_signed[zeros.size :] = kernel
    return _kernel.from_split(algebra, (basis_spectral, basis_signed))


def _times_linear(field: Any, polynomial: np.ndarray, root: np.ndarray) -> np.ndarray:
    """Multiply a polynomial, constant term first, by ``t - root``."""

    degree = polynomial.shape[0]
    shifted = field.zeros((degree + 1,))
    shifted[1:] = polynomial
    scaled = field.zeros((degree + 1,))
    scaled[:-1] = field.multiply(polynomial, root)
    result: np.ndarray = field.subtract(shifted, scaled)
    return result


def signed_polynomial(algebra: Any, d: np.ndarray) -> np.ndarray:
    """Return the minimal polynomial of an element of a Cayley--Dickson algebra."""

    field = algebra._field
    first = (0,) + (slice(None),) * field.tail
    if not field.nonzero(d[1:]).any():
        polynomial = field.zeros((2,))
        polynomial[0] = field.negate(d[first])
        polynomial[1] = field.scalar(1)
        return polynomial
    # A nonscalar d satisfies d^2 - 2 d_0 d + Q(d) = 0 and no linear relation
    polynomial = field.zeros((3,))
    polynomial[0] = algebra._signed.quadratic_norm(d)
    polynomial[1] = field.negate(field.multiply_int(d[first], 2))
    polynomial[2] = field.scalar(1)
    return polynomial


def _evaluate(field: Any, polynomial: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial at an array of field scalars by Horner's rule."""

    value = np.broadcast_to(polynomial[-1], points.shape).astype(field.dtype)
    for coefficient in polynomial[-2::-1]:
        value = field.add(field.multiply(value, points), coefficient)
    return value


def distinct_values(field: Any, spectral: np.ndarray) -> np.ndarray:
    """Return the distinct spectral values in a fixed order."""

    result: np.ndarray = np.unique(spectral, axis=0)
    return result


def minimal_polynomial(algebra: Any, split: Split) -> np.ndarray:
    """Return the minimal polynomial of one element, constant term first."""

    field = algebra._field
    spectral, d = split
    if field.index_shape(d) != (algebra._size,):
        raise ValueError("a minimal polynomial is computed for one element")
    # The element is the pair (spectrum, d) in R^D (+) A_n, so its minimal
    # polynomial is the least common multiple of prod (t - lambda) over the
    # distinct spectral values lambda and the minimal polynomial q of d.
    # That is q times the factors t - lambda with q(lambda) != 0.
    polynomial = signed_polynomial(algebra, d)
    values = distinct_values(field, spectral)
    keep = field.nonzero(_evaluate(field, polynomial, values))
    for value in values[keep]:
        polynomial = _times_linear(field, polynomial, value)
    return polynomial


def left_matrix(algebra: Any, split: Split, side: str, basis: str) -> np.ndarray:
    """Return the matrix of multiplication by one element.

    With ``basis="loop"`` the matrix acts on flattened loop-basis
    coefficients; with ``basis="split"`` it acts on the spectral coordinates
    followed by the signed coordinates, and is block diagonal.
    """

    field = algebra._field
    size = algebra._size
    spectral, d = split
    if basis == "split":
        matrix = field.zeros((2 * size, 2 * size))
        diagonal = np.arange(size)
        matrix[(diagonal, diagonal) + (slice(None),) * field.tail] = spectral
        matrix[size:, size:] = signed_matrix(algebra, d, side)
        return matrix
    # Multiply by every basis element at once: record j of the batch is the
    # image of basis element j, and becomes column j of the matrix
    identity = field.zeros((2 * size, 2 * size))
    diagonal = np.arange(2 * size)
    identity[(diagonal, diagonal) + (slice(None),) * field.tail] = field.scalar(1)
    record = (2 * size, 2, size) + field.shape
    basis_split = _kernel.to_split(algebra, identity.reshape(record))
    own = (spectral[None], d[None])
    operands = (own, basis_split) if side == "left" else (basis_split, own)
    images = _kernel.from_split(algebra, _kernel.multiply(algebra, "split", *operands))
    columns = images.reshape((2 * size, 2 * size) + field.shape)
    return np.ascontiguousarray(np.swapaxes(columns, 0, 1))


# Characteristic 2. The algebra has no split form there, so the routines
# below work with loop-basis coefficients.


def loop_matrix(algebra: Any, coefficients: np.ndarray, side: str) -> np.ndarray:
    """Return the matrix of multiplication by one element, in the loop basis.

    The matrix of ``y -> x y`` (left) or ``y -> y x`` (right) is assembled
    from translations: the product of the basis elements at positions ``i``
    and ``j`` is one basis element, so every entry of the matrix is a
    coefficient of ``x``.
    """

    field = algebra._field
    loop = algebra.loop
    order = loop.order
    positions = np.arange(order, dtype=np.int64)
    flat = coefficients.reshape((order,) + field.shape)
    if side == "left":
        # x [j] = sum of x_i [i j]: column j has x_i at the row of i j
        rows = loop._multiply_positions(positions[:, None], positions[None, :])
    else:
        # [j] x = sum of x_i [j i]: column j has x_i at the row of j i
        rows = loop._multiply_positions(positions[None, :], positions[:, None])
    matrix = field.zeros((order, order))
    columns = np.broadcast_to(positions[None, :], rows.shape)
    matrix[(rows, columns) + (slice(None),) * field.tail] = field.expand(flat, 1)
    return matrix


def binary_zero_divisor_marks(
    algebra: Any, coefficients: np.ndarray, side: str | None
) -> np.ndarray:
    """Mark the records that are zero divisors, in characteristic 2.

    Multiplication by ``x`` on either side maps the ideal of the elements
    whose coefficients of ``[e_t]`` and ``[-e_t]`` agree to itself, and acts
    on that ideal and on the quotient by it as multiplication by the group
    component of ``x`` in the group algebra. That map is invertible exactly
    when the augmentation of ``x`` is nonzero, so the zero divisors on each
    side are the elements of augmentation zero.
    """

    if side is not None:
        _choice(side, "side", _SIDES)
    marks: np.ndarray = ~algebra._field.nonzero(
        _kernel.augmentation(algebra, coefficients)
    )
    return marks


def binary_annihilator(algebra: Any, coefficients: np.ndarray, side: str) -> np.ndarray:
    """Return a basis of an annihilator of one element, in characteristic 2."""

    field = algebra._field
    record = (2, algebra._size) + field.shape
    if coefficients.shape != record:
        raise ValueError("an annihilator is computed for one element")
    # y x = 0 exactly when y lies in the null space of right multiplication
    # by x, and x y = 0 when it lies in that of left multiplication
    matrix = loop_matrix(algebra, coefficients, "right" if side == "left" else "left")
    kernel = linalg.nullspace(matrix, field=field.field)
    return kernel.reshape((kernel.shape[0],) + record)


def binary_minimal_polynomial(algebra: Any, coefficients: np.ndarray) -> np.ndarray:
    """Return the minimal polynomial of one element, in characteristic 2."""

    field = algebra._field
    record = (2, algebra._size) + field.shape
    if coefficients.shape != record:
        raise ValueError("a minimal polynomial is computed for one element")
    # With alpha the augmentation, y = x - alpha satisfies y^4 = 0, so the
    # minimal polynomial is (t - alpha)^j for the least j with y^j = 0
    alpha = _kernel.augmentation(algebra, coefficients)
    shifted = _kernel.add_scalar(algebra, "loop", coefficients, alpha)
    power = shifted
    degree = 1
    while field.nonzero(power).any():
        power = _kernel.multiply(algebra, "loop", power, shifted)
        degree += 1
    polynomial = field.zeros((1,))
    polynomial[0] = field.scalar(1)
    for _ in range(degree):
        polynomial = _times_linear(field, polynomial, alpha)
    return polynomial


def central_marks(algebra: Any, coefficients: np.ndarray) -> np.ndarray:
    """Mark the records that are central, from loop-basis coefficients.

    An element is central exactly when the coefficients of ``[e_t]`` and
    ``[-e_t]`` agree for every index ``t`` whose basis element is not
    central in the loop: conjugating by a loop element, or reassociating
    with two of them, either fixes a basis element or exchanges it with its
    negative, and it exchanges ``[e_t]`` with ``[-e_t]`` for some choice
    exactly when ``e_t`` is not central.
    """

    field = algebra._field
    plus, minus = _kernel._rows(algebra, coefficients)
    differ = ~field.equal(plus, minus)
    marks: np.ndarray = ~(differ & ~algebra._central).any(axis=-1)
    return marks


def binary_center_basis(algebra: Any) -> np.ndarray:
    """Return loop-basis coefficients of a basis of the center, in characteristic 2."""

    field = algebra._field
    size = algebra._size
    central = np.flatnonzero(algebra._central)
    others = np.flatnonzero(~algebra._central)
    count = 2 * central.size + others.size
    basis = field.zeros((count, 2, size))
    one = field.scalar(1)
    tail = (slice(None),) * field.tail
    rows = np.arange(central.size)
    # Central loop elements are central in the algebra, each on its own
    basis[(rows, 0, central) + tail] = one
    basis[(central.size + rows, 1, central) + tail] = one
    # Every other basis element contributes the sum with its negative
    rows = 2 * central.size + np.arange(others.size)
    basis[(rows, 0, others) + tail] = one
    basis[(rows, 1, others) + tail] = one
    return basis
