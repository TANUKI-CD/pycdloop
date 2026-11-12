"""Cayley--Dickson loops, their central products, and their elements."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from functools import cache
from typing import Any, final

import numpy as np

from ._validation import (
    DIFFERENT_LOOP_MESSAGE,
    _basis_index,
    _choice,
    _exponent,
    _index,
    _level,
    _sign,
)

# Largest number of index tuples that an identity test checks exhaustively,
# and the number of random tuples it draws beyond that
_EXHAUSTIVE_LIMIT = 1 << 22
_SAMPLE_COUNT = 1 << 18

# Largest quotient order for which full tables are built
_TABLE_LIMIT = 1 << 12

_SIDES = ("left", "middle", "right")


def _cd_twist(level: int, p: Any, q: Any) -> np.ndarray:
    """Return the signs ``sigma_n(p, q)`` of the Cayley--Dickson basis products.

    *p* and *q* are integer arrays of indices below ``2**level`` that
    broadcast against each other. The result holds +1 and -1.
    """

    # The doubling formula (a, b)(c, d) = (ac - conj(d) b, da + b conj(c))
    # gives, for N = 2M and i, j < M,
    #
    #     sigma_N(i,     j)     =  sigma_M(i, j)
    #     sigma_N(i,     M + j) =  sigma_M(j, i)
    #     sigma_N(M + i, j)     =  eps(j) sigma_M(i, j)
    #     sigma_N(M + i, M + j) = -eps(j) sigma_M(j, i),
    #
    # with eps(0) = 1 and eps(j) = -1 otherwise. The loop below peels off the
    # top bit of both indices at each level: it multiplies the sign by the
    # factor of the matching case and swaps the two remaining indices
    # whenever the second index had its top bit set.
    a, b = np.broadcast_arrays(np.asarray(p, dtype=np.int64), np.asarray(q, np.int64))
    a, b = a.copy(), b.copy()
    negative = np.zeros(a.shape, dtype=bool)
    for bit in range(level - 1, -1, -1):
        mask = (1 << bit) - 1
        high_a = (a >> bit) & 1 == 1
        high_b = (b >> bit) & 1 == 1
        a &= mask
        b &= mask
        negative ^= high_a & (b != 0)
        negative ^= high_a & high_b
        a, b = np.where(high_b, b, a), np.where(high_b, a, b)
    return np.where(negative, -1, 1).astype(np.int8)


@cache
def _factor_members(level: int, kind: str) -> np.ndarray:
    """Return the indices of one Cayley--Dickson loop in its commutant or a nucleus.

    The result is a read-only Boolean array over the ``2**level`` indices.
    Index 0 always belongs, because +1 and -1 are central.
    """

    size = 1 << level
    indices = np.arange(size, dtype=np.int64)

    def fails(p: Any, q: np.ndarray, r: np.ndarray | None) -> np.ndarray:
        """Mark the indices in *p* that fail against the given partners."""

        if kind == "commutant":
            return _cd_twist(level, p, q) != _cd_twist(level, q, p)
        assert r is not None
        triple = {"left": (p, q, r), "middle": (q, p, r), "right": (q, r, p)}[kind]
        return _cd_associator(level, *triple) == -1

    # A first pass against a few partners removes almost every index, and
    # only the survivors are then checked against all partners
    probe = indices[: min(size, 16)]
    member = np.ones(size, dtype=bool)
    if kind == "commutant":
        member &= ~fails(indices[:, None], probe[None, :], None).any(axis=1)
    else:
        failed = fails(
            indices[:, None, None], probe[None, :, None], probe[None, None, :]
        )
        member &= ~failed.any(axis=(1, 2))
    for p in np.flatnonzero(member)[1:]:
        if kind == "commutant":
            member[p] = not fails(np.int64(p), indices, None).any()
            continue
        for start in range(0, size, 1024):
            block = indices[start : start + 1024]
            if fails(np.int64(p), block[:, None], indices[None, :]).any():
                member[p] = False
                break
    member.setflags(write=False)
    return member


def _cd_associator(level: int, p: Any, q: Any, r: Any) -> np.ndarray:
    """Return the sign relating ``(e_p e_q) e_r`` to ``e_p (e_q e_r)``."""

    p, q, r = (np.asarray(value, dtype=np.int64) for value in (p, q, r))
    left = _cd_twist(level, p, q) * _cd_twist(level, p ^ q, r)
    right = _cd_twist(level, q, r) * _cd_twist(level, p, q ^ r)
    return left * right


class _SignedLoop:
    """A loop of signed basis elements over an elementary abelian 2-group.

    The elements are ``+e_t`` and ``-e_t`` for ``t`` in ``range(quotient_order)``
    and multiply by ``e_s e_t = sigma(s, t) e_(s ^ t)``, with ``-1`` central
    of order two. Subclasses supply the doubling levels of the factors, and
    the twist ``sigma`` is the product of the Cayley--Dickson signs of the
    factors.
    """

    __slots__ = ("levels", "level", "quotient_order", "order", "_shifts", "__weakref__")

    levels: tuple[int, ...]
    level: int
    quotient_order: int
    order: int
    _shifts: tuple[int, ...]

    def _setup(self, levels: tuple[int, ...]) -> None:
        """Record the factor levels and the bit layout of the flat index."""

        total = sum(levels)
        _level(total, "the total level")
        shifts = tuple(total - sum(levels[: i + 1]) for i in range(len(levels)))
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "level", total)
        object.__setattr__(self, "quotient_order", 1 << total)
        object.__setattr__(self, "order", 2 << total)
        object.__setattr__(self, "_shifts", shifts)

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation of a cached loop."""

        raise AttributeError("loops are immutable")

    def __delattr__(self, name: str) -> None:
        """Reject deletion of immutable loop state."""

        raise AttributeError("loops are immutable")

    # Elements

    def __call__(self, index: Any, sign: Any = 1) -> LoopElement:
        """Return the element ``sign * e_index``."""

        return LoopElement(self, index, sign)

    def one(self) -> LoopElement:
        """Return the identity element."""

        return LoopElement(self, 0, 1)

    def minus_one(self) -> LoopElement:
        """Return the central element -1."""

        return LoopElement(self, 0, -1)

    def basis(self, *indices: Any) -> LoopElement:
        """Return the positive basis element with the given index.

        Pass the flat index, or one index for each factor of a central
        product.
        """

        if len(indices) == 1:
            return LoopElement(self, indices[0], 1)
        return LoopElement(self, self.flat_index(*indices), 1)

    def flat_index(self, *indices: Any) -> int:
        """Return the flat index of a tuple with one index for each factor."""

        if len(indices) != len(self.levels):
            raise TypeError("pass one index for each factor of the loop")
        flat = 0
        for level, shift, value in zip(self.levels, self._shifts, indices, strict=True):
            flat |= _basis_index(value, 1 << level) << shift
        return flat

    def factor_indices(self, index: Any) -> tuple[int, ...]:
        """Return the index of each factor in a flat index."""

        flat = _basis_index(index, self.quotient_order)
        return tuple(
            (flat >> shift) & ((1 << level) - 1)
            for level, shift in zip(self.levels, self._shifts, strict=True)
        )

    def random(self, rng: Any = None) -> LoopElement:
        """Return a uniformly random element.

        *rng* is a ``numpy.random.Generator`` or a seed for one.
        """

        generator = np.random.default_rng(rng)
        index = int(generator.integers(self.quotient_order))
        return LoopElement(self, index, 1 if generator.integers(2) else -1)

    def __len__(self) -> int:
        """Return the order of the loop."""

        return self.order

    def __iter__(self) -> Iterator[LoopElement]:
        """Iterate over the elements, all positive ones first."""

        for sign in (1, -1):
            for index in range(self.quotient_order):
                yield LoopElement(self, index, sign)

    def __contains__(self, value: object) -> bool:
        """Return whether *value* is an element of this loop."""

        return isinstance(value, LoopElement) and value.loop is self

    def _own(self, value: Any) -> LoopElement:
        """Check that *value* is an element of this loop."""

        if not isinstance(value, LoopElement):
            raise TypeError("operand must be a LoopElement")
        if value.loop is not self:
            raise ValueError(DIFFERENT_LOOP_MESSAGE)
        return value

    def _element(self, position: int) -> LoopElement:
        """Return the element at a position of the standard enumeration."""

        size = self.quotient_order
        return LoopElement(self, position % size, 1 if position < size else -1)

    def _positions(self, elements: Iterable[Any]) -> np.ndarray:
        """Return the sorted positions of a collection of elements."""

        positions = {self._own(element).position for element in elements}
        return np.array(sorted(positions), dtype=np.int64)

    # Sign functions

    def _indices(self, value: Any, name: str) -> np.ndarray:
        """Validate an integer or an integer array of basis indices."""

        array = np.asarray(value)
        if array.dtype.kind not in "iu" or array.dtype.kind == "b":
            if array.ndim == 0:
                array = np.asarray(_index(value, name))
            else:
                raise TypeError(f"{name} must hold integers")
        array = array.astype(np.int64)
        if array.size and (array.min() < 0 or array.max() >= self.quotient_order):
            raise IndexError(f"{name} is out of range")
        return array

    def _twist(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Return the twist of validated index arrays."""

        result: np.ndarray | None = None
        for level, shift in zip(self.levels, self._shifts, strict=True):
            mask = (1 << level) - 1
            sign = _cd_twist(level, (s >> shift) & mask, (t >> shift) & mask)
            result = sign if result is None else result * sign
        assert result is not None
        return result

    def _associator(self, s: np.ndarray, t: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Return the associator sign of validated index arrays."""

        left = self._twist(s, t) * self._twist(s ^ t, u)
        return left * self._twist(t, u) * self._twist(s, t ^ u)

    @staticmethod
    def _result(signs: np.ndarray) -> Any:
        """Return a sign array, or a Python integer for scalar arguments."""

        return int(signs) if signs.ndim == 0 else signs

    def twist(self, s: Any, t: Any) -> Any:
        """Return ``sigma(s, t)``, the sign in ``e_s e_t = sigma(s, t) e_(s ^ t)``.

        The arguments are flat indices or integer arrays of them, which
        broadcast. The result is an ``int`` for two integers and an ``int8``
        array otherwise.
        """

        return self._result(self._twist(self._indices(s, "s"), self._indices(t, "t")))

    def commutator_sign(self, s: Any, t: Any) -> Any:
        """Return the sign ``c`` with ``e_s e_t = c e_t e_s``."""

        first, second = self._indices(s, "s"), self._indices(t, "t")
        return self._result(self._twist(first, second) * self._twist(second, first))

    def associator_sign(self, s: Any, t: Any, u: Any) -> Any:
        """Return the sign ``a`` with ``(e_s e_t) e_u = a e_s (e_t e_u)``."""

        triple = (self._indices(s, "s"), self._indices(t, "t"), self._indices(u, "u"))
        return self._result(self._associator(*triple))

    def _check_table(self) -> None:
        """Refuse to build a full table for a very large loop."""

        if self.quotient_order > _TABLE_LIMIT:
            raise ValueError(
                "the loop is too large for a full table; "
                "use the sign functions on index arrays instead"
            )

    def twist_table(self) -> np.ndarray:
        """Return the ``int8`` matrix of signs ``sigma(s, t)`` over all indices."""

        self._check_table()
        indices = np.arange(self.quotient_order, dtype=np.int64)
        return self._twist(indices[:, None], indices[None, :])

    def _multiply_positions(self, a: Any, b: Any) -> np.ndarray:
        """Return the positions of the products of elements given by position."""

        size = self.quotient_order
        s, t = a % size, b % size
        negative = (a >= size) ^ (b >= size) ^ (self._twist(s, t) < 0)
        return (s ^ t) + size * negative

    def cayley_table(self) -> np.ndarray:
        """Return the multiplication table of the loop.

        Elements are numbered in iteration order: ``+e_t`` at position ``t``
        and ``-e_t`` at position ``quotient_order + t``. Entry ``[i, j]`` is
        the position of the product of elements ``i`` and ``j``.
        """

        self._check_table()
        positions = np.arange(self.order, dtype=np.int64)
        return self._multiply_positions(positions[:, None], positions[None, :])

    # Substructure

    def _members(self, kind: str) -> np.ndarray:
        """Return the Boolean array of flat indices in the commutant or a nucleus."""

        # The commutator and associator signs of a central product are the
        # products of those of its factors. An index therefore commutes or
        # associates with everything exactly when each of its factor indices
        # does so in its own factor: if one factor index fails against some
        # partners there, the same partners with index 0 in every other
        # factor make the product fail.
        member = np.ones(1, dtype=bool)
        for level in self.levels:
            member = np.logical_and.outer(member, _factor_members(level, kind)).ravel()
        return member

    def _signed_set(self, member: np.ndarray) -> frozenset[LoopElement]:
        """Return both signs of every basis element marked in *member*."""

        indices = np.flatnonzero(member)
        return frozenset(
            LoopElement(self, int(index), sign) for index in indices for sign in (1, -1)
        )

    def commutant(self) -> frozenset[LoopElement]:
        """Return the elements that commute with every element."""

        return self._signed_set(self._members("commutant"))

    def nucleus(self, side: str | None = None) -> frozenset[LoopElement]:
        """Return the elements that associate with every pair of elements.

        *side* is ``"left"``, ``"middle"`` or ``"right"`` for the nucleus in
        which the element takes that position of the associator, and
        ``None`` for the intersection of the three.
        """

        if side is not None:
            return self._signed_set(self._members(_choice(side, "side", _SIDES)))
        member = self._members("left") & self._members("middle")
        return self._signed_set(member & self._members("right"))

    def center(self) -> frozenset[LoopElement]:
        """Return the center: the nucleus intersected with the commutant."""

        member = self._members("commutant")
        for side in _SIDES:
            member = member & self._members(side)
        return self._signed_set(member)

    def generate(self, elements: Iterable[Any]) -> frozenset[LoopElement]:
        """Return the subloop generated by *elements*."""

        size = self.quotient_order
        present = np.zeros(self.order, dtype=bool)
        present[0] = True
        present[self._positions(elements)] = True
        while True:
            # Once -1 is present the subloop is every signed basis element
            # whose index lies in the span of the indices found so far
            if present[size]:
                span = self._span(np.flatnonzero(present[:size] | present[size:]))
                present[:] = False
                present[span] = True
                present[span + size] = True
                break
            positions = np.flatnonzero(present)
            products = self._multiply_positions(positions[:, None], positions[None, :])
            before = int(present.sum())
            present[products.ravel()] = True
            if int(present.sum()) == before:
                break
        return frozenset(self._element(int(p)) for p in np.flatnonzero(present))

    @staticmethod
    def _span(indices: np.ndarray) -> np.ndarray:
        """Return the closure of a set of indices under XOR."""

        span = np.zeros(1, dtype=np.int64)
        seen = {0}
        for index in indices.tolist():
            if index not in seen:
                span = np.concatenate([span, span ^ index])
                seen.update(span.tolist())
        return np.sort(span)

    def is_subloop(self, elements: Iterable[Any]) -> bool:
        """Return whether *elements* form a subloop."""

        positions = self._positions(elements)
        if positions.size == 0 or positions[0] != 0:
            return False
        products = self._multiply_positions(positions[:, None], positions[None, :])
        return bool(np.isin(products, positions).all())

    def is_normal(self, elements: Iterable[Any]) -> bool:
        """Return whether a subloop is normal.

        A subloop ``S`` is normal when ``xS = Sx``, ``x(yS) = (xy)S`` and
        ``(Sx)y = S(xy)`` for all elements ``x`` and ``y``. The test
        examines up to ``quotient_order**2 * len(S)`` triples of elements,
        with eight sign evaluations each, when ``S`` does not contain -1.
        """

        positions = self._positions(elements)
        if not self.is_subloop(self._element(int(p)) for p in positions):
            raise ValueError("the elements do not form a subloop")
        size = self.quotient_order
        if size in positions:
            # A subloop containing -1 is a union of pairs {e_t, -e_t}, so each
            # set in the definition is such a union over one coset of indices
            return True
        # Otherwise S has one element s(w) e_w for each index w of a subgroup
        # W. All sets in the definition have one element for each index of a
        # coset of W, so two of them agree exactly when their signs agree,
        # and the signs of x and y cancel from every condition.
        w = positions % size
        indices = np.arange(size, dtype=np.int64)
        x = indices[:, None]
        if (self._twist(x, w[None, :]) != self._twist(w[None, :], x)).any():
            return False
        step = max(1, (1 << 22) // (size * w.size))
        for start in range(0, size, step):
            x = indices[start : start + step, None, None]
            y = indices[None, :, None]
            v = w[None, None, :]
            # x(y s_v e_v) and (xy)(s_v e_v) share the index x ^ y ^ v
            left = self._twist(y, v) * self._twist(x, y ^ v)
            right = self._twist(x, y) * self._twist(x ^ y, v)
            if (left != right).any():
                return False
            # ((s_v e_v) x) y and (s_v e_v)(xy) likewise
            left = self._twist(v, x) * self._twist(v ^ x, y)
            right = self._twist(x, y) * self._twist(v, x ^ y)
            if (left != right).any():
                return False
        return True

    def cosets(
        self, elements: Iterable[Any], *, side: str = "left"
    ) -> frozenset[frozenset[LoopElement]]:
        """Return the distinct cosets ``xS`` (left) or ``Sx`` (right) of a subloop.

        The cosets of a normal subloop partition the loop. Those of a
        subloop that is not normal can overlap.
        """

        _choice(side, "side", ("left", "right"))
        positions = self._positions(elements)
        if not self.is_subloop(self._element(int(p)) for p in positions):
            raise ValueError("the elements do not form a subloop")
        everything = np.arange(self.order, dtype=np.int64)
        if side == "left":
            products = self._multiply_positions(everything[:, None], positions[None, :])
        else:
            products = self._multiply_positions(positions[None, :], everything[:, None])
        distinct = {tuple(sorted(row)) for row in products.tolist()}
        return frozenset(
            frozenset(self._element(p) for p in coset) for coset in distinct
        )

    # Maps

    def left_translation(self, x: Any) -> np.ndarray:
        """Return the permutation ``y -> x y`` of the elements, by position."""

        position = np.int64(self._own(x).position)
        return self._multiply_positions(position, np.arange(self.order, dtype=np.int64))

    def right_translation(self, x: Any) -> np.ndarray:
        """Return the permutation ``y -> y x`` of the elements, by position."""

        position = np.int64(self._own(x).position)
        return self._multiply_positions(np.arange(self.order, dtype=np.int64), position)

    def character_automorphism(self, a: Any) -> np.ndarray:
        """Return the automorphism ``e_t -> (-1)**popcount(a & t) e_t``, by position.

        Each index ``a`` gives one character of the quotient group, and the
        twist is unchanged by it, so the map is an automorphism.
        """

        mask = _basis_index(a, self.quotient_order, "a")
        size = self.quotient_order
        indices = np.arange(size, dtype=np.int64)
        parity = np.zeros(size, dtype=np.int64)
        for bit in range(self.level):
            if (mask >> bit) & 1:
                parity ^= (indices >> bit) & 1
        positive = indices + size * parity
        return np.concatenate([positive, (positive + size) % (2 * size)])

    def _permutation(self, f: Any) -> np.ndarray:
        """Return a map of the loop to itself as an array of positions."""

        if callable(f):
            images = [self._own(f(element)).position for element in self]
            return np.array(images, dtype=np.int64)
        permutation = np.asarray(f)
        if permutation.dtype.kind not in "iu" or permutation.shape != (self.order,):
            raise ValueError("a map must be given as one position for each element")
        permutation = permutation.astype(np.int64)
        if permutation.min() < 0 or permutation.max() >= self.order:
            raise ValueError("positions are out of range")
        return permutation

    def is_automorphism(self, f: Any) -> bool:
        """Return whether *f* is an automorphism of the loop.

        *f* is a callable from elements to elements, or an array giving the
        position of the image of each element in iteration order.
        """

        permutation = self._permutation(f)
        if np.unique(permutation).size != self.order:
            return False
        positions = np.arange(self.order, dtype=np.int64)
        for start in range(0, self.order, 256):
            rows = positions[start : start + 256, None]
            products = self._multiply_positions(rows, positions[None, :])
            images = self._multiply_positions(permutation[rows], permutation[None, :])
            if (permutation[products] != images).any():
                return False
        return True

    # Identity tests

    def _tuples(self, arity: int, samples: int | None) -> tuple[np.ndarray, ...]:
        """Return index arrays covering all tuples, or a random sample of them."""

        size = self.quotient_order
        if samples is None and size**arity <= _EXHAUSTIVE_LIMIT:
            grids = np.meshgrid(*[np.arange(size)] * arity, indexing="ij", sparse=True)
            return tuple(grid.astype(np.int64) for grid in grids)
        count = _SAMPLE_COUNT if samples is None else _index(samples, "samples")
        if count <= 0:
            raise ValueError("samples must be positive")
        generator = np.random.default_rng(0)
        return tuple(generator.integers(size, size=count) for _ in range(arity))

    def is_commutative(self) -> bool:
        """Return whether the loop is commutative."""

        return bool(self._members("commutant").all())

    def is_associative(self) -> bool:
        """Return whether the loop is associative, that is, a group."""

        return all(bool(self._members(side).all()) for side in _SIDES)

    def is_moufang(self, samples: int | None = None) -> bool:
        """Return whether ``(xy)(zx) = (x(yz))x`` holds for all elements.

        Without *samples* the identity is checked on every triple of indices
        when there are at most ``2**22`` of them, and on ``2**18``
        pseudorandom triples otherwise. With *samples* it is checked on that
        many pseudorandom triples, whatever the size of the loop. The
        triples come from a fixed seed, so the result is reproducible. A
        sampled test that returns ``True`` has only failed to find a
        counterexample.
        """

        x, y, z = self._tuples(3, samples)
        left = self._twist(x, y) * self._twist(z, x) * self._twist(x ^ y, z ^ x)
        right = self._twist(y, z) * self._twist(x, y ^ z) * self._twist(x ^ y ^ z, x)
        return bool((left == right).all())

    def is_diassociative(self, samples: int | None = None) -> bool:
        """Return whether every two elements generate a group.

        Two basis elements generate the signed basis elements with indices
        in ``{0, p, q, p ^ q}``, so the test checks that all triples of such
        indices associate. Sampling follows :meth:`is_moufang`.
        """

        p, q = self._tuples(2, samples)
        members = (p, q, p ^ q)
        for triple in itertools.product(members, repeat=3):
            if (self._associator(*triple) == -1).any():
                return False
        return True

    # Export

    def to_gap(self) -> str:
        """Return a GAP expression that builds the loop with the LOOPS package.

        The expression is ``LoopByCayleyTable`` applied to the multiplication
        table with elements numbered from 1 in iteration order.
        """

        rows = (self.cayley_table() + 1).tolist()
        body = ",\n".join("  " + str(row).replace(" ", "") for row in rows)
        return f"LoopByCayleyTable([\n{body}\n]);"


@final
class CDLoop(_SignedLoop):
    """The Cayley--Dickson loop ``Q_n`` of order ``2**(n + 1)``.

    ``Q_n`` consists of the signed basis elements ``+e_p`` and ``-e_p``,
    ``0 <= p < 2**n``, of the standard Cayley--Dickson algebra of dimension
    ``2**n``, under the multiplication of that algebra. Basis elements
    follow the convention of pyfastcd.

    Parameters
    ----------
    n : int
        The doubling level, a nonnegative integer.

    Attributes
    ----------
    levels : tuple[int, ...]
        ``(n,)``.
    level : int
        ``n``.
    quotient_order : int
        ``2**n``, the number of basis elements and the order of the quotient
        by ``{+1, -1}``.
    order : int
        ``2**(n + 1)``.

    Notes
    -----
    Loops are immutable and cached, so ``CDLoop(3) is CDLoop(3)``. ``Q_1``
    is the cyclic group of order 4, ``Q_2`` the quaternion group and
    ``Q_3`` the octonion loop.

    Examples
    --------
    >>> import pycdloop as cdl
    >>> Q = cdl.CDLoop(3)
    >>> Q.basis(1) * Q.basis(2)
    e_3
    >>> Q.basis(2) * Q.basis(1)
    -e_3
    """

    __slots__ = ()
    _cache: dict[int, CDLoop] = {}

    def __new__(cls, n: Any) -> CDLoop:
        """Validate the level and retrieve the cached loop."""

        level = _level(n, "n")
        if level not in cls._cache:
            loop = object.__new__(cls)
            loop._setup((level,))
            cls._cache[level] = loop
        return cls._cache[level]

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle a loop by its level."""

        return (CDLoop, (self.level,))

    def __repr__(self) -> str:
        """Return the constructor expression of the loop."""

        return f"CDLoop({self.level})"

    def embed(self, x: Any) -> LoopElement:
        """Return the image of an element of a smaller Cayley--Dickson loop.

        ``Q_k`` is the subloop of ``Q_n`` on the indices below ``2**k``.
        """

        if not isinstance(x, LoopElement) or not isinstance(x.loop, CDLoop):
            raise TypeError("operand must be an element of a Cayley--Dickson loop")
        if x.loop.level > self.level:
            raise ValueError("the element belongs to a larger loop")
        return LoopElement(self, x.index, x.sign)


@final
class CentralProduct(_SignedLoop):
    """The central product ``Q_n1 * Q_n2 * ...`` of Cayley--Dickson loops.

    The central product is the direct product of the factors with their
    central elements -1 identified. Its elements are the signed pure tensors
    ``+e_p1 (x) e_p2 (x) ...`` and ``-e_p1 (x) e_p2 (x) ...`` of basis
    elements, and they multiply factor by factor.

    Parameters
    ----------
    *levels : int
        The doubling levels of the factors: at least two, each positive.

    Attributes
    ----------
    levels : tuple[int, ...]
        The doubling levels of the factors.
    level : int
        The sum of the levels.
    quotient_order : int
        ``2**level``, the number of basis elements.
    order : int
        ``2**(level + 1)``.

    Notes
    -----
    A basis element has one index for each factor. Its flat index
    concatenates the binary digits of those indices, first factor first, so
    for two factors the flat index of ``(p, q)`` is ``p * 2**m + q``.

    Examples
    --------
    >>> import pycdloop as cdl
    >>> L = cdl.CentralProduct(1, 1)
    >>> x = L.basis(1, 1)
    >>> x * x
    e_(0,0)
    """

    __slots__ = ()
    _cache: dict[tuple[int, ...], CentralProduct] = {}

    def __new__(cls, *levels: Any) -> CentralProduct:
        """Validate the levels and retrieve the cached loop."""

        if len(levels) < 2:
            raise TypeError("a central product needs at least two factors")
        key = tuple(_level(value, "a factor level", minimum=1) for value in levels)
        if key not in cls._cache:
            loop = object.__new__(cls)
            loop._setup(key)
            cls._cache[key] = loop
        return cls._cache[key]

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle a loop by its factor levels."""

        return (CentralProduct, self.levels)

    def __repr__(self) -> str:
        """Return the constructor expression of the loop."""

        return f"CentralProduct({', '.join(str(level) for level in self.levels)})"

    def factor_embedding(self, i: Any, x: Any) -> LoopElement:
        """Return the image of an element of factor *i* in the central product.

        The element ``s e_p`` of ``Q_ni`` maps to ``s`` times the basis
        element with index ``p`` in factor *i* and index 0 elsewhere.
        """

        position = _basis_index(i, len(self.levels), "the factor number")
        if not isinstance(x, LoopElement) or x.loop is not CDLoop(
            self.levels[position]
        ):
            raise ValueError("the element does not belong to that factor")
        return LoopElement(self, x.index << self._shifts[position], x.sign)


@final
class LoopElement:
    """One signed basis element of a loop.

    Parameters
    ----------
    loop : CDLoop or CentralProduct
        The loop that contains the element.
    index : int
        The flat index of the basis element.
    sign : int, optional
        ``+1`` (the default) or ``-1``.

    Attributes
    ----------
    loop
        The parent loop.
    index : int
        The flat basis index.
    sign : int
        ``+1`` or ``-1``.

    Notes
    -----
    Elements are immutable and hashable, and they combine only with
    elements of the same loop. ``x * y`` is the loop product, ``-x`` is
    ``x`` times the central element -1, and ``x ** k`` is an integer power.

    Examples
    --------
    >>> import pycdloop as cdl
    >>> Q = cdl.CDLoop(2)
    >>> i, j = Q.basis(1), Q.basis(2)
    >>> i * j, j * i, i ** 2
    (e_3, -e_3, -e_0)
    >>> i.order()
    4
    """

    __slots__ = ("loop", "index", "sign")

    loop: Any
    index: int
    sign: int

    def __init__(self, loop: Any, index: Any, sign: Any = 1) -> None:
        """Validate the parent, index and sign."""

        if not isinstance(loop, _SignedLoop):
            raise TypeError("loop must be a CDLoop or a CentralProduct")
        object.__setattr__(self, "loop", loop)
        object.__setattr__(self, "index", _basis_index(index, loop.quotient_order))
        object.__setattr__(self, "sign", _sign(sign))

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation."""

        raise AttributeError("LoopElement objects are immutable")

    def __delattr__(self, name: str) -> None:
        """Reject deletion of immutable state."""

        raise AttributeError("LoopElement objects are immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle an element by its parent, index and sign."""

        return (LoopElement, (self.loop, self.index, self.sign))

    @property
    def indices(self) -> tuple[int, ...]:
        """The basis index in each factor of the loop."""

        result: tuple[int, ...] = self.loop.factor_indices(self.index)
        return result

    @property
    def position(self) -> int:
        """The position of the element in the iteration order of the loop."""

        size: int = self.loop.quotient_order
        return self.index if self.sign > 0 else self.index + size

    def __repr__(self) -> str:
        """Return the element as a signed basis element."""

        if len(self.loop.levels) == 1:
            name = f"e_{self.index}"
        else:
            name = f"e_({','.join(str(index) for index in self.indices)})"
        return name if self.sign > 0 else "-" + name

    def __hash__(self) -> int:
        """Hash the parent, index and sign."""

        return hash((self.loop, self.index, self.sign))

    def __eq__(self, other: object) -> bool:
        """Return whether *other* is the same element of the same loop."""

        if not isinstance(other, LoopElement):
            return NotImplemented
        return (
            self.loop is other.loop
            and self.index == other.index
            and self.sign == other.sign
        )

    def _twist(self, other: LoopElement) -> int:
        """Return the twist of this element's index with another's."""

        return int(self.loop._twist(np.int64(self.index), np.int64(other.index)))

    def __mul__(self, other: object) -> LoopElement:
        """Return the loop product."""

        if not isinstance(other, LoopElement):
            return NotImplemented
        if other.loop is not self.loop:
            raise ValueError(DIFFERENT_LOOP_MESSAGE)
        sign = self.sign * other.sign * self._twist(other)
        return LoopElement(self.loop, self.index ^ other.index, sign)

    def __neg__(self) -> LoopElement:
        """Return the element times the central element -1."""

        return LoopElement(self.loop, self.index, -self.sign)

    def __pos__(self) -> LoopElement:
        """Return the element itself."""

        return self

    def _square_sign(self) -> int:
        """Return the sign of the square, which is +1 or -1 times the identity."""

        return self._twist(self)

    def inverse(self) -> LoopElement:
        """Return the two-sided inverse."""

        return LoopElement(self.loop, self.index, self.sign * self._square_sign())

    def order(self) -> int:
        """Return the order of the element: 1, 2 or 4."""

        if self.index == 0:
            return 1 if self.sign > 0 else 2
        return 2 if self._square_sign() > 0 else 4

    def __pow__(self, exponent: object) -> LoopElement:
        """Return an integer power; negative exponents use the inverse."""

        # The square of an element is central, so its powers do not depend
        # on bracketing, and the fourth power is the identity
        remainder = _exponent(exponent) % 4
        square = self._square_sign()
        if remainder == 0:
            return LoopElement(self.loop, 0, 1)
        if remainder == 1:
            return self
        if remainder == 2:
            return LoopElement(self.loop, 0, square)
        return LoopElement(self.loop, self.index, self.sign * square)

    def commutator(self, other: Any) -> LoopElement:
        """Return the central element ``c`` with ``x y = (y x) c``."""

        partner = self.loop._own(other)
        return LoopElement(self.loop, 0, self._twist(partner) * partner._twist(self))

    def associator(self, y: Any, z: Any) -> LoopElement:
        """Return the central element ``a`` with ``(x y) z = (x (y z)) a``."""

        second, third = self.loop._own(y), self.loop._own(z)
        indices = (np.int64(self.index), np.int64(second.index), np.int64(third.index))
        return LoopElement(self.loop, 0, int(self.loop._associator(*indices)))


def _loop_from(value: Any) -> Any:
    """Return a loop given as a loop object."""

    if not isinstance(value, _SignedLoop):
        raise TypeError("loop must be a CDLoop or a CentralProduct")
    return value


def _is_cd(loop: Any) -> bool:
    """Return whether a loop is a single Cayley--Dickson loop."""

    return isinstance(loop, CDLoop)
