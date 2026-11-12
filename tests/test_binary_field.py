"""Tests of the finite fields of characteristic 2 and their scalars."""

from __future__ import annotations

import copy
import itertools
import pickle

import numpy as np
import pytest

import pycdloop as cdl
from pycdloop import _binary

try:  # Support both unittest discovery and package-style test collection.
    from . import reference
except ImportError:  # pragma: no cover - depends on the collection mechanism
    import reference

DEGREES = range(1, 13)


def _oracle(field: cdl.BinaryField) -> reference.FiniteField:
    """Return the schoolbook arithmetic of the same presentation."""

    return reference.FiniteField(2, field.modulus)


def _coordinates(field: cdl.BinaryField, value: int) -> tuple[int, ...]:
    """Return the polynomial coordinates of a bit mask."""

    return tuple((value >> i) & 1 for i in range(field.deg))


def _scalar(field: cdl.BinaryField, value: int) -> cdl.BinaryFieldElement:
    """Return the scalar with the given bit mask."""

    if field.deg == 1:
        return field(value)
    return field(list(_coordinates(field, value)))


@pytest.mark.parametrize("degree", DEGREES)
def test_default_fields_have_the_documented_attributes(degree: int) -> None:
    """Expose the characteristic, degree, order, modulus and storage layout."""

    field = cdl.BinaryField(deg=degree)
    assert field is cdl.BinaryField(deg=degree)
    assert (field.char, field.deg, field.order) == (2, degree, 2**degree)
    assert field.dtype == np.dtype(np.uint32)
    assert field.coefficient_shape == (() if degree == 1 else (degree,))
    assert len(field.modulus) == degree + 1 and field.modulus[-1] == 1
    assert repr(field) == f"BinaryField(deg={degree})"
    assert pickle.loads(pickle.dumps(field)) is field
    assert copy.deepcopy(field) is field


def test_the_prime_field_has_the_modulus_of_pyfastcd_prime_fields() -> None:
    """Use ``(0, 1)`` at degree one, whatever linear polynomial is given."""

    prime = cdl.BinaryField()
    assert prime.modulus == (0, 1)
    assert cdl.BinaryField.from_modulus(modulus=(1, 1)) is prime
    assert cdl.BinaryField.from_modulus(modulus=(0, 1)) is prime
    assert cdl.BinaryField.from_modulus(modulus=[4, 3]) is prime


@pytest.mark.parametrize("degree", range(2, 13))
def test_default_moduli_are_irreducible_primitive_and_compatible(degree: int) -> None:
    """Check the three computable properties of a Conway polynomial."""

    modulus = cdl.BinaryField(deg=degree).modulus
    mask = _binary._mask(modulus)
    assert _binary._is_irreducible(mask, degree)
    # The class of t generates the multiplicative group
    size = 2**degree - 1
    value, order = 2, 1
    while value != 1:
        value = _binary._multiply(value, 2, mask, degree)
        order += 1
    assert order == size
    # For d dividing the degree, the matching power of t is a root of the
    # default polynomial of degree d
    for divisor in range(2, degree):
        if degree % divisor:
            continue
        root = 1
        for _ in range(size // (2**divisor - 1)):
            root = _binary._multiply(root, 2, mask, degree)
        total, power = 0, 1
        for coefficient in cdl.BinaryField(deg=divisor).modulus:
            if coefficient:
                total ^= power
            power = _binary._multiply(power, root, mask, degree)
        assert total == 0, (degree, divisor)


def test_irreducibility_test_counts_the_irreducible_polynomials() -> None:
    """Find the known number of monic irreducible polynomials of each degree."""

    for degree, count in ((2, 1), (3, 2), (4, 3), (5, 6), (6, 9), (7, 18), (8, 30)):
        found = sum(
            _binary._is_irreducible((1 << degree) | low, degree)
            for low in range(1 << degree)
        )
        assert found == count, degree


def test_from_modulus_validates_and_caches_presentations() -> None:
    """Accept irreducible polynomials and reject everything else."""

    other = cdl.BinaryField.from_modulus(modulus=(1, 0, 1, 1))
    assert other.deg == 3 and other is not cdl.BinaryField(deg=3)
    assert other is cdl.BinaryField.from_modulus(modulus=[3, 2, 1, 5])
    assert repr(other) == "BinaryField.from_modulus(modulus=(1, 0, 1, 1))"
    assert pickle.loads(pickle.dumps(other)) is other
    assert cdl.BinaryField.from_modulus(modulus=(1, 1, 1)) is cdl.BinaryField(deg=2)
    for reducible in ((1, 0, 1), (1, 0, 1, 0, 1), (0, 1, 1), (1, 1, 1, 1)):
        with pytest.raises(ValueError):
            cdl.BinaryField.from_modulus(modulus=reducible)
    with pytest.raises(ValueError):
        cdl.BinaryField.from_modulus(modulus=(1, 1, 2))
    with pytest.raises(ValueError):
        cdl.BinaryField.from_modulus(modulus=(1,))
    with pytest.raises(ValueError):
        cdl.BinaryField.from_modulus(modulus=(1,) + (0,) * 12 + (1,))
    for wrong in ("111", {1, 0}, 7, (1, 0.5, 1), (1, True, 1)):
        with pytest.raises(TypeError):
            cdl.BinaryField.from_modulus(modulus=wrong)


def test_invalid_degrees_are_rejected() -> None:
    """Raise ``ValueError`` outside 1..12 and ``TypeError`` for non-integers."""

    for degree in (0, 13, -1):
        with pytest.raises(ValueError):
            cdl.BinaryField(deg=degree)
    for degree in (True, 2.0, "2", None):
        with pytest.raises(TypeError):
            cdl.BinaryField(deg=degree)


def test_fields_are_immutable_and_final() -> None:
    """Reject attribute changes and subclasses."""

    field = cdl.BinaryField(deg=2)
    with pytest.raises(AttributeError):
        field.deg = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del field.modulus
    with pytest.raises(TypeError):
        type("Derived", (cdl.BinaryField,), {})


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_scalar_arithmetic_agrees_with_schoolbook_arithmetic(degree: int) -> None:
    """Check every sum, product, quotient and power in the small fields."""

    field = cdl.BinaryField(deg=degree)
    ring = _oracle(field)
    for a, b in itertools.product(range(field.order), repeat=2):
        x, y = _scalar(field, a), _scalar(field, b)
        ca, cb = _coordinates(field, a), _coordinates(field, b)
        assert tuple(np.atleast_1d((x + y).coefficients)) == ring.add(ca, cb)
        assert tuple(np.atleast_1d((x * y).coefficients)) == ring.mul(ca, cb)
        assert x - y == x + y and -x == x and +x == x
        if b:
            assert (x / y) * y == x
    for a in range(1, field.order):
        x = _scalar(field, a)
        assert x * x.reciprocal() == 1 and x**-1 == x.reciprocal()
        assert x ** (field.order - 1) == 1 and x**0 == 1 and x**3 == x * x * x
        assert x ** (10**30) == x ** (10**30 % (field.order - 1))


@pytest.mark.parametrize("degree", [5, 8, 11, 12])
def test_field_tables_agree_with_schoolbook_arithmetic(degree: int) -> None:
    """Check the table-driven array arithmetic on random residues."""

    field = cdl.BinaryField(deg=degree)
    ring = _oracle(field)
    rng = np.random.default_rng(degree)
    a = rng.integers(0, field.order, size=200)
    b = rng.integers(0, field.order, size=200)
    products = field._multiply_masks(a, b)
    for x, y, z in zip(a.tolist(), b.tolist(), products.tolist(), strict=True):
        expected = ring.mul(_coordinates(field, x), _coordinates(field, y))
        assert _coordinates(field, z) == expected
    nonzero = a[a != 0]
    assert (field._multiply_masks(nonzero, field._reciprocal_masks(nonzero)) == 1).all()
    assert (field._power_masks(a, 0) == 1).all()
    assert (field._power_masks(a, 2) == field._multiply_masks(a, a)).all()
    assert (field._unpack(field._pack(field._unpack(a))) == field._unpack(a)).all()


def test_scalars_are_created_from_integers_coordinates_and_scalars() -> None:
    """Embed integers in the prime subfield and read polynomial coordinates."""

    field = cdl.BinaryField(deg=3)
    assert field(0) == field.zero() == 0 and field(1) == field.one() == 1
    assert field(7) == 1 and field(-2) == 0 and field() == 0
    x = field([1, 0, 1])
    assert field(x) is x and field.scalar(x) is x
    assert field([3, 2, 5]) == x and field(np.array([1, 0, 1])) == x
    assert x.algebra is field
    assert x.coefficients.tolist() == [1, 0, 1]
    assert not x.coefficients.flags.writeable
    assert cdl.BinaryField()(1).coefficients.shape == ()
    with pytest.raises(ValueError):
        field([1, 0])
    with pytest.raises(ValueError):
        field(cdl.BinaryField(deg=2)([1, 1]))
    for wrong in (True, 1.0, "101", {1, 0}, [1, 0.5, 1]):
        with pytest.raises(TypeError):
            field(wrong)


def test_scalar_equality_and_hashing_follow_the_rules_of_pyfastcd() -> None:
    """Compare with unreduced integers and hash prime-subfield values like them."""

    field = cdl.BinaryField(deg=2)
    t = field([0, 1])
    assert field(1) == 1 and field(1) != 3 and field(0) == 0 and field(0) != 2
    assert t != 2 and t != 1 and t != "t" and t != None  # noqa: E711
    assert hash(field(1)) == hash(1) and hash(field(0)) == hash(0)
    assert {field(1), 1} == {1} and len({t, field([0, 1])}) == 1
    assert t != cdl.BinaryField.from_modulus(modulus=(1, 1, 0, 1))([0, 1, 0])
    assert field(1) != True and bool(t) and not bool(field(0))  # noqa: E712
    assert repr(t) == "BinaryFieldElement([0, 1], field=BinaryField(deg=2))"
    assert (
        repr(cdl.BinaryField()(1)) == "BinaryFieldElement(1, field=BinaryField(deg=1))"
    )


def test_scalar_arithmetic_with_integers_and_errors() -> None:
    """Reduce integer operands modulo 2 and raise on invalid operations."""

    field = cdl.BinaryField(deg=2)
    t = field([0, 1])
    assert t + 1 == field([1, 1]) and 1 + t == t + 1 and t - 3 == t + 1
    assert t * 2 == 0 and 3 * t == t and t / 1 == t and 1 / t == t.reciprocal()
    assert t + 2 == t
    with pytest.raises(ZeroDivisionError):
        field(0).reciprocal()
    with pytest.raises(ZeroDivisionError):
        t / 0
    with pytest.raises(ZeroDivisionError):
        field(0) ** -1
    with pytest.raises(ValueError):
        t + cdl.BinaryField(deg=3)(1)
    for operand in (1.5, "1", None, True):
        with pytest.raises(TypeError):
            t + operand
        with pytest.raises(TypeError):
            t * operand
    with pytest.raises(TypeError):
        t**True
    with pytest.raises(TypeError):
        t**1.5


def test_scalars_are_immutable_and_cannot_be_constructed_directly() -> None:
    """Reject mutation and direct construction, and pickle to equal scalars."""

    field = cdl.BinaryField(deg=4)
    x = field([1, 1, 0, 1])
    with pytest.raises(AttributeError):
        x._value = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del x._field
    with pytest.raises(TypeError):
        cdl.BinaryFieldElement(field, 3)
    restored = pickle.loads(pickle.dumps(x))
    assert restored == x and restored.algebra is field
