"""Static checks of the public annotations, read by ``mypy`` only.

Run ``python -m mypy tests/typing_checks.py``. The module is not collected by
the test runner; it only has to type-check. Results whose type depends on the
operands, such as operator results, are annotated in the package as ``Any``
and are not asserted here.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pyfastcd as fcd

import pycdloop as cdl
from pycdloop import codes, linalg, testing

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


def check_loops() -> None:
    """Check the annotations of loops and loop elements."""

    loop = cdl.CDLoop(3)
    assert_type(loop, cdl.CDLoop)
    product = cdl.CentralProduct(1, 2)
    assert_type(product, cdl.CentralProduct)
    assert_type(loop.levels, tuple[int, ...])
    assert_type(loop.order, int)
    assert_type(len(loop), int)
    x = loop.basis(1)
    assert_type(x, cdl.LoopElement)
    assert_type(loop(2, -1), cdl.LoopElement)
    assert_type(loop.one(), cdl.LoopElement)
    assert_type(loop.minus_one(), cdl.LoopElement)
    assert_type(loop.random(0), cdl.LoopElement)
    assert_type(x * x, cdl.LoopElement)
    assert_type(-x, cdl.LoopElement)
    assert_type(x**3, cdl.LoopElement)
    assert_type(x.inverse(), cdl.LoopElement)
    assert_type(x.commutator(x), cdl.LoopElement)
    assert_type(x.associator(x, x), cdl.LoopElement)
    assert_type(x.order(), int)
    assert_type(x.index, int)
    assert_type(x.sign, int)
    assert_type(x.position, int)
    assert_type(x.indices, tuple[int, ...])
    assert_type(loop.embed(cdl.CDLoop(1).basis(1)), cdl.LoopElement)
    assert_type(product.factor_embedding(0, cdl.CDLoop(1).basis(1)), cdl.LoopElement)
    assert_type(product.flat_index(1, 2), int)
    assert_type(product.factor_indices(5), tuple[int, ...])
    assert_type(loop.twist_table(), np.ndarray)
    assert_type(loop.cayley_table(), np.ndarray)
    assert_type(loop.left_translation(x), np.ndarray)
    assert_type(loop.character_automorphism(1), np.ndarray)
    assert_type(loop.center(), frozenset[cdl.LoopElement])
    assert_type(loop.nucleus("left"), frozenset[cdl.LoopElement])
    assert_type(loop.generate([x]), frozenset[cdl.LoopElement])
    assert_type(loop.cosets([loop.one()]), frozenset[frozenset[cdl.LoopElement]])
    assert_type(loop.is_subloop([loop.one()]), bool)
    assert_type(loop.is_moufang(), bool)
    assert_type(loop.is_automorphism(np.arange(16)), bool)
    assert_type(loop.to_gap(), str)
    sign: Any = loop.twist(1, 2)
    del sign


def check_algebra() -> None:
    """Check the annotations of loop algebras and their constructors."""

    loop = cdl.CDLoop(2)
    algebra = cdl.LoopAlgebra(loop)
    assert_type(algebra, cdl.LoopAlgebra)
    finite = cdl.LoopAlgebra(loop, field=fcd.FiniteField(char=7))
    assert_type(finite, cdl.LoopAlgebra)
    assert_type(algebra.dimension, int)
    assert_type(algebra.coefficient_shape, tuple[int, ...])
    assert_type(algebra.char, int)
    assert_type(algebra.deg, int)
    assert_type(algebra.default_algorithm, str)
    assert_type(algebra.signed_algorithms, tuple[str, ...])
    assert_type(algebra.default_signed_algorithm, str)
    assert_type(algebra(), cdl.Element)
    assert_type(algebra(3), cdl.Element)
    assert_type(algebra.zero(), cdl.Element)
    assert_type(algebra.one(), cdl.Element)
    assert_type(algebra.scalar(2.5), cdl.Element)
    assert_type(algebra.basis(loop.basis(1)), cdl.Element)
    assert_type(algebra.from_terms({loop.basis(1): 2.0}), cdl.Element)
    assert_type(algebra.from_bytes(bytes(64)), cdl.Element)
    assert_type(algebra.array(np.zeros((3, 2, 4))), cdl.ElementArray)
    assert_type(algebra.asarray(np.zeros((3, 2, 4)), copy=None), cdl.ElementArray)
    assert_type(algebra.split_array(np.zeros(4), np.zeros(4)), cdl.SplitArray)
    assert_type(algebra.central_idempotents(), tuple[cdl.Element, cdl.Element])
    assert_type(algebra.primitive_idempotent(0), cdl.Element)
    assert_type(algebra.primitive_idempotents(), cdl.ElementArray)
    assert_type(algebra.center_basis(), cdl.ElementArray)
    assert_type(algebra.left_matrix(algebra.one()), np.ndarray)
    assert_type(algebra.right_matrix(algebra.one(), basis="split"), np.ndarray)
    assert_type(algebra.structure_constants(), np.ndarray)
    assert_type(algebra.subalgebra_generated_by(algebra.one()), cdl.GeneratedSubalgebra)
    description = algebra.subalgebra_generated_by(algebra.one())
    assert_type(description.dimension, int)
    assert_type(description.minimal_polynomial, np.ndarray)
    product: Any = algebra.multiply(algebra.one(), algebra.one(), algorithm="table")
    del product


def check_binary_fields() -> None:
    """Check the annotations of the fields of characteristic 2 and their algebras."""

    field = cdl.BinaryField(deg=3)
    assert_type(field, cdl.BinaryField)
    assert_type(cdl.BinaryField.from_modulus(modulus=(1, 0, 1, 1)), cdl.BinaryField)
    assert_type(field.char, int)
    assert_type(field.deg, int)
    assert_type(field.order, int)
    assert_type(field.modulus, tuple[int, ...])
    assert_type(field.coefficient_shape, tuple[int, ...])
    scalar = field([1, 0, 1])
    assert_type(scalar, cdl.BinaryFieldElement)
    assert_type(field.scalar(1), cdl.BinaryFieldElement)
    assert_type(field.zero(), cdl.BinaryFieldElement)
    assert_type(field.one(), cdl.BinaryFieldElement)
    assert_type(scalar.algebra, cdl.BinaryField)
    assert_type(scalar.coefficients, np.ndarray)
    assert_type(-scalar, cdl.BinaryFieldElement)
    assert_type(scalar**5, cdl.BinaryFieldElement)
    assert_type(scalar.reciprocal(), cdl.BinaryFieldElement)
    assert_type(hash(scalar), int)
    algebra = cdl.LoopAlgebra(cdl.CentralProduct(1, 2), field=field)
    assert_type(algebra, cdl.LoopAlgebra)
    assert_type(cdl.LoopAlgebra(cdl.CDLoop(2), char=2, deg=3), cdl.LoopAlgebra)
    assert_type(algebra.algorithms, tuple[str, ...])
    assert_type(algebra.default_batch_algorithm, str)
    assert_type(algebra.scalar(scalar), cdl.Element)
    assert_type(algebra.center_basis(), cdl.ElementArray)
    assert_type(algebra.subalgebra_generated_by(algebra.one()), cdl.GeneratedSubalgebra)
    assert_type(linalg.rank(np.eye(2, dtype=np.uint32), field=field), int)
    assert_type(codes.generator_matrix(algebra.one()), np.ndarray)
    product: Any = algebra.multiply(algebra.one(), algebra.one(), algorithm="lift")
    del product


def check_values() -> None:
    """Check the annotations of elements and both kinds of array."""

    loop = cdl.CDLoop(2)
    algebra = cdl.LoopAlgebra(loop)
    x = algebra.one()
    assert_type(x.coefficients, np.ndarray)
    assert_type(x.shape, tuple[int, ...])
    assert_type(x.copy(), cdl.Element)
    assert_type(+x, cdl.Element)
    assert_type(x.to_numpy(copy=False), np.ndarray)
    assert_type(x.to_bytes(), bytes)
    assert_type(x.terms(), dict[cdl.LoopElement, Any])
    assert_type(x.support(), frozenset[cdl.LoopElement])
    assert_type(x.minimal_polynomial(), np.ndarray)
    assert_type(bool(x), bool)
    assert_type(hash(x), int)
    array = algebra.array(np.zeros((3, 2, 4)))
    assert_type(array.batch_shape, tuple[int, ...])
    assert_type(array.shape, tuple[int, ...])
    assert_type(array.coefficients, np.ndarray)
    assert_type(array.copy(), cdl.ElementArray)
    assert_type(+array, cdl.ElementArray)
    assert_type(array.to_numpy(), np.ndarray)
    assert_type(len(array), int)
    split = array.split()
    assert_type(split, cdl.SplitArray)
    assert_type(split.batch_shape, tuple[int, ...])
    assert_type(split.copy(), cdl.SplitArray)
    assert_type(split.to_loop(), cdl.ElementArray)
    total: Any = x + x * 2 - array[0]
    del total


def check_functions() -> None:
    """Check the annotations of the module-level functions and submodules."""

    assert_type(cdl.__version__, str)
    assert_type(cdl.wht([1.0, 0.0, 0.0, 0.0]), np.ndarray)
    assert_type(
        cdl.wht([1, 2, 0, 0], axis=-1, field=fcd.FiniteField(char=7)), np.ndarray
    )
    assert_type(cdl.dyadic_convolve([1.0, 0.0], [0.0, 1.0]), np.ndarray)
    matrix = np.eye(3)
    assert_type(linalg.rank(matrix), int)
    assert_type(linalg.row_reduce(matrix), tuple[np.ndarray, tuple[int, ...]])
    assert_type(linalg.nullspace(matrix), np.ndarray)
    algebra = cdl.LoopAlgebra(cdl.CDLoop(2), char=3)
    assert_type(codes.translate_matrix(algebra.one()), np.ndarray)
    assert_type(codes.generator_matrix(algebra.one(), side="right"), np.ndarray)
    assert_type(codes.to_gap(np.eye(2, dtype=np.int64), field=algebra.field), str)
    assert_type(testing.random_elements(algebra, 4, 0), cdl.ElementArray)
    assert_type(testing.is_flexible(algebra), bool)
    assert_type(testing.is_associative(algebra, samples=8, rng=1), bool)
