# Quick start

pycdloop requires CPython 3.10 or newer, NumPy 1.26 or newer and pyfastcd
1.4.0 or newer, below 2. To install it from PyPI with `pip`, run:

```sh
pip install pycdloop
```

To build and install it from a source checkout instead, you will also need
a C11 compiler and the Python development headers. Run the following command
in the repository root:

```sh
python -m pip install .
```

## Loops and their elements

```python
import pycdloop as cdl

Q = cdl.CDLoop(3)                   # the octonion loop Q_3, of order 16
e1, e2, e4 = Q.basis(1), Q.basis(2), Q.basis(4)

e1 * e2                             # e_3
e2 * e1                             # -e_3
(e1 * e2) * e4                      # e_7
e1 * (e2 * e4)                      # -e_7
e1 ** 2                             # -e_0, that is, -1
e1.inverse()                        # -e_1
e1.order()                          # 4

e1.commutator(e2)                   # -e_0: e1 e2 = -(e2 e1)
e1.associator(e2, e4)               # -e_0: (e1 e2) e4 = -(e1 (e2 e4))
```

A `LoopElement` is a signed basis element: it has an `index` and a `sign`,
is immutable and hashable, and combines only with elements of its own loop.
`CDLoop(n)` is cached, so `CDLoop(3) is CDLoop(3)`.

The sign functions take integers or whole arrays of indices:

```python
import numpy as np

Q.twist(1, 2)                       # 1:  e_1 e_2 = +e_3
Q.commutator_sign(1, 2)             # -1
Q.associator_sign(1, 2, 4)          # -1
p = np.arange(8)
Q.twist(p[:, None], p[None, :])     # the 8 x 8 table of signs, as int8

len(Q.center())                     # 2: the center is {+1, -1}
Q.is_moufang()                      # True
cdl.CDLoop(4).is_moufang()          # False
H = Q.generate([e1, e2])            # the quaternion subgroup, a frozenset
len(H), Q.is_normal(H)              # (8, True)
```

## Central products

```python
L = cdl.CentralProduct(1, 1)        # Q_1 * Q_1, of order 8
x = L.basis(1, 1)                   # one index for each factor
x * x                               # e_(0,0): the square is +1
sorted(y.order() for y in L)        # [1, 2, 2, 2, 4, 4, 4, 4]

M = cdl.CentralProduct(2, 3)        # quaternion group * octonion loop
M.order                             # 64
M.basis(1, 4).indices               # (1, 4)
M.basis(1, 4).index                 # 12, the flat index 1 * 8 + 4
```

A central product has the same methods as a Cayley–Dickson loop, with
`factor_embedding` in place of `embed`; indices are flat, and `basis`,
`flat_index` and `factor_indices` convert between a flat index and one
index for each factor.

## Loop algebras and elements

```python
Q = cdl.CDLoop(2)                   # the quaternion group
A = cdl.LoopAlgebra(Q)              # its loop algebra over R, of dimension 8
i, j = Q.basis(1), Q.basis(2)

x = A.basis(i) + 2 * A.basis(-j)    # [e_1] + 2 [-e_2]
y = A.from_terms({Q.one(): 1.0, i * j: -3.0})
z = A([[1, 0, 0, 2], [0, 5, 0, 0]]) # row 0: [e_t], row 1: [-e_t]

product = x * y                     # ordered product in the loop algebra
shifted = x + 3                     # add 3 times the identity [e_0]
scaled = 0.5 * x
power = x ** 5                      # ** is exponentiation; ^ is not
square = x.square()
inverse = (x + 4).inverse()         # two-sided inverse
comm = x.commutator(y)              # x y - y x

x.coefficients                      # read-only (2, 4) array
x.coefficient(-j)                   # 2.0
x.terms()                           # {e_1: 1.0, -e_2: 2.0}
x.weight()                          # 2 nonzero coefficients
```

An `Element` is immutable and hashable. Note that `[-e_t]` is a basis
element of its own: `A.basis(-j)` and `-A.basis(j)` are different elements.
Scalars of the coefficient field combine with elements as in pyfastcd: in
`x + 3` the scalar stands for 3 times the identity, and in `0.5 * x` it
scales every coefficient.

## The decomposition

Over R, C and the finite fields of odd characteristic, every element has a
*group component* and a *signed component*, and the two maps are
homomorphisms:

```python
x.kappa()                           # (a_t + b_t): array([0., 1., 2., 0.])
x.epsilon()                         # (a_t - b_t): a pyfastcd Element of A_2
x.spectrum()                        # Walsh–Hadamard transform of kappa
x.augmentation()                    # 3.0, the sum of all coefficients

(x * y).epsilon() == x.epsilon() * y.epsilon()        # True
A.from_components(x.kappa(), x.epsilon()) == x        # True

e_plus, e_minus = A.central_idempotents()             # (1 ± [-1]) / 2
e_plus * x == A.lift_group(x.kappa())                 # True
e_minus * x == A.lift_signed(x.epsilon())             # True
```

Products are evaluated in these coordinates: the spectrum multiplies
coordinatewise and the signed component multiplies in the Cayley–Dickson
algebra, through pyfastcd. An element keeps the transformed form after its
first product, and a product is created in that form, so chains of
operations transform each operand once and the result once, when its
coefficients are read.

## Batches

For a batch, the last two axes hold the coefficients of each element, and
the leading axes follow NumPy's broadcasting rules.

```python
rng = np.random.default_rng(1234)
A = cdl.LoopAlgebra(cdl.CDLoop(5))                # dimension 64
X = A.asarray(rng.normal(size=(1000, 2, 32)))     # shares the float64 storage
Y = A.random(rng, batch_shape=(1000,))
Z = X * Y                                         # batch_shape (1000,)
first = Z[0]                                      # an Element copy
Z[1] = A.one()                                    # assign one element

out = np.empty((1000, 2, 32))
A.multiply(X, Y, out=out)                         # reuse output storage
```

An `ElementArray` stores loop-basis coefficients, so each product transforms
its operands and its result. To run several operations without those
transforms, convert once:

```python
S, T = X.split(), Y.split()         # SplitArray: spectra and signed components
W = (S * T + S) * T                 # no transforms
result = W.to_loop()                # one conversion back

S.spectral.shape                    # (1000, 32)
S.signed                            # a pyfastcd ElementArray of A_5
```

Combining an `ElementArray` with a `SplitArray` raises `TypeError`; convert
one of them explicitly. Elements combine with both.

## Choosing the multiplication algorithm

```python
A = cdl.LoopAlgebra(cdl.CDLoop(4))
x, y = A.random(1), A.random(2)

p1 = A.multiply(x, y)                                    # the default
p2 = A.multiply(x, y, algorithm="table")                 # from the definition
p3 = A.multiply(x, y, signed_algorithm="quasilinear")    # passed to pyfastcd
A.signed_algorithms         # ('auto', 'direct', 'cariow_cariowa', 'quasilinear')
```

`algorithm` is `"auto"`, `"split"` or `"table"`. `"split"` multiplies
through the decomposition; `"table"` evaluates the definition with
`O(D^2)` multiplications and is useful as an independent check.
`signed_algorithm` names the method for the product of signed components.
`A.algorithms` and `A.signed_algorithms` list the names an algebra accepts;
they differ in [characteristic 2](#characteristic-2).

## Explicit coefficient fields

Everything above used real coefficients. Complex coefficients and finite
fields of odd characteristic are selected as in pyfastcd:

```python
C = cdl.LoopAlgebra(cdl.CDLoop(2), char=0, deg=2)       # complex coefficients
B = cdl.LoopAlgebra(cdl.CDLoop(3), char=13)             # over F_13
x = B.random(7)
y = B.random(8)
x * y == B.multiply(x, y, algorithm="table")            # True: arithmetic is exact
(x ** (10 ** 30)).coefficients.dtype                    # uint32

import pyfastcd as fcd
F = fcd.FiniteField(char=13, deg=5)
E = cdl.LoopAlgebra(cdl.CDLoop(3), field=F)
E.coefficient_shape                 # (2, 8, 5): one polynomial axis
c = F([1, 2, 0, 0, 0])
Q3 = cdl.CDLoop(3)
z = E.scalar(c) + E.basis(Q3.basis(1))
z.augmentation() == c + 1           # True: a scalar of the field
```

Over a finite field, coefficients are integers or scalars of the field,
`kappa()` and `spectrum()` return pyfastcd arrays of field scalars, and
`coefficient()` and `augmentation()` return field scalars. The fields
F_(2^k) are selected in the same way, with `char=2`, but their loop
algebras behave differently; they have a [section](#characteristic-2) of
their own below.

## Central-product algebras

```python
L = cdl.CentralProduct(2, 3)
A = cdl.LoopAlgebra(L)                  # dimension 64
x, y = A.random(1), A.random(2)
p = x * y
x.epsilon().shape                       # (4, 8): one axis for each factor
A.signed_algorithms                     # ('auto', 'rows', 'schedule')

x.left_power(3) == x * (x * x)          # True
x.right_power(3) == (x * x) * x         # True
# x ** 3 raises TypeError: powers depend on bracketing here

Q2 = cdl.LoopAlgebra(cdl.CDLoop(2))
u = Q2.random(3)
A.factor_embedding(0, u)                # the image of u from the first factor
```

The signed component of a central-product algebra lies in a tensor product
of Cayley–Dickson algebras, which in general is neither flexible nor
power-associative; [Conventions and results](mathematics.md#elements-of-the-loop-algebra-of-a-central-product)
lists what follows.

## Structure

```python
A = cdl.LoopAlgebra(cdl.CDLoop(4), char=7)
x = A.random(5)

x.minimal_polynomial()              # field coefficients, constant term first
x.is_zero_divisor()
x.has_inverse()
x.left_annihilator()                # an ElementArray: a basis of {y : y x = 0}
A.left_matrix(x)                    # the 32 x 32 matrix of y -> x y
A.center_basis()                    # an ElementArray spanning the center
A.primitive_idempotents()           # the 16 idempotents of the group component
```

## Characteristic 2

`char=2` selects the fields F_(2^k) for `k` up to 12, which pycdloop
provides as `BinaryField`:

```python
Q = cdl.CDLoop(3)
A = cdl.LoopAlgebra(Q, char=2)                  # over F_2, dimension 16
x = A.one() + A.basis(Q.basis(1)) + A.basis(Q.basis(2))

x.augmentation()                                # the scalar 1 of F_2
x * x == x + A.basis(Q.basis(3))                # False
x * x == A.one() + A.basis(Q.basis(3)) + A.basis(-Q.basis(3))   # True
x ** 4 == A.one()                               # True: x^4 = aug(x)^4
x.inverse() == x ** 3                           # True
(x + 1).is_nilpotent()                          # True: the augmentation is 0
x.kappa()                                       # array([1, 1, 1, 0, 0, 0, 0, 0], dtype=uint32)

F = cdl.BinaryField(deg=4)                      # F_16
F.modulus                                       # (1, 1, 0, 0, 1): 1 + t + t^4
t = F([0, 1, 0, 0])
B = cdl.LoopAlgebra(cdl.CentralProduct(2, 3), field=F)   # the same as char=2, deg=4
y = B.random(1)
(t * y).augmentation() == t * y.augmentation()  # True
y ** 3 == y * (y * y)                           # True: powers are defined here

B.algorithms                                    # ('auto', 'lift', 'table')
p1 = B.multiply(y, y, algorithm="lift")
p2 = B.multiply(y, y, algorithm="table")
p1 == p2                                        # True: both are exact
```

In characteristic 2 the algebra has no decomposition: `spectrum()`,
`epsilon()`, `split()` and the idempotents raise `TypeError`, and elements
and arrays stay in the loop basis. In exchange the algebra is simpler. The
square of every element is central, so powers, inverses and minimal
polynomials are defined for central products as well, and an element is
invertible exactly when its augmentation is nonzero. Products are computed
by `"lift"`, which runs the split algorithm over a field of large odd
characteristic and
reduces the result, or by `"table"`; `"auto"` picks the faster one for the
size at hand. [Conventions and results](mathematics.md#characteristic-2)
has the mathematics, and the [API reference](api.md#characteristic-2) lists
every difference.

## Codes

A subspace of a loop algebra is a linear code of length `len(loop)` through
the loop-basis coefficients. `pycdloop.codes` builds generator matrices of
one-sided ideals:

```python
from pycdloop import codes, linalg

Q = cdl.CDLoop(2)
A = cdl.LoopAlgebra(Q, char=3)
x = A.basis(Q.basis(0)) + A.basis(Q.basis(1))

T = codes.translate_matrix(x)                   # rows [g] x for g in Q
G = codes.generator_matrix(x)                   # RREF basis of the left ideal
G.shape                                         # (6, 8)
linalg.rank(T, field=A.field)                   # 6
x.weight()                                      # 2: the Hamming weight
print(codes.to_gap(G, field=A.field))           # a GUAVA expression
```

Binary codes come from the loop algebra over F_2 in the same way:

```python
A2 = cdl.LoopAlgebra(Q, char=2)
x2 = A2.basis(Q.basis(0)) + A2.basis(Q.basis(1))
G2 = codes.generator_matrix(x2)                 # a binary code of length 8
G2.shape                                        # (6, 8)
print(codes.to_sage(G2, field=A2.field))        # a SageMath expression
```

## Where to go next

- [Python API reference](api.md)
- [Conventions and results](mathematics.md)
- [Benchmark](benchmark.md) and [Numerical stability](stability.md)
