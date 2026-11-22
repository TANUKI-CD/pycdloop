# Quick start

pycdloop is a Python package for arithmetic in Cayley-Dickson loops, their
central products, and their loop algebras over the real numbers, the complex
numbers, and finite fields. A Cayley-Dickson loop consists of the basis elements
of a Cayley-Dickson algebra and their negatives, and its loop algebra has one
basis element for each element of the loop, so an element of the loop algebra is
an array of coefficients, which pycdloop multiplies with the help of pyfastcd.

This guide introduces the basics of how to use pycdloop one idea at a time, from
installation and the elements of a loop through loop algebras, their
decomposition, batches, and NumPy to other coefficient fields, central products,
characteristic 2, and codes. Each example builds on the examples that come
before it and shows its results in comments. Users of pycdloop are recommended
to start here and read this guide in its entirety, running the examples for
themselves before using the package in production. The [Python API
reference](api.md) gives the complete rules for everything shown here and should
be read for a more detailed reference if more information is needed than what
this quick start guide can provide.

## Installation

pycdloop requires CPython 3.10 or newer, NumPy 1.26 or newer, and pyfastcd 1.4.0
or a later 1.x release, which `pip` installs along with it. To install pycdloop
from PyPI, run:

```sh
pip install pycdloop
```

To build and install it from a source checkout instead, you also need a C11
compiler and the Python development headers. Run the following command from the
repository root, with the Python interpreter that will run your code:

```sh
python -m pip install .
```

## Loops and their elements

`CDLoop(n)` creates the Cayley-Dickson loop `Q_n`, whose elements are the basis
elements `e_0, ..., e_(N-1)` of the Cayley-Dickson algebra of dimension
`N = 2^n` and their negatives. The elements multiply as these basis elements do,
so multiplying a few of them is the quickest way to see how the loop behaves:

```python
import numpy as np
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
Q.twist(1, 2)                       # 1: e_1 e_2 = +e_3
```

The loop is not commutative, since `e_1 e_2` and `e_2 e_1` differ in sign, and
from level 3 on it is not associative either, so parentheses matter in every
product of three or more elements. Every element other than `+1` and `-1`
squares to `-1` and so has order 4. The sign of a product follows from the
indices alone: `twist(s, t)` is the sign in `e_s e_t = sigma(s, t) e_(s XOR t)`,
here `e_1 e_2 = +e_3`, and [The algebra and its
basis](mathematics.md#the-algebra-and-its-basis) gives the rule that produces
it.

A `LoopElement` such as `e1` has an `index` and a `sign`, is immutable and
hashable, and combines only with elements of its own loop, while loops
themselves are cached, so `CDLoop(3) is CDLoop(3)`.

## The structure of a loop

Since the loop is neither commutative nor associative, its structure is
described by how far its elements are from commuting and associating, and by its
center and its subloops, which a loop computes directly:

```python
e1.commutator(e2)                   # -e_0: e1 e2 = -(e2 e1)
e1.associator(e2, e4)               # -e_0: (e1 e2) e4 = -(e1 (e2 e4))
Q.commutator_sign(1, 2)             # -1
Q.associator_sign(1, 2, 4)          # -1
t = np.arange(8)
Q.twist(t[:, None], t[None, :]).shape   # (8, 8): the table of signs

len(Q.center())                     # 2: the center is {+1, -1}
H = Q.generate([e1, e2])            # the quaternion subgroup, a frozenset
len(H), Q.is_normal(H)              # (8, True)
Q.is_moufang()                      # True
cdl.CDLoop(4).is_moufang()          # False
```

The commutator of two elements and the associator of three are the central
elements that relate the two orders and the two bracketings of their product,
and the sign functions give the same signs from the indices, for single integers
or for whole arrays of indices, which broadcast as in NumPy. Subsets of a loop
are passed and returned as sets of `LoopElement` objects: the subloop generated
by `e_1` and `e_2` consists of the eight elements `+e_t` and `-e_t` with
`t < 4`, which form the quaternion group, a normal subloop. The octonion loop
satisfies the Moufang identity `(xy)(zx) = (x(yz))x`, while the loop of level 4
does not, and for loops too large to test exhaustively the test samples, as
[Identity tests](api.md#identity-tests) describes.

## Central products

A central product of Cayley-Dickson loops is their direct product with the
central elements `-1` of all factors identified, so its elements are signed
tensors of one basis element from each factor. `CentralProduct` creates it from
the levels of the factors:

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

Squares behave differently from those in a Cayley-Dickson loop: in `Q_1 * Q_1`
the element with index 1 in both factors squares to `+1`, so elements other than
`-1` can have order 2. Each element has a *flat index*, whose binary digits are
those of the factor indices, first factor first, and `basis`, `flat_index`, and
`factor_indices` convert between a flat index and one index for each factor.
Otherwise a central product has the same methods as a Cayley-Dickson loop, with
`factor_embedding` in place of `embed`.

## Elements of a loop algebra

The loop algebra of a loop has one basis element `[x]` for each loop element
`x`, with the product `[x][y] = [xy]`, and `LoopAlgebra` creates it, over R
unless another field is selected. An element is a linear combination of basis
elements, stored as two rows of coefficients, row 0 for the basis elements
`[e_t]` and row 1 for the basis elements `[-e_t]`:

```python
Q = cdl.CDLoop(2)                   # the quaternion group
A = cdl.LoopAlgebra(Q)              # its loop algebra over R, of dimension 8
i, j = Q.basis(1), Q.basis(2)

x = A.basis(i) + 2 * A.basis(-j)                 # [e_1] + 2 [-e_2]
y = A.from_terms({Q.one(): 1.0, i * j: -3.0})    # [e_0] - 3 [e_3]
z = A([[1, 0, 0, 2], [0, 5, 0, 0]])              # [e_0] + 2 [e_3] + 5 [-e_1]

x.coefficients                      # read-only (2, 4) array
x.coefficient(-j)                   # 2.0
x.terms()                           # {e_1: 1.0, -e_2: 2.0}
x.weight()                          # 2: two nonzero coefficients
```

The basis element `[-e_t]` is distinct from `-[e_t]`, so `A.basis(-j)` and
`-A.basis(j)` are different elements, and the loop algebra of a loop of order 8
has dimension 8. An `Element` is immutable and hashable, and its coefficients
can be read, through `coefficients`, `coefficient`, and `terms`, but not
changed.

## Arithmetic

Elements combine with the usual operators, and a scalar of the coefficient field
combines with an element as in pyfastcd, standing for that multiple of the
identity `[e_0]` in a sum and scaling every coefficient in a product:

```python
x * y                               # [e_1] - 6 [-e_1] - [-e_2]
x + 3                               # adds 3 [e_0]
0.5 * x                             # halves every coefficient
x.square()                          # x x = 2 [e_3] + 5 [-e_0] + 2 [-e_3]
x ** 5 == x * x * x * x * x         # True: ** is exponentiation, ^ is not
w = (x + 4).inverse()               # the two-sided inverse
w * (x + 4) == A.one()              # True
x.commutator(y)                     # x y - y x
x.associator(y, z)                  # (x y) z - x (y z), here 0
```

The product is the bilinear extension of the loop product, so it inherits the
properties of the loop: it is not commutative, and while the loop algebra of the
quaternion group is associative, as every group algebra is, so that the
associator is zero here, the loop algebras of larger Cayley-Dickson loops are
not.

Powers are written with `**`, since `^` is the bitwise exclusive or in Python,
and `inverse` returns the inverse that is a polynomial in the element, which is
two-sided, and raises `ZeroDivisionError` when there is none.

## The decomposition

Over R, C, and the finite fields of odd characteristic, the loop algebra
decomposes into two algebras, and this decomposition is what makes products
fast. Every element has a *group component*, the sums `a_t + b_t` of its
coefficients of `[e_t]` and `[-e_t]`, and a *signed component*, the differences
`a_t - b_t`, which form an element of the Cayley-Dickson algebra, and both maps
are homomorphisms:

```python
x.kappa()                           # array([0., 1., 2., 0.]): the group component
x.epsilon()                         # a pyfastcd Element of the quaternions
x.spectrum()                        # array([ 3.,  1., -1., -3.])
x.augmentation()                    # 3.0, the sum of all coefficients

(x * y).epsilon() == x.epsilon() * y.epsilon()        # True
A.from_components(x.kappa(), x.epsilon()) == x        # True

pi_plus, pi_minus = A.central_idempotents()           # (1 +/- [-1]) / 2
pi_plus * x == A.lift_group(x.kappa())                # True
pi_minus * x == A.lift_signed(x.epsilon())            # True
```

The spectrum is the Walsh-Hadamard transform of the group component, and
together with the signed component it forms the *split form* of an element, in
which products are evaluated: the spectra multiply coordinatewise and the signed
components multiply in the Cayley-Dickson algebra, through pyfastcd. An element
keeps its split form after its first product, and a product is created in split
form, so a chain of operations transforms each operand once and the result once,
when its coefficients are first read. The central idempotents project onto the
two components, and [The decomposition](mathematics.md#the-decomposition)
explains why the algebra splits in this way.

## Batches

An `ElementArray` holds a batch of elements in one NumPy array, whose trailing
axes hold the coefficients of each element, two axes or three over an extension
field, and whose leading axes follow NumPy's broadcasting rules:

```python
rng = np.random.default_rng(1234)
A = cdl.LoopAlgebra(cdl.CDLoop(5))                # dimension 64
X = A.asarray(rng.normal(size=(1000, 2, 32)))     # 1000 elements, storage shared
Y = A.random(rng, batch_shape=(1000,))            # 1000 random elements
Z = X * Y                                         # 1000 products
Z.batch_shape                                     # (1000,)
first = Z[0]                                      # an Element copy of record 0
Z[1] = A.one()                                    # assign one record

out = np.empty((1000, 2, 32))
A.multiply(X, Y, out=out)                         # writes the products into out
```

`asarray` shares the storage of a compatible array instead of copying it, while
`array` always copies, and `out` lets a product reuse existing storage, provided
it has exactly the shape and dtype of the result. An `ElementArray` stores
loop-basis coefficients, so every product transforms its operands and its
result, and to run several operations without those transforms, the arrays are
converted to split form once:

```python
S, T = X.split(), Y.split()         # SplitArray: spectra and signed components
W = (S * T + S) * T                 # no transforms
result = W.to_loop()                # one conversion back

S.spectral.shape                    # (1000, 32)
S.signed                            # a pyfastcd ElementArray of A_5
```

Combining an `ElementArray` with a `SplitArray` raises `TypeError`, so that
every transform is visible in the code, while elements combine with both kinds
of array.

## Working with NumPy

Elements and arrays hand their coefficients to NumPy, but they do not take part
in NumPy's own arithmetic, so that every operation of the algebra goes through
pycdloop:

```python
np.asarray(Z).shape                 # (1000, 2, 32): the storage of Z, not a copy
Z.coefficients.mean(axis=0).shape   # (2, 32): the mean of each coefficient
np.float64(2) * first == 2 * first  # True: the scalar reaches pycdloop
```

`np.asarray` returns the storage of an array, or a read-only view of the
coefficients of an element, and NumPy functions applied to these coefficient
arrays act on individual coefficients as usual. NumPy universal functions, such
as `np.multiply`, `np.add`, and `np.sum`, raise `TypeError` on elements and
arrays, which is what lets a NumPy scalar such as `np.float64(2)` combine with
an element through the operators of pycdloop, as the last line shows. [Arrays
and NumPy](api.md#arrays-and-numpy) lists these interfaces, together with
copying and pickling.

## Choosing the multiplication algorithm

pycdloop has two multiplication algorithms over these fields, the split
algorithm, which multiplies through the decomposition, and the table algorithm,
which evaluates the definition of the product with `4 D^2` multiplications, and
`*` uses the default, which is the split algorithm:

```python
A = cdl.LoopAlgebra(cdl.CDLoop(4))
x, y = A.random(1), A.random(2)

A.algorithms                        # ('auto', 'split', 'table')
A.default_algorithm                 # 'split'
p = A.multiply(x, y)                                     # the default
q = A.multiply(x, y, algorithm="table")                  # from the definition
np.allclose(p.coefficients, q.coefficients)              # True
r = A.multiply(x, y, signed_algorithm="quasilinear")     # passed to pyfastcd
A.signed_algorithms         # ('auto', 'direct', 'cariow_cariowa', 'quasilinear')
```

The default comes from benchmarks, which [How the results set the
defaults](benchmark.md#how-the-results-set-the-defaults) reports, and the table
algorithm is mainly useful as an independent check: both algorithms compute the
same product, although in floating point their results can differ in the last
bits. `signed_algorithm` names the method for the product of signed components,
which for a Cayley-Dickson loop is one of the algorithms of pyfastcd, and
`algorithms` and `signed_algorithms` list the names that an algebra accepts,
which depend on the loop and on the coefficient field.

## Other coefficient fields

Everything above uses real coefficients, which a loop algebra has unless its
coefficient field is selected explicitly. The selectors `char`, `deg`, and
`field` choose complex coefficients and the finite fields of odd characteristic
as in pyfastcd:

```python
C = cdl.LoopAlgebra(cdl.CDLoop(2), char=0, deg=2)       # complex coefficients
B = cdl.LoopAlgebra(cdl.CDLoop(3), char=13)             # over F_13
x, y = B.random(7), B.random(8)
x * y == B.multiply(x, y, algorithm="table")            # True: arithmetic is exact
(x ** (10 ** 30)).coefficients.dtype                    # uint32

import pyfastcd as fcd
F = fcd.FiniteField(char=13, deg=5)
E = cdl.LoopAlgebra(cdl.CDLoop(3), field=F)
E.coefficient_shape                 # (2, 8, 5): one polynomial axis
c = F([1, 2, 0, 0, 0])              # 1 + 2t
v = E.scalar(c) + E.basis(cdl.CDLoop(3).basis(1))
v.augmentation() == c + 1           # True: a scalar of the field
```

Over a finite field the coefficients are integers or scalars of the field, every
operation is exact, and a power can have an exponent of any size. Each
coefficient in an extension field is stored as its polynomial coordinates, so an
element of `E` is a `(2, 8, 5)` array of `uint32` residues, and `coefficient()`
and `augmentation()` return scalars of the field, while in odd characteristic
`kappa()` and `spectrum()` return pyfastcd arrays of field scalars. The fields
F_(2^k) are selected in the same way, with `char=2`, but their loop algebras
behave differently, as [Characteristic 2](#characteristic-2) shows.

## Loop algebras of central products

The loop algebra of a central product works in the same way, except that its
signed component lies in a tensor product of Cayley-Dickson algebras rather than
in one of them, which in general is neither flexible nor power-associative, so
powers depend on how they are bracketed:

```python
L = cdl.CentralProduct(2, 3)
A = cdl.LoopAlgebra(L)                  # dimension 64
x, y = A.random(1), A.random(2)
p = x * y
x.epsilon().shape                       # (4, 8): one axis for each factor
A.signed_algorithms                     # ('auto', 'rows', 'schedule')

x.left_power(3) == x * (x * x)          # True
x.right_power(3) == (x * x) * x         # True

Q2 = cdl.LoopAlgebra(cdl.CDLoop(2))
u = Q2.random(3)
A.factor_embedding(0, u)                # the image of u from the first factor
```

Here `x ** 3` raises `TypeError`, and so do the other operations that rely on
powers, while `left_power` and `right_power`, which fix the bracketing, are
defined in every loop algebra, and [Where powers are
defined](api.md#where-powers-are-defined) lists these operations. The signed
components are multiplied by one of the methods `"rows"` and `"schedule"`, or
`"native"` where it applies, and [Elements of the loop algebra of a central
product](mathematics.md#elements-of-the-loop-algebra-of-a-central-product)
describes what follows from the structure of the tensor product.

## Structure

An element answers structural questions about itself, such as whether it is
invertible or a zero divisor, and its algebra returns the matrices and bases
behind these answers:

```python
A = cdl.LoopAlgebra(cdl.CDLoop(4), char=7)
x = A.random(5)

x.has_inverse()                     # False
x.is_zero_divisor()                 # True
len(x.left_annihilator())           # 5: a basis of {y : y x = 0}
x.minimal_polynomial()              # array([0, 4, 6, 0, 0, 0, 0, 3, 1], dtype=uint32)
A.left_matrix(x).shape              # (32, 32): the matrix of y -> x y
len(A.center_basis())               # 17: a basis of the center
len(A.primitive_idempotents())      # 16
```

Three of the sixteen spectral coordinates of this element are zero, so it has no
inverse and is a zero divisor, and its minimal polynomial, given by its field
coefficients with the constant term first, has the root 0; over a finite field
all of these answers are exact.

The algebra provides the matrix of multiplication by an element and the bases
behind the structure of the algebra itself: its center is spanned by the sixteen
primitive idempotents of the group component and by `([e_0] - [-e_0]) / 2`,
which comes from the center `{+1, -1}` of the loop, and [Elements of the loop
algebra of `Q_n`](mathematics.md#elements-of-the-loop-algebra-of-q_n) explains
how all of these are read off the decomposition.

## Characteristic 2

`char=2` selects the fields F_(2^k), for `k` from 1 to 12, which pycdloop
provides as `BinaryField`. Over these fields 2 is zero, so the loop algebra has
no decomposition, but its structure is simpler instead:

```python
Q = cdl.CDLoop(3)
A = cdl.LoopAlgebra(Q, char=2)                  # over F_2, dimension 16
x = A.one() + A.basis(Q.basis(1)) + A.basis(Q.basis(2))

x.augmentation()                                # the scalar 1 of F_2
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

Without the decomposition, `spectrum()`, `epsilon()`, `split()`, and the
idempotents raise `TypeError`, and elements and arrays stay in the loop basis.
In exchange the square of every element is central, so powers, inverses, and
minimal polynomials are defined for central products as well, every fourth power
is the scalar `aug(x)^4`, and an element is invertible exactly when its
augmentation, the sum of its coefficients, is nonzero.

Products are computed by the lift algorithm, which runs the split algorithm over
a field of large odd characteristic and reduces the result, or by the table
algorithm, and `"auto"` picks the one expected to be faster for the size at
hand. [Characteristic 2](mathematics.md#characteristic-2) in the conventions has
the mathematics, and [Characteristic 2](api.md#characteristic-2) in the API
reference lists every difference from the other fields.

## Codes

Through its loop-basis coefficients, a subspace of a loop algebra is a linear
code of length `len(loop)`, and the module `pycdloop.codes` builds generator
matrices of left, right, and two-sided ideals and exports them to GAP and
SageMath:

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

y = x + A.basis(Q.basis(2))
L = codes.generator_matrix(y)                   # the left ideal, of dimension 5
B = codes.generator_matrix(y, side="two-sided") # the two-sided ideal
B.shape                                         # (7, 8)
codes.is_ideal(L, A)                            # True
codes.is_ideal(L, A, side="right")              # False
```

Since this algebra is associative, the left ideal generated by `x` is spanned by
its translates `[g] x`, so its dimension is the rank of the translate matrix,
and the weight of an element is its Hamming weight as a codeword. The ideal
generated by `y` on the left is not closed under translation on the right, so
the two-sided ideal it generates is larger. Binary codes come from the loop
algebra over F_2 in the same way:

```python
A2 = cdl.LoopAlgebra(Q, char=2)
x2 = A2.basis(Q.basis(0)) + A2.basis(Q.basis(1))
G2 = codes.generator_matrix(x2)                 # a binary code of length 8
G2.shape                                        # (6, 8)
print(codes.to_sage(G2, field=A2.field))        # a SageMath expression
```

The exports work over extension fields as well, and carry the modulus of the
field:

```python
A4 = cdl.LoopAlgebra(Q, char=2, deg=2)
x4 = A4.basis(Q.basis(0)) + A4.basis(Q.basis(1))
G4 = codes.generator_matrix(x4)                 # a code of length 8 over F_4
G4.shape                                        # (6, 8, 2)
print(codes.to_sage(G4, field=A4.field))        # over GF(4, 't', modulus=[1, 1, 1])
```

## Where to go next

- The [Python API reference](api.md) documents every class, method, and rule.
- [Conventions and results](mathematics.md) describes the loops, their loop
  algebras, the multiplication algorithms, and their operation counts.
- The [benchmark](benchmark.md) and the [stability study](stability.md) report
  the speed and the accuracy of the algorithms.
