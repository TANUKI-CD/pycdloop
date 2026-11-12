# Python API reference

`pycdloop` models Cayley–Dickson loops, their central products and their
loop algebras. Its public names are `CDLoop`, `CentralProduct`,
`LoopElement`, `LoopAlgebra`, `BinaryField`, `BinaryFieldElement`,
`Element`, `ElementArray`, `SplitArray`, `GeneratedSubalgebra`, `wht`,
`dyadic_convolve`, `__version__` and the modules `linalg`, `codes` and
`testing`; modules whose names begin with an underscore are private. Errors
are reported as Python exceptions.
[Conventions and results](mathematics.md) defines the mathematical objects
named here.

## Terms and notation

**Loops.** A *loop* is a `CDLoop` or a `CentralProduct`. Its elements are
the signed basis elements `+e_t` and `-e_t`, where `t` is a *flat index* in
`range(D)` and `D = loop.quotient_order`. For a central product the flat
index concatenates the binary digits of one index for each factor, first
factor first. The *position* of `+e_t` is `t` and that of `-e_t` is
`D + t`; loops iterate in position order.

**Coefficient field and selectors.** Every loop algebra has its
coefficients in R, C, or a finite field F_(p^d). The keyword arguments
`char`, `deg` and `field` choose the field with the meaning they have in
pyfastcd:

| Selectors | Coefficients | `dtype` | `coefficient_shape` |
|---|---|---|---|
| omitted, or `char=0, deg=1` | R | `float64` | `(2, D)` |
| `char=0, deg=2` | C | `complex128` | `(2, D)` |
| `char=p, deg=1` | F_p | `uint32` | `(2, D)` |
| `char=p, deg=d`, `d > 1` | F_(p^d) | `uint32` | `(2, D, d)` |
| `field=F` | the finite field `F` | `uint32` | `(2, D)` or `(2, D, F.deg)` |

For an odd prime `p` the field is a `pyfastcd.FiniteField`, and the rules
for the selectors, including the accepted characteristics and degrees and
the exceptions for invalid values, are those of `pyfastcd.Algebra`. For
`p = 2` it is a [`BinaryField`](#binaryfield-and-binaryfieldelement), which
pycdloop provides: `char=2, deg=k` selects `BinaryField(deg=k)` for `k`
from 1 to 12. `field=F` takes a field of either kind and, as in pyfastcd,
is passed without `char` or `deg`.

Some operations depend on the characteristic. Wherever this page says that
something is not defined in characteristic 2, it raises `TypeError` there;
[Characteristic 2](#characteristic-2) lists every difference in one place.

**Scalars.** A *field scalar* is a value of the coefficient field: a real
number (`int`, `float`, `Fraction`, `Decimal`, a real NumPy scalar or
zero-dimensional array) over R, additionally a complex number over C, and
an `int` or a scalar of the same field over a finite field: a `pyfastcd`
scalar in odd characteristic and a `BinaryFieldElement` in characteristic
2. Integers are reduced modulo `char` in arithmetic and construction;
floats are rejected over a finite field.

**Parent.** Every element and array belongs to the `LoopAlgebra` that
created it, and `x.algebra` returns it. Operations combine only objects with
the same parent, and field scalars. Nothing converts between parents, and a
`LoopElement` is never converted to an algebra element implicitly.

**Record.** A *record* is the coefficient storage of one element, of shape
`coefficient_shape`: row 0 holds the coefficients of the basis elements
`[e_t]`, row 1 those of `[-e_t]`. An `ElementArray` has shape
`batch_shape + coefficient_shape`.

**Split form.** The pair of the Walsh–Hadamard transform of `(a_t + b_t)`
and the array `(a_t - b_t)`, each of shape `(D,)` followed by the polynomial
axis of an extension field. Products need no transform in split form. It is
not defined in characteristic 2.

Below, `L` is a loop, `A` a loop algebra, `x`, `y`, `z` elements, `X` an
`ElementArray`, `S` a `SplitArray` and `r` a field scalar.

## Module contents

- `__version__: str`

  The version of the package.

- `wht(a, *, axis=-1, field=None) -> numpy.ndarray`

  Return the Walsh–Hadamard transform `H a` along one axis, whose length
  must be a power of two. The transform is not normalized: applying it twice
  multiplies by the length. Without `field` the input is converted to
  `float64`, or `complex128` when complex; with a `pyfastcd.FiniteField` it
  holds residues, with a trailing polynomial axis for an extension field
  that `axis` does not count. A length that is not a power of two or an axis
  out of range raises `ValueError`; nonnumeric input raises `TypeError`, and
  so does a `BinaryField`, over which the transform is not invertible.

- `dyadic_convolve(a, b, *, field=None) -> numpy.ndarray`

  Return `c[k] = sum(a[i] * b[i ^ k])` along the last coefficient axis, the
  product in the group algebra of the elementary abelian 2-group. Leading
  axes broadcast. `field` is a `pyfastcd.FiniteField` or a `BinaryField`.
  Operands of different or non-power-of-two length raise `ValueError`.

---

## CDLoop and CentralProduct

### Constructors

```python
CDLoop(n)
CentralProduct(n1, n2, ...)
```

`CDLoop(n)` is the Cayley–Dickson loop `Q_n` for a doubling level `n >= 0`.
`CentralProduct(n1, n2, ...)` is the central product of `Q_n1, Q_n2, ...`
and takes at least two levels, each at least 1. Loops are immutable and
cached, so `CDLoop(3) is CDLoop(3)`. A level that is not an integer, or is a
boolean, raises `TypeError`, and so does a central product with fewer than
two levels; a level below the minimum raises `ValueError`; a total level
above 62 raises `OverflowError`.

### Properties

- `levels: tuple[int, ...]`

  The doubling level of each factor; `(n,)` for `CDLoop(n)`.

- `level: int`

  The sum of the levels.

- `quotient_order: int`

  `D = 2**level`: the number of basis elements, and the order of the
  quotient of the loop by `{+1, -1}`.

- `order: int`

  `2 * quotient_order`, also `len(L)`. At total level 62 the order is
  `2**63`, which `len` cannot return: it raises `OverflowError`.

### Elements

- `L(index, sign=1) -> LoopElement`

  Return `sign * e_index`. An index outside `range(D)` raises `IndexError`;
  a sign other than `+1` or `-1` raises `ValueError`.

- `one()`, `minus_one() -> LoopElement`

  Return the identity and the central element `-1`.

- `basis(*indices) -> LoopElement`

  Return a positive basis element from its flat index, or from one index
  for each factor.

- `flat_index(*indices) -> int`, `factor_indices(index) -> tuple[int, ...]`

  Convert between one index for each factor and the flat index.

- `random(rng=None) -> LoopElement`

  Return a uniformly random element; `rng` is a `numpy.random.Generator` or
  a seed.

- `len(L)`, `iter(L)`, `x in L`

  The order, iteration in position order, and membership.

### Sign functions

The arguments are flat indices or integer arrays of them, which broadcast.
The result is an `int` when every argument is an integer or a
ze-dimensional array, and an `int8` array otherwise. Indices out of range
raise `IndexError`.

- `twist(s, t)`

  Return `sigma(s, t)`, the sign in `e_s e_t = sigma(s, t) e_(s ^ t)`.

- `commutator_sign(s, t)`

  Return the sign `c` with `e_s e_t = c e_t e_s`.

- `associator_sign(s, t, u)`

  Return the sign `a` with `(e_s e_t) e_u = a e_s (e_t e_u)`.

- `twist_table() -> numpy.ndarray`

  Return the `D x D` matrix of twists, as `int8`.

- `cayley_table() -> numpy.ndarray`

  Return the multiplication table: entry `[i, j]` is the position of the
  product of the elements at positions `i` and `j`.

Both tables raise `ValueError` when `D` exceeds 4096; the sign functions
have no such limit.

### Substructure

Subsets of a loop are passed as iterables of `LoopElement` and returned as
`frozenset` objects.

- `center()`, `commutant()`, `nucleus(side=None)`

  Return the center, the commutant, and a nucleus. `side` is `"left"`,
  `"middle"` or `"right"` for the elements that associate with every pair
  from that position of the associator, and `None` for the intersection of
  the three. They examine every index, so time and memory grow with `D`.

- `generate(elements)`

  Return the subloop generated by `elements`.

- `is_subloop(elements) -> bool`

  Return whether the elements contain the identity and are closed under
  multiplication.

- `is_normal(elements) -> bool`

  Return whether a subloop `S` satisfies `xS = Sx`, `x(yS) = (xy)S` and
  `(Sx)y = S(xy)` for all `x` and `y`. Elements that do not form a subloop
  raise `ValueError`.

- `cosets(elements, *, side="left")`

  Return the distinct cosets `xS` or `Sx` of a subloop, as a `frozenset` of
  `frozenset` objects. The cosets of a normal subloop partition the loop;
  those of other subloops can overlap. Elements that do not form a subloop
  raise `ValueError`.

### Maps

A map of the loop to itself is given as a callable on `LoopElement`
objects, or as an integer array with the position of the image of each
element in position order.

- `left_translation(x)`, `right_translation(x) -> numpy.ndarray`

  Return the permutations `y -> x y` and `y -> y x`, by position.

- `character_automorphism(a) -> numpy.ndarray`

  Return the automorphism `e_t -> (-1)**popcount(a & t) e_t`, by position,
  for an index `a`.

- `is_automorphism(f) -> bool`

  Return whether `f` is a bijection that preserves products.

- `CDLoop.embed(x) -> LoopElement`

  Return the image in `Q_n` of an element of `Q_k`, `k <= n`: the same
  index and sign.

- `CentralProduct.factor_embedding(i, x) -> LoopElement`

  Return the image of an element `x` of `CDLoop(levels[i])`: its index in
  factor `i` and index 0 elsewhere.

### Identity tests

- `is_commutative()`, `is_associative() -> bool`

  Exact and never sampled. They examine every index, so time and memory
  grow with `D`.

- `is_moufang(samples=None)`, `is_diassociative(samples=None) -> bool`

  Test `(xy)(zx) = (x(yz))x`, and that every two elements generate a group.
  With `samples=None` the test covers every tuple of indices when there are
  at most `2**22` of them and `2**18` pseudorandom tuples otherwise; a
  positive integer `samples` tests that many pseudorandom tuples, whatever
  the size of the loop. The tuples come from a fixed seed, so repeated
  calls test the same ones. A sampled `True` means only that no
  counterexample was found.

### Export

- `to_gap() -> str`

  Return a GAP expression that builds the loop with `LoopByCayleyTable` of
  the LOOPS package, with elements numbered from 1 in position order. It
  raises `ValueError` when `D` exceeds 4096, as the tables do.

---

## LoopElement

```python
LoopElement(loop, index, sign=1)
```

One signed basis element, immutable and hashable. `L(index, sign)` is the
usual constructor.

### Properties

- `loop`, `index: int`, `sign: int`

  The parent loop, the flat basis index, and `+1` or `-1`.

- `indices: tuple[int, ...]`

  The basis index in each factor.

- `position: int`

  The position in the iteration order of the loop.

### Operations

Operands must belong to the same loop; elements of different loops raise
`ValueError`.

- `x * y`, `-x`, `+x`

  The loop product, and `x` times the central element `-1`.

- `x ** k`

  An integer power; negative exponents use the inverse. The square of an
  element is central, so powers do not depend on bracketing.

- `x == y`, `hash(x)`

  Elements are equal when they have the same loop, index and sign.

- `inverse() -> LoopElement`

  Return the two-sided inverse.

- `order() -> int`

  Return the order: 1, 2 or 4.

- `commutator(other) -> LoopElement`

  Return the central element `c` with `x y = (y x) c`, where `y` is
  `other`.

- `associator(y, z) -> LoopElement`

  Return the central element `a` with `(x y) z = (x (y z)) a`.

---

## BinaryField and BinaryFieldElement

pyfastcd provides the finite fields of odd characteristic. `BinaryField`
provides F_(2^k) as a coefficient field for loop algebras and for `linalg`,
`codes` and `dyadic_convolve`. Its properties and scalar constructors,
listed below, have the names and meaning they have in
`pyfastcd.FiniteField`; it has no arrays or arithmetic functions of its
own.

### Constructors

```python
BinaryField(*, deg=1)
BinaryField.from_modulus(*, modulus)
```

`BinaryField(deg=k)` is F_(2^k) for `k` from 1 to 12, presented for `k >= 2`
by the Conway polynomial of degree `k`, which is also the default of GAP and
SageMath. `from_modulus` takes another monic irreducible polynomial over F_2
as a sequence of integers, constant coefficient first; entries are reduced
modulo 2. Fields are immutable, cannot be subclassed, and are cached by
their polynomial: two presentations of the same field are different parents,
and so are the loop algebras over them.

A degree or modulus coefficient that is not an integer, or is a boolean,
raises `TypeError`. A degree outside `1..12`, a leading coefficient other
than 1 and a reducible polynomial raise `ValueError`.

### Properties

- `char: int`, `deg: int`, `order: int`

  2, the extension degree `k`, and `2**k`.

- `modulus: tuple[int, ...]`

  The polynomial that presents the field, constant coefficient first. At
  degree 1 it is `(0, 1)`, as for the prime fields of pyfastcd.

- `dtype`, `coefficient_shape`

  `uint32`, and the shape of the coordinates of one scalar: `()` at degree
  1 and `(deg,)` otherwise. Each coordinate is 0 or 1.

### Scalars

- `F(value=0) -> BinaryFieldElement`, `F.scalar(value)`

  Create a scalar from an integer, which is reduced modulo 2 and lies in
  the prime subfield; from a sequence of `deg` polynomial coordinates,
  constant coefficient first; or from a scalar of `F`, which is returned.
  A wrong number of coordinates and a scalar of another field raise
  `ValueError`; a boolean, a float or a string raises `TypeError`.

- `F.zero()`, `F.one() -> BinaryFieldElement`

  Return the additive and the multiplicative identity.

A `BinaryFieldElement` is immutable and hashable, and is created only by
its field.

- `algebra`, `coefficients`

  The field, named as for pyfastcd scalars, and a read-only `uint32` array
  of the polynomial coordinates.

- `x + y`, `x - y`, `-x`, `x * y`, `x / y`, `x ** k`, `reciprocal()`

  Field arithmetic. An integer operand stands for an element of the prime
  subfield. Scalars of different fields raise `ValueError`, and division by
  zero raises `ZeroDivisionError`.

- `x == y`, `hash(x)`, `bool(x)`

  A scalar equals an integer only when it is 0 or 1 and the integer is the
  same; integers are not reduced for comparison, so that equality agrees
  with hashing.

---

## LoopAlgebra

### Constructor

```python
LoopAlgebra(loop, *, char=0, deg=1, field=None)
```

Create the loop algebra of `loop` over the field chosen by the selectors.
Algebras are immutable, cannot be subclassed, and are cached by loop and
coefficient field, so `LoopAlgebra(L, char=2, deg=3)` and
`LoopAlgebra(L, field=BinaryField(deg=3))` are the same object.

Every loop has a loop algebra in characteristic 2. Over the other fields
the signed components are held by pyfastcd, whose storage limits apply: a
factor of level 60 or more raises `OverflowError`, and finite fields reach
the limit a few levels earlier. For a central product the sign table of the
factor of smallest level is built when the algebra is created, or in
characteristic 2 when the algebra to which products are lifted is first
needed, so its memory grows with the square of that factor's order.

### Properties

- `loop`

  The loop.

- `dimension: int`

  The dimension of the algebra, `loop.order`.

- `char: int`, `deg: int`, `field`

  The characteristic and degree of the coefficient field, and the field
  object: a `pyfastcd.FiniteField` in odd characteristic, a `BinaryField`
  in characteristic 2, and `None` over R and C.

- `dtype`, `coefficient_shape`

  The storage type and the shape of one record, from the selector table.

- `cd_algebra`

  The `pyfastcd.Algebra` that holds the signed components of a
  Cayley–Dickson loop algebra; `None` for a central product and in
  characteristic 2.

- `algorithms: tuple[str, ...]`

  The accepted values of `algorithm` in `multiply`:
  `("auto", "split", "table")`, and `("auto", "lift", "table")` in
  characteristic 2.

- `default_algorithm: str`, `default_batch_algorithm: str`

  The algorithm that `"auto"` selects in `multiply` for a single product,
  and for a large batch. Both are `"split"` except in characteristic 2,
  where `"auto"` selects `"lift"` or `"table"` from the size of the loop,
  the number of products and the degree of the field (see
  [Choosing between the algorithms](mathematics.md#choosing-between-the-algorithms)).

- `signed_algorithms: tuple[str, ...]`

  The accepted values of `signed_algorithm` in `multiply`. For a
  Cayley–Dickson loop these are pyfastcd's algorithm names: `"auto"`,
  `"direct"`, `"quasilinear"` and, over R for `D` up to 1024,
  `"cariow_cariowa"`. For a central product they are `"auto"`, `"rows"`,
  `"schedule"` and, where it applies, `"native"` (see
  [The product of signed components](mathematics.md#the-product-of-signed-components)).
  In characteristic 2 the names are those of the loop algebra over the field
  to which products are lifted.

- `default_signed_algorithm: str`

  The method that `"auto"` selects for the signed components of a single
  product. For a Cayley–Dickson loop, a batch follows the batch choice of
  pyfastcd, `cd_algebra.default_batch_algorithm`.

### Construction

- `A(value=0) -> Element`

  Create an element from `value`, which is one of:

  - a field scalar `r`, giving `r` times the identity;
  - an `Element` with parent `A`, which is returned;
  - coefficients of shape `coefficient_shape`, from a nested sequence or an
    array.

  A `LoopElement`, a `set` or a `dict` raises `TypeError`; a wrong shape
  raises `ValueError`; complex coefficients over R and floats over a finite
  field raise `TypeError`.

- `zero()`, `one()`, `scalar(r) -> Element`

  Return the additive identity, the multiplicative identity `[e_0]`, and
  `r` times the identity.

- `basis(x) -> Element`

  Return the basis element `[x]` of a `LoopElement` `x` of `A.loop`.

- `from_terms(terms) -> Element`

  Return the combination of loop elements given by a mapping from
  `LoopElement` to field scalars, or by an iterable of such pairs.
  Coefficients of a repeated loop element add.

- `from_components(kappa, epsilon) -> Element | ElementArray`

  Return the element with group component `kappa` and signed component
  `epsilon`: `a_t = (c_t + d_t) / 2` and `b_t = (c_t - d_t) / 2`. `kappa`
  has shape `(D,)`, followed by the polynomial axis of an extension field.
  `epsilon` is a pyfastcd `Element` or `ElementArray` of `cd_algebra`, or
  coefficients of the shape of `kappa`; for a central product the
  coefficients may instead have one axis for each factor. Leading batch
  axes broadcast and give an `ElementArray`. Not defined in characteristic
  2.

- `lift_group(kappa)`, `lift_signed(epsilon) -> Element | ElementArray`

  Return the element with the given group component and signed component
  zero, and the reverse. Each map is an injective homomorphism of nonunital
  algebras onto an ideal. Not defined in characteristic 2.

- `split_array(spectral, signed) -> SplitArray`

  Return a `SplitArray` with the given parts, which are copied. Not defined
  in characteristic 2.

- `random(rng=None, *, batch_shape=None, secure=False) -> Element | ElementArray`

  Return a random element, or an `ElementArray` when `batch_shape` is given.
  Coefficients are independent: standard normal over R, standard normal
  real and imaginary parts over C, uniform residues over a finite field.
  `rng` is a `numpy.random.Generator` or a seed. With `secure=True` the
  residues of a finite field come from the operating system's
  cryptographic source through the `secrets` module and `rng` is ignored;
  over R and C it raises `ValueError`.

- `from_bytes(data) -> Element`

  Return the element encoded by `Element.to_bytes`. A wrong length, or a
  residue not below the characteristic, raises `ValueError`.

- `array(values) -> ElementArray`

  Copy `values` into a new array: an element, an array of the algebra, or
  coefficients of shape `batch_shape + coefficient_shape`.

- `asarray(values, *, copy=None) -> ElementArray`

  Share storage when `copy=None` and it is possible, always copy when
  `copy=True`, and require sharing when `copy=False`, raising `ValueError`
  when it is impossible. [Storage](#storage) gives the conditions.

### Products

- `multiply(x, y, *, out=None, algorithm="auto", signed_algorithm="auto")`

  Return the ordered product, broadcasting batch axes. One operand may be a
  field scalar, which scales the other.

  - `algorithm` is one of `algorithms`. `"split"` multiplies through the
    decomposition of the algebra. `"lift"`, its counterpart in
    characteristic 2, multiplies the operands by the split algorithm over
    a large prime field, or over an extension of it when the coefficient
    field is not F_2, and reduces the result (see
    [Multiplication by lifting](mathematics.md#multiplication-by-lifting)).
    `"table"` evaluates the definition in the loop basis with `O(D^2)`
    multiplications; it is available up to `D = 4096` and does not accept a
    `SplitArray`. `"auto"` selects `"split"`, and in characteristic 2
    whichever of `"lift"` and `"table"` is expected to be faster.
  - `signed_algorithm` is one of `signed_algorithms` and selects the method
    for the product of signed components under `"split"` and `"lift"`.
  - `out` is an ndarray or an `ElementArray` that receives a loop-basis
    result. It must have exactly the result's shape and the algebra's
    dtype, and be writable.

  The result is an `Element` when neither operand is an array, and otherwise
  an array of the operands' kind; with `out` it is `out` itself when that is
  an `ElementArray` and a new array around it when it is an ndarray. Names
  are case-sensitive: an unknown name raises `ValueError` and a name that is
  not a string raises `TypeError`.

- `left_operator_commutator(a, b, x)`

  Compute `a (b x) - b (a x)`.

- `right_operator_commutator(a, b, x)`

  Compute `(x b) a - (x a) b`.

### Idempotents and the center

The three methods that return idempotents are not defined in characteristic
2, where the only idempotents of the algebra are 0 and 1.

- `central_idempotents() -> tuple[Element, Element]`

  Return `(1 + [-1]) / 2` and `(1 - [-1]) / 2`: central orthogonal
  idempotents with sum 1, which project onto the group and the signed
  component.

- `primitive_idempotent(a) -> Element`

  Return the idempotent whose spectrum is 1 at coordinate `a` and 0
  elsewhere, with signed component zero.

- `primitive_idempotents() -> ElementArray`

  Return all `D` of them, with batch shape `(D,)`.

- `center_basis() -> ElementArray`

  Return a basis of the center of the algebra: the primitive idempotents,
  followed by `([e_t] - [-e_t]) / 2` for the central loop elements `e_t`.
  In characteristic 2 the basis is `[e_t]` for the central `e_t`, then
  `[-e_t]` for the same elements, then `[e_t] + [-e_t]` for the others.

### Maps

- `apply_automorphism(f, x)`

  Return the image of an element or `ElementArray` under the linear
  extension of a loop map `f`: the coefficient of `[y]` moves to `[f(y)]`.
  `f` must be a bijection of the loop, or `ValueError` is raised; that it
  preserves products is not checked.

- `character_automorphism(a, x)`

  Apply `e_t -> (-1)**popcount(a & t) e_t`: exchange the coefficients of
  `[e_t]` and `[-e_t]` wherever `popcount(a & t)` is odd. This is an
  algebra automorphism for every index `a`.

- `embed(x)`

  For a Cayley–Dickson loop `Q_n`: return the image of an element or array
  of the loop algebra of `Q_k`, `k <= n`, over the same field.

- `factor_embedding(i, x)`

  For a central product: return the image of an element or array of the
  loop algebra of factor `i`, over the same field. The map is an injective
  homomorphism of unital algebras.

### Linear algebra

- `left_matrix(x, *, basis="loop")`, `right_matrix(x, *, basis="loop")`

  Return the matrix of `y -> x y`, or of `y -> y x`, for one element. With
  `basis="loop"` the matrix `M` satisfies
  `(x * y).coefficients.ravel() == M @ y.coefficients.ravel()`: up to
  rounding over R and C, and exactly, with the right side reduced modulo
  `char`, over prime fields. With
  `basis="split"` it acts on the spectral coordinates followed by the
  signed coordinates and is block diagonal; this basis is not defined in
  characteristic 2. Over an extension field the matrix has a trailing
  polynomial axis.

- `structure_constants() -> numpy.ndarray`

  Return the `int8` array whose entry `[i, j, k]` is 1 when the product of
  the basis elements at positions `i` and `j` is the one at position `k`.
  Dimensions above 256 raise `ValueError`.

- `subalgebra_generated_by(x) -> GeneratedSubalgebra`

  Describe the unital subalgebra generated by one element. Defined where
  powers do not depend on bracketing: for a Cayley–Dickson loop, and for
  every loop in characteristic 2.

### GeneratedSubalgebra

The record that `subalgebra_generated_by` returns. Its attributes cannot
be reassigned: `dimension`, `minimal_polynomial`,
`spectral_values` (the distinct spectral coordinates of the element) and
`signed_polynomial` (the minimal polynomial of its signed component). The
subalgebra is isomorphic to `F[t] / (minimal_polynomial)`. In
characteristic 2 the last two attributes are `None`, and the minimal
polynomial is a power of `t - aug(x)` of degree at most 4.

---

## Element, ElementArray and SplitArray

### Construction

Create an `Element` with `A(...)` or another constructor of the algebra, an
`ElementArray` with `A.array`, `A.asarray` or `A.random`, and a
`SplitArray` with `X.split()` or `A.split_array`. Constructing any of the
three classes directly raises `TypeError`. In characteristic 2 there is no
split form, and so no `SplitArray`.

Elements are immutable and hashable. Arrays are unhashable and are writable
exactly when their NumPy storage is.

### Which operands combine

| Receiver | Accepts | Result |
|---|---|---|
| `Element` | `Element`, field scalar | `Element` |
| `ElementArray` | `ElementArray`, `Element`, field scalar | `ElementArray` |
| `SplitArray` | `SplitArray`, `Element`, field scalar | `SplitArray` |

Combining an `ElementArray` with a `SplitArray` raises `TypeError` in every
operator, method, comparison and assignment. The methods of an `Element`
take `Element` operands only; to combine an element with an array, call the
method on the array or use an operator.

### Properties

- `algebra`, `dtype`, `dimension`

  The parent, its storage type and its dimension.

- `shape`

  `coefficient_shape` for an element and
  `batch_shape + coefficient_shape` for an `ElementArray`.

- `batch_shape`

  The leading axes of an array.

- `coefficients`

  The loop-basis coefficients: a read-only array for an element, and the
  storage itself for an `ElementArray`.

- `SplitArray.spectral`, `SplitArray.signed`

  The two parts of a `SplitArray`, sharing its storage. `spectral` is an
  ndarray over R and C and a pyfastcd array of field scalars over a finite
  field. `signed` is a `pyfastcd.ElementArray` for a Cayley–Dickson loop
  and, for a central product, an ndarray of shape `batch_shape + (D,)`
  followed by the polynomial axis of an extension field.

### Conversion

- `to_numpy(*, copy=True)`

  Return the loop-basis coefficients of an element or `ElementArray`; with
  `copy=False` a read-only view of an element, or the storage of an array.

- `copy()`

  Return an equal object with independent storage.

- `Element.to_bytes() -> bytes`

  Return a canonical encoding of the coefficients in the order of
  `coefficients.ravel()`: little-endian binary64 values over R and C, and
  big-endian residues of the smallest whole number of bytes that holds
  `char - 1` over a finite field.

- `ElementArray.split() -> SplitArray`, `SplitArray.to_loop() -> ElementArray`

  Convert between the loop basis and split form. `split()` is not defined
  in characteristic 2.

### Decomposition

- `kappa()`

  Return the group component `(a_t + b_t)`. The map is a homomorphism onto
  the group algebra in every characteristic.

- `spectrum()`

  Return the Walsh–Hadamard transform of the group component. Not defined
  in characteristic 2.

- `epsilon()`

  Return the signed component `(a_t - b_t)`. Not defined in characteristic
  2, where it would coincide with the group component.

- `Element.augmentation()`

  Return the sum of all coefficients, a field scalar. The map is a
  homomorphism to the field.

| Method | R, C | Odd characteristic | Characteristic 2 |
|---|---|---|---|
| `kappa()` | ndarray | pyfastcd array of field scalars | ndarray of residues |
| `spectrum()` | ndarray | pyfastcd array of field scalars | not defined |
| `epsilon()`, Cayley–Dickson loop | pyfastcd `Element` or `ElementArray` | the same | not defined |
| `epsilon()` of an element, central product | read-only ndarray with one axis for each factor | the same, followed by the polynomial axis of an extension field | not defined |
| `epsilon()` of an array, central product | ndarray of shape `batch_shape + (D,)` | the same, followed by the polynomial axis of an extension field | not defined |
| `augmentation()`, `coefficient()` | `float` or `complex` | pyfastcd field scalar | `BinaryFieldElement` |

### Operators

Results are new objects, and arrays broadcast their batch axes.

- `+x`, `-x`

  The element itself (a copy of an array), and its negative.

- `x + y`, `x - y`, `x + r`, `r - x`, ...

  Add or subtract coefficients; a scalar `r` stands for `r` times the
  identity.

- `x * y`

  The ordered product, by the default algorithm.

- `r * x`, `x * r`

  Scale every coefficient.

- `x / r`

  Divide by a field scalar. Division by zero raises `ZeroDivisionError`.

- `x ** k`

  An integer power, where powers do not depend on bracketing: in the loop
  algebra of a Cayley–Dickson loop, and in every loop algebra of
  characteristic 2. Negative powers use the inverse. Any Python integer is
  accepted over a finite field; over R and C an exponent too large to
  convert to a `float` raises `OverflowError`. The cost grows with `log k`,
  and in characteristic 2 a power takes at most two products. In the loop
  algebra of a central product over R, C or a field of odd characteristic
  the operator raises `TypeError`. An exponent that is not an integer, or is
  a boolean, raises `TypeError`.

- `x == y`, `x != y`

  Compare exactly. For an `Element` the result is a `bool`; for an array it
  is a Boolean array over the broadcast batch shape.

  - An element or record equals a field scalar `r` when it is `r` times the
    identity. Over a finite field, integers are not reduced for comparison,
    so that equality agrees with hashing.
  - An object with a different parent compares unequal everywhere; the
    batch shapes of two arrays must still broadcast.
  - Elements and `ElementArray` records are compared by loop-basis
    coefficients, and `SplitArray` records by their two parts.
  - For any other operand the comparison returns `NotImplemented`, so
    ordinary objects compare unequal.

- `hash(x)`, `bool(x)`

  Equal elements hash equal, and an element equal to a field scalar hashes
  like that scalar. `bool(x)` is false only for the zero element; `bool` of
  an array raises `ValueError`.

NumPy universal functions are not supported: `__array_ufunc__` is disabled,
so that expressions such as `numpy.float64(2) * x` reach the operators
above. An ndarray that is not a zero-dimensional scalar is not an operand,
and combining one with an element or array raises `TypeError`; wrap it with
`A.asarray` first.

### Methods

These methods allocate new results. Operand rules are those of
[Which operands combine](#which-operands-combine); unlike the operators,
methods do not accept field scalars.

- `conjugate()`

  Return the image under the linear extension of `[x] -> [x^-1]`, which
  reverses products. It fixes the group component and conjugates the signed
  component. It does not conjugate complex coefficients or apply the
  Frobenius map.

- `square()`

  Return `x x`.

In the next three methods `y` stands for the operand `other`.

- `commutator(other)`

  Return `x y - y x`.

- `anticommutator(other)`

  Return `x y + y x`.

- `jordan_product(other)`

  Return `(x y + y x) / 2`. Not defined in characteristic 2.

- `associator(y, z)`

  Return `(x y) z - x (y z)`.

- `left_power(exponent)`, `right_power(exponent)`

  Return `x (x (... (x x)))` and `(((x x) x) ...) x` with `k = exponent`
  factors, for an integer `k >= 0`; `k = 0` gives the identity. They use
  `k - 1` products and are defined in every loop algebra. A negative
  exponent raises `ValueError`.

- `inverse()`

  Return the two-sided inverse. Over R, C and fields of odd characteristic
  it is defined for a Cayley–Dickson loop, and exists when every spectral
  coordinate is nonzero and the signed component has `Q(d) != 0`. In
  characteristic 2 it is defined for every loop, exists when the
  augmentation is nonzero, and equals `x**3 / aug(x)**4`. Where no inverse
  exists, `ZeroDivisionError` is raised. A batch checks every record before
  producing any result.

- `weight()`

  Return the number of nonzero loop-basis coefficients: an `int` for an
  element and for an array with an empty batch shape, and otherwise an
  integer array over the batch.

- `distance(other)`

  Return the number of loop-basis coefficients in which two operands
  differ.

For a Cayley–Dickson loop over R, C or a field of odd characteristic,
`square`, `anticommutator`, `jordan_product` and `conjugate` cost `O(D)`
operations on operands in split form, and `x ** k` costs `O(D + log k)`
beyond the coordinatewise powers of the spectrum.

### Structure

Each predicate returns a `bool` for an element and a Boolean array over the
batch for an array, which is a NumPy Boolean scalar when the batch shape is
empty.

- `has_inverse()`

  Return whether `inverse()` exists. Where powers are defined.

- `is_zero_divisor(side=None)`

  Return whether a nonzero `y` has `x y = 0` (`side="left"`), `y x = 0`
  (`side="right"`), or either (`None`). The zero element counts as a zero
  divisor. For a Cayley–Dickson loop of level at most 3 the test costs
  `O(D)` operations; otherwise it computes the rank of a `D x D` matrix for
  each record. In characteristic 2 the zero divisors on either side are the
  elements of augmentation zero, and the test costs `O(D)` operations for
  every loop.

- `is_idempotent()`

  Return whether `x x = x`, compared exactly: in split form, and in
  characteristic 2 in the loop basis.

- `is_nilpotent()`

  Return whether some power of `x` vanishes. Where powers are defined.

- `is_central()`

  Return whether `x` commutes and associates with every element of the
  algebra.

The following are methods of `Element` only.

- `coefficient(x)`

  Return the coefficient of the basis element `[x]` of a `LoopElement`.

- `terms() -> dict`

  Return the nonzero coefficients, keyed by `LoopElement`.

- `support() -> frozenset`

  Return the loop elements with a nonzero coefficient.

- `left_annihilator()`, `right_annihilator() -> ElementArray`

  Return a basis of `{y : y x = 0}` and of `{y : x y = 0}`. The computation
  finds the null space of a `D x D` matrix, and of a `2D x 2D` matrix in
  characteristic 2. Over R and C the null space is numerical, with the
  tolerance of `linalg.nullspace`.

- `minimal_polynomial() -> numpy.ndarray`

  Return the monic polynomial of least degree that vanishes at `x`, as
  field coefficients with the constant term first. Where powers are
  defined. Over R and C, coincidences between spectral values are detected
  by exact comparison of floating-point numbers. In characteristic 2 the
  polynomial is `(t - aug(x))**j` for some `j` from 1 to 4.

Methods marked "where powers are defined" are available in the loop algebra
of a Cayley–Dickson loop, and in every loop algebra of characteristic 2.
In the loop algebra of a central product over R, C or a field of odd
characteristic they raise `TypeError`.

### Indexing and iteration

- Indexing and iterating an array act on its batch axes. `X[i]` and `S[i]`
  return an `Element` copy when they select one record, and otherwise an
  array that shares storage where NumPy indexing does. An index cannot
  reach the coefficient axes: too many indices raise `IndexError`.
- `X[key] = value` assigns a field scalar, an `Element`, an `ElementArray`
  or a coefficient array. `S[key] = value` assigns a field scalar, an
  `Element` or a `SplitArray`.
- `len(X)` is the length of the first batch axis. As in NumPy, an array
  with an empty batch shape has no length and cannot be iterated.
- An `Element` is not a sequence; read its coefficients through
  `coefficients`, `coefficient` or `terms`.

---

## Characteristic 2

Over a `BinaryField` the loop algebra has no decomposition into a group
component and a signed component, and its elements behave differently (see
[Characteristic 2](mathematics.md#characteristic-2) in the conventions).
Loops, loop elements and everything not listed here work as over the other
fields.

| | Members |
|---|---|
| Not defined; raise `TypeError` | `SplitArray`, `ElementArray.split`, `LoopAlgebra.split_array`, `from_components`, `lift_group`, `lift_signed`; `spectrum`, `epsilon`; `central_idempotents`, `primitive_idempotent`, `primitive_idempotents`; `jordan_product`; `left_matrix` and `right_matrix` with `basis="split"`; `wht` with a `BinaryField` |
| Defined for every loop, central products included | `x ** k`, `inverse`, `has_inverse`, `is_nilpotent`, `minimal_polynomial`, `subalgebra_generated_by` |
| Different algorithm names | `multiply` takes `"auto"`, `"lift"` or `"table"`; `"split"` raises `ValueError`; `signed_algorithms` and `default_signed_algorithm` are those of the algebra to which products are lifted |
| Different contents | `center_basis` returns the basis described with it |
| Different result types | `kappa` returns an ndarray; `augmentation`, `coefficient` and the values of `terms` are `BinaryFieldElement` scalars; `cd_algebra` is `None`; `GeneratedSubalgebra.spectral_values` and `signed_polynomial` are `None` |

Every operation is exact. The results of `"lift"` and `"table"` are
identical, so `"auto"` chooses between them by expected running time, from
the quotient order `D`, the number of products in the call and the degree of
the field; `default_algorithm` and `default_batch_algorithm` give the choice
for one product and for a large batch, and
[Choosing between the algorithms](mathematics.md#choosing-between-the-algorithms)
states the rule. An `Element` keeps the transforms computed by its first
lifted product through `*` or `multiply`, and later such products reuse
them.

---

## pycdloop.linalg

Functions on matrices over a coefficient field. Without `field` a matrix is
real or complex and the results are numerical; with a finite field, a
`pyfastcd.FiniteField` or a `BinaryField`, the entries are residues, with a
trailing polynomial axis for an extension field, and the results are exact.

- `rank(matrix, *, field=None, tol=None) -> int`

  Return the rank. Over R and C it is the number of singular values above
  `tol`, which defaults to the largest singular value times
  `max(rows, columns)` times the machine epsilon.

- `row_reduce(matrix, *, field=None, tol=None) -> tuple[numpy.ndarray, tuple[int, ...]]`

  Return the reduced row echelon form and the pivot column of each nonzero
  row. Over R and C elimination uses partial pivoting, and entries of a
  pivot column at or below `tol` count as zero.

- `nullspace(matrix, *, field=None, tol=None) -> numpy.ndarray`

  Return an array whose rows are a basis of the vectors `v` with
  `matrix @ v = 0`. Over R and C the rows are orthonormal.

A matrix that does not have two index axes raises `ValueError`, and a
`field` that is not a field object raises `TypeError`.

---

## pycdloop.codes

A subspace of a loop algebra is a linear code of length `len(loop)` through
the loop-basis coefficients. Matrices have one row for each codeword and
one column for each loop element in position order, followed over an
extension field by the polynomial axis.

- `translate_matrix(x, *, side="left") -> numpy.ndarray`

  Return the matrix whose rows are `[g] x` (`"left"`) or `x [g]`
  (`"right"`) for `g` running over the loop.

- `generator_matrix(elements, *, side="left", closure=True, tol=None) -> numpy.ndarray`

  Return a generator matrix in reduced row echelon form of the code
  generated by one element or an iterable of elements. With `closure=True`
  the code is the one-sided ideal they generate; because the algebra need
  not be associative, translates are added until the dimension stops
  growing. With `closure=False` the code is the span of the translates of
  the generators.

- `to_gap(matrix, *, field) -> str`, `to_sage(matrix, *, field) -> str`

  Return a GAP expression (`GeneratorMatCode` of the GUAVA package) or a
  SageMath expression (`LinearCode`) for the code with the given generator
  matrix over a prime field: a `pyfastcd.FiniteField` of degree 1, or
  `BinaryField()` for binary codes. Other fields raise `ValueError`.

---

## pycdloop.testing

Sampled identity checks. Each draws `samples` random elements and returns
whether the identity held on all of them: `False` exhibits a counterexample
and is conclusive, `True` only reports that none was found. Over R and C
the samples have small integer coefficients, so the checks are exact.

- `random_elements(algebra, count, rng=None, *, exact=True) -> ElementArray`

  Return `count` random elements. With `exact=True`, coefficients over R,
  and the real and imaginary parts of coefficients over C, are integers
  between -3 and 3.

- `is_commutative(algebra, samples=32, rng=None)`
- `is_associative(algebra, samples=32, rng=None)`
- `is_alternative(algebra, samples=32, rng=None)`
- `is_flexible(algebra, samples=32, rng=None)`
- `is_power_associative(algebra, samples=32, rng=None)`

  Check `x y = y x`; `(x y) z = x (y z)`; `(x x) y = x (x y)` and
  `(y x) x = y (x x)`; `(x y) x = x (y x)`; and `(x x) x = x (x x)` with
  `((x x) x) x = (x x)(x x)`.

---

## Arrays and NumPy

### Storage

- **Shape.** An `ElementArray` has shape `batch_shape + coefficient_shape`.
  Only the batch axes broadcast.
- **Sharing.** `A.asarray` shares the storage of an `ElementArray` of the
  algebra, and of an ndarray with the algebra's dtype that, over a finite
  field, holds residues below the characteristic. Any other input is
  converted and copied, unless `copy=False` forbids it.
- **Reduction.** When finite-field coefficients are copied, integers are
  reduced modulo `char`.
- **Layout.** Strided and broadcast views are accepted. Operations copy
  operands that are not C-contiguous, and operands whose layout pyfastcd
  cannot read in place.

### NumPy interfaces

- `Element.__array__` returns the read-only coefficients, and
  `ElementArray.__array__` the storage, unless a dtype change or
  `copy=True` requires a copy. `np.asarray(X)` therefore shares storage
  with `X`.
- Universal functions are not supported; see [Operators](#operators).

### Copying and serialization

Pickling, `copy.copy` and `copy.deepcopy` of arrays create independent
storage. Elements are immutable, so `copy.copy` returns the element itself.
A pickled element records the form it was created with. Loops and algebras
pickle to the same cached objects.

---

## Exceptions

- `TypeError`

  An unsupported operand or scalar type, including a scalar of another
  field as an operand; an `ElementArray` combined with a
  `SplitArray`; direct construction of `Element`, `ElementArray`,
  `SplitArray` or `BinaryFieldElement`; `**`, `inverse`, `has_inverse`,
  `is_nilpotent`, `minimal_polynomial` or `subalgebra_generated_by` in the
  loop algebra of a central product outside characteristic 2; an operation
  that is not defined in characteristic 2; a non-integer exponent, index,
  level or degree; a `field` that is not a field object, or is combined
  with `char` or `deg`; a non-string algorithm name; a wrong `out` dtype.

- `ValueError`

  Mismatched parents; two `BinaryFieldElement` scalars of different fields,
  or a scalar of another field given as a coefficient; a wrong coefficient
  shape; a broadcast failure; an unknown algorithm name, including `"split"`
  in characteristic 2 and `"lift"` elsewhere; a subset that is not a subloop
  where one is required; a map that is not a bijection; a table requested
  for too large a loop; a degree out of range or a reducible modulus of a
  `BinaryField`; a lifted product for a loop beyond the bound of
  [Multiplication by lifting](mathematics.md#multiplication-by-lifting);
  `secure=True` over R or C; the truth value of an array.

- `IndexError`

  A basis index out of range, or too many indices for the batch axes of an
  array.

- `ZeroDivisionError`

  Division by a zero scalar, or the inverse or a negative power of an
  element that has none.

- `OverflowError`

  A total doubling level above 62; `len` of a loop of total level 62; a
  loop algebra outside characteristic 2 for a loop beyond the storage
  limits of pyfastcd; over R and C, an exponent of `**` too large to
  convert to a `float`.

- `numpy.linalg.LinAlgError`

  Over R and C, a rank or null space of a matrix with NaN or infinite
  entries: in `linalg`, the annihilators, and `is_zero_divisor` where it
  computes a rank.

Operators and products do not raise on NaN, infinity or overflow of
coefficients over R and C.

---

## Threads

Loops, algebras, fields, scalars and loop elements are immutable and can be
shared between threads. An `Element` derives its second form, or in
characteristic 2 the transforms of its lifted coefficients, on first use
and stores it; two threads that do so at once compute the same value, so
sharing elements is safe under the global interpreter lock. Do not modify
the storage of an array while an operation that uses it is running: the
compiled kernels of pycdloop release the global interpreter lock, the
transform on large batches and the change of coordinates always, and those
of pyfastcd release it on large batches, so other threads can run during
them. The
temporaries that a product of arrays reuses are kept separately by each
thread. pyfastcd does not support free-threaded CPython builds, and neither
does pycdloop.

---

## Versioning and stability

pycdloop follows semantic versioning. The public surface is the names in
`pycdloop.__all__` together with the members documented on this page;
modules and names with a leading underscore are private and may change
without notice. Before version 1.0.0, a minor release may change public
behavior, and every such change is recorded in `CHANGELOG.md`.

Two kinds of difference may come with any release that improves
performance or correctness: floating-point results may round differently,
and `"auto"` may select a different algorithm.

---

## Typing

pycdloop ships inline type annotations, a stub for its compiled module and
a `py.typed` marker. Loops, loop elements, `BinaryField` and the functions
of `linalg`, `codes` and `testing` have concrete return annotations, except
the sign functions `twist`, `commutator_sign` and `associator_sign`, whose
result depends on the arguments; most parameters are annotated `Any`. The
arithmetic of `Element`, `ElementArray` and
`SplitArray` is shared between the three classes and annotated with `Any`,
since the result type depends on the operand types and, for the
decomposition methods, on the coefficient field.
