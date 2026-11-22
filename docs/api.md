# Python API reference

pycdloop is a Python package for arithmetic in Cayley-Dickson loops, their
central products, and their loop algebras over the real numbers, the complex
numbers, and finite fields. Loops and their elements are immutable Python
objects, and the elements of a loop algebra are NumPy-compatible classes whose
products are computed by pyfastcd and by a small compiled extension of pycdloop;
errors are reported as Python exceptions.

This page documents every public name of pycdloop and the rules that govern
them, while the [quick start](quickstart.md) introduces them by example and is
the place to begin, and [Conventions and results](mathematics.md) defines the
mathematical objects they model. The public names are `CDLoop`,
`CentralProduct`, `LoopElement`, `LoopAlgebra`, `BinaryField`,
`BinaryFieldElement`, `Element`, `ElementArray`, `SplitArray`,
`GeneratedSubalgebra`, `wht`, `dyadic_convolve`, `__version__`, and the modules
`linalg`, `codes`, and `testing`, and modules whose names begin with an
underscore are private. After the dependencies and the terms used throughout,
the page documents the module functions, then the classes in the order a program
meets them, `CDLoop` and `CentralProduct`, `LoopElement`, `BinaryField`,
`LoopAlgebra`, and the three classes of algebra elements, then the differences
in characteristic 2 and the modules `linalg`, `codes`, and `testing`, and
finally the storage of arrays and their NumPy interfaces, floating point,
exceptions, threads, versioning, and typing.

---

## Dependencies

pycdloop depends on:

- CPython 3.10 or newer;
- NumPy 1.26 or newer;
- pyfastcd 1.4.0 or a later 1.x release.

pyfastcd provides the Cayley-Dickson products and the finite fields of odd
characteristic, while the compiled extension of pycdloop, which computes the
Walsh-Hadamard transform and the change of coordinates, is built into the
pycdloop wheels.

---

## Terms and notation

The rest of this page uses the following terms.

**Loops.** A *loop* is a `CDLoop` or a `CentralProduct`. Its elements are the
signed basis elements `+e_t` and `-e_t`, where `t` is a *flat index* in
`range(D)` and `D = loop.quotient_order`, and for a central product the flat
index concatenates the binary digits of one index for each factor, first factor
first. The *position* of `+e_t` is `t` and that of `-e_t` is `D + t`, and loops
iterate in position order.

**Coefficient field and selectors.** Every loop algebra has its coefficients in
R, C, or a finite field F_(p^d) (see [Loop
algebras](mathematics.md#loop-algebras)). Three keyword arguments, the
*selectors* `char`, `deg`, and `field`, choose the field with the meaning they
have in pyfastcd:

| Selectors | Coefficients | `dtype` | `coefficient_shape` |
|---|---|---|---|
| omitted, or `char=0, deg=1` | R | `float64` | `(2, D)` |
| `char=0, deg=2` | C | `complex128` | `(2, D)` |
| `char=p, deg=1` | F_p | `uint32` | `(2, D)` |
| `char=p, deg=d`, `d > 1` | F_(p^d) | `uint32` | `(2, D, d)` |
| `field=F` | the finite field `F` | `uint32` | `(2, D)` or `(2, D, F.deg)` |

- For an odd prime `p` the field is a `pyfastcd.FiniteField`, and the rules for
  the selectors, including the accepted characteristics and degrees and the
  exceptions for invalid values, are those of `pyfastcd.Algebra`.
- For `p = 2` the field is a
  [`BinaryField`](#binaryfield-and-binaryfieldelement), which pycdloop provides,
  and `char=2, deg=k` selects `BinaryField(deg=k)` for `k` from 1 to 12.
- `field` takes a field of either kind and, as in pyfastcd, is passed on its
  own: combining it with `char` or `deg` raises `TypeError`.

**Characteristic 2.** Some operations depend on the characteristic. Wherever
this page says that something is not defined in characteristic 2, it raises
`TypeError` there, and [Characteristic 2](#characteristic-2) lists every
difference in one place.

**Scalars.** A *field scalar* is a value of the coefficient field. Over R it is
a real number, which may be an `int`, `float`, `Fraction`, `Decimal`, real NumPy
scalar, or zero-dimensional array, and over C it may also be complex. Over a
finite field it is an `int` or a scalar of the same field, a `pyfastcd` scalar
in odd characteristic and a `BinaryFieldElement` in characteristic 2, and
integers are reduced modulo `char` in arithmetic and construction, while floats
are rejected.

**Parent.** Every element and array belongs to the `LoopAlgebra` that created
it, which `x.algebra` returns. Operations combine only objects with the same
parent, and field scalars, so nothing converts between parents, and a
`LoopElement` is never converted to an algebra element implicitly.

**Record.** A *record* is the coefficient storage of one element, and its shape
is `coefficient_shape`: row 0 holds the coefficients `a_t` of the basis elements
`[e_t]` and row 1 the coefficients `b_t` of `[-e_t]`, followed over F_(p^d) by
an axis of `d` polynomial coordinates. An `ElementArray` has shape
`batch_shape + coefficient_shape`, where the *batch shape* consists of the
leading axes.

**Split form.** Outside characteristic 2 an element can also be held in *split
form*, the pair of the Walsh-Hadamard transform of `(a_t + b_t)` and the array
`(a_t - b_t)`, each of shape `(D,)` followed by the polynomial axis of an
extension field (see [Split form](mathematics.md#split-form)). Products in split
form need no transform.

Below, `L` is a loop, `A` a loop algebra, `x`, `y`, and `z` are elements, `X` is
an `ElementArray`, `S` a `SplitArray`, and `r` a field scalar.

---

## Module contents

- `__version__: str`

  The version of pycdloop.

- `wht(a, *, axis=-1, field=None) -> numpy.ndarray`

  Return the Walsh-Hadamard transform `H a` along one axis.

  - **Parameters:** `a` is array-like, and the length of the axis `axis` is a
    power of two. Without `field` the input is converted to `float64`, or to
    `complex128` when it is complex, and with a `pyfastcd.FiniteField` it holds
    residues, with a trailing polynomial axis for an extension field that `axis`
    does not count.
  - **Returns:** a new ndarray of the shape of `a`.
  - **Notes:** The transform is not normalized, so applying it twice multiplies
    by the length. A length that is not a power of two or an axis out of range
    raises `ValueError`, and nonnumeric input raises `TypeError`, as does a
    `BinaryField`, over which the transform is not invertible.

- `dyadic_convolve(a, b, *, field=None) -> numpy.ndarray`

  Return `c[k] = sum(a[i] * b[i ^ k])` along the last coefficient axis, which is
  the product in the group algebra of the elementary abelian 2-group.

  - **Parameters:** `a` and `b` are array-like with the same power-of-two length
    along that axis, and their leading axes broadcast. `field` is a
    `pyfastcd.FiniteField` or a `BinaryField`, and without it the operands are
    real, or complex when either is.
  - **Returns:** a new ndarray with the broadcast shape.
  - **Notes:** Over an extension field the convolved axis is the one before the
    polynomial axis. Operands of different lengths, or of a length that is not a
    power of two, raise `ValueError`.

---

## CDLoop and CentralProduct

The two loop classes describe the loops themselves, with no coefficient field:
their elements, their sign functions, their substructures and maps, and tests of
their identities. Elements of the loop algebra over a field are created through
[`LoopAlgebra`](#loopalgebra).

### Constructors

```python
CDLoop(n)
CentralProduct(n1, n2, ...)
```

`CDLoop(n)` is the Cayley-Dickson loop `Q_n` for a doubling level `n >= 0`, and
`CentralProduct(n1, n2, ...)` is the central product of `Q_n1, Q_n2, ...`, which
takes at least two levels, each at least 1. Loops are immutable and cached, so
`CDLoop(3) is CDLoop(3)`.

A level that is not an integer, or is a boolean, raises `TypeError`, and so does
a central product with fewer than two levels, while a level below the minimum
raises `ValueError` and a total level above 62 raises `OverflowError`.

### Properties

- `levels: tuple[int, ...]`

  The doubling level of each factor, which is `(n,)` for `CDLoop(n)`.

- `level: int`

  The sum of the levels.

- `quotient_order: int`

  `D = 2**level`, the number of basis elements and the order of the quotient of
  the loop by `{+1, -1}`.

- `order: int`

  `2 * quotient_order`, which is also `len(L)`. At total level 62 the order is
  `2**63`, which `len` cannot return, so `len` raises `OverflowError`.

### Elements

- `L(index, sign=1) -> LoopElement`

  Return `sign * e_index`, raising `IndexError` for an index outside `range(D)`
  and `ValueError` for a sign other than `+1` or `-1`.

- `one()`, `minus_one() -> LoopElement`

  Return the identity and the central element `-1`.

- `basis(*indices) -> LoopElement`

  Return a positive basis element from its flat index, or from one index for
  each factor.

- `flat_index(*indices) -> int`, `factor_indices(index) -> tuple[int, ...]`

  Convert between one index for each factor and the flat index.

- `random(rng=None) -> LoopElement`

  Return a uniformly random element, where `rng` is a `numpy.random.Generator`
  or a seed.

- `len(L)`, `iter(L)`, `x in L`

  The order, iteration in position order, and membership.

### Sign functions

The sign functions give the signs of the multiplication of basis elements (see
[The loop](mathematics.md#the-loop)). Their arguments are flat indices or
integer arrays of them, which broadcast, and the result is an `int` when every
argument is an integer or a zero-dimensional array and an `int8` array
otherwise. Indices out of range raise `IndexError`.

- `twist(s, t)`

  Return `sigma(s, t)`, the sign in `e_s e_t = sigma(s, t) e_(s XOR t)`.

- `commutator_sign(s, t)`

  Return the sign `comm(s, t)` with `e_s e_t = comm(s, t) e_t e_s`.

- `associator_sign(s, t, u)`

  Return the sign `assoc(s, t, u)` with `(e_s e_t) e_u = assoc(s, t, u) e_s (e_t
  e_u)`.

- `twist_table() -> numpy.ndarray`

  Return the `D x D` matrix of twists, as `int8`.

- `cayley_table() -> numpy.ndarray`

  Return the multiplication table, whose entry `[i, j]` is the position of the
  product of the elements at positions `i` and `j`.

Both tables raise `ValueError` when `D` exceeds 4096, while the sign functions
have no such limit.

### Substructure

Subsets of a loop are passed as iterables of `LoopElement` and returned as
`frozenset` objects.

- `center()`, `commutant()`, `nucleus(side=None)`

  Return the center, the commutant, and a nucleus, where `side` is `"left"`,
  `"middle"`, or `"right"` for the elements that associate with every pair from
  that position of the associator, and `None` for the intersection of the three.
  They examine every index, so their time and memory grow with `D`.

- `generate(elements)`

  Return the subloop generated by `elements`.

- `is_subloop(elements) -> bool`

  Return whether the elements contain the identity and are closed under
  multiplication.

- `is_normal(elements) -> bool`

  Return whether a subloop `S` satisfies `xS = Sx`, `x(yS) = (xy)S`, and
  `(Sx)y = S(xy)` for all `x` and `y`, raising `ValueError` for elements that do
  not form a subloop.

- `cosets(elements, *, side="left")`

  Return the distinct cosets `xS` or `Sx` of a subloop, as a `frozenset` of
  `frozenset` objects, raising `ValueError` for elements that do not form a
  subloop. The cosets of a normal subloop partition the loop, while those of
  other subloops can overlap.

### Maps

A map of the loop to itself is given as a callable on `LoopElement` objects, or
as an integer array with the position of the image of each element in position
order.

- `left_translation(x)`, `right_translation(x) -> numpy.ndarray`

  Return the permutations `y -> x y` and `y -> y x`, by position.

- `character_automorphism(a) -> numpy.ndarray`

  Return the automorphism `e_t -> (-1)**popcount(a & t) e_t` for an index `a`,
  by position.

- `is_automorphism(f) -> bool`

  Return whether `f` is a bijection that preserves products.

- `CDLoop.embed(x) -> LoopElement`

  Return the image in `Q_n` of an element of `Q_k`, `k <= n`, which has the same
  index and sign.

- `CentralProduct.factor_embedding(i, x) -> LoopElement`

  Return the image of an element `x` of `CDLoop(levels[i])`, which has the index
  of `x` in factor `i` and index 0 elsewhere.

### Identity tests

- `is_commutative()`, `is_associative() -> bool`

  Test the identity exactly, never by sampling. Both examine every index, so
  their time and memory grow with `D`.

- `is_moufang(samples=None)`, `is_diassociative(samples=None) -> bool`

  Test `(xy)(zx) = (x(yz))x`, and that every two elements generate a group.

  - **Parameters:** With `samples=None` the test covers every tuple of indices
    when there are at most `2**22` of them and `2**18` pseudorandom tuples
    otherwise, and a positive integer `samples` tests that many pseudorandom
    tuples, whatever the size of the loop.
  - **Returns:** `False` when a counterexample was found, and otherwise `True`,
    which for a sampled test means only that no counterexample was found.
  - **Notes:** The tuples come from a fixed seed, so repeated calls test the
    same ones.

### Export

- `to_gap() -> str`

  Return a GAP expression that builds the loop with `LoopByCayleyTable` of the
  LOOPS package, with elements numbered from 1 in position order. Like the
  tables, it raises `ValueError` when `D` exceeds 4096.

---

## LoopElement

```python
LoopElement(loop, index, sign=1)
```

A `LoopElement` is one signed basis element of a loop, immutable and hashable,
and is usually created as `L(index, sign)`.

### Properties

- `loop`, `index: int`, `sign: int`

  The parent loop, the flat basis index, and `+1` or `-1`.

- `indices: tuple[int, ...]`

  The basis index in each factor.

- `position: int`

  The position in the iteration order of the loop.

### Operations

Operands must belong to the same loop, and elements of different loops raise
`ValueError`.

- `x * y`, `-x`, `+x`

  The loop product, `x` times the central element `-1`, and `x` itself.

- `x ** k`

  An integer power, which uses the inverse for negative exponents. The square of
  an element is central, so powers do not depend on bracketing.

- `x == y`, `hash(x)`

  Elements are equal when they have the same loop, index, and sign, and equal
  elements hash equal.

- `inverse() -> LoopElement`

  Return the two-sided inverse.

- `order() -> int`

  Return the order, which is 1, 2, or 4.

- `commutator(other) -> LoopElement`

  Return the central element `c` with `x y = (y x) c`, where `y` is `other`.

- `associator(y, z) -> LoopElement`

  Return the central element `a` with `(x y) z = (x (y z)) a`.

---

## BinaryField and BinaryFieldElement

pyfastcd provides the finite fields of odd characteristic, and `BinaryField`
provides the fields F_(2^k) as coefficient fields for loop algebras and for
`linalg`, `codes`, and `dyadic_convolve`. Its properties and scalar constructors
have the names and meaning they have in `pyfastcd.FiniteField`, but it has no
arrays or arithmetic functions of its own.

### Constructors

```python
BinaryField(*, deg=1)
BinaryField.from_modulus(*, modulus)
```

`BinaryField(deg=k)` is F_(2^k) for `k` from 1 to 12, presented for `k >= 2` by
the Conway polynomial of degree `k`, which is also the default of GAP and
SageMath, while `from_modulus` takes another monic irreducible polynomial over
F_2 as a sequence of integers, constant coefficient first, whose entries are
reduced modulo 2.

Fields are immutable, cannot be subclassed, and are cached by their polynomial,
so two presentations of the same field are different parents, and so are the
loop algebras over them. A degree or modulus coefficient that is not an integer,
or is a boolean, raises `TypeError`, while a degree outside `1..12`, a leading
coefficient other than 1, and a reducible polynomial raise `ValueError`.

### Properties

- `char: int`, `deg: int`, `order: int`

  2, the extension degree `k`, and `2**k`.

- `modulus: tuple[int, ...]`

  The polynomial that presents the field, constant coefficient first. At degree
  1 it is `(0, 1)`, as for the prime fields of pyfastcd.

- `dtype`, `coefficient_shape`

  `uint32`, and the shape of the coordinates of one scalar, which is `()` at
  degree 1 and `(deg,)` otherwise. Each coordinate is 0 or 1.

### Scalars

- `F(value=0) -> BinaryFieldElement`, `F.scalar(value)`

  Create a scalar of `F`.

  - **Parameters:** `value` is an integer, which is reduced modulo 2 and lies in
    the prime subfield; a sequence of `deg` polynomial coordinates, constant
    coefficient first; or a scalar of `F`, which is returned.
  - **Returns:** a `BinaryFieldElement` of `F`.
  - **Notes:** A wrong number of coordinates and a scalar of another field raise
    `ValueError`, and a boolean, a float, or a string raises `TypeError`.

- `F.zero()`, `F.one() -> BinaryFieldElement`

  Return the additive and the multiplicative identity.

A `BinaryFieldElement` is immutable and hashable, and it is created only by its
field.

- `algebra`, `coefficients`

  The field, named as for pyfastcd scalars, and a read-only `uint32` array of
  the polynomial coordinates.

- `x + y`, `x - y`, `-x`, `x * y`, `x / y`, `x ** k`, `reciprocal()`

  Field arithmetic, in which an integer operand stands for an element of the
  prime subfield. Scalars of different fields raise `ValueError`, and division
  by zero raises `ZeroDivisionError`.

- `x == y`, `hash(x)`, `bool(x)`

  A scalar equals an integer only when it is 0 or 1 and the integer is the same,
  because integers are not reduced for comparison, so that equality agrees with
  hashing.

---

## LoopAlgebra

A `LoopAlgebra` is the loop algebra of one loop over one coefficient field. It
creates elements and arrays, multiplies them, and provides the idempotents,
maps, and matrices of the algebra.

### Constructor

```python
LoopAlgebra(loop, *, char=0, deg=1, field=None)
```

Create the loop algebra of `loop` over the field that the selectors choose.
Algebras are immutable, cannot be subclassed, and are cached by loop and
coefficient field, so `LoopAlgebra(L, char=2, deg=3)` and `LoopAlgebra(L,
field=BinaryField(deg=3))` are the same object.

Every loop has a loop algebra in characteristic 2, while over the other fields
two limits apply:

- **Level.** The signed components are held by pyfastcd, whose storage limits
  apply: over R a factor of level 60 or more raises `OverflowError`, and over C
  and over finite fields the limit is up to a few levels lower, depending on the
  characteristic and the degree. A central product with a factor of level 1 over
  R also uses the algebra over C.
- **Memory.** Creating the algebra of a central product allocates one byte for
  each of the `D` basis indices and a sign table for every factor but one of
  largest level, whose memory grows with the square of that factor's order, and
  a product too large for this raises `MemoryError`. In characteristic 2 these
  are built when the algebra to which products are lifted is first needed.

### Properties

- `loop`

  The loop.

- `dimension: int`

  The dimension of the algebra, `loop.order`.

- `char: int`, `deg: int`, `field`

  The characteristic and degree of the coefficient field, and the field object,
  which is a `pyfastcd.FiniteField` in odd characteristic, a `BinaryField` in
  characteristic 2, and `None` over R and C.

- `dtype`, `coefficient_shape`

  The storage dtype and the shape of one record, from the selector table.

- `cd_algebra`

  The `pyfastcd.Algebra` that holds the signed components of a Cayley-Dickson
  loop algebra, or `None` for a central product and in characteristic 2.

- `algorithms: tuple[str, ...]`

  The accepted values of `algorithm` in `multiply`, which are
  `("auto", "split", "table")`, and `("auto", "lift", "table")` in
  characteristic 2.

- `default_algorithm: str`, `default_batch_algorithm: str`

  The algorithm that `"auto"` selects in `multiply` for a single product and for
  a large batch. Both are `"split"` except in characteristic 2, where `"auto"`
  selects `"lift"` or `"table"` from the size of the loop, the number of
  products, and the degree of the field; [How the results set the
  defaults](benchmark.md#how-the-results-set-the-defaults) explains these
  choices.

- `signed_algorithms: tuple[str, ...]`

  The accepted values of `signed_algorithm` in `multiply`. For a Cayley-Dickson
  loop these are pyfastcd's algorithm names, `"auto"`, `"direct"`,
  `"quasilinear"`, and, over R for `D` up to 1024, `"cariow_cariowa"`, and for a
  central product they are `"auto"`, `"rows"`, `"schedule"`, and, where it
  applies, `"native"` (see [The product of signed
  components](mathematics.md#the-product-of-signed-components)). In
  characteristic 2 the names are those of the loop algebra over the field to
  which products are lifted.

- `default_signed_algorithm: str`

  The method that `"auto"` selects for the signed components of a single
  product. For a Cayley-Dickson loop outside characteristic 2 a batch follows
  the batch choice of pyfastcd, `cd_algebra.default_batch_algorithm`, and in
  characteristic 2 the choice is that of the algebra to which products are
  lifted.

### Construction

These methods create elements and arrays of `A`, in the loop basis or from the
components of the decomposition.

- `A(value=0) -> Element`

  Create an element of `A`.

  - **Parameters:** `value` is a field scalar `r`, giving `r` times the
    identity; an `Element` with parent `A`, which is returned; or coefficients
    of shape `coefficient_shape`, from a nested sequence or an array.
  - **Returns:** an `Element` with parent `A`.
  - **Notes:** A `LoopElement`, a `set`, or a `dict` raises `TypeError`, and so
    do complex coefficients over R and floats over a finite field, while a wrong
    shape raises `ValueError`.

- `zero()`, `one()`, `scalar(r) -> Element`

  Return the additive identity, the multiplicative identity `[e_0]`, and `r`
  times the identity.

- `basis(x) -> Element`

  Return the basis element `[x]` of a `LoopElement` `x` of `A.loop`.

- `from_terms(terms) -> Element`

  Return the combination of loop elements given by a mapping from `LoopElement`
  to field scalars, or by an iterable of such pairs, in which the coefficients
  of a repeated loop element add.

- `from_components(kappa, epsilon) -> Element | ElementArray`

  Return the element with group component `kappa` and signed component
  `epsilon`, whose coefficients are `a_t = (c_t + d_t) / 2` and
  `b_t = (c_t - d_t) / 2`, where `c` and `d` are the coefficients of `kappa` and
  `epsilon`. Not defined in characteristic 2.

  - **Parameters:** `kappa` has shape `(D,)`, followed by the polynomial axis of
    an extension field. `epsilon` is a pyfastcd `Element` or `ElementArray` of
    `cd_algebra`, or coefficients of the shape of `kappa`, and for a central
    product the coefficients may instead have one axis for each factor.
  - **Returns:** an `Element`, or an `ElementArray` when leading batch axes are
    given, which broadcast.

- `lift_group(kappa)`, `lift_signed(epsilon) -> Element | ElementArray`

  Return the element with the given group component and signed component zero,
  and the reverse. Each map is an injective homomorphism of nonunital algebras
  onto an ideal. Not defined in characteristic 2.

- `split_array(spectral, signed) -> SplitArray`

  Return a `SplitArray` with the given parts, which are copied and, over a
  finite field, reduced, as for `from_components`. Not defined in characteristic
  2.

- `random(rng=None, *, batch_shape=None, secure=False) -> Element |
  ElementArray`

  Return a random element, or an `ElementArray` when `batch_shape` is given.

  - **Parameters:** `rng` is a `numpy.random.Generator` or a seed. With
    `secure=True` the residues of a finite field come from the operating
    system's cryptographic source through the `secrets` module, and `rng` is
    ignored.
  - **Returns:** an element or array whose coefficients are independent:
    standard normal over R, with standard normal real and imaginary parts over
    C, and uniform residues over a finite field.
  - **Notes:** `secure=True` over R or C raises `ValueError`.

- `from_bytes(data) -> Element`

  Return the element encoded by `Element.to_bytes`, from a bytes-like object. A
  wrong length, or a residue not below the characteristic, raises `ValueError`,
  and anything that is not bytes-like raises `TypeError`.

- `array(values) -> ElementArray`

  Copy `values`, which is an element, an `ElementArray` of the algebra, or
  coefficients of shape `batch_shape + coefficient_shape`, into a new array. A
  `SplitArray` raises `TypeError` and is converted with `to_loop()` instead.

- `asarray(values, *, copy=None) -> ElementArray`

  Create an array that shares the storage of `values` where possible:
  `copy=None` shares storage when it can, `copy=True` always copies, and
  `copy=False` requires sharing and raises `ValueError` when it is impossible.
  [Storage](#storage) gives the conditions for sharing.

### Products

- `multiply(x, y, *, out=None, algorithm="auto", signed_algorithm="auto")`

  Return the ordered product `x y`, broadcasting batch axes.

  - **Parameters:** `x` and `y` are elements or arrays of this algebra, and one
    of them may be a field scalar, which scales the other. `algorithm` is one of
    `algorithms`: `"split"` multiplies through the decomposition of the algebra,
    `"lift"`, its counterpart in characteristic 2, multiplies by the split
    algorithm over a large prime field, or over an extension of it when the
    coefficient field is not F_2, and reduces the result (see [Multiplication by
    lifting](mathematics.md#multiplication-by-lifting)), and `"table"` evaluates
    the definition in the loop basis with `4 D^2` multiplications, up to
    `D = 4096`, and does not accept a `SplitArray`. `signed_algorithm` is one of
    `signed_algorithms` and selects the method for the product of signed
    components under `"split"` and `"lift"`. `out` is an ndarray or an
    `ElementArray` that receives a loop-basis result, with exactly the shape of
    the result and the dtype of the algebra, and it must be writable.
  - **Returns:** an `Element` when neither operand is an array, and otherwise an
    array of the operands' kind. With `out` the result is `out` itself when that
    is an `ElementArray`, and a new array around it when it is an ndarray.
  - **Notes:** `"auto"` selects `"split"`, and in characteristic 2 whichever of
    `"lift"` and `"table"` is expected to be faster for the number of products
    in the call. An operand of another algebra raises `ValueError`, any other
    operand that is not an element, array, or field scalar raises `TypeError`,
    and a wrong `out` dtype raises `TypeError`. Names are case-sensitive, so an
    unknown name raises `ValueError`, while a name that is not a string raises
    `TypeError`.

- `left_operator_commutator(a, b, x)`

  Compute `a (b x) - b (a x)`.

- `right_operator_commutator(a, b, x)`

  Compute `(x b) a - (x a) b`.

### Idempotents and the center

The idempotents come from the decomposition (see [Idempotents and
ideals](mathematics.md#idempotents-and-ideals)), so the three methods that
return them are not defined in characteristic 2, where the only idempotents of
the algebra are 0 and 1.

- `central_idempotents() -> tuple[Element, Element]`

  Return `(1 + [-1]) / 2` and `(1 - [-1]) / 2`, central orthogonal idempotents
  with sum 1, which project onto the group and the signed component.

- `primitive_idempotent(a) -> Element`

  Return the idempotent whose spectrum is 1 at coordinate `a` and 0 elsewhere,
  with signed component zero.

- `primitive_idempotents() -> ElementArray`

  Return all `D` of them, with batch shape `(D,)`.

- `center_basis() -> ElementArray`

  Return a basis of the center of the algebra: the primitive idempotents,
  followed by `([e_t] - [-e_t]) / 2` for the central loop elements `e_t`. In
  characteristic 2 the basis is `[e_t]` for the central `e_t`, then `[-e_t]` for
  the same elements, and then `[e_t] + [-e_t]` for the others.

### Maps

The maps act on elements and arrays through the loop basis.

- `apply_automorphism(f, x)`

  Return the image of an element or `ElementArray` under the linear extension of
  a loop map `f`, which moves the coefficient of `[y]` to `[f(y)]`. `f` must be
  a bijection of the loop, or `ValueError` is raised, but that it preserves
  products is not checked.

- `character_automorphism(a, x)`

  Apply `e_t -> (-1)**popcount(a & t) e_t` to an element or `ElementArray`,
  which exchanges the coefficients of `[e_t]` and `[-e_t]` wherever
  `popcount(a & t)` is odd. This is an algebra automorphism for every index `a`.

- `embed(x)`

  For a Cayley-Dickson loop `Q_n`, return the image of an element or
  `ElementArray` of the loop algebra of `Q_k`, `k <= n`, over the same field.
  For a central product it raises `TypeError`.

- `factor_embedding(i, x)`

  For a central product, return the image of an element or `ElementArray` of the
  loop algebra of factor `i` over the same field, under an injective
  homomorphism of unital algebras. For a Cayley-Dickson loop it raises
  `TypeError`.

### Linear algebra

- `left_matrix(x, *, basis="loop")`, `right_matrix(x, *, basis="loop")`

  Return the matrix of `y -> x y`, or of `y -> y x`, for one element, with a
  trailing polynomial axis over an extension field.

  - **Parameters:** With `basis="loop"` the matrix `M` of `y -> x y` satisfies
    `(x * y).coefficients.ravel() == M @ y.coefficients.ravel()`, and that of
    `y -> y x` the same with `y * x`. With `basis="split"` the matrix acts on
    the spectral coordinates followed by the signed coordinates and is block
    diagonal, and this basis is not defined in characteristic 2.
  - **Notes:** The identity holds up to rounding over R and C, and exactly over
    prime fields when the right side is computed in integers that do not
    overflow and then reduced modulo `char`, since the entries of `M` are
    `uint32` there and their products can exceed that type.

- `structure_constants() -> numpy.ndarray`

  Return the `int8` array whose entry `[i, j, k]` is 1 when the product of the
  basis elements at positions `i` and `j` is the one at position `k`. Dimensions
  above 256 raise `ValueError`.

- `subalgebra_generated_by(x) -> GeneratedSubalgebra`

  Describe the unital subalgebra generated by one element. [Where powers are
  defined](#where-powers-are-defined) only.

### GeneratedSubalgebra

A `GeneratedSubalgebra` is the record that `subalgebra_generated_by` returns.
The subalgebra is isomorphic to `F[z] / (minimal_polynomial)`, and the
attributes of the record cannot be reassigned.

- `dimension`, `minimal_polynomial`

  The dimension of the subalgebra and the minimal polynomial of the element,
  which in characteristic 2 is a power of `z - aug(x)` of degree at most 4.

- `spectral_values`

  The distinct spectral coordinates of the element, or `None` in characteristic
  2.

- `signed_polynomial`

  The minimal polynomial of the signed component of the element, or `None` in
  characteristic 2.

---

## Element, ElementArray, and SplitArray

The elements of a loop algebra have two representations, the loop basis, in
which coefficients are read and written, and, outside characteristic 2, the
split form, in which arithmetic is cheapest, and the three classes differ in how
they hold the two:

- An `Element` is one immutable element. It holds its loop-basis coefficients,
  its split form, or both: whichever it was created with is never replaced, and
  the other is derived from it on first use and kept. A product is evaluated in
  split form and its result is created in split form, so a chain such as
  `((x * y) * z) * w` performs one forward transform for each operand that does
  not have its split form yet and one inverse transform when the coefficients of
  the result are first read.
- An `ElementArray` is a writable batch in the loop basis. Its storage can be
  shared with NumPy arrays, so nothing derived from it is kept, and every
  product transforms its array operands and its result.
- A `SplitArray` is a writable batch in split form, whose arithmetic needs no
  transform. `ElementArray.split()` and `SplitArray.to_loop()` convert between
  the two kinds of array, and combining them raises `TypeError`, so that every
  transform is visible in the calling code.

The coordinates in which an `Element` operation is evaluated depend only on how
its operands were created: sums, scalings, and `conjugate()` of elements created
in the loop basis are evaluated there, and every other arithmetic operation in
split form. Since a derived form is a function of the form the element was
created with, an operation returns the same result, bit for bit, whether or not
its operands have derived forms already.

In characteristic 2 there is no split form and no `SplitArray`, so elements and
arrays hold loop-basis coefficients and every operation is evaluated there. An
`Element` keeps one derived form of its own, the transforms of its lifted
coefficients over the larger field, which its first lifted product through `*`
or `multiply` computes and later such products reuse, while an `ElementArray`
keeps nothing, as over the other fields.

The three classes share most of their properties, methods, and operators, so
this section documents them together and notes where they differ.

### Construction

An `Element` is created with `A(...)` or another constructor of the algebra, an
`ElementArray` with `A.array`, `A.asarray`, or `A.random`, and a `SplitArray`
with `X.split()` or `A.split_array`, while constructing any of the three classes
directly raises `TypeError`.

Elements are immutable and hashable, while arrays are unhashable and are
writable exactly when their NumPy storage is.

### Which operands combine

| Receiver | Accepts | Result |
|---|---|---|
| `Element` | `Element`, field scalar | `Element` |
| `ElementArray` | `ElementArray`, `Element`, field scalar | `ElementArray` |
| `SplitArray` | `SplitArray`, `Element`, field scalar | `SplitArray` |

Combining an `ElementArray` with a `SplitArray` raises `TypeError` in every
operator, method, comparison, and assignment. The methods of an `Element` take
`Element` operands only, so an element is combined with an array by calling the
method on the array or by using an operator.

### Where powers are defined

Powers depend on bracketing in a general loop algebra, so the operations that
rely on them are defined only where they do not: in the loop algebra of a
Cayley-Dickson loop, and in every loop algebra of characteristic 2, central
products included (see [Elements of the loop algebra of a central
product](mathematics.md#elements-of-the-loop-algebra-of-a-central-product)).
These operations are `x ** k`, `inverse`, `has_inverse`, `is_nilpotent`,
`minimal_polynomial`, and `LoopAlgebra.subalgebra_generated_by`, and in the loop
algebra of a central product over R, C, or a field of odd characteristic they
raise `TypeError`. `square`, `left_power`, and `right_power`, which fix the
bracketing, are defined in every loop algebra.

### Properties

- `algebra`, `dtype`, `dimension`

  The parent, its storage dtype, and its dimension.

- `shape`

  `coefficient_shape` for an element, and `batch_shape + coefficient_shape` for
  an `ElementArray`.

- `batch_shape`

  The leading axes of an array.

- `coefficients`

  The loop-basis coefficients, as a read-only array for an element and as the
  storage itself for an `ElementArray`. The array of an element cannot be made
  writable, and each access returns a new array, so nothing done to one array
  reaches the element or a later access.

- `SplitArray.spectral`, `SplitArray.signed`

  The two parts of a `SplitArray`, which share its storage. `spectral` is an
  ndarray over R and C and a pyfastcd array of field scalars over a finite
  field, while `signed` is a `pyfastcd.ElementArray` for a Cayley-Dickson loop
  and, for a central product, an ndarray of shape `batch_shape + (D,)` followed
  by the polynomial axis of an extension field.

### Conversion

- `to_numpy(*, copy=True)`

  Return the loop-basis coefficients of an element or `ElementArray`, and with
  `copy=False` a read-only view of an element or the storage of an array.

- `copy()`

  Return an equal object with independent storage.

- `Element.to_bytes() -> bytes`

  Return a canonical encoding of the coefficients in the order of
  `coefficients.ravel()`: little-endian binary64 values over R and C, and over a
  finite field big-endian residues of the smallest whole number of bytes that
  holds `char - 1`.

- `ElementArray.split() -> SplitArray`, `SplitArray.to_loop() -> ElementArray`

  Convert between the loop basis and split form. `split()` is not defined in
  characteristic 2.

### Decomposition

These methods return the components of the decomposition (see [Group and signed
components](mathematics.md#group-and-signed-components)), and the table after
them gives the types of their results.

- `kappa()`

  Return the group component `(a_t + b_t)`. The map is a homomorphism onto the
  group algebra in every characteristic.

- `spectrum()`

  Return the Walsh-Hadamard transform of the group component. Not defined in
  characteristic 2.

- `epsilon()`

  Return the signed component `(a_t - b_t)`. Not defined in characteristic 2,
  where it would coincide with the group component.

- `Element.augmentation()`

  Return the sum of all coefficients, a field scalar. The map is a homomorphism
  to the field.

| Method | R, C | Odd characteristic | Characteristic 2 |
|---|---|---|---|
| `kappa()` | ndarray | pyfastcd array of field scalars | ndarray of residues |
| `spectrum()` | ndarray | pyfastcd array of field scalars | not defined |
| `epsilon()`, Cayley-Dickson loop | pyfastcd `Element` or `ElementArray` | the same | not defined |
| `epsilon()` of an element, central product | read-only ndarray with one axis for each factor | the same, followed by the polynomial axis of an extension field | not defined |
| `epsilon()` of an array, central product | ndarray of shape `batch_shape + (D,)` | the same, followed by the polynomial axis of an extension field | not defined |
| `augmentation()`, `coefficient()` | `float` or `complex` | pyfastcd field scalar | `BinaryFieldElement` |

### Operators

Operators accept the operands of [Which operands
combine](#which-operands-combine), field scalars included. Their results are new
objects, and arrays broadcast their batch axes.

- `+x`, `-x`

  The element itself, or a copy of an array, and its negative.

- `x + y`, `x - y`, `x + r`, `r - x`, ...

  Add or subtract coefficients, where a scalar `r` stands for `r` times the
  identity.

- `x * y`

  The ordered product, by the default algorithm.

- `r * x`, `x * r`

  Scale every coefficient.

- `x / r`

  Divide by a field scalar, raising `ZeroDivisionError` for division by zero.

- `x ** k`

  An integer power, which uses the inverse for negative exponents. [Where powers
  are defined](#where-powers-are-defined) only.

  - **Parameters:** Any Python integer is accepted as an exponent, and an
    exponent that is not an integer, or is a boolean, raises `TypeError`.
  - **Notes:** The cost grows with `log k`, and in characteristic 2 a power
    takes at most two products. Over R and C a power whose value leaves the
    range of the floats holds infinities, zeros, or NaNs, and [Floating
    point](#floating-point) describes how the power is evaluated.

- `x == y`, `x != y`

  Compare exactly, as described under [Equality and
  hashing](#equality-and-hashing).

NumPy universal functions are not supported: `__array_ufunc__` is disabled, so
that expressions such as `numpy.float64(2) * x` reach the operators above. An
ndarray that is not a zero-dimensional scalar is not an operand, so combining
one with an element or array raises `TypeError`, and it is wrapped with
`A.asarray` first.

### Equality and hashing

Equality compares exactly, and its rules for the different kinds of operand keep
it consistent with hashing. For an `Element` the result is a `bool`, and for an
array it is a Boolean array over the broadcast batch shape, which is a NumPy
Boolean scalar when that shape is empty.

- **Representations.** Elements and `ElementArray` records are compared by their
  loop-basis coefficients, and `SplitArray` records by their two parts.
- **Field scalars.** An element or record equals a field scalar `r` when it is
  `r` times the identity, and an element equal to a field scalar hashes like
  that scalar. Over a finite field, integers are not reduced for comparison, so
  that equality agrees with hashing.
- **Other parents.** An object with a different parent compares unequal
  everywhere, although the batch shapes of two arrays must still broadcast.
- **Other Python objects.** For any other operand the comparison returns
  `NotImplemented`, so ordinary objects compare unequal.
- **Hashing and truth.** Equal elements hash equal, `bool(x)` is false only for
  the zero element, and `bool` of an array raises `ValueError`.

### Methods

These methods allocate new results. Their operand rules are those of [Which
operands combine](#which-operands-combine), but unlike the operators, methods do
not accept field scalars.

- `conjugate()`

  Return the image under the linear extension of `[x] -> [x^-1]`, which reverses
  products, fixes the group component, and conjugates the signed component. It
  does not conjugate complex coefficients or apply the Frobenius map.

- `square()`

  Return `x x`.

- `commutator(other)`, `anticommutator(other)`

  Return `x y - y x` and `x y + y x`, where `y` is `other`.

- `jordan_product(other)`

  Return `(x y + y x) / 2`, where `y` is `other`. Not defined in characteristic
  2.

- `associator(y, z)`

  Return `(x y) z - x (y z)`.

- `left_power(exponent)`, `right_power(exponent)`

  Return `x (x (... (x x)))` and `(((x x) x) ...) x` with `k = exponent`
  factors, for an integer `k >= 0`, where `k = 0` gives the identity. They use
  `k - 1` products, are defined in every loop algebra, and raise `ValueError`
  for a negative exponent.

- `inverse()`

  Return the inverse of `x` that is a polynomial in `x`, which is a two-sided
  inverse, raising `ZeroDivisionError` where it does not exist. [Where powers
  are defined](#where-powers-are-defined) only.

  - **Notes:** Over R, C, and fields of odd characteristic the inverse is the
    pair of the coordinatewise reciprocal of the spectrum and
    `conjugate(d) / Q(d)` of the signed component `d`, and it exists exactly
    when every spectral coordinate is nonzero and `Q(d) != 0`. From level 4 on
    an element can have other two-sided inverses, and over C and finite fields
    also where this one does not exist, which are not computed. In
    characteristic 2 the inverse exists when the augmentation is nonzero and
    equals `x**3 / aug(x)**4`. A batch checks every record before producing any
    result.

- `weight()`

  Return the number of nonzero loop-basis coefficients, as an `int` for an
  element and for an array with an empty batch shape, and otherwise as an
  integer array over the batch.

- `distance(other)`

  Return the number of loop-basis coefficients in which two operands differ.

For a Cayley-Dickson loop over R, C, or a field of odd characteristic, `square`,
`anticommutator`, `jordan_product`, and `conjugate` cost `O(D)` operations on
operands in split form, and `x ** k` costs `O(D + log k)` beyond the
coordinatewise powers of the spectrum.

### Structure

Each predicate returns a `bool` for an element, and for an array a Boolean array
over the batch, which is a NumPy Boolean scalar when the batch shape is empty.

- `has_inverse()`

  Return whether `inverse()` exists. [Where powers are
  defined](#where-powers-are-defined) only.

- `is_zero_divisor(side=None)`

  Return whether a nonzero `y` has `x y = 0` (`side="left"`), `y x = 0`
  (`side="right"`), or either (`None`), counting the zero element as a zero
  divisor. For a Cayley-Dickson loop of level at most 3 the test costs `O(D)`
  operations, and otherwise it computes the rank of a `D x D` matrix for each
  record. In characteristic 2 the zero divisors on either side are the elements
  of augmentation zero, and the test costs `O(D)` operations for every loop.

- `is_idempotent()`

  Return whether `x x = x`, compared exactly in split form, and in
  characteristic 2 in the loop basis.

- `is_nilpotent()`

  Return whether some power of `x` vanishes. [Where powers are
  defined](#where-powers-are-defined) only.

- `is_central()`

  Return whether `x` commutes and associates with every element of the algebra.

The remaining methods belong to `Element` only.

- `coefficient(x)`

  Return the coefficient of the basis element `[x]` of a `LoopElement`.

- `terms() -> dict`

  Return the nonzero coefficients, keyed by `LoopElement`.

- `support() -> frozenset`

  Return the loop elements with a nonzero coefficient.

- `left_annihilator()`, `right_annihilator() -> ElementArray`

  Return a basis of `{y : y x = 0}` and of `{y : x y = 0}`. The computation
  finds the null space of a `D x D` matrix, and of a `2D x 2D` matrix in
  characteristic 2, and over R and C the null space is numerical, with the
  tolerance of `linalg.nullspace`.

- `minimal_polynomial() -> numpy.ndarray`

  Return the monic polynomial of least degree that vanishes at `x`, as field
  coefficients with the constant term first. [Where powers are
  defined](#where-powers-are-defined) only.

  - **Notes:** In characteristic 2 the polynomial is `(z - aug(x))**m` for some
    `m` from 1 to 4. Over R and C it is computed from the spectrum and the
    signed component as they are held, coincidences between spectral values are
    detected by exact comparison of floating-point numbers, and whether a
    spectral value is a root of the polynomial of the signed component is
    decided exactly. For an element given by loop-basis coefficients these are
    rounded sums of the coefficients, so a coincidence that holds for the
    coefficients can be missed.

### Indexing and iteration

- Indexing and iterating an array act on its batch axes, with the meaning the
  same index has for a NumPy array of the batch shape. `X[i]` and `S[i]` return
  an `Element` copy when they select one record, and otherwise an array that
  shares storage where NumPy indexing does. An index cannot reach the
  coefficient axes, so too many indices raise `IndexError`.
- `X[key] = value` assigns a field scalar, an `Element`, an `ElementArray`, or a
  coefficient array, and `S[key] = value` assigns a field scalar, an `Element`,
  or a `SplitArray`.
- `len(X)` is the length of the first batch axis, and, as in NumPy, an array
  with an empty batch shape has no length and cannot be iterated.
- An `Element` is not a sequence, and its coefficients are read through
  `coefficients`, `coefficient`, or `terms`.

---

## Characteristic 2

Over a `BinaryField` the loop algebra has no decomposition into a group
component and a signed component, so its elements behave differently (see
[Characteristic 2](mathematics.md#characteristic-2) in the conventions). This
section lists every difference, and loops, loop elements, and everything not
listed here work as over the other fields.

| | Members |
|---|---|
| Not defined; raise `TypeError` | `SplitArray`, `ElementArray.split`, `LoopAlgebra.split_array`, `from_components`, `lift_group`, `lift_signed`; `spectrum`, `epsilon`; `central_idempotents`, `primitive_idempotent`, `primitive_idempotents`; `jordan_product`; `left_matrix` and `right_matrix` with `basis="split"`; `wht` with a `BinaryField` |
| Defined for every loop, central products included | `x ** k`, `inverse`, `has_inverse`, `is_nilpotent`, `minimal_polynomial`, `subalgebra_generated_by` |
| Different algorithm names | `multiply` takes `"auto"`, `"lift"`, or `"table"`; `"split"` raises `ValueError`; `signed_algorithms` and `default_signed_algorithm` are those of the algebra to which products are lifted |
| Different contents | `center_basis` returns the basis described with it |
| Different result types | `kappa` returns an ndarray; `augmentation`, `coefficient`, and the values of `terms` are `BinaryFieldElement` scalars; `cd_algebra` is `None`; `GeneratedSubalgebra.spectral_values` and `signed_polynomial` are `None` |

Every operation is exact, and the results of `"lift"` and `"table"` are
identical, so `"auto"` chooses between them by expected running time, from the
quotient order `D`, the number of products in the call, and the degree of the
field. `default_algorithm` and `default_batch_algorithm` give the choice for one
product and for a large batch, and [How the results set the
defaults](benchmark.md#how-the-results-set-the-defaults) states the rule and the
measurements behind it.

---

## pycdloop.linalg

The module `linalg` provides functions on matrices over a coefficient field,
which the annihilators and the zero-divisor tests use. Without `field` a matrix
is real or complex and the results are numerical, while with a finite field, a
`pyfastcd.FiniteField` or a `BinaryField`, the entries are residues, with a
trailing polynomial axis for an extension field, and the results are exact.

- `rank(matrix, *, field=None, tol=None) -> int`

  Return the rank. Over R and C it is the number of singular values above `tol`,
  which defaults to the largest singular value times `max(rows, columns)` times
  the machine epsilon. The decomposition resolves singular values down to about
  the machine epsilon times the largest, so a smaller `tol` does not separate
  more.

- `row_reduce(matrix, *, field=None, tol=None) -> tuple[numpy.ndarray,
  tuple[int, ...]]`

  Return the reduced row echelon form and the pivot column of each nonzero row.

  - **Parameters:** Over R and C the result is numerical, and `tol` decides
    which entries of a pivot column count as zero in the rows that do not hold a
    pivot yet. With `tol`, the entries at or below `tol` count as zero, and
    `tol=0` treats every nonzero entry as significant. Without it, an entry
    counts as zero when it is no larger than the rounding that elimination can
    have left in its place, which is `max(rows, columns)` times the machine
    epsilon times the largest entry of its row, multiplied by the growth of the
    reduction, which is large where the pivot columns are nearly dependent. A
    pivot must in addition be above `max(rows, columns)` times the machine
    epsilon times the largest entry of the matrix, so a row that depends on the
    others, exactly or up to rounding, reduces to a row of zeros.
  - **Returns:** the reduced matrix and the tuple of pivot columns.
  - **Notes:** An entry that counts as zero is zero in the elimination too and
    is not multiplied by the pivot row, while the rows that already hold a pivot
    are always cleared, whatever the size of their entries. Elimination uses
    scaled partial pivoting: of the entries that can be pivots it takes the one
    that is largest relative to the largest entry of its row. Apart from the
    comparisons with `tol` and with the largest entry of the matrix, the result
    therefore does not depend on the scale of each row, and rows may lie
    anywhere in the range of the floats beside one another, because where a step
    overflows or underflows, the reduction is done on numbers whose exponents
    cannot leave the range. NaN and infinite entries raise nothing, and the
    result is then without meaning.

- `nullspace(matrix, *, field=None, tol=None) -> numpy.ndarray`

  Return an array whose rows are a basis of the vectors `v` with
  `matrix @ v = 0`. Over R and C the rows are orthonormal, and `tol` is the
  singular value threshold of `rank`.

A matrix that does not have two index axes raises `ValueError`, and a `field`
that is not a field object raises `TypeError`.

---

## pycdloop.codes

The module `codes` treats a subspace of a loop algebra as a linear code of
length `len(loop)`, through the loop-basis coefficients. Matrices have one row
for each codeword and one column for each loop element in position order,
followed over an extension field by the polynomial axis.

- `translate_matrix(x, *, side="left") -> numpy.ndarray`

  Return the matrix whose rows are `[g] x` (`"left"`) or `x [g]` (`"right"`) for
  `g` running over the loop.

- `generator_matrix(elements, *, side="left", closure=True, tol=None) ->
  numpy.ndarray`

  Return a generator matrix, in reduced row echelon form, of the code generated
  by one element or an iterable of elements.

  - **Parameters:** `elements` holds at least one element, or `ValueError` is
    raised. `side` is `"left"`, `"right"`, or `"two-sided"`. With `closure=True`
    the code is the ideal that the elements generate on that side, the smallest
    subspace that contains them and is closed under translation by every loop
    element on the left, on the right, or on both sides, and because the algebra
    need not be associative, translates are added until the dimension stops
    growing. With `closure=False` the code is the span of the translates of the
    generators, on both sides for `"two-sided"`. Over R and C a vector of norm 1
    counts as lying in a space when its distance from the space is at most
    `tol`, which defaults to the square root of the machine epsilon and decides
    the dimension.
  - **Notes:** The translates are formed and reduced a few rows at a time rather
    than all at once, which bounds the working memory. Over a finite field the
    result is exact.

- `is_ideal(matrix, algebra, *, side="left", tol=None) -> bool`

  Return whether the row space of `matrix` is closed under translation by every
  loop element on the given side, `"left"`, `"right"`, or `"two-sided"`.

  - **Parameters:** `matrix` has one column for each loop element, followed over
    an extension field by the polynomial axis, and its rows need not be
    independent, so a matrix without rows spans the zero ideal. The entries are
    read as coefficients of the algebra, so integers are reduced over a finite
    field. `tol` is that of `generator_matrix`.
  - **Notes:** A complex matrix for an algebra over R raises `TypeError`, and so
    does an `algebra` that is not a `LoopAlgebra`, while a matrix that does not
    have two index axes or has another number of columns raises `ValueError`.

- `to_gap(matrix, *, field) -> str`, `to_sage(matrix, *, field) -> str`

  Return a GAP expression (`GeneratorMatCode` of the GUAVA package) or a
  SageMath expression (`LinearCode`) for the code with the given generator
  matrix over a finite field, a `pyfastcd.FiniteField` or a `BinaryField`.

  - **Returns:** an expression that carries the length of the code, so a matrix
    without rows gives the zero code of that length, which is `NullCode` in GAP.
  - **Notes:** Over an extension field the entries are the polynomials of the
    matrix in a root `t` of `field.modulus`. The SageMath expression builds the
    field with that modulus and the generator `t`, while GAP presents the field
    by its Conway polynomial, with root `Z(q)`, so the GAP expression takes
    `t = Z(q)` when that is a root of the modulus, as it is for the default
    moduli of `BinaryField`, and otherwise the first root that GAP finds. The
    roots differ by automorphisms of the field, and the codes they give have the
    same length, dimension, and weights. R and C (`field=None`) and a matrix
    that is not an integer array with two index axes, followed by the polynomial
    axis of an extension field, raise `ValueError`, and a `field` that is not a
    field object raises `TypeError`.

---

## pycdloop.testing

The module `testing` checks identities by sampling. Each check draws `samples`
random elements for each operand of the identity and returns whether the
identity held on all of them, so `False` exhibits a counterexample and is
conclusive, while `True` only reports that none was found. Over R and C the
samples have small integer coefficients, so the checks are exact.

- `random_elements(algebra, count, rng=None, *, exact=True) -> ElementArray`

  Return `count` random elements. With `exact=True` the coefficients over R, and
  the real and imaginary parts of the coefficients over C, are integers between
  -3 and 3, and with `exact=False` they are those of `LoopAlgebra.random`. Over
  a finite field `exact` changes nothing.

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

- **Shape.** An `ElementArray` has shape `batch_shape + coefficient_shape`, and
  only the batch axes broadcast.
- **Sharing.** `A.asarray` shares the storage of an `ElementArray` of the
  algebra, and of an ndarray with the algebra's dtype that, over a finite field,
  holds residues below the characteristic. Any other input is converted and
  copied, unless `copy=False` forbids it.
- **Reduction.** When finite-field coefficients are copied, integers of any size
  and NumPy integer type are reduced modulo `char`, while booleans, floats, and
  strings are not coefficients of a finite field and raise `TypeError`. The
  coefficients of a pyfastcd object given as a signed component are copied and
  reduced in the same way.
- **Layout.** Strided, broadcast, and unaligned views are accepted, and
  operations copy operands that are not C-contiguous or not aligned, as well as
  operands whose layout pyfastcd cannot read in place.

### Shared storage

The storage of an `ElementArray` or `SplitArray` can be written to from outside
once it has been handed out, through `coefficients`, `to_numpy(copy=False)`, a
NumPy conversion that does not copy, the parts of a `SplitArray`, `asarray` of
an ndarray, or `out=`.

From then on the storage is checked before it is read, for that array and for
every array that shares it. The check runs in every arithmetic operation,
comparison, predicate, and decomposition, and in `copy()`, `copy.copy`,
`copy.deepcopy`, and pickling, and it raises `ValueError` for storage

- whose dtype was changed,
- whose trailing axes are no longer those of the coefficients, or
- that over a finite field holds a value that is not below the characteristic.

`to_numpy()` and the NumPy conversions return the storage as it is, without this
check.

An `Element`, by contrast, is immutable. The views of its coefficients that it
returns, from `coefficients`, `to_numpy(copy=False)`, and NumPy conversions that
do not copy, are read-only and cannot be made writable, its other methods return
independent arrays, and writing to the array it was created from does not change
it.

### NumPy interfaces

- `Element.__array__` returns the read-only coefficients, and
  `ElementArray.__array__` the storage, unless a dtype change or `copy=True`
  requires a copy, so `np.asarray(X)` shares storage with `X`. A dtype change
  with `copy=False` raises `ValueError`.
- Universal functions are not supported, as described under
  [Operators](#operators).

### Copying and serialization

Pickling, `copy.copy`, and `copy.deepcopy` of arrays create independent storage,
while elements are immutable, so `copy.copy` returns the element itself. A
pickled element records the form it was created with, and loops and algebras
pickle to the same cached objects.

Unpickling into an array or an element raises `ValueError` for data that does
not have the dtype and shape of the algebra, that over a finite field holds a
value that is not below the characteristic, or that is in split form for an
algebra of characteristic 2.

---

## Floating point

Over R and C coefficients are binary64 values, so every result is formed by
rounded operations, while arithmetic over finite fields is exact. This section
describes how accurate results are, how they are kept in range, and where sums
can overflow, and the [stability study](stability.md) measures the accuracy of
products and chains against an exact reference.

**Accuracy.** Accuracy is that of floating-point arithmetic: relative to the
largest of the terms that are added, so a coefficient that is a small difference
of large terms carries their rounding. A product is accurate relative to the
largest coefficients of its operands, not to each coefficient.

**Cancellation in the quadratic norm.** Over C the quadratic norm `Q(d)` of a
signed component is a sum of squares that can cancel. Where the floating-point
sum is not well above its rounding error, it is formed in rational arithmetic
from the stored coefficients and rounded once, and whether it vanishes is
decided exactly, in `has_inverse`, `inverse`, `is_nilpotent`, and, through level
3, `is_zero_divisor`. Inverses, powers, and minimal polynomials use the rounded
exact value, so a complex result is accurate relative to the modulus of each
coefficient.

**Results stay in range.** Inverses, powers, the predicates, minimal
polynomials, scalar division, `linalg.row_reduce`, and `pycdloop.codes` return a
coefficient that is representable whatever the sizes of the values formed on the
way, whether the coefficients of an operand are all large, all small, or of very
different sizes. Squares, norms, quotients, and the recurrence of a power are
evaluated with exponents that cannot leave the range, or in rational arithmetic,
and operands are rescaled by a power of two only where that loses nothing. A
coefficient that is itself out of range is an infinity, a zero, or a NaN, and
over C it is out of range when either of its parts is.

**Powers.** A power in the loop algebra of `Q_n` raises the spectrum and the
signed component separately (see [Elements of the loop algebra of
`Q_n`](mathematics.md#elements-of-the-loop-algebra-of-q_n)). Each spectral
coordinate is raised by binary exponentiation, with `O(log k)` multiplications,
so over R its power is exact whenever it is representable, while over C each
part of a product is rounded and the power need not be exact. The signed
component `d = d_0 + d'` is raised in the ring `F[z] / (z^2 + Q(d'))`, which
over R with `d' != 0` is a copy of C, where the modulus of a product is the
product of the moduli, so each product is accurate relative to its modulus. The
values of this recurrence can be far larger or smaller than the power itself, so
outside the ordinary range they are evaluated on numbers whose exponents are
kept apart as integers, and a coefficient of the power is lost only where it is
itself out of the range of the floats.

**Sums can overflow.** Sums are plain floating-point sums. A product, and the
conversion between loop-basis coefficients and split form that products, powers,
inverses, and the predicates of loop-basis operands pass through, form sums of
up to `2D` coefficients, and a complex product forms a difference of two
products, so they can overflow when coefficients are within a factor `4D` of the
largest float, and operands should be scaled away from the exponent limits. An
infinity or a NaN in one coordinate of the split form propagates through the
transform and makes every loop-basis coefficient of the record an infinity or a
NaN.

---

## Exceptions

pycdloop reports errors with standard Python exceptions:

- `TypeError`

  - An unsupported operand or scalar type, including a scalar of another field
    as an operand, or given to `scalar` or `from_terms`.
  - An `ElementArray` combined with a `SplitArray`.
  - Direct construction of `Element`, `ElementArray`, `SplitArray`, or
    `BinaryFieldElement`.
  - An operation outside the loop algebras [where powers are
    defined](#where-powers-are-defined).
  - An operation that is not defined in characteristic 2.
  - A non-integer exponent, index, level, or degree.
  - A `field` that is not a field object, or is combined with `char` or `deg`.
  - A non-string algorithm name.
  - A wrong `out` dtype.

- `ValueError`

  - Mismatched parents.
  - Two `BinaryFieldElement` scalars of different fields, or a scalar of another
    field among coefficients given to a constructor or assigned to an item.
  - A wrong coefficient shape, or a broadcast failure.
  - An unknown algorithm name, including `"split"` in characteristic 2 and
    `"lift"` elsewhere.
  - A subset that is not a subloop where one is required, or a map that is not a
    bijection.
  - A table requested for too large a loop.
  - A degree out of range or a reducible modulus of a `BinaryField`.
  - A lifted product for a loop beyond the bound of [Multiplication by
    lifting](mathematics.md#multiplication-by-lifting).
  - `secure=True` over R or C.
  - The truth value of an array.

- `IndexError`

  A basis index out of range, or too many indices for the batch axes of an
  array.

- `ZeroDivisionError`

  Division by a zero scalar, or the inverse or a negative power of an element
  that has none.

- `OverflowError`

  A total doubling level above 62, `len` of a loop of total level 62, or a loop
  algebra outside characteristic 2 for a loop beyond the storage limits of
  pyfastcd.

- `MemoryError`

  The algebra of a central product too large for its index and sign tables.

- `numpy.linalg.LinAlgError`

  Over R and C, a rank or null space of a matrix with NaN or infinite entries,
  in `linalg`, the annihilators, and `is_zero_divisor` where it computes a rank.

### Floating-point conditions

Operators and products do not raise on NaN, infinity, or overflow of
coefficients over R and C. They follow NumPy's floating-point error state, so by
default an overflow emits a `RuntimeWarning`. `linalg.row_reduce` and
`pycdloop.codes` do not raise on NaN or infinite entries either, and their
results are then without meaning.

---

## Threads

**Immutable objects.** Loops, algebras, fields, scalars, and loop elements are
immutable and can be shared between threads. Loops, fields, and algebras are
cached, and each is built once, so threads that construct the same loop, field,
or algebra at the same time all receive the same object.

**Elements.** An `Element` derives its second form, or in characteristic 2 the
transforms of its lifted coefficients, on first use and stores it. Two threads
that do so at once compute the same value, so sharing elements is safe under the
global interpreter lock (GIL).

**Arrays.** The storage of an array must not be modified while an operation that
uses it is running, because the compiled kernels release the GIL, so other
threads can run during them: those of pycdloop release it for the transform on
large batches and always for the change of coordinates, and those of pyfastcd on
large batches. The temporaries that a product of arrays reuses are kept
separately by each thread.

pyfastcd does not support free-threaded CPython builds, and neither does
pycdloop.

---

## Versioning and stability

pycdloop follows semantic versioning. The public surface consists of the names
in `pycdloop.__all__` together with the members documented on this page, while
modules and names with a leading underscore are private and may change without
notice. Before version 1.0.0 a minor release may change public behavior, and
every such change is recorded in `CHANGELOG.md`.

Two kinds of difference may come with any release that improves performance or
correctness:

- floating-point results may round differently;
- `"auto"` may select a different algorithm.

---

## Typing

pycdloop ships inline type annotations, a stub for its compiled module, and a
`py.typed` marker. Loops, loop elements, `BinaryField`, and the functions of
`linalg`, `codes`, and `testing` have concrete return annotations, except the
sign functions `twist`, `commutator_sign`, and `associator_sign`, whose result
depends on the arguments, and most parameters are annotated `Any`. The
arithmetic of `Element`, `ElementArray`, and `SplitArray` is shared between the
three classes and annotated with `Any`, since the result type depends on the
operand types and, for the decomposition methods, on the coefficient field.
