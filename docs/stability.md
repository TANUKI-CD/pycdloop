# Numerical stability

Over the real and complex numbers, pycdloop computes a loop-algebra product in
IEEE binary64 arithmetic, where every addition and multiplication rounds its
result, so its two algorithms return slightly different results for the same
operands. The split algorithm transforms its operands, multiplies two components
separately, and recombines them, while the table algorithm evaluates the
definition of the product directly, and this page investigates whether the split
algorithm loses accuracy compared with the table algorithm, and whether a chain
of products kept in split form stays as accurate as the same chain converted
back to the loop basis after every product. Over finite fields, in odd
characteristic and in characteristic 2, every result is exact, so the question
does not arise there.

The conclusion is that, measured against exact products, the split algorithm
stays accurate at every dimension tested. On inputs with random coefficients,
its median error grows only from 0.78 at dimension 4 to 2.73 at dimension 2048
over R, in the units of [Error measures](#error-measures), no faster than the
logarithm of the dimension, while that of the table algorithm grows from 0.67 to
9.68, so that the table algorithm is the more accurate one through dimension 64
and the split algorithm from dimension 128 on. The sections below explain where
the split algorithm rounds, describe how accuracy was measured, report the
results for single products and for chains, and end with what the results mean
for code that uses pycdloop and with the limitations of the study.

## Where rounding enters

The split algorithm rounds in five places: in forming the two components, in the
Walsh-Hadamard transforms, in the coordinatewise product, in the Cayley-Dickson
product, and in the recombination, while its divisions by the quotient order and
by 2 are exact unless they underflow (see [The split
algorithm](mathematics.md#the-split-algorithm)). Three properties of these steps
predict the measurements below.

**The transform is a balanced sum.** Each output of a Walsh-Hadamard transform
is a signed sum of `D` terms, formed by butterflies as a balanced tree of depth
`log2 D`, so the number of additions that any one term passes through grows only
with the logarithm of the dimension. The compiled kernel changes the order of
the loops but not of the additions on any coordinate, so this holds for it as
well, and the sums inside the Cayley-Dickson product are those of pyfastcd,
which are balanced too (see [fastCD's stability
study](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/stability.md)).

**The table algorithm sums sequentially.** Each coefficient of a product by the
table algorithm is a sum of `2D` products of coefficients, added by NumPy along
one axis, so its rounding depth grows with the dimension, which is why its error
overtakes that of the split algorithm.

**The two components are rounded separately.** The split algorithm forms `a + b`
and `a - b`, multiplies the two components independently, and recombines them,
so when one component is much smaller than the other, its rounding error is
measured against the larger one. The families `signed_cancel`, `group_cancel`,
`group_ideal`, and `signed_ideal` below exercise this case.

## How accuracy was measured

### Error measures

The study measures errors with the Euclidean norm `||.||_2` of the loop-basis
coefficients, which over C is the Euclidean norm of the complex coefficients.
For operands `x` and `y` with exact product `z` and computed product `z_hat`,
and with `u = 2^-53`, the unit roundoff of binary64, it records four quantities:

```text
E_abs = ||z_hat - z||_2
E_rel = E_abs / ||z||_2                    (z nonzero)
eta   = E_abs / (u ||x||_2 ||y||_2)
chi   = ||x||_2 ||y||_2 / ||z||_2          (z nonzero)
```

The main measure is `eta`, the error in units of `u ||x||_2 ||y||_2`, the scale
set by the sizes of the operands, so that an `eta` of 1 is an error of one unit
roundoff times the product of the operand norms; for a chain of `k` operands,
`eta` divides `E_abs` by `u` times the product of the `k` norms. Unlike the
relative error, `eta` is defined for every operand pair, which matters because a
loop algebra has zero divisors at every dimension: the product of an element
whose signed component is zero with one whose group component is zero vanishes.

The other two measures are related by `E_rel = u eta chi`, where `chi` is large
when the exact product is much smaller than its operands, as it is near a zero
divisor, so that there even a small `eta` means a large relative error.

### Exact reference

`benchmarks/stability.py` computes every reference product exactly, without
pycdloop or pyfastcd arithmetic, by converting each binary64 coefficient to an
integer times a power of two and summing the product of the loop algebra from
its definition in arbitrary-precision integers. The basis signs used in this
reference come from the doubling formula in integer arithmetic and are checked
on all basis pairs through dimension 16 against a separate rational
implementation of the formula. Norms and errors are evaluated with MPFR at 256
bits, and selected products are repeated at 512 bits as a check. The script
needs gmpy2, which the package itself does not use.

### Computations compared

The study compares two algorithms on single products and two ways of evaluating
chains:

- **S** is the split algorithm and **T** the table algorithm, both applied to
  loop-basis arrays of one record. The Cayley-Dickson product inside S uses
  pyfastcd's automatic selection, and the study records the algorithm at every
  level: over R it is the single-product choice, which is quasilinear at level 1
  and from level 5 on and direct at levels 2 through 4, and over C it is the
  batch choice, which is direct through level 2 and quasilinear above.
- A left-nested chain `((x_1 x_2) x_3) ... x_k` of `k = 2`, `4`, or `8` operands
  is evaluated twice, and both results are compared with the exact value of the
  chain. **L** multiplies `Element` operands, so that every intermediate result
  stays in split form and only the final result is converted to the loop basis,
  while **R** multiplies `ElementArray` operands, so that every intermediate
  result is converted to the loop basis and back.

### Inputs

A *case* is one operand pair at one dimension, multiplied by S and T. The study
has 4,764 cases, 2,440 real from 22 families and 2,324 complex from 20 families,
each family exercising one kind of input:

- **Dense:** unit-norm Gaussian coefficients, uniform coefficients in `[-1, 1)`,
  and exactly representable Rademacher coefficients. The dense rows of the
  tables pool the first two.
- **Sparse:** supports of one, four, and `D/8` loop elements, the last of them
  at least one.
- **Correlated:** `y = x`, and `y = +/-x + 2^-40 noise`.
- **Magnitude:** alternating magnitudes over the exponents `-24` to `24`.
- **Component cancellation:** `signed_cancel`, where `b = a + 2^-40 noise` and
  the signed component is tiny, and `group_cancel`, where `b` is nearly `-a` and
  the group component is tiny.
- **Ideals:** `group_ideal` (`b = a`, signed component zero), `signed_ideal`
  (`b = -a`, group component zero), and `cross_ideal`, one operand from each,
  whose exact product is zero.
- **Structure:** `idempotent` (a sum of primitive idempotents times a dense
  element), `character` (an operand whose spectrum has one nonzero coordinate),
  and `scalar_only`.
- **Zero divisors (real, dimension 32 and above):** the sedenion zero divisor
  `(e1 + e10)(e4 - e15) = 0` lifted to the signed components, and perturbations
  of it by `2^-20`, `2^-40`, and `2^-52`.
- **Isotropic (complex):** signed components with `Q(d) = 0`, exact and
  perturbed by `2^-40`.

Real dimensions run from 4 to 2048 and complex dimensions from 4 to 512, with
256 dense cases per dimension through 64, 64 at dimensions 128 and 256, and 16
above. The chain experiment has 1,068 chains over R, through dimension 512, and
1,008 over C, through dimension 128. Seeds derive from
`numpy.random.SeedSequence(20261111, ...)`, and every operand is stored with the
raw results of the study.

### Host and software

The study ran on an Intel Xeon Gold 6148 under Linux with Python 3.14.8, NumPy
2.5.4, pyfastcd 1.4.0, and pycdloop 0.1.0, and its record lists the hash of
every measured source file.

## Accuracy of products and chains

The tables below are generated by `benchmarks/stability_report.py` from the
archived results of the *full profile* of the study, its complete run, as
opposed to the short smoke profile that only tests the tooling. They come in a
real set and a complex set, each with four views: the two algorithms on dense
inputs at every dimension, the growth of the error of S with the dimension,
every family at dimension 512, and the chains.

In every table, `N` is the dimension of the loop algebra, which the other pages
of this documentation write as `2D`, and a cell of the form "S / T" or "L / R"
gives one statistic for the two computations. The column "pairs" counts cases,
"p99" is the 99th percentile, "S better / tied" gives the fraction of cases in
which S has a smaller exact error than T and the fraction in which the two are
equal, and "L at least as accurate / tied" does the same for the chains, of `k`
operands each. The growth tables divide the median `eta` by `log2 N`, so that a
constant ratio means logarithmic growth.

<!-- BEGIN STABILITY ACCURACY -->

4,764 operand pairs and 2,076 chains from the full profile. S is the split algorithm and T is the table algorithm; both multiply the same stored operands. N is the dimension of the loop algebra, twice the dimension of the Cayley–Dickson algebra. Dense rows pool the Gaussian and uniform families, and "S better / tied" compares the exact errors of S and T. In the chain tables, L keeps a left-nested product of k operands in split form and converts once at the end, R returns to the loop basis after every product, and eta divides the error by the product of the operand norms. Empirical p99 values and maxima describe this corpus only.

**Real dense inputs**

| N | pairs | median eta S / T | p99 eta S / T | max eta S / T | S better / tied |
|---:|---:|:---|:---|:---|:---|
| 4 | 256 | 0.78 / 0.668 | 2.31 / 1.36 | 2.81 / 1.5 | 39% / 5% |
| 8 | 256 | 1.02 / 0.819 | 2.42 / 1.54 | 4.84 / 1.9 | 31% / 0% |
| 16 | 256 | 1.28 / 1.04 | 2.53 / 1.82 | 2.74 / 2 | 29% / 0% |
| 32 | 256 | 1.54 / 1.36 | 2.42 / 2.28 | 2.66 / 2.47 | 32% / 0% |
| 64 | 256 | 1.9 / 1.86 | 2.84 / 2.59 | 3.05 / 2.76 | 44% / 0% |
| 128 | 64 | 2.1 / 2.54 | 2.71 / 3.32 | 2.86 / 3.4 | 92% / 0% |
| 256 | 64 | 2.25 / 3.58 | 2.59 / 4.23 | 2.61 / 4.37 | 100% / 0% |
| 512 | 16 | 2.47 / 4.92 | 2.85 / 5.49 | 2.87 / 5.5 | 100% / 0% |
| 1024 | 16 | 2.61 / 6.91 | 2.88 / 7.23 | 2.9 / 7.24 | 100% / 0% |
| 2048 | 16 | 2.73 / 9.68 | 2.79 / 10.1 | 2.79 / 10.1 | 100% / 0% |

**Real growth with dimension**, S on dense inputs

| N | median eta | eta / log2 N | median relative error | max relative error |
|---:|---:|---:|---:|---:|
| 4 | 0.78 | 0.39 | 9.48e-17 | 2.56e-16 |
| 8 | 1.02 | 0.34 | 1.19e-16 | 5.86e-16 |
| 16 | 1.28 | 0.321 | 1.49e-16 | 2.9e-16 |
| 32 | 1.54 | 0.308 | 1.74e-16 | 3.05e-16 |
| 64 | 1.9 | 0.317 | 2.13e-16 | 3.29e-16 |
| 128 | 2.1 | 0.3 | 2.36e-16 | 3.03e-16 |
| 256 | 2.25 | 0.282 | 2.55e-16 | 2.99e-16 |
| 512 | 2.47 | 0.274 | 2.71e-16 | 3.05e-16 |
| 1024 | 2.61 | 0.261 | 2.84e-16 | 3.22e-16 |
| 2048 | 2.73 | 0.248 | 2.99e-16 | 3.16e-16 |

**Real families at N=512**

| family | pairs | median eta S / T | max relative error S | S better / tied |
|:---|---:|:---|---:|:---|
| gaussian | 8 | 2.57 / 4.86 | 3.05e-16 | 100% / 0% |
| uniform | 8 | 2.37 / 5.04 | 2.88e-16 | 100% / 0% |
| rademacher | 2 | 0 / 0 | 0 | 0% / 100% |
| sparse_one | 2 | 0.527 / 0.527 | 6.75e-17 | 0% / 100% |
| sparse_four | 2 | 1.49 / 0.419 | 1.8e-16 | 0% / 0% |
| sparse_eighth | 2 | 2 / 0.729 | 2.22e-16 | 0% / 0% |
| repeated | 2 | 3.21 / 7.89 | 2.58e-16 | 100% / 0% |
| correlated_plus | 2 | 2.78 / 5.77 | 2.4e-16 | 100% / 0% |
| correlated_minus | 2 | 2.65 / 6.66 | 2.14e-16 | 100% / 0% |
| alternating_magnitude | 2 | 1.68 / 0.619 | 1.77e-16 | 0% / 0% |
| signed_cancel | 2 | 2.98 / 6.48 | 2.51e-16 | 100% / 0% |
| group_cancel | 2 | 3.64 / 6.55 | 3.09e-16 | 100% / 0% |
| group_ideal | 2 | 2.87 / 6.7 | 2.44e-16 | 100% / 0% |
| signed_ideal | 2 | 3.5 / 6.55 | 2.73e-16 | 100% / 0% |
| cross_ideal | 2 | 0 / 0 | — | 0% / 100% |
| idempotent | 2 | 1.69 / 5.68 | 1.94e-16 | 100% / 0% |
| character | 2 | 0.5 / 2.39 | 6.94e-16 | 100% / 0% |
| scalar_only | 2 | 0.566 / 0.566 | 9.13e-17 | 0% / 100% |
| zero_divisor | 2 | 0 / 0 | — | 0% / 100% |
| near_zero_divisor_20 | 2 | 3.03 / 1.85 | 2.47e-10 | 0% / 0% |
| near_zero_divisor_40 | 2 | 2.71 / 0.75 | 0.000251 | 0% / 0% |
| near_zero_divisor_52 | 2 | 1.82 / 4.91e-16 | 0.615 | 0% / 0% |

**Real chains**

| N | k | chains | median eta L / R | max eta L / R | L at least as accurate / tied |
|---:|---:|---:|:---|:---|:---|
| 4 | 2 | 64 | 0.841 / 0.841 | 2.65 / 2.65 | 100% / 100% |
| 4 | 4 | 64 | 1.11 / 1.46 | 4.48 / 4.48 | 83% / 19% |
| 4 | 8 | 64 | 0.641 / 1.17 | 6.2 / 6.86 | 86% / 5% |
| 8 | 2 | 64 | 1.12 / 1.12 | 2.88 / 2.88 | 100% / 100% |
| 8 | 4 | 64 | 1.37 / 1.81 | 3.61 / 4 | 80% / 3% |
| 8 | 8 | 64 | 1.47 / 1.86 | 5.87 / 6.07 | 75% / 2% |
| 16 | 2 | 64 | 1.43 / 1.43 | 2.38 / 2.38 | 100% / 100% |
| 16 | 4 | 64 | 1.75 / 2.08 | 10.5 / 10.6 | 80% / 0% |
| 16 | 8 | 64 | 2.36 / 2.61 | 6.68 / 18.1 | 64% / 0% |
| 32 | 2 | 64 | 1.61 / 1.61 | 2.53 / 2.53 | 100% / 100% |
| 32 | 4 | 64 | 2.08 / 2.46 | 6.6 / 6.24 | 75% / 0% |
| 32 | 8 | 64 | 2.57 / 3.2 | 7.09 / 10 | 72% / 0% |
| 64 | 2 | 64 | 2.09 / 2.09 | 2.81 / 2.81 | 100% / 100% |
| 64 | 4 | 64 | 2.96 / 3.33 | 6.5 / 6.52 | 80% / 0% |
| 64 | 8 | 64 | 3.59 / 4.25 | 45.4 / 61.2 | 81% / 0% |
| 128 | 2 | 16 | 2.16 / 2.16 | 2.5 / 2.5 | 100% / 100% |
| 128 | 4 | 16 | 3.17 / 3.61 | 4.13 / 4.38 | 81% / 0% |
| 128 | 8 | 16 | 4.85 / 5.22 | 11.8 / 28.9 | 81% / 0% |
| 256 | 2 | 16 | 2.31 / 2.31 | 2.68 / 2.68 | 100% / 100% |
| 256 | 4 | 16 | 3.44 / 3.83 | 4.62 / 5.04 | 100% / 0% |
| 256 | 8 | 16 | 4.81 / 5.15 | 10.3 / 11.3 | 75% / 0% |
| 512 | 2 | 4 | 2.47 / 2.47 | 2.56 / 2.56 | 100% / 100% |
| 512 | 4 | 4 | 3.53 / 4.2 | 4.11 / 4.6 | 100% / 0% |
| 512 | 8 | 4 | 5.93 / 6.63 | 9.25 / 11.7 | 100% / 0% |

**Complex dense inputs**

| N | pairs | median eta S / T | p99 eta S / T | max eta S / T | S better / tied |
|---:|---:|:---|:---|:---|:---|
| 4 | 256 | 0.987 / 0.785 | 2.07 / 1.4 | 2.84 / 1.53 | 32% / 0% |
| 8 | 256 | 1.17 / 0.954 | 2.04 / 1.49 | 2.51 / 1.99 | 26% / 0% |
| 16 | 256 | 1.61 / 1.14 | 2.59 / 1.74 | 3.26 / 1.86 | 8% / 0% |
| 32 | 256 | 1.81 / 1.45 | 2.45 / 1.98 | 2.76 / 2.17 | 10% / 0% |
| 64 | 256 | 2.02 / 1.91 | 2.53 / 2.31 | 2.63 / 2.45 | 34% / 0% |
| 128 | 64 | 2.22 / 2.57 | 2.64 / 2.97 | 2.86 / 2.99 | 94% / 0% |
| 256 | 64 | 2.38 / 3.59 | 2.61 / 3.89 | 2.63 / 3.93 | 100% / 0% |
| 512 | 16 | 2.47 / 4.84 | 2.66 / 5.05 | 2.66 / 5.05 | 100% / 0% |

**Complex growth with dimension**, S on dense inputs

| N | median eta | eta / log2 N | median relative error | max relative error |
|---:|---:|---:|---:|---:|
| 4 | 0.987 | 0.493 | 1.11e-16 | 2.73e-16 |
| 8 | 1.17 | 0.391 | 1.33e-16 | 2.79e-16 |
| 16 | 1.61 | 0.402 | 1.8e-16 | 3.53e-16 |
| 32 | 1.81 | 0.362 | 2.02e-16 | 3.24e-16 |
| 64 | 2.02 | 0.337 | 2.25e-16 | 2.82e-16 |
| 128 | 2.22 | 0.317 | 2.46e-16 | 3.11e-16 |
| 256 | 2.38 | 0.297 | 2.63e-16 | 2.97e-16 |
| 512 | 2.47 | 0.275 | 2.78e-16 | 3.02e-16 |

**Complex families at N=512**

| family | pairs | median eta S / T | max relative error S | S better / tied |
|:---|---:|:---|---:|:---|
| gaussian | 8 | 2.61 / 4.83 | 3.02e-16 | 100% / 0% |
| uniform | 8 | 2.44 / 4.87 | 2.79e-16 | 100% / 0% |
| rademacher | 2 | 0 / 0 | 0 | 0% / 100% |
| sparse_one | 2 | 1.52 / 0.624 | 1.95e-16 | 0% / 0% |
| sparse_four | 2 | 1.6 / 0.466 | 1.79e-16 | 0% / 0% |
| sparse_eighth | 2 | 2.18 / 0.723 | 2.48e-16 | 0% / 0% |
| repeated | 2 | 2.63 / 4.73 | 3.1e-16 | 100% / 0% |
| correlated_plus | 2 | 2.5 / 4.86 | 2.82e-16 | 100% / 0% |
| correlated_minus | 2 | 2.44 / 4.94 | 2.76e-16 | 100% / 0% |
| alternating_magnitude | 2 | 2.79 / 8.9 | 1.69e-16 | 100% / 0% |
| signed_cancel | 2 | 3.42 / 7.26 | 2.7e-16 | 100% / 0% |
| group_cancel | 2 | 3.93 / 6.74 | 3.11e-16 | 100% / 0% |
| group_ideal | 2 | 3.31 / 6.74 | 2.62e-16 | 100% / 0% |
| signed_ideal | 2 | 3.55 / 6.89 | 2.86e-16 | 100% / 0% |
| cross_ideal | 2 | 0 / 0 | — | 0% / 100% |
| idempotent | 2 | 1.81 / 5.76 | 1.98e-16 | 100% / 0% |
| character | 2 | 1.33 / 3.94 | 3.17e-16 | 100% / 0% |
| scalar_only | 2 | 0.38 / 0.38 | 5.36e-17 | 0% / 100% |
| isotropic | 2 | 1.16 / 0.671 | 1.33e-16 | 0% / 0% |
| near_isotropic | 2 | 1.93 / 6.94 | 2.29e-16 | 100% / 0% |

**Complex chains**

| N | k | chains | median eta L / R | max eta L / R | L at least as accurate / tied |
|---:|---:|---:|:---|:---|:---|
| 4 | 2 | 64 | 1.07 / 1.07 | 2.07 / 2.07 | 100% / 100% |
| 4 | 4 | 64 | 1.53 / 1.63 | 6.25 / 4.21 | 67% / 6% |
| 4 | 8 | 64 | 1.2 / 1.58 | 6.85 / 5.27 | 83% / 2% |
| 8 | 2 | 64 | 1.29 / 1.29 | 2.14 / 2.14 | 100% / 100% |
| 8 | 4 | 64 | 1.56 / 1.85 | 4.18 / 4.45 | 77% / 0% |
| 8 | 8 | 64 | 1.79 / 2.33 | 5.69 / 6.52 | 78% / 0% |
| 16 | 2 | 64 | 1.69 / 1.69 | 2.85 / 2.85 | 100% / 100% |
| 16 | 4 | 64 | 2.36 / 2.81 | 6.02 / 5.94 | 81% / 0% |
| 16 | 8 | 64 | 3.02 / 3.62 | 8.73 / 11.3 | 69% / 0% |
| 32 | 2 | 64 | 1.84 / 1.84 | 2.74 / 2.74 | 100% / 100% |
| 32 | 4 | 64 | 2.9 / 3.24 | 8.55 / 9.2 | 83% / 0% |
| 32 | 8 | 64 | 3.82 / 4.29 | 9.07 / 9.76 | 78% / 0% |
| 64 | 2 | 64 | 2.12 / 2.12 | 2.56 / 2.56 | 100% / 100% |
| 64 | 4 | 64 | 3.09 / 3.5 | 5.37 / 5.49 | 78% / 0% |
| 64 | 8 | 64 | 4.38 / 5.04 | 8.02 / 9.57 | 80% / 0% |
| 128 | 2 | 16 | 2.29 / 2.29 | 2.44 / 2.44 | 100% / 100% |
| 128 | 4 | 16 | 3.32 / 3.89 | 4.04 / 4.83 | 94% / 0% |
| 128 | 8 | 16 | 4.65 / 5.21 | 6.03 / 7.05 | 81% / 0% |

The archived CSV summary also gives, for every family and chain length, the count; the median, mean, p95, p99, and maximum of eta; the relative errors; and the paired comparison fractions. Groups with fewer than 100 samples have an empirical p99 close to their maximum and large sampling uncertainty.

<!-- END STABILITY ACCURACY -->

### Reading the tables

**Dense inputs.** The ratio of the median `eta` of S to `log2 N` falls slowly,
from 0.39 to 0.25 over R, so the error of S grows no faster than the logarithm
of the dimension on this corpus, and at dimension 2048 its median relative error
is `3.0e-16`. The median `eta` of T, by contrast, grows by a factor of about 1.4
for each doubling of the dimension at the largest dimensions.

**Small dimensions.** Through dimension 64, T is the more accurate algorithm on
dense inputs, because its sums are still short, while S always pays for forming
and recombining the two components, although the difference is at most a factor
of about 1.4 in the median.

**Sparse and scaled inputs.** T is also more accurate on sparse operands, except
those with a single nonzero coefficient, on which the two agree over R, and over
R on operands with alternating magnitudes. T never combines coefficients that
the product does not combine, whereas the transforms of S mix every coefficient
of an operand before multiplying.

**Component cancellation.** On the four families where one component is small or
zero, the median `eta` of S at dimension 512 lies between 2.9 and 3.6 over R,
against 2.5 on dense inputs and about half that of T, so rounding one component
separately from the other enlarges the error in units of `u ||x|| ||y||` by less
than a factor of 1.5. The factor is 1.6 over C at the same dimension and 2.0
over R at dimension 16.

**Exact zeros.** All 86 real and 50 complex pairs whose exact product is zero
gave an exactly zero result with both algorithms: the `cross_ideal` and
`zero_divisor` pairs and, over R, two `rademacher` pairs at dimensions 4 and 8.
For `cross_ideal` this is structural in S, since one operand has an exactly zero
signed component and the other an exactly zero spectrum.

**Near zero divisors.** For the perturbed zero divisors the product is small
compared with the operands, and the relative error of S is correspondingly
large: at dimension 512 it reaches 0.6 at a perturbation of `2^-52`, while the
median `eta` stays between 1.8 and 3.0. This is cancellation in the exact
product, not an instability, since `chi` is large for these inputs, and T is
more accurate here because the perturbed terms enter its sums directly.

**Chains.** A chain of two operands is one product, so L and R agree bit for bit
there. For four and eight operands the median `eta` of L is smaller than that of
R at every dimension, by 7% to 45% over R, and L is at least as accurate in 64%
to 100% of the real chains of a given length and dimension. The largest `eta` of
a chain is 45 for L and 61 for R, both at dimension 64 with eight operands over
R, because errors compound along a chain whichever way it is evaluated.

## What this means for your code

- **Use the default `split` algorithm.** On this corpus its error grows with the
  logarithm of the dimension, and from dimension 128 on it is the more accurate
  algorithm as well as the faster one.
- **Keep chains of products in split form,** with `Element` operands or a
  `SplitArray`: they are faster, and on this corpus usually more accurate.
- **Judge products that nearly vanish by `eta`.** When a product is expected to
  be small compared with its operands, the relative error can be large even
  though `eta` is small, as for any floating-point product with cancellation.
- **Scale operands away from the exponent limits,** because spectral coordinates
  are sums of `2D` coefficients and can overflow before the result does.
- **Expect exact results on integer coefficients** while no intermediate value
  exceeds `2^53`.

## Limitations

- The study is empirical: its medians and maxima describe this corpus, not all
  inputs, and no bound on the error of either algorithm is proved. Only the
  dense rows through dimension 64 have 100 samples or more, and every other
  group, every group of chains included, is too small to estimate tails.
- Nothing was measured above real dimension 2048 or complex dimension 512, and
  the operands have controlled exponent ranges, so overflow, underflow, and
  exceptional values were not studied.
- Only products and left-nested chains in the loop algebras of `Q_n` were
  measured: neither the loop algebras of central products nor powers, inverses,
  or other operations built from products.

## Checking and reproducing the results

The study is archived in
[benchmark-results/stability-20261115/](../benchmark-results/stability-20261115/),
with its record `study.json`, the summary tables and key figures, the charts in
PNG, SVG, and PDF, and `SHA256SUMS`, while the raw per-case results and the
operand corpus are not distributed.

To check the study, run from the repository root:

```sh
python tools/check_archives.py
```

The checker uses only the Python standard library. It compares every published
file with `SHA256SUMS` and checks that the tables on this page are the tables of
the study, and with `--full` it also requires the raw files and regenerates the
tables from the raw results, which needs NumPy.

To repeat the study, install the `benchmark` extras and run:

```sh
python benchmarks/stability_study.py --profile full --jobs 8
```

`--profile smoke` runs a few cases at small dimensions in seconds, to test the
tooling, and `--update-docs` rewrites the tables on this page from the new
results.
