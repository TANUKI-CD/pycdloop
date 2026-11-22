![pycdloop](https://git.tanuki-cd.com/algebraity/pycdloop/raw/branch/main/assets/pycdloop.png)

# pycdloop - v0.1.1

pycdloop is a Python package for arithmetic in Cayley-Dickson loops, their central products, and the loop algebras of both over the real numbers, the complex numbers, and finite fields. It is built on [pyfastcd](https://pypi.org/project/pyfastcd), the Python interface of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD), and uses its quasilinear Cayley-Dickson multiplication to multiply in the loop algebra of a Cayley-Dickson loop, of dimension `2N`, with `O(N log N)` operations. The loop algebras of central products are multiplied by the same methods, with a further logarithmic factor for each further factor. The algorithms are those of the accompanying paper *Quasilinear multiplication in Cayley-Dickson loop algebras* (H. Lemley), which will be released publicly in the near future.

The canonical source repository is
[git.tanuki-cd.com/algebraity/pycdloop](https://git.tanuki-cd.com/algebraity/pycdloop).

## Convention

The Cayley-Dickson loop `Q_n` consists of the basis elements of the Cayley-Dickson algebra `A_n` of dimension `N = 2^n` and their negatives, which multiply as they do in `A_n`, with the basis and signs of fastCD:

```text
Q_n := { +e_i, -e_i : 0 <= i < 2^n },     e_i e_j = sigma_n(i, j) e_(i XOR j),
```

where the sign `sigma_n(i, j)` is `+1` or `-1` and follows from the doubling formula of `A_n`.

The loop algebra of `Q_n` over a field `F` has one basis element `[x]` for each loop element `x`, so its dimension is `2N`, and pycdloop stores an element as an array of shape `(2, N)`, with a further axis for the polynomial coordinates over an extension field: row 0 holds the coefficients of `[e_i]` and row 1 those of `[-e_i]`.

Where 2 is invertible in `F`, the algebra decomposes as `F^N (+) A_n`, and products are evaluated in that decomposition: a Walsh-Hadamard transform turns the product in the first summand into coordinatewise multiplication, and pyfastcd multiplies in the second. Over the fields F_(2^k) there is no such decomposition, so a product is instead lifted to a field of large odd characteristic, computed there in the same way, and reduced.

[Conventions and results](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/mathematics.md) gives the details, including central products and characteristic 2.

## Installation

pycdloop requires CPython 3.10 or newer, NumPy 1.26 or newer, and pyfastcd 1.4.0 or a later 1.x release. To install it from PyPI with `pip`, run:

```sh
pip install pycdloop
```

Wheels are provided for CPython 3.10 through 3.14 on Linux, macOS, and Windows. To build and install it from a checkout of this repository, which also requires a C11 compiler and the Python development headers, run:

```sh
python -m pip install .
```

## Usage

After importing `pycdloop`, you can create loops with `CDLoop` and `CentralProduct` and their loop algebras with `LoopAlgebra`. Here is an example over the real numbers, which ends by creating the loop algebra of a central product over F_13 and that of the octonion loop over F_256:

```python
import numpy as np
import pycdloop as cdl

Q = cdl.CDLoop(3)                          # the octonion loop, of order 16
e1, e2 = Q.basis(1), Q.basis(2)
e1 * e2, e2 * e1                           # (e_3, -e_3)

A = cdl.LoopAlgebra(Q)                     # its loop algebra over R
x = A.basis(e1) + 2 * A.basis(-e2)         # [e_1] + 2 [-e_2]
y = A.random(0)

product = x * y
power = x ** 5
x.kappa(), x.epsilon()                     # group and signed components

X = A.asarray(np.ones((1000, 2, 8)))       # a batch of 1000 elements
Z = X * y                                  # batch product, leading axes broadcast

B = cdl.LoopAlgebra(cdl.CentralProduct(2, 3), char=13)   # Q_2 * Q_3 over F_13
C = cdl.LoopAlgebra(Q, char=2, deg=8)                    # Q_3 over F_256
```

An `Element` is one immutable element of a loop algebra. An `ElementArray` holds a batch of elements in a NumPy coefficient array of shape `(..., 2, N)`, followed over an extension field by the polynomial axis, and a `SplitArray` holds a batch in split form, the coordinates of the decomposition, where products need no transform. See the [quick start](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/quickstart.md) and the [API reference](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/api.md) for more information on how each class works.

## Benchmark

pycdloop chooses its multiplication algorithm from single-core benchmarks of pycdloop 0.1.0 as it ships, timed through the Python interface on an Intel Xeon Gold 6148 in a Dell PowerEdge T440, with every timed product verified against an independent implementation. The [benchmark documentation](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/benchmark.md) describes the method, gives the full results, and explains how to rerun them.

Mean microseconds per product in batches over real coefficients, where `2N` is the dimension of the loop algebra, split form is the product of operands already in split form, and the last column is the Cayley-Dickson product of dimension `N` that every product contains:

| 2N | split | split form | table | Cayley-Dickson product alone |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.20 | 0.088 | 0.82 | 0.012 |
| 32 | 0.31 | 0.15 | 8.95 | 0.062 |
| 128 | 1.13 | 0.61 | 197.2 | 0.45 |
| 512 | 5.69 | 3.34 | 3,731 | 2.59 |
| 2048 | 28.2 | 17.7 | 43,789 | 14.5 |

The split algorithm is faster than the table algorithm at every tested dimension, by a factor of 4 at dimension 8 and of 1555 at dimension 2048, and it is the default over R, C, and the finite fields of odd characteristic. In characteristic 2 the default is the lift or the table algorithm, depending on the size of the loop, the number of products, and the degree of the field.

## Numerical stability

The split algorithm forms its long sums as balanced trees, in its Walsh-Hadamard transforms and in the Cayley-Dickson product of pyfastcd, while the table algorithm adds the terms of each coefficient along one axis, so the error of the split algorithm grows more slowly with the dimension. Measured against exact products on 4,764 operand pairs, with the error `eta = ||fl(xy) - xy||_2 / (u ||x||_2 ||y||_2)` and `u = 2^-53`, the median error on dense real inputs is:

| 2N | 4 | 64 | 512 | 2048 |
| --- | ---: | ---: | ---: | ---: |
| split | 0.78 | 1.90 | 2.47 | 2.73 |
| table | 0.67 | 1.86 | 4.92 | 9.68 |

The largest error of the split algorithm in the whole corpus is 5.8 over R and 4.3 over C, against 15.8 and 9.4 for the table algorithm; these figures describe the inputs tested, not a worst case, and [Numerical stability](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/stability.md) gives the method and the full results. Arithmetic over finite fields is exact and needs no error analysis.

## Documentation

- [Quick start](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/quickstart.md)
- [Python API reference](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/api.md)
- [Conventions and results](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/mathematics.md)
- [Benchmarks](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/benchmark.md)
- [Numerical stability](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/stability.md)

## License and Attribution

All code is released under the [MIT License](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/LICENSE) and is presented with no warranty or guarantees. To cite the software, use the metadata in [CITATION.cff](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/CITATION.cff). The changelog is in [CHANGELOG.md](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/CHANGELOG.md).

The loop-algebra algorithms are due to Harrison Lemley and will be available in the forthcoming paper *Quasilinear multiplication in Cayley-Dickson loop algebras*. The Cayley-Dickson arithmetic is that of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD) and its quasilinear algorithm, [*Quasilinear multiplication in the Cayley-Dickson tower*](https://arxiv.org/abs/2609.11588).
