![pycdloop](https://git.tanuki-cd.com/algebraity/pycdloop/raw/branch/main/assets/pycdloop.png)

# pycdloop - v0.1.0

pycdloop is a Python package for arithmetic in Cayley–Dickson loops, their central products, and the loop algebras of both over the real numbers, the complex numbers, and finite fields. It is built on [pyfastcd](https://pypi.org/project/pyfastcd), the Python interface of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD), and uses its quasilinear Cayley–Dickson multiplication to multiply in the loop algebra of a Cayley–Dickson loop, of dimension `2N`, with `O(N log N)` operations. The loop algebras of central products are multiplied by the same methods, with a further logarithmic factor for each further factor that grows. The algorithms are those of the accompanying paper *Quasilinear multiplication in Cayley–Dickson loop algebras* (H. Lemley), which will be publicly available in the near future.

The canonical source repository is
[git.tanuki-cd.com/algebraity/pycdloop](https://git.tanuki-cd.com/algebraity/pycdloop).

## Convention

```text
Q_n := { +e_p, -e_p : 0 <= p < 2^n },     e_p e_q = sigma_n(p, q) e_(p XOR q)
```

The Cayley–Dickson loop `Q_n` consists of the signed basis elements of the Cayley–Dickson algebra `A_n` of dimension `N = 2^n`, with the basis and signs of fastCD. Its loop algebra has one basis element `[x]` for each loop element `x`, and an element is stored as an array of shape `(2, N)`, with a further axis for the polynomial coefficients over an extension field: row 0 holds the coefficients of `[e_p]` and row 1 those of `[-e_p]`. Where 2 is invertible in the coefficient field `F`, the algebra decomposes as `F^N ⊕ A_n`, and products are evaluated in that decomposition, so that a Walsh–Hadamard transform turns the first summand into coordinatewise multiplication, and pyfastcd multiplies in the second. Over the fields F_(2^k) there is no such decomposition, and a product is instead lifted to a field of large odd characteristic, computed there in the same way, and reduced. [Conventions and results](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/mathematics.md) gives the details, including central products and characteristic 2.

## Installation

pycdloop requires CPython 3.10 or newer, NumPy 1.26 or newer and pyfastcd 1.4.0 or newer, below 2. To install it from PyPI with `pip`, run:

```sh
pip install pycdloop
```

Wheels are provided for CPython 3.10 through 3.14 on Linux, macOS, and Windows. To build and install it from a checkout of this repository, which also requires a C11 compiler and the Python development headers, run:

```sh
python -m pip install .
```

## Usage

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

An `Element` is one immutable element of a loop algebra, an `ElementArray` holds a batch in a NumPy coefficient array of shape `(..., 2, N)`, followed over an extension field by the polynomial axis, and a `SplitArray` holds a batch in the decomposed coordinates, where products need no transform. See the [quick start](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/quickstart.md) and the [API reference](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/api.md).

## Benchmark

The data in this table contains the mean microseconds per measured product in batches, over real coefficients, measured on 2026-11-14 on one logical CPU of an Intel Xeon Gold 6148. `2N` is the dimension of the loop algebra. `split` is the default algorithm on operands in the loop basis, `split form` is the same product on operands already in the decomposed coordinates, `table` evaluates the definition, and the last column is the pyfastcd product that every loop-algebra product contains. Timing is done in Python and includes the call overhead of the package. The protocol, the single-product and finite-field results, and the central-product results are in [Benchmark](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/benchmark.md).

| 2N | split | split form | table | Cayley–Dickson product alone |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.20 | 0.085 | 0.83 | 0.011 |
| 32 | 0.31 | 0.15 | 8.84 | 0.063 |
| 128 | 1.15 | 0.62 | 194.3 | 0.45 |
| 512 | 5.50 | 3.29 | 3,735 | 2.61 |
| 2048 | 28.3 | 18.1 | 43,635 | 15.0 |

## Numerical stability

Accuracy was measured against an exact reference on 4,764 operand pairs. The error of a computed product `fl(xy)` is reported as `eta = ||fl(xy) - xy||_2 / (u ||x||_2 ||y||_2)`, with `u = 2^-53`. The median `eta` on dense real inputs is:

| 2N | 4 | 64 | 512 | 2048 |
| --- | ---: | ---: | ---: | ---: |
| split | 0.78 | 1.90 | 2.47 | 2.73 |
| table | 0.67 | 1.86 | 4.92 | 9.68 |

On this corpus the error of `split` grows with the logarithm of the dimension, and its largest value is 5.8 over R and 4.3 over C. Arithmetic over finite fields is exact and thus does not need an error analysis. The method and the full results are in [Numerical stability](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/stability.md).

## Documentation

- [Quick start](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/quickstart.md)
- [Python API reference](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/api.md)
- [Conventions and results](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/mathematics.md)
- [Benchmark](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/benchmark.md)
- [Numerical stability](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/docs/stability.md)

## License and Attribution

All code is released under the [MIT License](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/LICENSE) and is presented with no warranty or guarantees. To cite the software, use the metadata in [CITATION.cff](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/CITATION.cff). The changelog is in [CHANGELOG.md](https://git.tanuki-cd.com/algebraity/pycdloop/src/branch/main/CHANGELOG.md).

The loop-algebra algorithms are due to Harrison Lemley and will be available in the forthcoming paper *Quasilinear multiplication in Cayley–Dickson loop algebras*. The Cayley–Dickson arithmetic is that of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD) and its quasilinear algorithm, [*Quasilinear multiplication in the Cayley–Dickson tower*](https://arxiv.org/abs/2609.11588).

