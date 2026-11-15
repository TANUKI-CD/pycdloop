# Changelog

All notable changes to pycdloop are recorded in this file. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); the format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The version is
defined in `src/pycdloop/_version.py`.

## [0.1.0] - 2026-11-14

### Added

- `CDLoop` and `CentralProduct`: the Cayley–Dickson loops `Q_n` and their
  central products, with element arithmetic, vectorised sign functions
  (`twist`, `commutator_sign`, `associator_sign`), multiplication tables,
  the center, nuclei and commutant, subloop generation, normality and
  cosets, translations and automorphism checks, identity tests, and export
  to GAP.
- `LoopAlgebra`: loop algebras of both kinds of loop over R, C and finite
  fields, with the selectors of pyfastcd. The fields of odd characteristic
  are those of pyfastcd; `char=2` selects F_(2^k) for `k` up to 12.
- `BinaryField` and `BinaryFieldElement`: the fields F_(2^k) and their
  scalars, with the properties and scalar constructors of
  `pyfastcd.FiniteField` and Conway polynomials as default moduli.
- `Element`, `ElementArray` and `SplitArray`: immutable elements, NumPy-backed
  batches in the loop basis, and batches in the split form in which products
  need no transform.
- The split multiplication algorithm, which uses three Walsh–Hadamard
  transforms, one coordinatewise product and one product of signed
  components, and the table algorithm, which evaluates the definition.
- The lift algorithm for characteristic 2, where the algebra has no
  decomposition: a product over F_(2^k) is computed by the split algorithm
  over a large prime field or an extension of it, and reduced. `"auto"`
  chooses between it and the table algorithm by size.
- Multiplication of signed components of central-product algebras by three
  methods: through a field extension for a factor of level 1 (`"native"`),
  row by row (`"rows"`), and by the quasilinear schedule with coefficients in
  the remaining factors (`"schedule"`).
- The decomposition maps `kappa`, `epsilon` and `spectrum`, the central and
  primitive idempotents, a basis of the center, inverses, integer powers in
  `O(D + log k)` operations beyond the coordinatewise powers of the
  spectrum, zero-divisor tests, annihilators, minimal polynomials and
  multiplication matrices.
- In characteristic 2, powers, inverses, nilpotency and zero-divisor tests,
  and minimal polynomials for every loop, central products included, from
  the identity `x^4 = aug(x)^4`. The operations that need the decomposition
  raise `TypeError` there.
- `wht` over R, C and finite fields of odd characteristic, and
  `dyadic_convolve` over every supported field.
- The extension module `pycdloop._core`, with compiled kernels for the
  Walsh–Hadamard transform and for the change of coordinates between the
  loop basis and the split form, over R, C and finite fields. Their results
  are bit-identical to the reference evaluation in `_transform.py`.
- `pycdloop.linalg` (rank, row reduction and null spaces over the coefficient
  fields), `pycdloop.codes` (translate matrices, generator matrices of left,
  right and two-sided ideals, a test for ideals, and export to GAP and
  SageMath over every finite field) and `pycdloop.testing` (sampled identity
  checks).
- Benchmark drivers for three protocols and a numerical stability study
  against an exact reference, with `tools/check_operation_counts.py` and
  `tools/check_archives.py`.
