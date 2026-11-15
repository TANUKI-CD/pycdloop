# Stability study, 2026-11-15

This directory holds the measurements behind [docs/stability.md](../../docs/stability.md), which defines the error measures and interprets the results.

## What was measured

The floating-point accuracy of the product of the loop algebra of a Cayley–Dickson loop, over the real and the complex numbers, against the exact product of the stored operands.

- **S, the split algorithm:** three Walsh–Hadamard transforms, one coordinatewise product, and one Cayley–Dickson product.
- **T, the table algorithm:** the direct evaluation of the definition in the loop basis.
- **L and R, chains of products:** a left-nested product of 2, 4, or 8 operands, kept in split form with one conversion at the end (L), or returned to the loop basis after every product (R).

The full profile measured 4,764 operand pairs and 2,076 chains. The reference converts every binary64 coefficient to an integer times a power of two, sums the product from its definition in arbitrary-precision integers, and evaluates norms and errors with MPFR at 256 bits.

## Host and software

| | |
|:---|:---|
| CPU | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz |
| Operating system | Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 |
| Python | 3.14.8 |
| NumPy, gmpy2, MPFR | 2.5.4, 2.3.2, MPFR 4.2.2 |
| pyfastcd | 1.4.0 |
| pycdloop | 0.1.0 |
| Matplotlib | 3.11.2 |
| Worker processes | 8 |

## How it was produced

```sh
python3 benchmarks/stability_study.py --profile full --seed 20261111 --jobs 8 --output benchmark-results/stability-20261115 --update-docs
```

| Step | What it does | Duration |
|:---|:---|---:|
| accuracy | Measure every error against the exact product. | 12 s |
| tables | Tabulate the accuracy results. | 0 s |
| plots | Draw the charts. | 4 s |

The operands derive from the root seed 20261111. study.json records the SHA-256 of every file of the measured package.

## Files

| File | Contents |
|:---|:---|
| README.md | This description, generated from the results. |
| SHA256SUMS | SHA-256 of every published file of the study. |
| study.json | The command, the host and software, the hashes of the measured sources and of the raw files, and every step with its duration. |
| key-figures.json | The figures quoted in docs/stability.md, computed from the results. |
| accuracy-tables.md | The accuracy tables shown in docs/stability.md. |
| accuracy-summary.csv | Error statistics for every domain, dimension, family, chain length, and method. |
| PLOTS.md | Description of the charts and the command that drew them. |
| plot-data.csv | Every plotted value in tabular form. |
| chain-eta-complex, chain-eta-real, eta-complex, eta-real, relative-error-complex, relative-error-real | The charts, each as PNG, SVG, and PDF. |

The raw per-case results and the operand corpus are written to `raw/`. They are not distributed; study.json records their SHA-256, and `tools/check_archives.py --full` checks them where they are present.
