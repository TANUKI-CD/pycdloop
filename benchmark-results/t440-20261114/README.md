# Benchmark run, 2026-11-14

This directory holds the measurements behind the archived results in
[docs/benchmark.md](../../docs/benchmark.md), which describes the protocols
and how to read the reports.

## Host and software

| | |
|:---|:---|
| Machine | Dell PowerEdge T440 |
| CPU | Intel Xeon Gold 6148, one logical CPU per run (CPU 36) |
| Operating system | Linux 7.2.11 |
| Python | 3.14.8 |
| NumPy | 2.5.4 |
| pyfastcd | 1.4.0 |
| pycdloop | 0.1.0 |

The host is shared, and no governor, turbo, or SMT settings were changed.
`host-load.log` records the load average before each run and after the
last one.

## How it was produced

The three protocols ran one after another, followed by a second run of the
`domains` protocol over the fields of characteristic 2:

```sh
python benchmarks/benchmark.py --output benchmark-results/t440-20261114/real.json --cpu 36
python benchmarks/domains.py --output benchmark-results/t440-20261114/domains.json --cpu 36
python benchmarks/central.py --output benchmark-results/t440-20261114/central.json --cpu 36 \
    --factors 1x4,1x8,2x2,2x4,2x6,3x3,3x5,3x7,4x4,4x6,4x8,5x5,5x7,6x6,7x7
python benchmarks/domains.py --output benchmark-results/t440-20261114/binary.json --cpu 36 \
    --fields 2:1,2:2,2:4,2:6,2:7,2:8,2:12 --levels 1,2,3,4,5,6,7,8
```

Each used 24 blocks and the default corpus seed. Every verified product
agreed with the independent reference: 25,760 products across the
four runs.

## Files

| File | Contents |
|:---|:---|
| `real.json`, `real.md` | `Q_n` over R, levels 1 through 10: `split` and `table` in batch and single mode, products in split form, the conversions, and the Cayley–Dickson product alone |
| `domains.json`, `domains.md` | `Q_n` over C, six finite fields of odd characteristic and F_2, F_4 and F_256, levels 2, 4, 6 and 8 |
| `central.json`, `central.md` | Central products over R: the signed-component methods, and the full product |
| `binary.json`, `binary.md` | `Q_n` over F_(2^k) for `k` = 1, 2, 4, 6, 7, 8 and 12, levels 1 through 8: `lift` and `table` in batch and single mode |
| `host-load.log` | The load average of the host around each run |

Each JSON document holds the raw observations of every block, the corpus
hashes, the SHA-256 of every source file of the measured package, the
software versions, and the state of the CPU.

In the reports of the `domains` protocol, the columns named `split` hold the
timings of `lift` for the fields of characteristic 2.
