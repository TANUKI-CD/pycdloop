# Benchmark

pycdloop multiplies in a loop algebra with `split`, which works through the
decomposition of the algebra, or with `table`, which evaluates the
definition of the product. In characteristic 2, where the algebra has no
decomposition, `lift` takes the place of `split`: it runs the split
algorithm over a field of large odd characteristic and reduces the
result. For central
products the `split` algorithm multiplies signed components by one of three
methods. `"auto"` chooses among these, and the benchmark measures them as
the package ships them, so that the choices rest on reproducible data. The
drivers in `benchmarks/` run three *protocols*:

| `--protocol` | Measurements | Driver |
| --- | --- | --- |
| `real` (default) | `split` and `table` for `Q_n` over R, in batch and single mode; products in split form; the conversions; the Cayley–Dickson product alone | `benchmark.py` |
| `domains` | `split` and `table` for `Q_n` over C and finite fields of odd characteristic; `lift` and `table` in characteristic 2 | `domains.py` |
| `central` | The three signed-component methods and the full product for central products over R | `central.py` |
| `all` | All three protocols, one after another with separate reports | `protocols.py` |

This page first describes how the protocols measure and report, which they
share, then each protocol and how to run it, and finally the archived
results. The accuracy of the algorithms is covered by the
[stability study](stability.md).

## How measurements are made

### What is timed

pycdloop is a Python package: its transform and its change of coordinates
are compiled, its Cayley–Dickson products are pyfastcd's, and everything
that connects them is Python and NumPy. The benchmark times calls to the
public interface with `time.perf_counter_ns`, with the garbage collector
disabled inside each timed region. The call overhead of the package, the
allocation of results and the validation of operands are therefore part of
every measurement; they are what a caller pays. Verification is not.

### Corpora and verification

Every protocol times a fixed *corpus*: a deterministic set of operand pairs
for each level, and for each field or pair of factors where there is one.
The driver generates it with NumPy from a recorded seed, stores it for
reuse, and binds it to a manifest with its parameters and hash, so a run
never times inputs different from the ones it reports. Larger levels use
fewer pairs, because one `table` product at dimension 2048 takes tens of
milliseconds; the schedule is recorded with the run.

Before any timing, every loop-algebra product of every task is compared
with an independent evaluation of the definition, written in the driver: it derives
the basis signs from the doubling formula and sums the products of
coefficients directly, without pycdloop arithmetic. Over R and C a
coordinate may differ from the reference by at most

```text
128 * DBL_EPSILON * D * ||x||_2 * ||y||_2,
```

and over finite fields the comparison is exact, against schoolbook
polynomial arithmetic. A failed check is reported as a failure, never as a
timing.

### Observations, modes and blocks

An *observation* times complete passes over the corpus. After two untimed
passes, the driver calibrates the number of passes so that the observation
lasts at least 20 ms, and divides the elapsed time by the number of
products.

Products are timed in two *modes*. Batch mode makes one call on arrays that
hold the whole corpus. Single mode makes one call for each pair, on arrays
of one record, and so includes the fixed cost of a call, which dominates at
small dimensions.

A run is divided into 24 *blocks*. Each block takes one observation of every
*task*, and the order of the tasks rotates from block to block, so that a
slow change in the state of the host affects all tasks alike. Ratios are
formed within a block.

### Reading the reports

Each run writes a JSON file with the raw observations, the corpus hashes,
the hash of every source file of the package, the versions of Python, NumPy,
pyfastcd and pycdloop, and the state of the CPU, and a Markdown report
beside it. A time is the mean over the blocks, in nanoseconds per product,
with a two-sided 95% Student-t interval whose degrees of freedom are capped
at 30. A ratio is the geometric mean of the per-block ratios with its 95%
interval. A "fastest mean" column names the lowest mean, or a tie when the
interval of the paired ratio of the two lowest means contains one.

An interval describes the variation between the blocks of one run in one
process. It does not cover differences between processes, between runs or
between hosts, and on a shared host those can exceed it.

## Real coefficients

The `real` protocol times the loop algebra of `Q_n` over R for levels 1
through 10, that is, dimensions 4 through 2048. Its tasks are:

| Task | What one product is |
| --- | --- |
| `split/batch`, `split/single` | The `split` algorithm on `ElementArray` operands: both operands are converted to split form, multiplied there, and the result converted back |
| `table/batch`, `table/single` | The `table` algorithm on the same operands |
| `splitform/batch` | The product of two `SplitArray` operands, with no conversion |
| `cached/single` | The product of two `Element` operands that already hold their split forms; the result is not converted back |
| `forward/batch`, `inverse/batch` | The conversions alone: both operands to split form, and one result back |
| `pyfastcd/batch`, `pyfastcd/single` | The Cayley–Dickson product of the signed components alone, through pyfastcd, on its arrays and on its elements |

The last two are for scale: a loop-algebra product of dimension `N` contains
one Cayley–Dickson product of dimension `N / 2`.

```sh
python benchmarks/benchmark.py --output results/real.json --cpu 2
```

`--levels`, `--pairs`, `--blocks`, `--min-time-ms`, `--seed` and `--tasks`
change the defaults, and `--cpu` pins the process to one logical CPU.

## Complex and finite fields

The `domains` protocol times `split` and `table` in both modes for `Q_n`
over C and over the finite fields F_3, F_13, F_65521, F_2147483647, F_(3^5)
and F_(13^5), at levels 2, 4, 6 and 8. Over F_2, F_4 and F_256 it times
`lift` in place of `split`, at the same levels; the reports keep one set of
columns, so there the columns named `split` hold the timings of `lift`.
`--fields` takes other fields as `char:deg` pairs.

```sh
python benchmarks/domains.py --output results/domains.json --cpu 2
```

In characteristic 2 `"auto"` chooses between `lift` and `table` from the
size of the loop, the number of products and the degree of the field. The
data behind that choice is a second run of this protocol, over F_(2^k)
alone, for `k` = 1, 2, 4, 6, 7, 8 and 12 at every level from 1 to 8:

```sh
python benchmarks/domains.py --output results/binary.json --cpu 2 \
    --fields 2:1,2:2,2:4,2:6,2:7,2:8,2:12 --levels 1,2,3,4,5,6,7,8
```

A batch holds 64 products at levels 1 through 4, 16 at levels 5 and 6, and
4 at levels 7 and 8.

## Central products

The `central` protocol times central products `Q_n * Q_m` over R in two
parts. The first compares the signed-component methods `native`, `rows` and
`schedule` on `SplitArray` operands, where no transform is involved, so the
tasks differ only in the method. The second times the full product, `split`
with the method that `"auto"` selects against `table`, on `ElementArray`
operands, for the pairs where `table` is affordable.

```sh
python benchmarks/central.py --output results/central.json --cpu 2
```

`--factors` takes the pairs as `nxm` items, and `--max-table-level` bounds
the total level of the pairs whose full product is timed.

## Archived results

The results below were measured on 2026-11-12 on a Dell PowerEdge T440 with an
Intel Xeon Gold 6148, pinned to one logical CPU, with Python 3.14.8,
NumPy 2.5.4 and pyfastcd 1.4.0 on Linux. The host is shared, and
no governor, turbo, or SMT settings were changed; the load average during
the runs was between 2.8 and 4.1. Every verified product
agreed with the independent reference: 25,760 products in all. The
JSON documents and full reports are archived in
[benchmark-results/t440-20261112/](../benchmark-results/t440-20261112/).

### Real coefficients

Mean nanoseconds per product with 95% intervals. `N` is the dimension of
the loop algebra of `Q_n`.

#### Batch products (ns per product)

| N | n | pairs | split | table | splitform | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 183.6 [181.0, 186.2] | 593.8 [585.4, 602.1] | 66.59 [65.64, 67.55] | split | split |
| 8 | 2 | 256 | 205.0 [187.3, 222.7] | 832.8 [819.1, 846.6] | 71.03 [70.07, 71.98] | split | split |
| 16 | 3 | 256 | 212.1 [208.8, 215.3] | 1717.1 [1681.8, 1752.3] | 80.11 [78.90, 81.32] | split | split |
| 32 | 4 | 256 | 310.6 [295.9, 325.3] | 8855.4 [8693.9, 9016.9] | 133.5 [130.5, 136.4] | split | split |
| 64 | 5 | 256 | 546.8 [532.1, 561.5] | 48506 [46670, 50342] | 266.3 [262.8, 269.9] | split | split |
| 128 | 6 | 256 | 1082.6 [1063.3, 1101.8] | 183579 [181954, 185204] | 594.2 [578.4, 610.0] | split | split |
| 256 | 7 | 64 | 2699.3 [2668.2, 2730.3] | 713142 [706043, 720241] | 1473.0 [1450.9, 1495.0] | split | split |
| 512 | 8 | 64 | 5506.1 [5419.0, 5593.3] | 3785276 [3761010, 3809543] | 3287.2 [3205.3, 3369.1] | split | split |
| 1024 | 9 | 16 | 13436 [13222, 13650] | 14788687 [14680545, 14896830] | 8235.7 [8017.8, 8453.5] | split | split |
| 2048 | 10 | 16 | 28224 [27582, 28865] | 44572127 [44060960, 45083295] | 17334 [17031, 17637] | split | split |

#### Single products (ns per product)

| N | n | pairs | split | table | cached | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 30021 [29816, 30227] | 48679 [48218, 49140] | 15515 [15325, 15704] | split | split |
| 8 | 2 | 256 | 30735 [28860, 32611] | 48257 [47815, 48699] | 15494 [15357, 15631] | split | split |
| 16 | 3 | 256 | 29982 [29730, 30233] | 50198 [49377, 51018] | 15586 [15373, 15798] | split | split |
| 32 | 4 | 256 | 31510 [30684, 32335] | 60510 [56985, 64036] | 16438 [16057, 16818] | split | split |
| 64 | 5 | 256 | 30551 [30116, 30985] | 66769 [61626, 71913] | 16076 [15915, 16236] | split | split |
| 128 | 6 | 256 | 31362 [30853, 31871] | 119885 [118010, 121760] | 16627 [16457, 16798] | split | split |
| 256 | 7 | 64 | 34640 [34065, 35215] | 366792 [360365, 373220] | 18836 [18504, 19169] | split | split |
| 512 | 8 | 64 | 38032 [37444, 38620] | 1382884 [1352335, 1413432] | 20727 [20181, 21272] | split | split |
| 1024 | 9 | 16 | 47142 [45063, 49222] | 5068743 [5001454, 5136032] | 26221 [25201, 27242] | split | split |
| 2048 | 10 | 16 | 59427 [57577, 61277] | 34466855 [34302721, 34630990] | 35674 [35007, 36341] | split | split |

`split` is faster than `table` at every dimension in both modes, by a factor
that grows from 3.23 at dimension 4 to 1581 at
dimension 2048 in batch mode and from 1.62 to 581
in single mode. `"auto"` therefore selects `split` at every dimension.

A single call costs about 31 microseconds at small
dimensions, nearly all of it the fixed cost of a call through the Python
layers; a product of two elements that hold their split forms costs about
half that.

#### Comparison with the Cayley–Dickson product

| N | n | pyfastcd batch (ns) | splitform / pyfastcd | split / pyfastcd | pyfastcd single (ns) | cached / pyfastcd |
| ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 9.22 [9.15, 9.30] | 7.22 [7.1, 7.34] | 19.9 [19.6, 20.2] | 97.28 [96.23, 98.33] | 159 [157, 162] |
| 8 | 2 | 11.41 [11.26, 11.56] | 6.22 [6.12, 6.34] | 17.7 [16.6, 18.9] | 108.8 [98.40, 119.1] | 145 [136, 154] |
| 16 | 3 | 19.78 [19.57, 19.99] | 4.05 [3.98, 4.12] | 10.7 [10.5, 10.9] | 120.3 [119.1, 121.5] | 130 [128, 131] |
| 32 | 4 | 64.09 [62.39, 65.79] | 2.08 [2.02, 2.15] | 4.83 [4.66, 5.01] | 234.1 [230.6, 237.7] | 70.2 [68.8, 71.5] |
| 64 | 5 | 184.6 [183.0, 186.3] | 1.44 [1.42, 1.47] | 2.96 [2.88, 3.04] | 335.5 [331.7, 339.3] | 47.9 [47.4, 48.4] |
| 128 | 6 | 451.7 [447.3, 456.0] | 1.31 [1.28, 1.35] | 2.4 [2.35, 2.44] | 630.6 [623.5, 637.6] | 26.4 [26.2, 26.6] |
| 256 | 7 | 1130.9 [1120.0, 1141.9] | 1.3 [1.28, 1.33] | 2.39 [2.35, 2.42] | 1360.9 [1337.7, 1384.1] | 13.8 [13.6, 14.1] |
| 512 | 8 | 2606.1 [2571.7, 2640.4] | 1.26 [1.23, 1.29] | 2.11 [2.08, 2.15] | 2953.0 [2916.8, 2989.2] | 7.01 [6.86, 7.16] |
| 1024 | 9 | 6522.8 [6449.6, 6596.0] | 1.26 [1.23, 1.29] | 2.06 [2.02, 2.1] | 6716.9 [6640.5, 6793.3] | 3.89 [3.76, 4.02] |
| 2048 | 10 | 15042 [14681, 15403] | 1.15 [1.12, 1.18] | 1.88 [1.82, 1.94] | 14994 [14823, 15165] | 2.38 [2.34, 2.42] |

In batch mode a product in split form costs 7.22 times the
Cayley–Dickson product it contains at dimension 4 and 1.15 times at
dimension 2048, and a product in the loop basis costs 19.9 and
1.88 times. The conversions to and from split form account for
between 31% and 54% of the time of the conversions and
the product in split form together:

| N | n | forward (ns) | inverse (ns) | splitform (ns) | transform share |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 45.24 [44.56, 45.92] | 29.55 [29.05, 30.05] | 66.59 [65.64, 67.55] | 52.9% [52.5%, 53.3%] |
| 8 | 2 | 51.78 [51.16, 52.40] | 32.59 [32.25, 32.93] | 71.03 [70.07, 71.98] | 54.3% [53.9%, 54.6%] |
| 16 | 3 | 52.63 [52.10, 53.16] | 34.50 [34.05, 34.95] | 80.11 [78.90, 81.32] | 52.1% [51.8%, 52.5%] |
| 32 | 4 | 79.20 [77.88, 80.52] | 44.11 [43.03, 45.19] | 133.5 [130.5, 136.4] | 48.0% [47.6%, 48.5%] |
| 64 | 5 | 133.5 [122.5, 144.5] | 72.09 [70.78, 73.41] | 266.3 [262.8, 269.9] | 43.4% [42.2%, 44.6%] |
| 128 | 6 | 249.4 [244.8, 254.0] | 141.7 [138.0, 145.5] | 594.2 [578.4, 610.0] | 39.7% [39.4%, 40.0%] |
| 256 | 7 | 602.2 [594.0, 610.4] | 300.4 [294.1, 306.8] | 1473.0 [1450.9, 1495.0] | 38.0% [37.5%, 38.5%] |
| 512 | 8 | 1105.8 [1089.8, 1121.7] | 588.3 [577.2, 599.4] | 3287.2 [3205.3, 3369.1] | 34.0% [33.7%, 34.4%] |
| 1024 | 9 | 2660.3 [2588.2, 2732.3] | 1290.9 [1250.3, 1331.5] | 8235.7 [8017.8, 8453.5] | 32.4% [32.0%, 32.9%] |
| 2048 | 10 | 5363.9 [5233.1, 5494.7] | 2589.8 [2516.7, 2662.8] | 17334 [17031, 17637] | 31.5% [30.9%, 32.0%] |

In single mode the comparison is with pyfastcd's element type, which is
implemented in C; the fixed cost of a pycdloop call dominates until the
arithmetic outgrows it.

### Complex and finite fields

Each cell gives the `split` mean in nanoseconds per product, or the `lift`
mean in the rows of F_2, F_4 and F_256, and, in parentheses, the paired
ratio of `table` to it, so a ratio above 1 favours `split` or `lift`.

#### Batch products: split or lift ns per product (ratio of table to it)

| Field | N = 8 | N = 32 | N = 128 | N = 512 |
| --- | ---: | ---: | ---: | ---: |
| C | 689.4 (2.19x) | 1135.0 (6.4x) | 5098.3 (57.6x) | 23148 (186x) |
| F_2 | 1194.9 (1.14x) | 2249.9 (1.7x) | 8982.8 (6.05x) | 35746 (34.3x) |
| F_(2^2) | 2898.2 (2.46x) | 7934.8 (7.38x) | 29967 (31x) | 119017 (156x) |
| F_(2^8) | 18209 (0.466x) | 53265 (1.65x) | 196159 (10.4x) | 803261 (41.1x) |
| F_3 | 1053.1 (2.81x) | 2121.8 (9.95x) | 8267.3 (38.9x) | 29471 (181x) |
| F_13 | 1051.7 (2.71x) | 2149.9 (9.17x) | 8640.7 (35.5x) | 30552 (161x) |
| F_65521 | 1060.9 (2.7x) | 2167.9 (8.88x) | 8486.9 (35x) | 33617 (147x) |
| F_2147483647 | 1054.9 (2.71x) | 2195.3 (8.84x) | 8345.8 (35.5x) | 32857 (148x) |
| F_(3^5) | 3473.8 (3.72x) | 8313.2 (17.3x) | 31297 (72.1x) | 122123 (325x) |
| F_(13^5) | 3634.0 (3.64x) | 11360 (12.9x) | 34339 (66.6x) | 141944 (286x) |

#### Single products: split or lift ns per product (ratio of table to it)

| Field | N = 8 | N = 32 | N = 128 | N = 512 |
| --- | ---: | ---: | ---: | ---: |
| C | 35534 (1.4x) | 37236 (1.54x) | 41219 (3.28x) | 52022 (32.7x) |
| F_2 | 53769 (0.914x) | 52877 (1x) | 62036 (1.63x) | 79501 (12.1x) |
| F_(2^2) | 88900 (1.75x) | 101588 (2.21x) | 127066 (8.34x) | 197945 (72.4x) |
| F_(2^8) | 262450 (0.62x) | 329164 (0.725x) | 470457 (3.02x) | 1039577 (21.1x) |
| F_3 | 44080 (1.69x) | 45187 (2.16x) | 53572 (7.48x) | 64409 (79.3x) |
| F_13 | 44098 (1.68x) | 44698 (2.11x) | 54641 (7.06x) | 65446 (73x) |
| F_65521 | 44072 (1.7x) | 45272 (2.09x) | 53331 (7.04x) | 70290 (68.6x) |
| F_2147483647 | 45048 (1.65x) | 46329 (2.08x) | 53783 (6.93x) | 69794 (68.1x) |
| F_(3^5) | 72750 (2.81x) | 81535 (4.27x) | 102504 (23.4x) | 179275 (190x) |
| F_(13^5) | 72276 (2.81x) | 88327 (3.95x) | 103961 (23.2x) | 199106 (176x) |

**C and odd characteristic.** `split` is faster in every cell, in both
modes. For single products its advantage at dimension 8 is a factor of
1.4 over C and between 1.6 and 1.7 over
the prime fields. A single `split` call over C and the prime fields costs
36 to 45 microseconds at dimension 8, nearly all of it the fixed
cost of a call: in a batch the same product takes 0.7 to 1.1
microseconds.

**Characteristic 2.** Over F_4 `lift` is faster in every cell. Over F_2 and
F_256 it is faster in the large loops and `table` in some of the small
ones. The next section looks at these fields more closely.

Intervals for every cell are in the archived report.

### Characteristic 2

Each cell gives the `lift` mean in nanoseconds per product and, in
parentheses, the paired ratio of `table` to it, so a ratio above 1 favours
`lift`. These are the results of the second `domains` run described above.

#### Batch products: lift ns per product (table / lift)

| Field | N = 4 | N = 8 | N = 16 | N = 32 | N = 64 | N = 128 | N = 256 | N = 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_2 | 1067.0 (1.06x) | 1220.3 (1.12x) | 1564.0 (1.23x) | 2325.9 (1.71x) | 6099.1 (2.44x) | 9250.4 (6.07x) | 26418 (12x) | 35917 (53.2x) |
| F_(2^2) | 2175.7 (1.99x) | 2956.5 (2.46x) | 5124.2 (3.44x) | 8267.1 (7.16x) | 18798 (12.4x) | 29766 (40.9x) | 74316 (65.7x) | 126945 (150x) |
| F_(2^4) | 3217.4 (1.38x) | 5929.3 (1.29x) | 13511 (1.41x) | 18473 (3.82x) | 37672 (7.37x) | 64581 (20.1x) | 139415 (37.1x) | 262038 (91.5x) |
| F_(2^6) | 4868.1 (0.906x) | 10677 (0.722x) | 27673 (0.729x) | 32480 (2.35x) | 62212 (4.88x) | 106681 (11x) | 229703 (20.3x) | 425359 (68.1x) |
| F_(2^7) | 9690.2 (0.475x) | 18008 (0.455x) | 41299 (0.542x) | 54525 (1.52x) | 112316 (2.98x) | 194967 (6.53x) | 432797 (11.9x) | 789874 (38.3x) |
| F_(2^8) | 9620.4 (0.467x) | 18007 (0.464x) | 40862 (0.58x) | 54236 (1.61x) | 113706 (3.04x) | 197844 (7.45x) | 432002 (13.4x) | 791197 (41.5x) |
| F_(2^12) | 15020 (0.333x) | 32443 (0.299x) | 83898 (0.351x) | 99601 (1.1x) | 190056 (2.34x) | 337374 (5.41x) | 715095 (10.3x) | 1315610 (33.9x) |

#### Single products: lift ns per product (table / lift)

| Field | N = 4 | N = 8 | N = 16 | N = 32 | N = 64 | N = 128 | N = 256 | N = 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_2 | 53616 (0.9x) | 53763 (0.905x) | 53834 (0.911x) | 55410 (0.923x) | 59659 (1.01x) | 66974 (1.52x) | 71124 (4x) | 82070 (11.8x) |
| F_(2^2) | 95192 (1.59x) | 94327 (1.64x) | 96573 (1.74x) | 108627 (2.09x) | 110951 (3.56x) | 127468 (8.22x) | 156505 (24.1x) | 208500 (70.3x) |
| F_(2^4) | 94211 (1.64x) | 96981 (1.61x) | 104614 (1.65x) | 119081 (1.95x) | 135455 (3.09x) | 160446 (7.25x) | 225344 (19x) | 335166 (49.2x) |
| F_(2^6) | 95207 (1.55x) | 99614 (1.54x) | 118965 (1.45x) | 135177 (1.73x) | 155898 (2.78x) | 205183 (6.05x) | 310363 (15.6x) | 539704 (35x) |
| F_(2^7) | 258269 (0.583x) | 270771 (0.569x) | 301776 (0.585x) | 332098 (0.716x) | 389320 (1.18x) | 471996 (2.75x) | 656868 (7.95x) | 1027729 (19.4x) |
| F_(2^8) | 255849 (0.584x) | 265461 (0.59x) | 295694 (0.597x) | 332335 (0.736x) | 375330 (1.25x) | 479767 (3.01x) | 652933 (8.35x) | 1005341 (22.6x) |
| F_(2^12) | 265986 (0.571x) | 283833 (0.546x) | 341122 (0.52x) | 394615 (0.646x) | 473403 (1.16x) | 634744 (2.8x) | 946892 (7.19x) | 1559044 (20.3x) |

`lift` is the faster algorithm in large loops, where its advantage reaches
a factor of 150 in batch mode and 70.2 for
single products at dimension 512. In small loops the outcome depends on the
degree of the field, which sets the cost of a lifted product, and on the
number of products in a call. For single products `lift` has the lower mean
at every dimension over F_4, F_16 and F_64, and from dimension 64 over the
other fields. In batch mode it has the lower mean at every dimension over
F_2, F_4 and F_16, and from dimension 32 over the other fields.

`"auto"` follows the rule of
[Choosing between the algorithms](mathematics.md#choosing-between-the-algorithms),
which these measurements test: in 108 of the 112 cells
it selects the algorithm with the lower mean, or one tied with it, and in
the others the algorithm it selects is slower by a factor of at most
1.39. The largest differences are for batches over F_64 in loops of
dimension at most 16, where the rule selects `lift`.

### Central products

#### Signed-component methods (ns per product)

| n x m | D | pairs | native | rows | schedule | fastest mean | auto |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| 1 x 4 | 32 | 64 | 1845.7 [1815.7, 1875.8] | 2418.0 [2381.0, 2454.9] | 2749.0 [2702.0, 2796.0] | native | native |
| 1 x 8 | 512 | 4 | 35320 [34277, 36363] | 43093 [42201, 43985] | 52384 [51107, 53661] | native | native |
| 2 x 2 | 16 | 64 | unavailable | 2465.9 [2411.2, 2520.5] | 2888.0 [2820.3, 2955.7] | rows | rows |
| 2 x 4 | 64 | 64 | unavailable | 4659.3 [4579.8, 4738.8] | 4422.4 [4335.8, 4509.1] | schedule | rows |
| 2 x 6 | 256 | 16 | unavailable | 19522 [19100, 19945] | 20805 [20331, 21278] | rows | rows |
| 3 x 3 | 64 | 64 | unavailable | 6321.6 [6197.7, 6445.6] | 8583.2 [8463.9, 8702.5] | rows | rows |
| 3 x 5 | 256 | 16 | unavailable | 29604 [29201, 30007] | 39490 [38715, 40266] | rows | rows |
| 3 x 7 | 1024 | 4 | unavailable | 133472 [131954, 134991] | 178383 [175935, 180830] | rows | rows |
| 4 x 4 | 256 | 16 | unavailable | 53324 [52556, 54092] | 58887 [57581, 60193] | rows | rows |
| 4 x 6 | 1024 | 4 | unavailable | 223863 [220736, 226989] | 272706 [267915, 277497] | rows | rows |
| 4 x 8 | 4096 | 4 | unavailable | 990954 [980494, 1001413] | 869498 [850914, 888082] | schedule | rows |
| 5 x 5 | 1024 | 4 | unavailable | 399770 [392850, 406690] | 383642 [374401, 392883] | schedule | schedule |
| 5 x 7 | 4096 | 4 | unavailable | 1733375 [1708859, 1757891] | 1067602 [1046340, 1088864] | schedule | schedule |
| 6 x 6 | 4096 | 4 | unavailable | 4005734 [3959912, 4051557] | 1215182 [1182212, 1248151] | schedule | schedule |
| 7 x 7 | 16384 | 2 | unavailable | 35595187 [35393981, 35796393] | 5865333 [5739084, 5991583] | schedule | schedule |

#### Paired ratios of the signed-component methods

| n x m | outer level | schedule / rows | native / rows |
| --- | ---: | --- | --- |
| 1 x 4 | 1 | 1.14 [1.12, 1.15] | 0.763 [0.753, 0.773] |
| 1 x 8 | 1 | 1.22 [1.2, 1.23] | 0.819 [0.797, 0.841] |
| 2 x 2 | 2 | 1.17 [1.15, 1.19] | unavailable |
| 2 x 4 | 2 | 0.949 [0.932, 0.967] | unavailable |
| 2 x 6 | 2 | 1.07 [1.05, 1.08] | unavailable |
| 3 x 3 | 3 | 1.36 [1.33, 1.39] | unavailable |
| 3 x 5 | 3 | 1.33 [1.31, 1.36] | unavailable |
| 3 x 7 | 3 | 1.34 [1.32, 1.35] | unavailable |
| 4 x 4 | 4 | 1.1 [1.08, 1.13] | unavailable |
| 4 x 6 | 4 | 1.22 [1.19, 1.24] | unavailable |
| 4 x 8 | 4 | 0.877 [0.859, 0.895] | unavailable |
| 5 x 5 | 5 | 0.959 [0.94, 0.979] | unavailable |
| 5 x 7 | 5 | 0.616 [0.604, 0.627] | unavailable |
| 6 x 6 | 6 | 0.303 [0.296, 0.31] | unavailable |
| 7 x 7 | 7 | 0.165 [0.161, 0.168] | unavailable |

`native` is the fastest method wherever it applies. Through outer level 4,
`rows` and `schedule` are within a factor of 1.4 of each other, and
which is faster depends on the inner level. `schedule` is faster in every
measured pair from outer level 5, by a factor that reaches 6.1 at
`7 x 7`. `"auto"` therefore selects `native` where it applies, and otherwise
`rows` through outer level 4 and `schedule` above.

#### Full product (ns per product)

| n x m | N | pairs | split | table | table / split | fastest mean |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 x 4 | 64 | 64 | 2293.8 [2253.6, 2334.0] | 30583 [30145, 31022] | 13.3 [13.2, 13.5] | split |
| 1 x 8 | 1024 | 4 | 42870 [41796, 43945] | 10456602 [10333840, 10579365] | 244 [238, 251] | split |
| 2 x 2 | 32 | 64 | 2786.8 [2716.9, 2856.7] | 5202.1 [5084.6, 5319.7] | 1.87 [1.83, 1.91] | split |
| 2 x 4 | 128 | 64 | 5333.1 [5211.6, 5454.5] | 113815 [111831, 115799] | 21.4 [20.9, 21.8] | split |
| 2 x 6 | 512 | 16 | 22861 [22241, 23482] | 2832736 [2813728, 2851743] | 124 [121, 127] | split |
| 3 x 3 | 128 | 64 | 7032.4 [6861.6, 7203.1] | 109405 [108253, 110557] | 15.6 [15.2, 15.9] | split |
| 3 x 5 | 512 | 16 | 32460 [31719, 33200] | 2794034 [2767897, 2820172] | 86.2 [84.3, 88.1] | split |
| 3 x 7 | 2048 | 4 | 143621 [141659, 145583] | 44011877 [43851955, 44171800] | 307 [302, 311] | split |
| 4 x 4 | 512 | 16 | 56667 [55865, 57469] | 2705596 [2681281, 2729911] | 47.8 [46.9, 48.6] | split |
| 4 x 6 | 2048 | 4 | 240036 [236162, 243909] | 44035324 [43819761, 44250887] | 184 [181, 187] | split |
| 5 x 5 | 2048 | 4 | 400275 [388279, 412271] | 44225541 [44024226, 44426856] | 111 [108, 114] | split |

## Reproducing a run

```sh
python -m pip install -e ".[dev]"
python benchmarks/protocols.py --protocol all --output results/run.json --cpu 2
```

The three protocols write `run.real.json`, `run.domains.json` and
`run.central.json` with their reports, and take about twelve minutes
together on the archive's host; the second `domains` run over the fields of
characteristic 2 takes about seven minutes more. The archived `central` run
was given a longer list of factor pairs with `--factors`; the README of the
archive records the exact commands. The corpus is written to
`benchmark-corpora/` on first use. `benchmarks/test_benchmark.py` tests the
drivers themselves: argument validation, the statistics, the independent
reference, and a short run of every protocol.
