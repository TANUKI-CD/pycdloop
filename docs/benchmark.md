# Benchmark

pycdloop multiplies in a loop algebra with three algorithms:

- `split` works through the decomposition of the algebra.
- `table` evaluates the definition of the product.
- `lift` takes the place of `split` in characteristic 2, where the algebra
  has no decomposition: it runs the split algorithm over a field of large
  odd characteristic and reduces the result.

For central products the `split` algorithm multiplies signed components by
one of three methods. `"auto"` chooses among these, and the benchmark
measures them as the package ships them, so that the choices rest on
reproducible data. The drivers in `benchmarks/` run three *protocols*:

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
allocation of results, and the validation of operands are therefore part of
every measurement, as they are what a caller pays.

### Corpora and verification

Every protocol times a fixed *corpus*: a deterministic set of operand pairs
for each level, and for each field or pair of factors where there is one.
The driver generates it with NumPy from a recorded seed, stores it for
reuse, and binds it to a manifest with its parameters and hash, so a run
never times inputs different from the ones it reports. Larger levels use
fewer pairs, because one `table` product at dimension 2048 takes tens of
milliseconds; the schedule is recorded with the run.

Before any timing, every loop-algebra product of every task is compared with
an independent evaluation of the definition, written in the driver: it
derives the basis signs from the doubling formula and sums the products of
coefficients directly, without pycdloop arithmetic. Over R and C a
coordinate may differ from the reference by at most

```text
128 * DBL_EPSILON * D * ||x||_2 * ||y||_2,
```

and over finite fields the comparison is exact, against schoolbook
polynomial arithmetic. A failed check is reported as a failure, never as a
timing.

### Observations, modes, and blocks

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
pyfastcd, and pycdloop, and the state of the CPU, and a Markdown report
beside it. A time is the mean over the blocks, in nanoseconds per product,
with a two-sided 95% Student-t interval whose degrees of freedom are capped
at 30. A ratio is the geometric mean of the per-block ratios with its 95%
interval. A "fastest mean" column names the lowest mean, or a tie when the
interval of the paired ratio of the two lowest means contains one.

An interval describes the variation between the blocks of one run in one
process. It does not cover differences between processes, between runs, or
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

`--levels`, `--pairs`, `--blocks`, `--min-time-ms`, `--seed`, and `--tasks`
change the defaults, and `--cpu` pins the process to one logical CPU.

## Complex and finite fields

The `domains` protocol times `split` and `table` in both modes for `Q_n`
over C and over the finite fields F_3, F_13, F_65521, F_2147483647, F_(3^5),
and F_(13^5), at levels 2, 4, 6, and 8. Over F_2, F_4, and F_256 it times
`lift` in place of `split`, at the same levels; the reports keep one set of
columns, so there the columns named `split` hold the timings of `lift`.
`--fields` takes other fields as `char:deg` pairs.

```sh
python benchmarks/domains.py --output results/domains.json --cpu 2
```

In characteristic 2 `"auto"` chooses between `lift` and `table` from the
size of the loop, the number of products, and the degree of the field. The
data behind that choice is a second run of this protocol, over F_(2^k)
alone, for `k` = 1, 2, 4, 6, 7, 8, and 12 at every level from 1 to 8:

```sh
python benchmarks/domains.py --output results/binary.json --cpu 2 \
    --fields 2:1,2:2,2:4,2:6,2:7,2:8,2:12 --levels 1,2,3,4,5,6,7,8
```

A batch holds 64 products at levels 1 through 4, 16 at levels 5 and 6, and
4 at levels 7 and 8.

## Central products

The `central` protocol times central products `Q_n * Q_m` over R in two
parts. The first compares the signed-component methods `native`, `rows`, and
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

The results below were measured on 2026-11-15 on a Dell PowerEdge T440 with
an Intel Xeon Gold 6148, pinned to one logical CPU, with Python 3.14.8,
NumPy 2.5.4, and pyfastcd 1.4.0 on Linux. The host is shared, and no
governor, turbo, or SMT settings were changed; the load average during the
runs was between 1.6 and 2.3. Every verified product agreed with the
independent reference: 25,760 products in all. The JSON documents and full
reports are archived in
[benchmark-results/t440-20261115/](../benchmark-results/t440-20261115/).

### Real coefficients

Mean nanoseconds per product with 95% intervals. `N` is the dimension of
the loop algebra of `Q_n`.

#### Batch products (ns per product)

| N | n | pairs | split | table | splitform | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 188.6 [184.6, 192.5] | 588.2 [572.0, 604.4] | 82.60 [80.27, 84.94] | split | split |
| 8 | 2 | 256 | 203.0 [200.5, 205.5] | 819.8 [801.3, 838.4] | 87.85 [83.83, 91.88] | split | split |
| 16 | 3 | 256 | 211.6 [207.0, 216.2] | 1682.2 [1642.4, 1721.9] | 96.47 [94.33, 98.61] | split | split |
| 32 | 4 | 256 | 305.3 [296.6, 313.9] | 8950.4 [8751.9, 9148.8] | 147.8 [144.8, 150.9] | split | split |
| 64 | 5 | 256 | 553.0 [542.0, 564.1] | 48355 [47492, 49219] | 286.3 [280.1, 292.5] | split | split |
| 128 | 6 | 256 | 1128.5 [1106.1, 1150.9] | 197239 [193905, 200572] | 613.3 [602.1, 624.4] | split | split |
| 256 | 7 | 64 | 2768.5 [2674.5, 2862.6] | 727264 [718274, 736253] | 1517.1 [1473.7, 1560.5] | split | split |
| 512 | 8 | 64 | 5685.4 [5523.7, 5847.2] | 3730925 [3716147, 3745704] | 3335.1 [3231.9, 3438.4] | split | split |
| 1024 | 9 | 16 | 13475 [13170, 13779] | 14807170 [14737892, 14876448] | 8522.9 [8356.4, 8689.4] | split | split |
| 2048 | 10 | 16 | 28205 [27420, 28990] | 43788816 [43610330, 43967301] | 17709 [17385, 18033] | split | split |

#### Single products (ns per product)

| N | n | pairs | split | table | cached | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 30923 [30442, 31403] | 46734 [46086, 47382] | 15583 [15268, 15899] | split | split |
| 8 | 2 | 256 | 30454 [29989, 30920] | 46970 [46288, 47652] | 15446 [15139, 15754] | split | split |
| 16 | 3 | 256 | 30360 [29926, 30793] | 49839 [48881, 50798] | 15284 [15017, 15550] | split | split |
| 32 | 4 | 256 | 31216 [30726, 31706] | 55352 [53347, 57358] | 15856 [15587, 16125] | split | split |
| 64 | 5 | 256 | 31809 [31559, 32059] | 65007 [63493, 66522] | 16027 [15837, 16217] | split | split |
| 128 | 6 | 256 | 33777 [33276, 34278] | 121367 [118744, 123989] | 16672 [16375, 16969] | split | split |
| 256 | 7 | 64 | 35023 [34280, 35766] | 355695 [347782, 363609] | 18749 [18257, 19241] | split | split |
| 512 | 8 | 64 | 38023 [37345, 38702] | 1345210 [1317672, 1372748] | 20700 [20010, 21390] | split | split |
| 1024 | 9 | 16 | 46132 [44775, 47490] | 5036054 [4991707, 5080402] | 25831 [25257, 26405] | split | split |
| 2048 | 10 | 16 | 59140 [57625, 60654] | 33963383 [33732098, 34194668] | 35456 [34617, 36295] | split | split |

`split` is faster than `table` at every dimension in both modes, by a factor
that grows from 3.12 at dimension 4 to 1555 at dimension 2048 in batch mode
and from 1.51 to 575 in single mode. `"auto"` therefore selects `split` at
every dimension.

A single call costs about 31 microseconds at small dimensions, nearly all of
it the fixed cost of a call through the Python layers; a product of two
elements that hold their split forms costs about half that.

#### Comparison with the Cayley–Dickson product

| N | n | pyfastcd batch (ns) | splitform / pyfastcd | split / pyfastcd | pyfastcd single (ns) | cached / pyfastcd |
| ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 9.48 [9.34, 9.62] | 8.7 [8.4, 9.01] | 19.9 [19.3, 20.4] | 98.23 [96.75, 99.72] | 159 [155, 162] |
| 8 | 2 | 11.55 [11.31, 11.78] | 7.58 [7.21, 7.96] | 17.6 [17.2, 18] | 105.4 [104.1, 106.8] | 146 [144, 149] |
| 16 | 3 | 19.41 [19.03, 19.79] | 4.97 [4.83, 5.11] | 10.9 [10.6, 11.2] | 119.1 [117.3, 120.9] | 128 [127, 130] |
| 32 | 4 | 62.09 [60.72, 63.47] | 2.38 [2.31, 2.45] | 4.91 [4.73, 5.1] | 228.2 [224.4, 232.1] | 69.5 [68.5, 70.4] |
| 64 | 5 | 185.6 [183.4, 187.8] | 1.54 [1.51, 1.58] | 2.98 [2.91, 3.05] | 338.0 [334.8, 341.3] | 47.4 [46.8, 48] |
| 128 | 6 | 448.3 [440.1, 456.5] | 1.37 [1.34, 1.4] | 2.52 [2.45, 2.58] | 636.4 [628.0, 644.7] | 26.2 [25.9, 26.5] |
| 256 | 7 | 1105.1 [1091.0, 1119.2] | 1.37 [1.33, 1.41] | 2.5 [2.41, 2.6] | 1329.8 [1311.5, 1348.0] | 14.1 [13.8, 14.4] |
| 512 | 8 | 2593.9 [2555.7, 2632.0] | 1.28 [1.25, 1.32] | 2.19 [2.12, 2.26] | 2925.7 [2858.5, 2992.9] | 7.07 [6.94, 7.2] |
| 1024 | 9 | 6495.7 [6414.9, 6576.6] | 1.31 [1.28, 1.34] | 2.07 [2.03, 2.11] | 6619.5 [6529.7, 6709.3] | 3.9 [3.82, 3.98] |
| 2048 | 10 | 14522 [14277, 14766] | 1.22 [1.19, 1.25] | 1.94 [1.87, 2.01] | 14801 [14611, 14992] | 2.39 [2.35, 2.44] |

In batch mode a product in split form costs 8.7 times the Cayley–Dickson
product it contains at dimension 4 and 1.22 times at dimension 2048, and a
product in the loop basis costs 19.9 and 1.94 times. The conversions to and
from split form account for between 31% and 50% of the time of the
conversions and the product in split form together:

| N | n | forward (ns) | inverse (ns) | splitform (ns) | transform share |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 46.49 [45.76, 47.22] | 30.60 [30.02, 31.18] | 82.60 [80.27, 84.94] | 48.3% [47.6%, 49.0%] |
| 8 | 2 | 54.11 [53.26, 54.96] | 34.55 [33.90, 35.19] | 87.85 [83.83, 91.88] | 50.3% [49.4%, 51.3%] |
| 16 | 3 | 54.14 [53.39, 54.89] | 36.20 [35.60, 36.79] | 96.47 [94.33, 98.61] | 48.4% [47.9%, 48.9%] |
| 32 | 4 | 79.29 [78.34, 80.24] | 45.97 [44.35, 47.60] | 147.8 [144.8, 150.9] | 45.9% [45.3%, 46.4%] |
| 64 | 5 | 134.9 [133.0, 136.8] | 73.88 [72.10, 75.65] | 286.3 [280.1, 292.5] | 42.2% [41.8%, 42.6%] |
| 128 | 6 | 253.0 [248.7, 257.2] | 141.1 [137.0, 145.2] | 613.3 [602.1, 624.4] | 39.1% [38.8%, 39.4%] |
| 256 | 7 | 579.0 [570.4, 587.6] | 303.7 [295.7, 311.8] | 1517.1 [1473.7, 1560.5] | 36.8% [36.2%, 37.4%] |
| 512 | 8 | 1213.4 [1189.4, 1237.3] | 589.3 [573.3, 605.2] | 3335.1 [3231.9, 3438.4] | 35.1% [34.5%, 35.8%] |
| 1024 | 9 | 2817.7 [2748.9, 2886.5] | 1318.5 [1288.6, 1348.5] | 8522.9 [8356.4, 8689.4] | 32.7% [32.2%, 33.2%] |
| 2048 | 10 | 5383.1 [5288.2, 5478.0] | 2558.8 [2503.9, 2613.8] | 17709 [17385, 18033] | 31.0% [30.6%, 31.4%] |

In single mode the comparison is with pyfastcd's element type, which is
implemented in C; the fixed cost of a pycdloop call dominates until the
arithmetic outgrows it.

### Complex and finite fields

Each cell gives the `split` mean in nanoseconds per product, or the `lift`
mean in the rows of F_2, F_4, and F_256, and, in parentheses, the paired
ratio of `table` to it, so a ratio above 1 favours `split` or `lift`.

#### Batch products: split or lift ns per product (ratio of table to it)

| Field | N = 8 | N = 32 | N = 128 | N = 512 |
| --- | ---: | ---: | ---: | ---: |
| C | 723.2 (2.12x) | 1215.5 (6.17x) | 5178.3 (59.3x) | 23892 (182x) |
| F_2 | 1216.9 (1.09x) | 2334.2 (1.73x) | 9200.2 (5.99x) | 38027 (32.8x) |
| F_(2^2) | 3034.2 (2.47x) | 8768.6 (6.97x) | 30384 (29.7x) | 120163 (154x) |
| F_(2^8) | 18285 (0.466x) | 54186 (1.62x) | 200135 (10.3x) | 832906 (38.3x) |
| F_3 | 1095.7 (2.69x) | 2151.3 (10x) | 8952.6 (37.1x) | 31247 (177x) |
| F_13 | 1120.3 (2.59x) | 2176.3 (8.71x) | 8542.6 (34.9x) | 31049 (162x) |
| F_65521 | 1088.1 (2.62x) | 2202.7 (8.73x) | 8604.5 (34.6x) | 34447 (145x) |
| F_2147483647 | 1092.8 (2.66x) | 2243.9 (8.65x) | 8491.0 (34.7x) | 33122 (150x) |
| F_(3^5) | 3564.2 (3.66x) | 8285.2 (17.1x) | 31201 (71.9x) | 124150 (324x) |
| F_(13^5) | 3663.5 (3.56x) | 11324 (13.1x) | 34574 (68.2x) | 137450 (300x) |

#### Single products: split or lift ns per product (ratio of table to it)

| Field | N = 8 | N = 32 | N = 128 | N = 512 |
| --- | ---: | ---: | ---: | ---: |
| C | 37563 (1.29x) | 40065 (1.51x) | 43732 (3.11x) | 54065 (31.2x) |
| F_2 | 54033 (0.885x) | 55597 (0.911x) | 64696 (1.56x) | 86437 (11.4x) |
| F_(2^2) | 96313 (1.66x) | 112634 (2.09x) | 130151 (8.09x) | 209184 (70.5x) |
| F_(2^8) | 269759 (0.59x) | 338711 (0.709x) | 510517 (2.83x) | 1044165 (20.6x) |
| F_3 | 45660 (1.61x) | 48894 (1.99x) | 57567 (7.35x) | 72351 (73.6x) |
| F_13 | 47562 (1.56x) | 46530 (2x) | 56656 (6.79x) | 67486 (71.4x) |
| F_65521 | 45528 (1.59x) | 46523 (2.06x) | 55324 (6.89x) | 72050 (65.8x) |
| F_2147483647 | 45853 (1.59x) | 47589 (2.03x) | 54766 (6.92x) | 72629 (65.9x) |
| F_(3^5) | 75818 (2.74x) | 84566 (4.1x) | 105198 (22.5x) | 185725 (188x) |
| F_(13^5) | 74917 (2.67x) | 92759 (3.77x) | 108980 (22.4x) | 198427 (182x) |

**C and odd characteristic.** `split` is faster in every cell, in both
modes. For single products its advantage at dimension 8 is a factor of 1.3
over C and about 1.6 over the prime fields. A single `split` call over C and
the prime fields costs 38 to 48 microseconds at dimension 8, nearly all of
it the fixed cost of a call: in a batch the same product takes 0.7 to 1.1
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
| F_2 | 1091.3 (1.05x) | 1225.7 (1.07x) | 1520.4 (1.22x) | 2333.5 (1.66x) | 6164.4 (2.32x) | 9158.6 (6.11x) | 26288 (12x) | 37617 (51.3x) |
| F_(2^2) | 2189.6 (2.01x) | 2946.1 (2.49x) | 4957.2 (3.44x) | 8131.4 (7.26x) | 18830 (12.4x) | 29468 (40.2x) | 73913 (65.4x) | 125894 (149x) |
| F_(2^4) | 3194.9 (1.39x) | 5868.7 (1.3x) | 13287 (1.42x) | 18227 (3.87x) | 35656 (7.71x) | 64712 (20x) | 142380 (37.2x) | 276983 (82.5x) |
| F_(2^6) | 4880.0 (0.912x) | 10714 (0.749x) | 28026 (0.752x) | 33394 (2.38x) | 64827 (4.79x) | 109962 (11.1x) | 237365 (20.3x) | 420703 (67.8x) |
| F_(2^7) | 9757.9 (0.471x) | 18405 (0.465x) | 41478 (0.529x) | 53735 (1.53x) | 108564 (3.03x) | 196412 (6.55x) | 429496 (12.1x) | 777004 (38.9x) |
| F_(2^8) | 9711.6 (0.47x) | 18026 (0.457x) | 41088 (0.577x) | 55812 (1.62x) | 117131 (2.98x) | 200685 (7.05x) | 438306 (12.6x) | 792053 (41.6x) |
| F_(2^12) | 15006 (0.329x) | 32497 (0.294x) | 84369 (0.35x) | 101312 (1.08x) | 189765 (2.29x) | 342710 (5.18x) | 740484 (9.65x) | 1328956 (33.6x) |

#### Single products: lift ns per product (table / lift)

| Field | N = 4 | N = 8 | N = 16 | N = 32 | N = 64 | N = 128 | N = 256 | N = 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_2 | 54364 (0.883x) | 52658 (0.897x) | 53214 (0.916x) | 55778 (0.912x) | 58727 (1.02x) | 65537 (1.56x) | 72584 (3.93x) | 89657 (11.1x) |
| F_(2^2) | 94923 (1.64x) | 93229 (1.64x) | 94524 (1.74x) | 110694 (2x) | 114996 (3.45x) | 129938 (8.22x) | 157024 (23.8x) | 206892 (71.6x) |
| F_(2^4) | 93306 (1.6x) | 95085 (1.61x) | 103772 (1.62x) | 123183 (1.88x) | 133877 (3.09x) | 162192 (7.26x) | 223693 (19.8x) | 350117 (51.3x) |
| F_(2^6) | 96027 (1.58x) | 101718 (1.54x) | 121247 (1.42x) | 139777 (1.68x) | 160668 (2.81x) | 209405 (6.09x) | 314243 (15.5x) | 510199 (36.3x) |
| F_(2^7) | 265126 (0.581x) | 274321 (0.587x) | 295446 (0.598x) | 338815 (0.701x) | 383760 (1.18x) | 480005 (2.75x) | 687634 (7.76x) | 1013181 (19.7x) |
| F_(2^8) | 256464 (0.584x) | 263443 (0.581x) | 293796 (0.609x) | 349707 (0.684x) | 393091 (1.19x) | 500050 (2.93x) | 660016 (8.14x) | 1013752 (22.6x) |
| F_(2^12) | 267320 (0.562x) | 283639 (0.553x) | 339917 (0.535x) | 381842 (0.676x) | 463210 (1.14x) | 669123 (2.7x) | 985059 (7.1x) | 1562769 (20.4x) |

`lift` is the faster algorithm in large loops, where its advantage reaches a
factor of 149 in batch mode and 71.6 for single products at dimension 512.
In small loops the outcome depends on the degree of the field, which sets
the cost of a lifted product, and on the number of products in a call. For
single products `lift` has the lower mean at every dimension over F_4, F_16,
and F_64, and from dimension 64 over the other fields. In batch mode it has
the lower mean at every dimension over F_2, F_4, and F_16, and from
dimension 32 over the other fields.

`"auto"` follows the rule of [Choosing between the
algorithms](mathematics.md#choosing-between-the-algorithms), which these
measurements test: in 108 of the 112 cells it selects the algorithm with the
lower mean, or one tied with it, and in the others the algorithm it selects
is slower by a factor of at most 1.33. The largest differences are for
batches over F_64 in loops of dimension at most 16, where the rule selects
`lift`.

### Central products

#### Signed-component methods (ns per product)

| n x m | D | pairs | native | rows | schedule | fastest mean | auto |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| 1 x 4 | 32 | 64 | 1801.4 [1773.5, 1829.2] | 2322.9 [2287.7, 2358.1] | 2665.1 [2606.9, 2723.3] | native | native |
| 1 x 8 | 512 | 4 | 34561 [33613, 35509] | 42656 [41648, 43664] | 51925 [50715, 53135] | native | native |
| 2 x 2 | 16 | 64 | unavailable | 2434.0 [2400.7, 2467.4] | 2833.3 [2783.2, 2883.4] | rows | rows |
| 2 x 4 | 64 | 64 | unavailable | 4763.2 [4658.5, 4868.0] | 4477.4 [4370.1, 4584.7] | schedule | rows |
| 2 x 6 | 256 | 16 | unavailable | 19402 [19093, 19711] | 20729 [20467, 20991] | rows | rows |
| 3 x 3 | 64 | 64 | unavailable | 6362.4 [6235.0, 6489.8] | 8513.9 [8307.6, 8720.2] | rows | rows |
| 3 x 5 | 256 | 16 | unavailable | 29071 [28143, 29999] | 37904 [37264, 38543] | rows | rows |
| 3 x 7 | 1024 | 4 | unavailable | 131095 [128686, 133504] | 174554 [170639, 178469] | rows | rows |
| 4 x 4 | 256 | 16 | unavailable | 52397 [51306, 53488] | 57376 [56237, 58515] | rows | rows |
| 4 x 6 | 1024 | 4 | unavailable | 222998 [216652, 229345] | 271586 [263377, 279795] | rows | rows |
| 4 x 8 | 4096 | 4 | unavailable | 911857 [898192, 925521] | 771936 [759434, 784438] | schedule | rows |
| 5 x 5 | 1024 | 4 | unavailable | 382102 [377023, 387180] | 350873 [343628, 358118] | schedule | schedule |
| 5 x 7 | 4096 | 4 | unavailable | 1680408 [1634049, 1726768] | 999042 [975667, 1022416] | schedule | schedule |
| 6 x 6 | 4096 | 4 | unavailable | 3884691 [3812286, 3957095] | 1151597 [1116734, 1186460] | schedule | schedule |
| 7 x 7 | 16384 | 2 | unavailable | 34844261 [34519817, 35168704] | 5744267 [5650608, 5837925] | schedule | schedule |

#### Paired ratios of the signed-component methods

| n x m | outer level | schedule / rows | native / rows |
| --- | ---: | --- | --- |
| 1 x 4 | 1 | 1.15 [1.12, 1.17] | 0.775 [0.764, 0.787] |
| 1 x 8 | 1 | 1.22 [1.19, 1.24] | 0.81 [0.795, 0.825] |
| 2 x 2 | 2 | 1.16 [1.14, 1.19] | unavailable |
| 2 x 4 | 2 | 0.94 [0.915, 0.965] | unavailable |
| 2 x 6 | 2 | 1.07 [1.05, 1.09] | unavailable |
| 3 x 3 | 3 | 1.34 [1.3, 1.37] | unavailable |
| 3 x 5 | 3 | 1.31 [1.27, 1.35] | unavailable |
| 3 x 7 | 3 | 1.33 [1.3, 1.36] | unavailable |
| 4 x 4 | 4 | 1.1 [1.07, 1.12] | unavailable |
| 4 x 6 | 4 | 1.22 [1.17, 1.26] | unavailable |
| 4 x 8 | 4 | 0.846 [0.834, 0.859] | unavailable |
| 5 x 5 | 5 | 0.918 [0.903, 0.933] | unavailable |
| 5 x 7 | 5 | 0.595 [0.583, 0.607] | unavailable |
| 6 x 6 | 6 | 0.296 [0.287, 0.306] | unavailable |
| 7 x 7 | 7 | 0.165 [0.162, 0.168] | unavailable |

`native` is the fastest method wherever it applies. Through outer level 4,
`rows` and `schedule` are within a factor of 1.3 of each other, and which is
faster depends on the inner level. `schedule` is faster in every measured
pair from outer level 5, by a factor that reaches 6.1 at `7 x 7`. `"auto"`
therefore selects `native` where it applies, and otherwise `rows` through
outer level 4 and `schedule` above.

#### Full product (ns per product)

| n x m | N | pairs | split | table | table / split | fastest mean |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 x 4 | 64 | 64 | 2239.4 [2199.0, 2279.7] | 31068 [30108, 32028] | 13.9 [13.5, 14.2] | split |
| 1 x 8 | 1024 | 4 | 42746 [41476, 44015] | 10359237 [10275580, 10442893] | 243 [235, 251] | split |
| 2 x 2 | 32 | 64 | 2808.8 [2765.7, 2851.8] | 5277.3 [5181.5, 5373.1] | 1.88 [1.85, 1.91] | split |
| 2 x 4 | 128 | 64 | 5457.7 [5382.3, 5533.0] | 118236 [116597, 119876] | 21.7 [21.4, 22] | split |
| 2 x 6 | 512 | 16 | 22726 [22360, 23092] | 2853617 [2829235, 2878000] | 126 [124, 128] | split |
| 3 x 3 | 128 | 64 | 7084.4 [6913.1, 7255.7] | 111590 [109857, 113322] | 15.8 [15.4, 16.1] | split |
| 3 x 5 | 512 | 16 | 31974 [31164, 32783] | 2829542 [2794911, 2864173] | 88.6 [86.5, 90.8] | split |
| 3 x 7 | 2048 | 4 | 150022 [146757, 153287] | 43871862 [43568734, 44174990] | 293 [286, 300] | split |
| 4 x 4 | 512 | 16 | 55926 [54563, 57289] | 2729314 [2707134, 2751494] | 48.9 [47.8, 50] | split |
| 4 x 6 | 2048 | 4 | 236249 [229505, 242992] | 43405767 [43169403, 43642130] | 184 [180, 189] | split |
| 5 x 5 | 2048 | 4 | 361473 [355984, 366962] | 43561391 [43395788, 43726995] | 121 [119, 122] | split |

## Reproducing a run

```sh
python -m pip install -e ".[dev]"
python benchmarks/protocols.py --protocol all --output results/run.json --cpu 2
```

The three protocols write `run.real.json`, `run.domains.json`, and
`run.central.json` with their reports, and take about ten minutes together
on the archive's host; the second `domains` run over the fields of
characteristic 2 takes about six minutes more. The archived `central` run
was given a longer list of factor pairs with `--factors`; the README of the
archive records the exact commands. The corpus is written to
`benchmark-corpora/` on first use.

`benchmarks/test_benchmark.py` tests the drivers themselves: argument
validation, the statistics, the independent reference, and a short run of
every protocol.
