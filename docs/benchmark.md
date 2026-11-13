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

The results below were measured on 2026-11-13 on a Dell PowerEdge T440 with an
Intel Xeon Gold 6148, pinned to one logical CPU, with Python 3.14.8,
NumPy 2.5.4 and pyfastcd 1.4.0 on Linux. The host is shared, and
no governor, turbo, or SMT settings were changed; the load average during
the runs was between 2.3 and 3.8. Every verified product
agreed with the independent reference: 25,760 products in all. The
JSON documents and full reports are archived in
[benchmark-results/t440-20261113/](../benchmark-results/t440-20261113/).

### Real coefficients

Mean nanoseconds per product with 95% intervals. `N` is the dimension of
the loop algebra of `Q_n`.

#### Batch products (ns per product)

| N | n | pairs | split | table | splitform | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 206.7 [194.1, 219.3] | 648.3 [612.0, 684.6] | 78.63 [70.58, 86.68] | split | split |
| 8 | 2 | 256 | 219.2 [192.1, 246.3] | 930.8 [808.5, 1053.0] | 76.56 [69.26, 83.87] | split | split |
| 16 | 3 | 256 | 212.7 [205.8, 219.6] | 1693.6 [1657.9, 1729.2] | 79.42 [78.14, 80.71] | split | split |
| 32 | 4 | 256 | 295.4 [291.4, 299.4] | 8879.7 [8726.6, 9032.8] | 128.1 [126.2, 130.0] | split | split |
| 64 | 5 | 256 | 536.2 [526.7, 545.8] | 46773 [45957, 47590] | 269.9 [263.6, 276.1] | split | split |
| 128 | 6 | 256 | 1082.5 [1066.8, 1098.1] | 184011 [181840, 186182] | 579.6 [571.3, 588.0] | split | split |
| 256 | 7 | 64 | 2636.8 [2584.7, 2689.0] | 709910 [703908, 715913] | 1471.9 [1439.8, 1503.9] | split | split |
| 512 | 8 | 64 | 5639.6 [5515.0, 5764.3] | 3767871 [3747529, 3788214] | 3288.8 [3203.9, 3373.6] | split | split |
| 1024 | 9 | 16 | 12811 [12609, 13012] | 14594981 [14534451, 14655512] | 7980.2 [7775.6, 8184.8] | split | split |
| 2048 | 10 | 16 | 27713 [27285, 28140] | 43475940 [43351362, 43600518] | 17286 [16992, 17581] | split | split |

#### Single products (ns per product)

| N | n | pairs | split | table | cached | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 30972 [30348, 31596] | 48802 [47518, 50087] | 16136 [15577, 16695] | split | split |
| 8 | 2 | 256 | 32437 [28526, 36349] | 52726 [45649, 59804] | 16782 [14571, 18992] | split | split |
| 16 | 3 | 256 | 29206 [28908, 29504] | 47270 [46724, 47816] | 15277 [14993, 15562] | split | split |
| 32 | 4 | 256 | 29945 [29619, 30271] | 53394 [52445, 54342] | 15623 [15449, 15796] | split | split |
| 64 | 5 | 256 | 30041 [29651, 30430] | 62016 [61120, 62912] | 15661 [15356, 15966] | split | split |
| 128 | 6 | 256 | 31721 [30782, 32659] | 116140 [113360, 118920] | 15895 [15655, 16135] | split | split |
| 256 | 7 | 64 | 33203 [32630, 33777] | 353772 [345186, 362358] | 18326 [17695, 18957] | split | split |
| 512 | 8 | 64 | 38219 [36821, 39617] | 1353414 [1323163, 1383665] | 20395 [19812, 20978] | split | split |
| 1024 | 9 | 16 | 43406 [42693, 44119] | 4964909 [4915770, 5014048] | 25062 [24390, 25734] | split | split |
| 2048 | 10 | 16 | 59113 [57928, 60298] | 33646177 [33498984, 33793370] | 35637 [34671, 36603] | split | split |

`split` is faster than `table` at every dimension in both modes, by a factor
that grows from 3.14 at dimension 4 to 1570 at
dimension 2048 in batch mode and from 1.57 to 570
in single mode. `"auto"` therefore selects `split` at every dimension.

A single call costs about 31 microseconds at small
dimensions, nearly all of it the fixed cost of a call through the Python
layers; a product of two elements that hold their split forms costs about
half that.

#### Comparison with the Cayley–Dickson product

| N | n | pyfastcd batch (ns) | splitform / pyfastcd | split / pyfastcd | pyfastcd single (ns) | cached / pyfastcd |
| ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 9.51 [9.26, 9.75] | 8.11 [7.52, 8.74] | 21.6 [20.4, 22.8] | 102.9 [100.2, 105.6] | 157 [152, 161] |
| 8 | 2 | 12.42 [10.72, 14.13] | 6.26 [5.86, 6.69] | 17.7 [16.4, 19.1] | 120.6 [98.19, 143.0] | 142 [138, 147] |
| 16 | 3 | 19.33 [18.96, 19.69] | 4.11 [4.02, 4.2] | 11 [10.6, 11.4] | 121.2 [119.8, 122.5] | 126 [124, 128] |
| 32 | 4 | 62.26 [61.64, 62.89] | 2.06 [2.02, 2.09] | 4.74 [4.67, 4.82] | 229.5 [227.1, 231.8] | 68.1 [67.1, 69.1] |
| 64 | 5 | 185.1 [182.9, 187.4] | 1.46 [1.42, 1.49] | 2.9 [2.84, 2.96] | 333.8 [328.3, 339.3] | 46.9 [46.1, 47.7] |
| 128 | 6 | 449.4 [441.9, 457.0] | 1.29 [1.26, 1.32] | 2.41 [2.36, 2.46] | 623.7 [617.3, 630.0] | 25.5 [25.1, 25.9] |
| 256 | 7 | 1109.9 [1097.7, 1122.1] | 1.32 [1.3, 1.35] | 2.37 [2.32, 2.43] | 1326.2 [1312.7, 1339.7] | 13.8 [13.3, 14.2] |
| 512 | 8 | 2629.2 [2595.8, 2662.6] | 1.25 [1.22, 1.28] | 2.14 [2.09, 2.2] | 2980.1 [2931.2, 3029.1] | 6.83 [6.66, 7.01] |
| 1024 | 9 | 6422.9 [6331.4, 6514.4] | 1.24 [1.2, 1.28] | 1.99 [1.96, 2.03] | 6506.3 [6429.0, 6583.7] | 3.85 [3.76, 3.93] |
| 2048 | 10 | 14674 [14473, 14874] | 1.18 [1.15, 1.21] | 1.89 [1.85, 1.93] | 14818 [14665, 14970] | 2.4 [2.34, 2.46] |

In batch mode a product in split form costs 8.11 times the
Cayley–Dickson product it contains at dimension 4 and 1.18 times at
dimension 2048, and a product in the loop basis costs 21.6 and
1.89 times. The conversions to and from split form account for
between 31% and 54% of the time of the conversions and
the product in split form together:

| N | n | forward (ns) | inverse (ns) | splitform (ns) | transform share |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 48.00 [46.14, 49.86] | 30.51 [29.73, 31.30] | 78.63 [70.58, 86.68] | 50.5% [48.7%, 52.2%] |
| 8 | 2 | 54.73 [48.41, 61.04] | 34.18 [30.47, 37.89] | 76.56 [69.26, 83.87] | 53.7% [52.7%, 54.6%] |
| 16 | 3 | 51.94 [51.29, 52.59] | 34.01 [33.55, 34.47] | 79.42 [78.14, 80.71] | 52.0% [51.6%, 52.4%] |
| 32 | 4 | 77.69 [76.81, 78.58] | 43.66 [42.69, 44.62] | 128.1 [126.2, 130.0] | 48.7% [48.2%, 49.1%] |
| 64 | 5 | 125.0 [123.2, 126.8] | 70.59 [69.33, 71.86] | 269.9 [263.6, 276.1] | 42.1% [41.6%, 42.5%] |
| 128 | 6 | 248.2 [245.4, 251.0] | 145.6 [142.6, 148.7] | 579.6 [571.3, 588.0] | 40.5% [40.1%, 40.8%] |
| 256 | 7 | 581.7 [573.4, 590.1] | 297.0 [289.2, 304.8] | 1471.9 [1439.8, 1503.9] | 37.4% [36.9%, 37.9%] |
| 512 | 8 | 1217.8 [1200.6, 1235.0] | 593.7 [580.5, 606.8] | 3288.8 [3203.9, 3373.6] | 35.6% [35.1%, 36.0%] |
| 1024 | 9 | 2667.3 [2616.2, 2718.4] | 1298.1 [1272.6, 1323.5] | 7980.2 [7775.6, 8184.8] | 33.2% [32.5%, 33.9%] |
| 2048 | 10 | 5332.8 [5272.3, 5393.3] | 2547.5 [2506.3, 2588.8] | 17286 [16992, 17581] | 31.3% [31.0%, 31.6%] |

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
| C | 682.1 (2.2x) | 1128.9 (6.12x) | 5011.4 (58.7x) | 22466 (185x) |
| F_2 | 1191.2 (1.13x) | 2276.4 (1.7x) | 8775.8 (6.2x) | 36005 (34x) |
| F_(2^2) | 2952.7 (2.47x) | 8010.3 (7.3x) | 29859 (30.5x) | 121135 (153x) |
| F_(2^8) | 17930 (0.46x) | 58434 (1.57x) | 200890 (10.3x) | 785457 (41.5x) |
| F_3 | 1056.3 (2.81x) | 2111.6 (9.97x) | 8208.2 (38.8x) | 28794 (183x) |
| F_13 | 1063.8 (2.73x) | 2126.8 (8.97x) | 8296.6 (35.9x) | 30207 (164x) |
| F_65521 | 1042.5 (2.74x) | 2179.3 (8.74x) | 8481.6 (35x) | 33252 (148x) |
| F_2147483647 | 1070.1 (2.69x) | 2162.1 (8.76x) | 8402.2 (35.3x) | 33578 (147x) |
| F_(3^5) | 3510.0 (3.7x) | 8264.2 (17.2x) | 30931 (72.4x) | 123808 (318x) |
| F_(13^5) | 3729.8 (3.54x) | 11381 (12.8x) | 35323 (65.2x) | 140480 (288x) |

#### Single products: split or lift ns per product (ratio of table to it)

| Field | N = 8 | N = 32 | N = 128 | N = 512 |
| --- | ---: | ---: | ---: | ---: |
| C | 35633 (1.36x) | 35983 (1.5x) | 40097 (3.3x) | 50433 (32.7x) |
| F_2 | 53095 (0.906x) | 52932 (0.969x) | 61328 (1.61x) | 79884 (12.1x) |
| F_(2^2) | 91141 (1.76x) | 104643 (2.11x) | 129228 (8.13x) | 198992 (72.2x) |
| F_(2^8) | 265737 (0.592x) | 352378 (0.739x) | 481392 (2.99x) | 1030819 (20.7x) |
| F_3 | 44012 (1.67x) | 45721 (2.11x) | 52549 (7.65x) | 64595 (78.3x) |
| F_13 | 44370 (1.69x) | 45134 (2.08x) | 53070 (7.09x) | 66491 (71.9x) |
| F_65521 | 44113 (1.69x) | 45924 (2.07x) | 53697 (6.99x) | 70200 (68.5x) |
| F_2147483647 | 44294 (1.69x) | 45827 (2.06x) | 54942 (6.83x) | 72404 (66.1x) |
| F_(3^5) | 74422 (2.76x) | 82343 (4.19x) | 104687 (22.4x) | 180254 (188x) |
| F_(13^5) | 75291 (2.71x) | 92772 (3.75x) | 107604 (22.5x) | 198036 (177x) |

**C and odd characteristic.** `split` is faster in every cell, in both
modes. For single products its advantage at dimension 8 is a factor of
1.4 over C and about 1.7 over
the prime fields. A single `split` call over C and the prime fields costs
36 to 44 microseconds at dimension 8, nearly all of it the fixed
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
| F_2 | 1060.5 (1.05x) | 1242.5 (1.08x) | 1524.9 (1.22x) | 2258.6 (1.69x) | 6068.7 (2.36x) | 8939.3 (6.15x) | 25173 (12.3x) | 35769 (52.6x) |
| F_(2^2) | 2139.9 (1.97x) | 2893.5 (2.48x) | 5142.4 (3.44x) | 8008.7 (7.45x) | 18218 (12.8x) | 30076 (40.4x) | 71530 (67.5x) | 122010 (148x) |
| F_(2^4) | 3164.4 (1.35x) | 5856.4 (1.3x) | 13468 (1.42x) | 17744 (3.91x) | 34925 (7.78x) | 63502 (19.9x) | 138054 (37.3x) | 258037 (92.9x) |
| F_(2^6) | 4897.2 (0.902x) | 10563 (0.727x) | 27692 (0.732x) | 32981 (2.38x) | 61530 (4.92x) | 105579 (11x) | 233063 (19.9x) | 412988 (68.9x) |
| F_(2^7) | 9639.4 (0.463x) | 17688 (0.46x) | 40229 (0.541x) | 53147 (1.53x) | 105205 (3.05x) | 189767 (6.58x) | 431295 (11.8x) | 776971 (38.7x) |
| F_(2^8) | 9627.4 (0.465x) | 18082 (0.462x) | 41066 (0.583x) | 54984 (1.61x) | 108503 (3.13x) | 197918 (7.21x) | 439508 (12.9x) | 782171 (41.4x) |
| F_(2^12) | 15238 (0.332x) | 33010 (0.298x) | 85103 (0.357x) | 103613 (1.09x) | 188164 (2.29x) | 330635 (5.75x) | 719388 (10.7x) | 1288117 (34.6x) |

#### Single products: lift ns per product (table / lift)

| Field | N = 4 | N = 8 | N = 16 | N = 32 | N = 64 | N = 128 | N = 256 | N = 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_2 | 52840 (0.898x) | 53358 (0.91x) | 52979 (0.904x) | 53900 (0.929x) | 57718 (1.03x) | 64209 (1.55x) | 68857 (4.06x) | 81472 (11.8x) |
| F_(2^2) | 93011 (1.62x) | 92017 (1.66x) | 95651 (1.72x) | 104627 (2.1x) | 111734 (3.54x) | 131174 (8.19x) | 148600 (25.2x) | 202650 (71.9x) |
| F_(2^4) | 92807 (1.63x) | 94759 (1.64x) | 104835 (1.67x) | 120390 (1.93x) | 129459 (3.23x) | 156375 (7.36x) | 219040 (19.6x) | 334790 (50x) |
| F_(2^6) | 93435 (1.59x) | 99648 (1.51x) | 118277 (1.46x) | 132830 (1.75x) | 158286 (2.76x) | 206541 (5.89x) | 307568 (15.7x) | 496565 (37.1x) |
| F_(2^7) | 256231 (0.595x) | 264502 (0.571x) | 291850 (0.594x) | 319647 (0.726x) | 376022 (1.19x) | 452422 (2.82x) | 651678 (8.1x) | 1005218 (19.8x) |
| F_(2^8) | 257351 (0.582x) | 265142 (0.592x) | 293326 (0.597x) | 334288 (0.714x) | 371028 (1.25x) | 469244 (3.08x) | 646958 (8.5x) | 1036474 (22.3x) |
| F_(2^12) | 266299 (0.573x) | 286852 (0.558x) | 343017 (0.53x) | 398081 (0.649x) | 457470 (1.18x) | 603213 (2.91x) | 923439 (7.6x) | 1547332 (20.5x) |

`lift` is the faster algorithm in large loops, where its advantage reaches
a factor of 149 in batch mode and 71.5 for
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
1.38. The largest differences are for batches over F_64 in loops of
dimension at most 16, where the rule selects `lift`.

### Central products

#### Signed-component methods (ns per product)

| n x m | D | pairs | native | rows | schedule | fastest mean | auto |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| 1 x 4 | 32 | 64 | 1817.5 [1774.4, 1860.7] | 2339.6 [2298.9, 2380.3] | 2660.5 [2591.5, 2729.4] | native | native |
| 1 x 8 | 512 | 4 | 33041 [32436, 33646] | 40863 [40240, 41486] | 49935 [48800, 51071] | native | native |
| 2 x 2 | 16 | 64 | unavailable | 2446.2 [2396.5, 2496.0] | 2773.1 [2722.9, 2823.4] | rows | rows |
| 2 x 4 | 64 | 64 | unavailable | 4520.1 [4442.4, 4597.8] | 4282.9 [4208.2, 4357.6] | schedule | rows |
| 2 x 6 | 256 | 16 | unavailable | 19054 [18787, 19322] | 20419 [19930, 20907] | rows | rows |
| 3 x 3 | 64 | 64 | unavailable | 6081.7 [5990.1, 6173.4] | 8372.5 [8227.0, 8518.0] | rows | rows |
| 3 x 5 | 256 | 16 | unavailable | 28719 [28108, 29330] | 38730 [38014, 39446] | rows | rows |
| 3 x 7 | 1024 | 4 | unavailable | 128990 [126547, 131433] | 171041 [168970, 173113] | rows | rows |
| 4 x 4 | 256 | 16 | unavailable | 52824 [51717, 53932] | 57344 [56075, 58613] | rows | rows |
| 4 x 6 | 1024 | 4 | unavailable | 225889 [220492, 231286] | 270132 [262540, 277723] | rows | rows |
| 4 x 8 | 4096 | 4 | unavailable | 917986 [906831, 929140] | 783561 [771308, 795814] | schedule | rows |
| 5 x 5 | 1024 | 4 | unavailable | 378496 [373085, 383907] | 341760 [331183, 352338] | schedule | schedule |
| 5 x 7 | 4096 | 4 | unavailable | 1613719 [1588245, 1639193] | 992419 [970234, 1014603] | schedule | schedule |
| 6 x 6 | 4096 | 4 | unavailable | 3880501 [3818776, 3942225] | 1151667 [1134160, 1169174] | schedule | schedule |
| 7 x 7 | 16384 | 2 | unavailable | 35324294 [35056218, 35592369] | 5876363 [5776103, 5976622] | schedule | schedule |

#### Paired ratios of the signed-component methods

| n x m | outer level | schedule / rows | native / rows |
| --- | ---: | --- | --- |
| 1 x 4 | 1 | 1.14 [1.11, 1.17] | 0.776 [0.759, 0.794] |
| 1 x 8 | 1 | 1.22 [1.2, 1.24] | 0.808 [0.794, 0.823] |
| 2 x 2 | 2 | 1.13 [1.11, 1.16] | unavailable |
| 2 x 4 | 2 | 0.948 [0.934, 0.961] | unavailable |
| 2 x 6 | 2 | 1.07 [1.04, 1.1] | unavailable |
| 3 x 3 | 3 | 1.38 [1.35, 1.4] | unavailable |
| 3 x 5 | 3 | 1.35 [1.33, 1.37] | unavailable |
| 3 x 7 | 3 | 1.33 [1.31, 1.35] | unavailable |
| 4 x 4 | 4 | 1.09 [1.06, 1.11] | unavailable |
| 4 x 6 | 4 | 1.2 [1.17, 1.22] | unavailable |
| 4 x 8 | 4 | 0.853 [0.841, 0.866] | unavailable |
| 5 x 5 | 5 | 0.901 [0.876, 0.927] | unavailable |
| 5 x 7 | 5 | 0.615 [0.604, 0.626] | unavailable |
| 6 x 6 | 6 | 0.297 [0.291, 0.303] | unavailable |
| 7 x 7 | 7 | 0.166 [0.163, 0.169] | unavailable |

`native` is the fastest method wherever it applies. Through outer level 4,
`rows` and `schedule` are within a factor of 1.4 of each other, and
which is faster depends on the inner level. `schedule` is faster in every
measured pair from outer level 5, by a factor that reaches 6.0 at
`7 x 7`. `"auto"` therefore selects `native` where it applies, and otherwise
`rows` through outer level 4 and `schedule` above.

#### Full product (ns per product)

| n x m | N | pairs | split | table | table / split | fastest mean |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 x 4 | 64 | 64 | 2239.3 [2173.4, 2305.2] | 30664 [30059, 31268] | 13.7 [13.4, 14] | split |
| 1 x 8 | 1024 | 4 | 40271 [39497, 41044] | 10238071 [10149224, 10326919] | 254 [249, 260] | split |
| 2 x 2 | 32 | 64 | 2794.6 [2719.2, 2870.1] | 5039.7 [4960.5, 5118.9] | 1.81 [1.76, 1.85] | split |
| 2 x 4 | 128 | 64 | 5181.8 [5111.4, 5252.3] | 110789 [109434, 112143] | 21.4 [21.1, 21.6] | split |
| 2 x 6 | 512 | 16 | 22320 [21881, 22759] | 2808996 [2790878, 2827113] | 126 [123, 129] | split |
| 3 x 3 | 128 | 64 | 6717.5 [6629.1, 6805.8] | 106426 [105579, 107272] | 15.8 [15.6, 16.1] | split |
| 3 x 5 | 512 | 16 | 31743 [31248, 32238] | 2787642 [2767484, 2807801] | 87.9 [86.2, 89.6] | split |
| 3 x 7 | 2048 | 4 | 144905 [141151, 148660] | 43821420 [43657425, 43985416] | 303 [296, 310] | split |
| 4 x 4 | 512 | 16 | 55611 [54350, 56872] | 2693134 [2668359, 2717909] | 48.5 [47.3, 49.7] | split |
| 4 x 6 | 2048 | 4 | 236623 [230666, 242581] | 43486770 [43292751, 43680788] | 184 [180, 189] | split |
| 5 x 5 | 2048 | 4 | 352845 [345612, 360078] | 43116388 [42924813, 43307962] | 122 [120, 125] | split |

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
