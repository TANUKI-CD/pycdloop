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

The results below were measured on 2026-11-14 on a Dell PowerEdge T440 with an
Intel Xeon Gold 6148, pinned to one logical CPU, with Python 3.14.8,
NumPy 2.5.4 and pyfastcd 1.4.0 on Linux. The host is shared, and
no governor, turbo, or SMT settings were changed; the load average during
the runs was between 1.9 and 3.6. Every verified product
agreed with the independent reference: 25,760 products in all. The
JSON documents and full reports are archived in
[benchmark-results/t440-20261114/](../benchmark-results/t440-20261114/).

### Real coefficients

Mean nanoseconds per product with 95% intervals. `N` is the dimension of
the loop algebra of `Q_n`.

#### Batch products (ns per product)

| N | n | pairs | split | table | splitform | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 190.3 [185.1, 195.6] | 602.1 [578.3, 625.9] | 83.04 [80.40, 85.68] | split | split |
| 8 | 2 | 256 | 203.3 [196.5, 210.1] | 834.1 [813.1, 855.1] | 84.72 [82.52, 86.93] | split | split |
| 16 | 3 | 256 | 212.8 [207.8, 217.9] | 1740.6 [1698.4, 1782.7] | 95.78 [93.34, 98.21] | split | split |
| 32 | 4 | 256 | 307.7 [295.0, 320.3] | 8842.4 [8605.5, 9079.3] | 146.4 [142.4, 150.3] | split | split |
| 64 | 5 | 256 | 540.1 [529.5, 550.7] | 46956 [46522, 47390] | 279.7 [274.0, 285.4] | split | split |
| 128 | 6 | 256 | 1149.9 [1122.9, 1176.8] | 194279 [189335, 199223] | 624.9 [614.2, 635.5] | split | split |
| 256 | 7 | 64 | 2734.3 [2685.3, 2783.2] | 647863 [642389, 653337] | 1529.4 [1507.1, 1551.8] | split | split |
| 512 | 8 | 64 | 5503.7 [5380.8, 5626.6] | 3734639 [3719464, 3749814] | 3294.1 [3236.5, 3351.6] | split | split |
| 1024 | 9 | 16 | 13413 [13203, 13624] | 14744094 [14694059, 14794129] | 8490.9 [8217.8, 8764.0] | split | split |
| 2048 | 10 | 16 | 28270 [27469, 29071] | 43634747 [43487652, 43781841] | 18146 [17755, 18537] | split | split |

#### Single products (ns per product)

| N | n | pairs | split | table | cached | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 30759 [30332, 31185] | 47298 [46560, 48037] | 15348 [15076, 15619] | split | split |
| 8 | 2 | 256 | 30530 [30088, 30972] | 47550 [46918, 48182] | 15609 [15284, 15934] | split | split |
| 16 | 3 | 256 | 30626 [30263, 30990] | 50165 [48857, 51472] | 15232 [14989, 15475] | split | split |
| 32 | 4 | 256 | 31086 [30677, 31495] | 54044 [52795, 55293] | 15697 [15403, 15990] | split | split |
| 64 | 5 | 256 | 31669 [31197, 32141] | 63259 [62080, 64438] | 15875 [15687, 16064] | split | split |
| 128 | 6 | 256 | 34926 [34322, 35530] | 124524 [121247, 127802] | 16781 [16473, 17089] | split | split |
| 256 | 7 | 64 | 36042 [35375, 36708] | 360453 [354995, 365912] | 18731 [18211, 19251] | split | split |
| 512 | 8 | 64 | 39011 [38032, 39989] | 1356510 [1337980, 1375040] | 20591 [19960, 21223] | split | split |
| 1024 | 9 | 16 | 45329 [44298, 46360] | 5092589 [5027865, 5157313] | 25600 [24968, 26232] | split | split |
| 2048 | 10 | 16 | 61153 [59247, 63059] | 34554684 [34368620, 34740749] | 35661 [34428, 36895] | split | split |

`split` is faster than `table` at every dimension in both modes, by a factor
that grows from 3.16 at dimension 4 to 1547 at
dimension 2048 in batch mode and from 1.54 to 566
in single mode. `"auto"` therefore selects `split` at every dimension.

A single call costs about 31 microseconds at small
dimensions, nearly all of it the fixed cost of a call through the Python
layers; a product of two elements that hold their split forms costs about
half that.

#### Comparison with the Cayley–Dickson product

| N | n | pyfastcd batch (ns) | splitform / pyfastcd | split / pyfastcd | pyfastcd single (ns) | cached / pyfastcd |
| ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 9.58 [9.39, 9.78] | 8.65 [8.42, 8.9] | 19.8 [19.2, 20.5] | 98.27 [96.66, 99.88] | 156 [153, 159] |
| 8 | 2 | 11.42 [11.25, 11.58] | 7.41 [7.21, 7.62] | 17.8 [17.1, 18.4] | 106.9 [105.4, 108.4] | 146 [144, 148] |
| 16 | 3 | 19.53 [19.16, 19.90] | 4.9 [4.76, 5.05] | 10.9 [10.6, 11.2] | 119.8 [117.8, 121.8] | 127 [125, 129] |
| 32 | 4 | 62.73 [61.80, 63.67] | 2.33 [2.26, 2.41] | 4.89 [4.68, 5.1] | 222.7 [218.8, 226.6] | 70.5 [69, 72] |
| 64 | 5 | 182.6 [179.6, 185.6] | 1.53 [1.49, 1.57] | 2.96 [2.88, 3.03] | 331.8 [327.6, 336.0] | 47.8 [47.2, 48.5] |
| 128 | 6 | 451.4 [446.1, 456.7] | 1.38 [1.36, 1.4] | 2.54 [2.48, 2.61] | 647.6 [641.9, 653.2] | 25.9 [25.5, 26.3] |
| 256 | 7 | 1122.9 [1108.9, 1137.0] | 1.36 [1.34, 1.38] | 2.43 [2.38, 2.49] | 1348.4 [1333.2, 1363.6] | 13.9 [13.5, 14.2] |
| 512 | 8 | 2613.7 [2585.5, 2642.0] | 1.26 [1.23, 1.29] | 2.1 [2.05, 2.15] | 2953.3 [2902.8, 3003.8] | 6.96 [6.79, 7.13] |
| 1024 | 9 | 6599.5 [6516.5, 6682.4] | 1.28 [1.24, 1.33] | 2.03 [2, 2.07] | 6665.8 [6574.3, 6757.2] | 3.84 [3.76, 3.92] |
| 2048 | 10 | 14983 [14795, 15171] | 1.21 [1.18, 1.24] | 1.88 [1.83, 1.94] | 14940 [14736, 15144] | 2.38 [2.31, 2.45] |

In batch mode a product in split form costs 8.65 times the
Cayley–Dickson product it contains at dimension 4 and 1.21 times at
dimension 2048, and a product in the loop basis costs 19.8 and
1.88 times. The conversions to and from split form account for
between 31% and 51% of the time of the conversions and
the product in split form together:

| N | n | forward (ns) | inverse (ns) | splitform (ns) | transform share |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 46.96 [46.08, 47.83] | 30.85 [30.24, 31.46] | 83.04 [80.40, 85.68] | 48.4% [47.8%, 49.0%] |
| 8 | 2 | 54.39 [53.60, 55.18] | 33.52 [32.94, 34.09] | 84.72 [82.52, 86.93] | 51.0% [50.4%, 51.5%] |
| 16 | 3 | 55.26 [54.35, 56.16] | 36.21 [35.85, 36.56] | 95.78 [93.34, 98.21] | 48.9% [48.3%, 49.5%] |
| 32 | 4 | 78.67 [77.68, 79.66] | 45.79 [44.79, 46.79] | 146.4 [142.4, 150.3] | 46.0% [45.4%, 46.5%] |
| 64 | 5 | 129.5 [127.2, 131.8] | 72.90 [71.84, 73.96] | 279.7 [274.0, 285.4] | 42.0% [41.5%, 42.4%] |
| 128 | 6 | 254.6 [251.3, 257.9] | 144.9 [142.0, 147.8] | 624.9 [614.2, 635.5] | 39.0% [38.6%, 39.4%] |
| 256 | 7 | 600.4 [591.4, 609.3] | 313.0 [306.3, 319.7] | 1529.4 [1507.1, 1551.8] | 37.4% [36.9%, 37.9%] |
| 512 | 8 | 1201.9 [1182.1, 1221.7] | 595.4 [582.7, 608.1] | 3294.1 [3236.5, 3351.6] | 35.3% [34.9%, 35.8%] |
| 1024 | 9 | 2708.5 [2672.4, 2744.6] | 1346.5 [1313.6, 1379.4] | 8490.9 [8217.8, 8764.0] | 32.4% [31.8%, 33.0%] |
| 2048 | 10 | 5517.8 [5433.8, 5601.9] | 2666.3 [2592.7, 2739.8] | 18146 [17755, 18537] | 31.1% [30.6%, 31.6%] |

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
| C | 714.5 (2.12x) | 1167.0 (6.42x) | 5071.8 (59.3x) | 23046 (185x) |
| F_2 | 1230.2 (1.11x) | 2299.3 (1.71x) | 9289.8 (6.06x) | 37146 (33.1x) |
| F_(2^2) | 2950.2 (2.49x) | 8222.4 (7.28x) | 31965 (29.2x) | 129177 (147x) |
| F_(2^8) | 18170 (0.481x) | 58437 (1.54x) | 203912 (10.4x) | 841704 (38.6x) |
| F_3 | 1123.0 (2.68x) | 2185.0 (9.54x) | 8189.5 (38.9x) | 29459 (178x) |
| F_13 | 1100.1 (2.64x) | 2259.0 (8.88x) | 9168.8 (33.5x) | 31997 (158x) |
| F_65521 | 1095.7 (2.68x) | 2189.6 (8.68x) | 8805.0 (34.3x) | 34426 (144x) |
| F_2147483647 | 1106.8 (2.65x) | 2201.3 (8.93x) | 8812.5 (33.8x) | 34757 (142x) |
| F_(3^5) | 3555.8 (3.7x) | 8361.5 (17x) | 31477 (72.1x) | 131231 (312x) |
| F_(13^5) | 3727.1 (3.56x) | 11644 (12.8x) | 35469 (66.6x) | 137251 (297x) |

#### Single products: split or lift ns per product (ratio of table to it)

| Field | N = 8 | N = 32 | N = 128 | N = 512 |
| --- | ---: | ---: | ---: | ---: |
| C | 37137 (1.34x) | 38719 (1.48x) | 42811 (3.15x) | 52334 (32.2x) |
| F_2 | 53887 (0.914x) | 56133 (0.945x) | 67387 (1.51x) | 82649 (11.6x) |
| F_(2^2) | 93867 (1.67x) | 112808 (2.04x) | 135480 (8.06x) | 208817 (71.7x) |
| F_(2^8) | 272136 (0.589x) | 348761 (0.719x) | 505255 (2.84x) | 1054060 (20.6x) |
| F_3 | 46283 (1.64x) | 47045 (2.05x) | 54205 (7.48x) | 66906 (76.4x) |
| F_13 | 45482 (1.64x) | 48547 (2.02x) | 58862 (6.8x) | 70551 (70x) |
| F_65521 | 46885 (1.62x) | 46639 (2.07x) | 56336 (6.77x) | 75680 (64.1x) |
| F_2147483647 | 46814 (1.63x) | 48545 (2.01x) | 57322 (6.64x) | 75766 (64x) |
| F_(3^5) | 77821 (2.74x) | 87575 (3.99x) | 106064 (22.3x) | 193388 (184x) |
| F_(13^5) | 79092 (2.69x) | 97139 (3.78x) | 110134 (22.8x) | 199993 (177x) |

**C and odd characteristic.** `split` is faster in every cell, in both
modes. For single products its advantage at dimension 8 is a factor of
1.3 over C and about 1.6 over
the prime fields. A single `split` call over C and the prime fields costs
37 to 47 microseconds at dimension 8, nearly all of it the fixed
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
| F_2 | 1069.1 (1.06x) | 1244.7 (1.09x) | 1585.9 (1.19x) | 2372.9 (1.64x) | 6145.2 (2.38x) | 9177.0 (5.97x) | 27539 (11.3x) | 35505 (53.3x) |
| F_(2^2) | 2172.2 (2.04x) | 3042.0 (2.43x) | 5213.8 (3.46x) | 8593.6 (7x) | 19015 (12.2x) | 31156 (39.3x) | 75430 (65.3x) | 124873 (151x) |
| F_(2^4) | 3188.2 (1.36x) | 6003.6 (1.31x) | 13589 (1.43x) | 18076 (3.83x) | 36787 (7.3x) | 65718 (19.7x) | 141317 (36.5x) | 273314 (89.1x) |
| F_(2^6) | 4951.0 (0.91x) | 10906 (0.747x) | 28291 (0.741x) | 33131 (2.35x) | 64155 (4.79x) | 107807 (11.1x) | 234094 (19.9x) | 425613 (67.7x) |
| F_(2^7) | 9786.5 (0.462x) | 18121 (0.459x) | 41673 (0.54x) | 56129 (1.5x) | 111419 (2.97x) | 196111 (6.49x) | 439560 (11.8x) | 783611 (39x) |
| F_(2^8) | 9779.2 (0.469x) | 18084 (0.472x) | 41176 (0.572x) | 54482 (1.6x) | 113289 (3.08x) | 205664 (6.98x) | 450834 (12.8x) | 786777 (41.4x) |
| F_(2^12) | 15172 (0.331x) | 33007 (0.299x) | 84718 (0.347x) | 101558 (1.09x) | 196174 (2.21x) | 332248 (5.43x) | 725510 (10.1x) | 1340815 (33.3x) |

#### Single products: lift ns per product (table / lift)

| Field | N = 4 | N = 8 | N = 16 | N = 32 | N = 64 | N = 128 | N = 256 | N = 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_2 | 55061 (0.881x) | 55640 (0.875x) | 55527 (0.892x) | 56740 (0.903x) | 59309 (1.02x) | 65181 (1.56x) | 75813 (3.78x) | 85875 (11.3x) |
| F_(2^2) | 94774 (1.59x) | 97143 (1.64x) | 100521 (1.69x) | 113592 (2.03x) | 114988 (3.41x) | 139299 (7.68x) | 161342 (23.4x) | 211285 (68.7x) |
| F_(2^4) | 95166 (1.6x) | 98897 (1.59x) | 109084 (1.59x) | 125941 (1.86x) | 137139 (3.07x) | 166077 (6.95x) | 228228 (19.2x) | 356486 (47.6x) |
| F_(2^6) | 97906 (1.56x) | 104590 (1.53x) | 123318 (1.46x) | 140554 (1.7x) | 164176 (2.69x) | 221399 (5.7x) | 319567 (15.2x) | 516492 (36.2x) |
| F_(2^7) | 261356 (0.595x) | 273372 (0.575x) | 302095 (0.594x) | 348812 (0.7x) | 395538 (1.17x) | 476182 (2.78x) | 684117 (7.79x) | 1011798 (19.7x) |
| F_(2^8) | 261351 (0.588x) | 274763 (0.572x) | 296167 (0.616x) | 339500 (0.715x) | 396476 (1.19x) | 492261 (2.94x) | 723535 (7.73x) | 1083191 (20.8x) |
| F_(2^12) | 279517 (0.557x) | 289522 (0.574x) | 345876 (0.537x) | 396401 (0.64x) | 483502 (1.14x) | 626352 (2.84x) | 979202 (7.03x) | 1639659 (19.5x) |

`lift` is the faster algorithm in large loops, where its advantage reaches
a factor of 151 in batch mode and 68.7 for
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
1.35. The largest differences are for batches over F_64 in loops of
dimension at most 16, where the rule selects `lift`.

### Central products

#### Signed-component methods (ns per product)

| n x m | D | pairs | native | rows | schedule | fastest mean | auto |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| 1 x 4 | 32 | 64 | 1861.4 [1805.0, 1917.7] | 2425.7 [2362.8, 2488.6] | 2755.8 [2680.0, 2831.6] | native | native |
| 1 x 8 | 512 | 4 | 33978 [32943, 35014] | 41906 [40627, 43184] | 50957 [49856, 52057] | native | native |
| 2 x 2 | 16 | 64 | unavailable | 2340.1 [2301.5, 2378.8] | 2774.2 [2728.2, 2820.1] | rows | rows |
| 2 x 4 | 64 | 64 | unavailable | 4636.0 [4479.2, 4792.8] | 4275.1 [4189.8, 4360.5] | schedule | rows |
| 2 x 6 | 256 | 16 | unavailable | 19582 [19309, 19856] | 21006 [20527, 21485] | rows | rows |
| 3 x 3 | 64 | 64 | unavailable | 6363.7 [6270.2, 6457.2] | 8692.3 [8527.6, 8857.0] | rows | rows |
| 3 x 5 | 256 | 16 | unavailable | 29510 [28963, 30057] | 38981 [38410, 39551] | rows | rows |
| 3 x 7 | 1024 | 4 | unavailable | 129912 [127442, 132383] | 177582 [172857, 182306] | rows | rows |
| 4 x 4 | 256 | 16 | unavailable | 52578 [51523, 53633] | 58511 [57297, 59725] | rows | rows |
| 4 x 6 | 1024 | 4 | unavailable | 222628 [219003, 226253] | 269723 [261036, 278411] | rows | rows |
| 4 x 8 | 4096 | 4 | unavailable | 905611 [891183, 920039] | 788146 [769587, 806704] | schedule | rows |
| 5 x 5 | 1024 | 4 | unavailable | 376410 [370273, 382548] | 340347 [331129, 349564] | schedule | schedule |
| 5 x 7 | 4096 | 4 | unavailable | 1656751 [1628222, 1685279] | 1041602 [1008986, 1074218] | schedule | schedule |
| 6 x 6 | 4096 | 4 | unavailable | 3931599 [3886864, 3976333] | 1160088 [1148895, 1171280] | schedule | schedule |
| 7 x 7 | 16384 | 2 | unavailable | 34892275 [34579607, 35204943] | 5828510 [5666627, 5990394] | schedule | schedule |

#### Paired ratios of the signed-component methods

| n x m | outer level | schedule / rows | native / rows |
| --- | ---: | --- | --- |
| 1 x 4 | 1 | 1.14 [1.1, 1.17] | 0.767 [0.747, 0.787] |
| 1 x 8 | 1 | 1.22 [1.18, 1.25] | 0.811 [0.795, 0.827] |
| 2 x 2 | 2 | 1.19 [1.17, 1.2] | unavailable |
| 2 x 4 | 2 | 0.924 [0.895, 0.953] | unavailable |
| 2 x 6 | 2 | 1.07 [1.05, 1.1] | unavailable |
| 3 x 3 | 3 | 1.37 [1.34, 1.39] | unavailable |
| 3 x 5 | 3 | 1.32 [1.3, 1.35] | unavailable |
| 3 x 7 | 3 | 1.37 [1.32, 1.41] | unavailable |
| 4 x 4 | 4 | 1.11 [1.09, 1.14] | unavailable |
| 4 x 6 | 4 | 1.21 [1.18, 1.24] | unavailable |
| 4 x 8 | 4 | 0.87 [0.853, 0.887] | unavailable |
| 5 x 5 | 5 | 0.903 [0.882, 0.925] | unavailable |
| 5 x 7 | 5 | 0.628 [0.613, 0.642] | unavailable |
| 6 x 6 | 6 | 0.295 [0.291, 0.299] | unavailable |
| 7 x 7 | 7 | 0.167 [0.162, 0.171] | unavailable |

`native` is the fastest method wherever it applies. Through outer level 4,
`rows` and `schedule` are within a factor of 1.4 of each other, and
which is faster depends on the inner level. `schedule` is faster in every
measured pair from outer level 5, by a factor that reaches 6.0 at
`7 x 7`. `"auto"` therefore selects `native` where it applies, and otherwise
`rows` through outer level 4 and `schedule` above.

#### Full product (ns per product)

| n x m | N | pairs | split | table | table / split | fastest mean |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 x 4 | 64 | 64 | 2285.7 [2235.8, 2335.6] | 31144 [30439, 31848] | 13.6 [13.5, 13.8] | split |
| 1 x 8 | 1024 | 4 | 41383 [40454, 42312] | 10332562 [10193911, 10471214] | 250 [244, 256] | split |
| 2 x 2 | 32 | 64 | 2842.2 [2779.1, 2905.2] | 5187.0 [5065.8, 5308.1] | 1.82 [1.78, 1.87] | split |
| 2 x 4 | 128 | 64 | 5248.3 [5110.8, 5385.8] | 113131 [111190, 115071] | 21.6 [21.1, 22.1] | split |
| 2 x 6 | 512 | 16 | 23124 [22729, 23519] | 2858846 [2838265, 2879426] | 124 [122, 126] | split |
| 3 x 3 | 128 | 64 | 7195.0 [7075.3, 7314.8] | 112953 [111140, 114765] | 15.7 [15.4, 16] | split |
| 3 x 5 | 512 | 16 | 32584 [31843, 33325] | 2835687 [2804456, 2866918] | 87.1 [85.2, 89.1] | split |
| 3 x 7 | 2048 | 4 | 143956 [140616, 147296] | 43521000 [43279902, 43762099] | 303 [295, 310] | split |
| 4 x 4 | 512 | 16 | 55901 [54657, 57146] | 2718702 [2692574, 2744831] | 48.7 [47.6, 49.8] | split |
| 4 x 6 | 2048 | 4 | 240817 [233059, 248575] | 43130455 [42812245, 43448665] | 180 [174, 185] | split |
| 5 x 5 | 2048 | 4 | 347098 [341586, 352610] | 43162866 [42950890, 43374842] | 124 [123, 126] | split |

## Reproducing a run

```sh
python -m pip install -e ".[dev]"
python benchmarks/protocols.py --protocol all --output results/run.json --cpu 2
```

The three protocols write `run.real.json`, `run.domains.json` and
`run.central.json` with their reports, and take about ten minutes
together on the archive's host; the second `domains` run over the fields of
characteristic 2 takes about six minutes more. The archived `central` run
was given a longer list of factor pairs with `--factors`; the README of the
archive records the exact commands. The corpus is written to
`benchmark-corpora/` on first use. `benchmarks/test_benchmark.py` tests the
drivers themselves: argument validation, the statistics, the independent
reference, and a short run of every protocol.
