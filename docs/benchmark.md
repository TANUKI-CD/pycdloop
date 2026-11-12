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

The results below were measured on 2026-11-11 on a Dell PowerEdge T440 with an
Intel Xeon Gold 6148, pinned to one logical CPU, with Python 3.14.8,
NumPy 2.5.4 and pyfastcd 1.3.1 on Linux. The host is shared, and
no governor, turbo, or SMT settings were changed; the load average during
the runs was between 2.7 and 4.1. Every verified product
agreed with the independent reference: 25,760 products in all. The
JSON documents and full reports are archived in
[benchmark-results/t440-20261111/](../benchmark-results/t440-20261111/).

### Real coefficients

Mean nanoseconds per product with 95% intervals. `N` is the dimension of
the loop algebra of `Q_n`.

#### Batch products (ns per product)

| N | n | pairs | split | table | splitform | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 372.0 [358.9, 385.1] | 587.0 [572.4, 601.6] | 254.2 [246.7, 261.8] | split | split |
| 8 | 2 | 256 | 379.9 [373.0, 386.7] | 832.0 [816.5, 847.5] | 250.5 [244.9, 256.0] | split | split |
| 16 | 3 | 256 | 396.0 [389.4, 402.7] | 1675.2 [1648.9, 1701.6] | 257.4 [252.7, 262.1] | split | split |
| 32 | 4 | 256 | 487.9 [478.9, 497.0] | 8803.1 [8595.4, 9010.8] | 315.4 [306.7, 324.1] | split | split |
| 64 | 5 | 256 | 765.8 [750.8, 780.8] | 46746 [45938, 47554] | 458.4 [452.0, 464.8] | split | split |
| 128 | 6 | 256 | 1329.4 [1306.2, 1352.6] | 186075 [184323, 187826] | 810.6 [799.6, 821.7] | split | split |
| 256 | 7 | 64 | 3618.6 [3554.4, 3682.7] | 704150 [699159, 709140] | 2271.4 [2218.8, 2324.1] | split | split |
| 512 | 8 | 64 | 6461.9 [6322.2, 6601.5] | 3767752 [3755732, 3779771] | 4091.6 [4028.3, 4154.9] | split | split |
| 1024 | 9 | 16 | 16833 [16588, 17079] | 14384721 [14331087, 14438355] | 11360 [11149, 11572] | split | split |
| 2048 | 10 | 16 | 31172 [30588, 31755] | 43506362 [43313503, 43699220] | 21171 [20851, 21491] | split | split |

#### Single products (ns per product)

| N | n | pairs | split | table | cached | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 32242 [31795, 32690] | 48389 [47612, 49166] | 16777 [16539, 17014] | split | split |
| 8 | 2 | 256 | 32006 [31537, 32475] | 47689 [47131, 48248] | 16374 [16153, 16595] | split | split |
| 16 | 3 | 256 | 31446 [31023, 31869] | 51081 [50254, 51908] | 16557 [16264, 16851] | split | split |
| 32 | 4 | 256 | 32330 [31902, 32758] | 55441 [53908, 56973] | 16910 [16661, 17159] | split | split |
| 64 | 5 | 256 | 32651 [32176, 33126] | 64377 [63304, 65450] | 17039 [16831, 17246] | split | split |
| 128 | 6 | 256 | 34371 [34008, 34734] | 122685 [120201, 125169] | 17976 [17697, 18255] | split | split |
| 256 | 7 | 64 | 36906 [36150, 37663] | 345996 [339820, 352171] | 19730 [19381, 20079] | split | split |
| 512 | 8 | 64 | 40470 [39228, 41713] | 1353612 [1328786, 1378439] | 21834 [21411, 22257] | split | split |
| 1024 | 9 | 16 | 47004 [46347, 47661] | 5090128 [5020133, 5160124] | 26667 [26273, 27061] | split | split |
| 2048 | 10 | 16 | 61128 [59865, 62390] | 34063829 [33838793, 34288866] | 36958 [36027, 37889] | split | split |

`split` is faster than `table` at every dimension in both modes, by a factor
that grows from 1.58 at dimension 4 to 1397 at
dimension 2048 in batch mode and from 1.5 to 558
in single mode. `"auto"` therefore selects `split` at every dimension.

A single call costs about 32 microseconds at small
dimensions, nearly all of it the fixed cost of a call through the Python
layers; a product of two elements that hold their split forms costs about
half that.

#### Comparison with the Cayley–Dickson product

| N | n | pyfastcd batch (ns) | splitform / pyfastcd | split / pyfastcd | pyfastcd single (ns) | cached / pyfastcd |
| ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 91.64 [90.23, 93.05] | 2.77 [2.7, 2.84] | 4.05 [3.92, 4.19] | 97.70 [96.31, 99.08] | 172 [170, 174] |
| 8 | 2 | 92.25 [90.83, 93.67] | 2.71 [2.64, 2.79] | 4.12 [4.01, 4.22] | 104.6 [103.4, 105.9] | 156 [155, 158] |
| 16 | 3 | 99.80 [98.67, 100.9] | 2.58 [2.52, 2.64] | 3.97 [3.88, 4.06] | 121.3 [119.7, 122.9] | 136 [134, 139] |
| 32 | 4 | 144.6 [142.4, 146.7] | 2.18 [2.12, 2.24] | 3.37 [3.3, 3.45] | 222.3 [218.7, 225.8] | 76.1 [75, 77.2] |
| 64 | 5 | 266.0 [262.8, 269.2] | 1.72 [1.69, 1.75] | 2.88 [2.81, 2.95] | 338.8 [335.0, 342.6] | 50.3 [49.9, 50.7] |
| 128 | 6 | 537.0 [531.3, 542.7] | 1.51 [1.48, 1.54] | 2.47 [2.42, 2.53] | 649.1 [639.5, 658.6] | 27.7 [27.1, 28.3] |
| 256 | 7 | 1447.0 [1430.5, 1463.4] | 1.57 [1.52, 1.61] | 2.5 [2.44, 2.56] | 1326.5 [1308.0, 1345.0] | 14.9 [14.6, 15.1] |
| 512 | 8 | 2962.5 [2924.1, 3001.0] | 1.38 [1.35, 1.41] | 2.18 [2.12, 2.24] | 2921.3 [2892.9, 2949.7] | 7.47 [7.35, 7.58] |
| 1024 | 9 | 7806.7 [7713.7, 7899.7] | 1.45 [1.43, 1.48] | 2.16 [2.11, 2.2] | 6536.5 [6458.6, 6614.3] | 4.08 [4.04, 4.12] |
| 2048 | 10 | 16229 [15968, 16489] | 1.3 [1.28, 1.33] | 1.92 [1.87, 1.98] | 14828 [14605, 15050] | 2.49 [2.44, 2.54] |

In batch mode a product in split form costs 2.77 times the
Cayley–Dickson product it contains at dimension 4 and 1.3 times at
dimension 2048, and a product in the loop basis costs 4.05 and
1.92 times. The conversions to and from split form account for
between 23% and 32% of the time of the conversions and
the product in split form together:

| N | n | forward (ns) | inverse (ns) | splitform (ns) | transform share |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 45.90 [45.42, 46.39] | 29.67 [29.20, 30.14] | 254.2 [246.7, 261.8] | 23.0% [22.5%, 23.4%] |
| 8 | 2 | 51.99 [51.31, 52.67] | 32.27 [31.81, 32.73] | 250.5 [244.9, 256.0] | 25.2% [24.8%, 25.6%] |
| 16 | 3 | 51.65 [51.03, 52.27] | 34.67 [34.17, 35.16] | 257.4 [252.7, 262.1] | 25.1% [24.8%, 25.4%] |
| 32 | 4 | 77.94 [76.88, 79.01] | 44.18 [42.74, 45.63] | 315.4 [306.7, 324.1] | 27.9% [27.4%, 28.5%] |
| 64 | 5 | 128.1 [126.7, 129.6] | 69.86 [68.65, 71.07] | 458.4 [452.0, 464.8] | 30.2% [29.9%, 30.5%] |
| 128 | 6 | 250.6 [246.2, 255.0] | 139.4 [136.6, 142.3] | 810.6 [799.6, 821.7] | 32.5% [32.2%, 32.8%] |
| 256 | 7 | 553.0 [544.9, 561.0] | 295.2 [289.1, 301.4] | 2271.4 [2218.8, 2324.1] | 27.2% [26.8%, 27.6%] |
| 512 | 8 | 1238.0 [1222.0, 1254.0] | 595.3 [581.9, 608.8] | 4091.6 [4028.3, 4154.9] | 31.0% [30.7%, 31.2%] |
| 1024 | 9 | 2668.6 [2622.2, 2715.1] | 1285.2 [1259.4, 1311.1] | 11360 [11149, 11572] | 25.8% [25.4%, 26.3%] |
| 2048 | 10 | 5407.6 [5341.3, 5473.8] | 2558.6 [2512.8, 2604.4] | 21171 [20851, 21491] | 27.3% [27.1%, 27.6%] |

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
| C | 1395.5 (1.15x) | 1779.1 (3.93x) | 7905.7 (38.7x) | 33437 (130x) |
| F_2 | 2233.9 (0.609x) | 3384.7 (1.15x) | 13835 (4.07x) | 53632 (22.9x) |
| F_(2^2) | 5369.7 (1.37x) | 12246 (4.8x) | 47085 (19.5x) | 180569 (100x) |
| F_(2^8) | 25485 (0.324x) | 66187 (1.37x) | 247156 (8.54x) | 984708 (33.2x) |
| F_3 | 2027.0 (1.48x) | 3229.2 (6.53x) | 12702 (25.3x) | 45754 (116x) |
| F_13 | 2036.8 (1.41x) | 3268.3 (6.07x) | 12750 (23.9x) | 48075 (105x) |
| F_65521 | 2108.5 (1.42x) | 3267.2 (5.92x) | 13247 (22.9x) | 51299 (96.4x) |
| F_2147483647 | 2070.6 (1.44x) | 3306.2 (5.88x) | 12866 (23.3x) | 54301 (92.5x) |
| F_(3^5) | 5957.5 (4.49x) | 12082 (24.6x) | 46953 (99.7x) | 184123 (425x) |
| F_(13^5) | 6069.6 (4.37x) | 16176 (18.6x) | 49785 (94.1x) | 204321 (393x) |

#### Single products: split or lift ns per product (ratio of table to it)

| Field | N = 8 | N = 32 | N = 128 | N = 512 |
| --- | ---: | ---: | ---: | ---: |
| C | 74214 (0.675x) | 74384 (0.773x) | 82581 (1.63x) | 91573 (18.3x) |
| F_2 | 111580 (0.442x) | 111096 (0.468x) | 131420 (0.775x) | 148476 (6.51x) |
| F_(2^2) | 211423 (0.759x) | 225367 (1.01x) | 265464 (4.13x) | 358301 (40.6x) |
| F_(2^8) | 585895 (0.276x) | 714673 (0.354x) | 861348 (1.68x) | 1486428 (14.4x) |
| F_3 | 102589 (0.745x) | 101013 (0.985x) | 114818 (3.51x) | 125734 (40.8x) |
| F_13 | 100379 (0.763x) | 105772 (0.925x) | 117752 (3.28x) | 130724 (37.8x) |
| F_65521 | 104916 (0.746x) | 104286 (0.951x) | 121065 (3.2x) | 136539 (35.3x) |
| F_2147483647 | 103908 (0.742x) | 103146 (0.962x) | 117225 (3.27x) | 142423 (34.5x) |
| F_(3^5) | 181690 (2.44x) | 195769 (3.81x) | 229140 (22.9x) | 332660 (219x) |
| F_(13^5) | 183263 (2.37x) | 210761 (3.62x) | 233747 (22.3x) | 357472 (207x) |

**C and odd characteristic.** In batch mode `split` is faster in every
cell. In single mode `table` is faster at dimension 8 over C and over the
prime fields, by factors between 1.3 and 1.5.
At dimension 32 it is faster by a factor of 1.3 over C, and the
two are within 8% of each other over the prime fields. A
single `split` call over these fields costs 74 to 105 microseconds at dimension 8, much of
it in the array interface of pyfastcd for these fields, which the `table`
algorithm does not use. `"auto"` still selects `split` there, so that the
result of a product does not depend on how many records its operands hold;
`algorithm="table"` selects the other explicitly.

**Characteristic 2.** Over F_2, F_4 and F_256 neither algorithm is faster
throughout: `table` is faster in small loops and `lift` in large ones. The
next section looks at these fields more closely.

Intervals for every cell are in the archived report.

### Characteristic 2

Each cell gives the `lift` mean in nanoseconds per product and, in
parentheses, the paired ratio of `table` to it, so a ratio above 1 favours
`lift`. These are the results of the second `domains` run described above.

#### Batch products: lift ns per product (table / lift)

| Field | N = 4 | N = 8 | N = 16 | N = 32 | N = 64 | N = 128 | N = 256 | N = 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_2 | 2028.3 (0.56x) | 2227.4 (0.615x) | 2615.3 (0.723x) | 3364.5 (1.15x) | 10069 (1.42x) | 13798 (4.04x) | 43132 (7.29x) | 53019 (35.9x) |
| F_(2^2) | 4264.0 (1.03x) | 5425.7 (1.36x) | 8158.3 (2.21x) | 12472 (4.91x) | 30641 (7.69x) | 45152 (26.5x) | 122281 (40.4x) | 184535 (100x) |
| F_(2^4) | 5433.5 (0.815x) | 8458.5 (0.926x) | 17240 (1.12x) | 21269 (3.24x) | 48754 (5.49x) | 78900 (16x) | 188191 (27.6x) | 310747 (75.9x) |
| F_(2^6) | 7225.0 (0.62x) | 13478 (0.593x) | 31922 (0.647x) | 37678 (2.08x) | 75226 (4.1x) | 124477 (9.4x) | 283533 (16.6x) | 491208 (59.3x) |
| F_(2^7) | 16223 (0.289x) | 25588 (0.325x) | 51747 (0.423x) | 67660 (1.23x) | 147805 (2.26x) | 242905 (5.28x) | 583749 (8.78x) | 964637 (31.5x) |
| F_(2^8) | 16410 (0.282x) | 25895 (0.328x) | 52596 (0.461x) | 65424 (1.36x) | 150288 (2.33x) | 243003 (5.72x) | 580908 (9.61x) | 987833 (33.3x) |
| F_(2^12) | 22594 (0.23x) | 41399 (0.236x) | 98750 (0.305x) | 118535 (0.953x) | 235148 (1.86x) | 388499 (4.68x) | 888165 (8.15x) | 1555595 (28.4x) |

#### Single products: lift ns per product (table / lift)

| Field | N = 4 | N = 8 | N = 16 | N = 32 | N = 64 | N = 128 | N = 256 | N = 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_2 | 112516 (0.435x) | 111153 (0.441x) | 113185 (0.443x) | 116042 (0.449x) | 116599 (0.524x) | 126926 (0.791x) | 133703 (2.12x) | 144726 (6.73x) |
| F_(2^2) | 202522 (0.761x) | 214247 (0.745x) | 217369 (0.809x) | 232017 (1.01x) | 237127 (1.7x) | 251334 (4.17x) | 295208 (12.9x) | 360343 (40.2x) |
| F_(2^4) | 206658 (0.759x) | 208312 (0.749x) | 219218 (0.799x) | 237292 (0.973x) | 256167 (1.67x) | 283816 (4.12x) | 365078 (11.6x) | 484528 (34.1x) |
| F_(2^6) | 204962 (0.755x) | 213885 (0.747x) | 237227 (0.767x) | 256217 (0.935x) | 287059 (1.59x) | 346634 (3.65x) | 456227 (10.6x) | 689907 (27.3x) |
| F_(2^7) | 589494 (0.267x) | 601365 (0.266x) | 643479 (0.283x) | 685915 (0.349x) | 756602 (0.611x) | 866798 (1.52x) | 1110353 (4.76x) | 1530966 (13x) |
| F_(2^8) | 597740 (0.265x) | 610656 (0.272x) | 613987 (0.298x) | 677861 (0.355x) | 749043 (0.63x) | 851755 (1.69x) | 1083770 (5.19x) | 1529578 (15.3x) |
| F_(2^12) | 591940 (0.28x) | 642405 (0.259x) | 704372 (0.272x) | 791376 (0.34x) | 870764 (0.638x) | 1071718 (1.66x) | 1394047 (5.01x) | 2194207 (15.2x) |

`table` is faster in small loops and `lift` in large ones, where its
advantage reaches a factor of 100 in batch mode and
40.2 for single products at dimension 512. Where the two
cross depends on the degree of the field, which sets the cost of a lifted
product, and on the number of products in a call. A single product is
faster by `lift` from dimension 256 over F_2, from dimension 64 for the
degrees 2 through 6, and from dimension 128 for the degrees 7 through 12;
in batches the crossing comes at smaller dimensions.

`"auto"` follows the rule of
[Choosing between the algorithms](mathematics.md#choosing-between-the-algorithms),
which these measurements test: in 108 of the 112 cells
it selects the algorithm with the lower mean, or one tied with it, and in
the others the algorithm it selects is slower by a factor of at most
1.55.

### Central products

#### Signed-component methods (ns per product)

| n x m | D | pairs | native | rows | schedule | fastest mean | auto |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| 1 x 4 | 32 | 64 | 2627.1 [2564.1, 2690.1] | 3446.2 [3365.9, 3526.5] | 3751.7 [3689.8, 3813.7] | native | native |
| 1 x 8 | 512 | 4 | 45611 [44748, 46473] | 55599 [53756, 57442] | 68465 [66854, 70076] | native | native |
| 2 x 2 | 16 | 64 | unavailable | 4247.9 [4134.6, 4361.1] | 3952.6 [3843.0, 4062.1] | schedule | rows |
| 2 x 4 | 64 | 64 | unavailable | 6628.5 [6514.6, 6742.4] | 5622.7 [5502.0, 5743.5] | schedule | rows |
| 2 x 6 | 256 | 16 | unavailable | 23638 [23239, 24037] | 25751 [24950, 26551] | rows | rows |
| 3 x 3 | 64 | 64 | unavailable | 11916 [11539, 12292] | 10732 [10477, 10986] | schedule | rows |
| 3 x 5 | 256 | 16 | unavailable | 37192 [36686, 37699] | 47363 [46078, 48648] | rows | rows |
| 3 x 7 | 1024 | 4 | unavailable | 151005 [148979, 153031] | 209624 [205946, 213302] | rows | rows |
| 4 x 4 | 256 | 16 | unavailable | 74768 [73102, 76434] | 70032 [68675, 71390] | schedule | rows |
| 4 x 6 | 1024 | 4 | unavailable | 252377 [248175, 256579] | 325871 [316426, 335315] | rows | rows |
| 4 x 8 | 4096 | 4 | unavailable | 947317 [936808, 957827] | 824191 [813887, 834495] | schedule | rows |
| 5 x 5 | 1024 | 4 | unavailable | 455011 [447621, 462401] | 421102 [408851, 433353] | schedule | schedule |
| 5 x 7 | 4096 | 4 | unavailable | 1698702 [1673587, 1723817] | 1054109 [1033957, 1074261] | schedule | schedule |
| 6 x 6 | 4096 | 4 | unavailable | 4142602 [4071389, 4213814] | 1243742 [1223296, 1264189] | schedule | schedule |
| 7 x 7 | 16384 | 2 | unavailable | 36603306 [36378033, 36828578] | 6061988 [5992071, 6131906] | schedule | schedule |

#### Paired ratios of the signed-component methods

| n x m | outer level | schedule / rows | native / rows |
| --- | ---: | --- | --- |
| 1 x 4 | 1 | 1.09 [1.06, 1.12] | 0.762 [0.742, 0.783] |
| 1 x 8 | 1 | 1.23 [1.2, 1.26] | 0.822 [0.799, 0.845] |
| 2 x 2 | 2 | 0.93 [0.912, 0.949] | unavailable |
| 2 x 4 | 2 | 0.848 [0.83, 0.867] | unavailable |
| 2 x 6 | 2 | 1.09 [1.06, 1.12] | unavailable |
| 3 x 3 | 3 | 0.902 [0.866, 0.939] | unavailable |
| 3 x 5 | 3 | 1.27 [1.24, 1.3] | unavailable |
| 3 x 7 | 3 | 1.39 [1.37, 1.4] | unavailable |
| 4 x 4 | 4 | 0.937 [0.916, 0.958] | unavailable |
| 4 x 6 | 4 | 1.29 [1.25, 1.33] | unavailable |
| 4 x 8 | 4 | 0.87 [0.861, 0.879] | unavailable |
| 5 x 5 | 5 | 0.924 [0.904, 0.945] | unavailable |
| 5 x 7 | 5 | 0.62 [0.61, 0.631] | unavailable |
| 6 x 6 | 6 | 0.3 [0.295, 0.305] | unavailable |
| 7 x 7 | 7 | 0.166 [0.163, 0.168] | unavailable |

`native` is the fastest method wherever it applies. Through outer level 4,
`rows` and `schedule` are within a factor of 1.4 of each other, and
which is faster depends on the inner level. `schedule` is faster in every
measured pair from outer level 5, by a factor that reaches 6.0 at
`7 x 7`. `"auto"` therefore selects `native` where it applies, and otherwise
`rows` through outer level 4 and `schedule` above.

#### Full product (ns per product)

| n x m | N | pairs | split | table | table / split | fastest mean |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 x 4 | 64 | 64 | 3050.9 [2991.1, 3110.6] | 31943 [31232, 32654] | 10.5 [10.3, 10.6] | split |
| 1 x 8 | 1024 | 4 | 54213 [52659, 55768] | 10393686 [10304391, 10482980] | 192 [186, 198] | split |
| 2 x 2 | 32 | 64 | 4636.0 [4538.9, 4733.1] | 5356.7 [5229.2, 5484.1] | 1.16 [1.13, 1.18] | split |
| 2 x 4 | 128 | 64 | 7384.8 [7235.9, 7533.7] | 114016 [111742, 116289] | 15.4 [15.2, 15.7] | split |
| 2 x 6 | 512 | 16 | 27066 [26570, 27562] | 2803279 [2783404, 2823154] | 104 [102, 106] | split |
| 3 x 3 | 128 | 64 | 12446 [12138, 12754] | 110800 [108600, 112999] | 8.91 [8.72, 9.1] | split |
| 3 x 5 | 512 | 16 | 40153 [39514, 40791] | 2789619 [2772168, 2807071] | 69.5 [68.5, 70.5] | split |
| 3 x 7 | 2048 | 4 | 165549 [162754, 168344] | 43787294 [43638972, 43935615] | 265 [260, 269] | split |
| 4 x 4 | 512 | 16 | 76620 [75269, 77971] | 2725514 [2703773, 2747256] | 35.6 [35, 36.3] | split |
| 4 x 6 | 2048 | 4 | 267284 [262777, 271791] | 43580408 [43332554, 43828263] | 163 [160, 166] | split |
| 5 x 5 | 2048 | 4 | 425297 [415491, 435102] | 43230345 [42982919, 43477771] | 102 [99.6, 104] | split |

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
