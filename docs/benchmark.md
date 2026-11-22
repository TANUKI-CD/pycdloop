# Benchmarks

pycdloop multiplies in a loop algebra with three algorithms, and it chooses
among them automatically. The split algorithm multiplies through the
decomposition of the algebra, the table algorithm evaluates the definition of
the product, and the lift algorithm takes the place of the split algorithm in
characteristic 2 (see [Multiplication
algorithms](mathematics.md#multiplication-algorithms)). For a loop algebra of
dimension `2D`, where `D` is the quotient order of the loop, their operation
counts differ widely, from `O(D log D)` for the split algorithm with a
quasilinear Cayley-Dickson product to `4 D^2` multiplications for the table
algorithm, but a smaller operation count does not by itself make an
implementation faster at a given size: pycdloop is a Python package, so every
call also pays a fixed cost in the Python layers, and its transforms, its
Cayley-Dickson products, and the NumPy operations that connect them each have
their own overheads. The only way to know which algorithm is faster at a given
size is to time the implementations themselves.

The benchmark does this in order to justify the choices of `"auto"` with
reproducible data: between the split and the table algorithm, between the lift
and the table algorithm in characteristic 2, and among the three methods for the
signed components of central products. It does not try to prove that one
algorithm is faster than another in general; instead, it measures pycdloop's
implementations as the package ships them, on one host, and asks which of them
the package should call.

The measurements show that the split algorithm is faster than the table
algorithm at every dimension tested over R, C, and the finite fields of odd
characteristic, by a factor that reaches 1555 at dimension 2048 over R, that in
characteristic 2 the lift algorithm is faster in large loops while the table
algorithm can win in small ones, depending on the degree of the field and the
number of products in a call, and that for central products the fastest method
for the signed components depends on the level of the outer factor. The sections
below describe what is measured and how the timings are made, report the results
for real coefficients, for complex coefficients and finite fields, for
characteristic 2, and for central products, show how these results set the
automatic choices, and end with the limitations of the measurements and with
instructions for running the benchmark.

## What is measured

The benchmark measures the algorithms and methods among which `"auto"` chooses,
and not all of them are available for every field and loop: the split and table
algorithms multiply over R, C, and the finite fields of odd characteristic, the
lift and table algorithms over the fields F_(2^k), and for a central product the
split and lift algorithms multiply the signed components by one of the methods
`native`, `rows`, and `schedule` (see [The product of signed
components](mathematics.md#the-product-of-signed-components)).

The benchmark times calls to the public interface with `time.perf_counter_ns`,
with the garbage collector disabled inside each timed region. pycdloop's
transform and change of coordinates are compiled, its Cayley-Dickson products
are pyfastcd's, and everything that connects them is Python and NumPy, so the
call overhead of the package, the allocation of results, and the validation of
operands are part of every measurement, as they are what a caller pays.

Products are timed in two *modes*, which can rank the algorithms differently.
*Batch mode* makes one call on arrays that hold the whole corpus, while *single
mode* makes one call for each pair, on arrays of one record, and so includes the
fixed cost of a call, which dominates at small dimensions.

The drivers in `benchmarks/` run three experiments, called *protocols*, each of
which answers one question:

- **Real coefficients** (`real`): how much faster is the split algorithm than
  the table algorithm over R, at each dimension from 4 through 2048, and how
  much of its time goes to the transforms and to the Cayley-Dickson product?
- **Complex coefficients and finite fields** (`domains`): does the split
  algorithm keep its advantage over C and the finite fields of odd
  characteristic, and where does the lift algorithm overtake the table algorithm
  in characteristic 2?
- **Central products** (`central`): which method for the signed components of a
  central product is fastest, and how much faster is the full product than the
  table algorithm?

## How timings are made

### Corpora and verification

Every protocol times a fixed *corpus*, a deterministic set of operand pairs for
each level and, where the protocol has them, for each field or pair of factors.
The driver generates the corpus with NumPy from a recorded seed, stores it for
reuse, and binds it to a manifest of its parameters and its hash, so that a run
never times inputs other than the ones it reports. Larger levels use fewer
pairs, because one product by the table algorithm at dimension 2048 takes tens
of milliseconds, and the schedule of pairs is recorded with the run.

Before any timing, every loop-algebra product of every task is compared with an
independent evaluation of the definition, written in the driver, which derives
the basis signs from the doubling formula and sums the products of coefficients
directly, without pycdloop arithmetic. Over R and C a coordinate may differ from
this reference by at most

```text
128 * DBL_EPSILON * D * ||x||_2 * ||y||_2,
```

a bound on the disagreement between two floating-point evaluations rather than
on the error against the exact product, and over finite fields the comparison is
exact, against schoolbook polynomial arithmetic. A failed check is reported as a
failure, never as a timing.

### Observations, blocks, and tasks

An *observation* times complete passes over the corpus: after two untimed
passes, the driver calibrates the number of passes so that the observation lasts
at least 20 ms, and divides the elapsed time by the number of products. A *task*
is one thing being timed, such as one algorithm in one mode.

A run is divided into 24 *blocks*, and each block takes one observation of every
task. The order of the tasks rotates from block to block, so that a slow change
in the state of the host affects all tasks alike, and every ratio between two
tasks is formed within a block.

### Statistics

A time is the mean over the blocks, in nanoseconds per product, with a two-sided
95% Student-t interval whose degrees of freedom are capped at 30. A ratio is the
geometric mean of the per-block ratios, with its own 95% interval, and each
table below states which side a ratio above 1 favors. A "fastest mean" column
names the algorithm with the lowest mean, or a tie when the interval of the
paired ratio between the two lowest means contains 1.

An interval describes only the variation between the blocks of one run in one
process. It does not cover differences between processes, between runs, or
between hosts, and on a shared host those can exceed it.

## The archived run

All results on this page come from the four runs of pycdloop 0.1.0 archived in
[benchmark-results/t440-20261115/](../benchmark-results/t440-20261115/), one for
each protocol and a second `domains` run over the fields of characteristic 2.
They were measured on 2026-11-15 on a Dell PowerEdge T440 with an Intel Xeon
Gold 6148, pinned to one logical CPU, with Python 3.14.8, NumPy 2.5.4, and
pyfastcd 1.4.0 on Linux 7.2.11. The host is shared with other work, and its
frequency governor, turbo, and SMT settings were left unchanged, so the archive
records the load average around each run, which stayed between 1.6 and 2.3.
Every verified product agreed with the independent reference, 25,760 products in
all, and every JSON file of the archive records the hash of every source file of
the measured package.

## Real coefficients

The `real` protocol times the loop algebra of `Q_n` over R at levels 1 through
10, that is, at dimensions `2D` from 4 through 2048. Besides the two algorithms,
it times the pieces that a product by the split algorithm consists of, so that
the cost of each can be read off:

| Task | What one product is |
| --- | --- |
| `split/batch`, `split/single` | The split algorithm on `ElementArray` operands: both operands are converted to split form, multiplied there, and the result converted back |
| `table/batch`, `table/single` | The table algorithm on the same operands |
| `splitform/batch` | The product of two `SplitArray` operands, with no conversion |
| `cached/single` | The product of two `Element` operands that already hold their split forms, with no conversion of the result |
| `forward/batch`, `inverse/batch` | The conversions alone: both operands to split form, and one result back |
| `pyfastcd/batch`, `pyfastcd/single` | The Cayley-Dickson product of the signed components alone, through pyfastcd, on its arrays and on its elements |

The tables give the mean time per product with its 95% interval, the fastest
mean as defined in [Statistics](#statistics), and the algorithm that `"auto"`
selects.

**Batch products** (ns per product)

| 2D | n | pairs | split | table | splitform | fastest mean | default |
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

**Single products** (ns per product)

| 2D | n | pairs | split | table | cached | fastest mean | default |
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

**Split against table.** The split algorithm is faster than the table algorithm
at every dimension in both modes, by a factor that grows from 3.12 at dimension
4 to 1555 at dimension 2048 in batch mode and from 1.51 to 575 in single mode.

**The cost of a call.** A single call costs about 31 microseconds at small
dimensions, nearly all of it the fixed cost of a call through the Python layers,
and a product of two elements that already hold their split forms costs about
half of that, because it skips the conversions.

The next two tables take a product by the split algorithm apart. The first
compares it with the Cayley-Dickson product of dimension `D` that it contains,
where a ratio divides the first time by the second, and the second measures the
conversions to and from split form, where the *transform share* is the fraction
of the time of the conversions and the product in split form together that the
conversions take.

| 2D | n | pyfastcd batch (ns) | splitform / pyfastcd | split / pyfastcd | pyfastcd single (ns) | cached / pyfastcd |
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

| 2D | n | forward (ns) | inverse (ns) | splitform (ns) | transform share |
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

**Where the time goes.** In batch mode a product in split form costs 8.7 times
the Cayley-Dickson product it contains at dimension 4 and 1.22 times at
dimension 2048, and a product in the loop basis costs 19.9 and 1.94 times, while
the transform share falls from 50% at dimension 8 to 31% at dimension 2048. In
single mode the comparison is with pyfastcd's element type, which is implemented
in C, so the fixed cost of a pycdloop call dominates until the arithmetic
outgrows it.

## Complex coefficients and finite fields

The split algorithm's advantage over R does not by itself carry over to other
fields, because the cost of each coefficient operation, and in characteristic 2
the cost of lifting, differs from field to field. The `domains` protocol
therefore times both algorithms in both modes for `Q_n` over C and over the
finite fields F_3, F_13, F_65521, F_2147483647, F_(3^5), and F_(13^5), and the
lift algorithm in place of the split algorithm over F_2, F_4, and F_256, at
levels 2, 4, 6, and 8.

Each cell of the following tables gives the mean time per product of the split
algorithm, or of the lift algorithm in the rows of characteristic 2, and in
parentheses the paired ratio of the table algorithm's time to it, so that a
ratio above 1 favors the split or the lift algorithm, and the archived report
gives the intervals of every cell:

**Batch products:** split or lift ns per product (table / split or lift)

| Field | 2D = 8 | 2D = 32 | 2D = 128 | 2D = 512 |
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

**Single products:** split or lift ns per product (table / split or lift)

| Field | 2D = 8 | 2D = 32 | 2D = 128 | 2D = 512 |
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

**C and odd characteristic.** The split algorithm is faster in every cell, in
both modes, and for single products its advantage at dimension 8 is a factor of
1.3 over C and about 1.6 over the prime fields. A single call over C and the
prime fields costs 38 to 48 microseconds at dimension 8, nearly all of it the
fixed cost of a call, since in a batch the same product takes 0.7 to 1.1
microseconds.

**Characteristic 2.** Over F_4 the lift algorithm is faster in every cell, while
over F_2 and F_256 it is faster in the large loops and the table algorithm in
some of the small ones, which the next section examines more closely.

## Characteristic 2

Because the outcome in characteristic 2 depends on the field and on the size of
the loop, a second run of the `domains` protocol times the lift and table
algorithms over F_(2^k) alone, for `k` = 1, 2, 4, 6, 7, 8, and 12, at every
level from 1 to 8, with batches of 64 products at levels 1 through 4, 16 at
levels 5 and 6, and 4 at levels 7 and 8. Each cell gives the mean time per
product of the lift algorithm and, in parentheses, the paired ratio of the table
algorithm's time to it, so that a ratio above 1 favors the lift algorithm:

**Batch products:** lift ns per product (table / lift)

| Field | 2D = 4 | 2D = 8 | 2D = 16 | 2D = 32 | 2D = 64 | 2D = 128 | 2D = 256 | 2D = 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_2 | 1091.3 (1.05x) | 1225.7 (1.07x) | 1520.4 (1.22x) | 2333.5 (1.66x) | 6164.4 (2.32x) | 9158.6 (6.11x) | 26288 (12x) | 37617 (51.3x) |
| F_(2^2) | 2189.6 (2.01x) | 2946.1 (2.49x) | 4957.2 (3.44x) | 8131.4 (7.26x) | 18830 (12.4x) | 29468 (40.2x) | 73913 (65.4x) | 125894 (149x) |
| F_(2^4) | 3194.9 (1.39x) | 5868.7 (1.3x) | 13287 (1.42x) | 18227 (3.87x) | 35656 (7.71x) | 64712 (20x) | 142380 (37.2x) | 276983 (82.5x) |
| F_(2^6) | 4880.0 (0.912x) | 10714 (0.749x) | 28026 (0.752x) | 33394 (2.38x) | 64827 (4.79x) | 109962 (11.1x) | 237365 (20.3x) | 420703 (67.8x) |
| F_(2^7) | 9757.9 (0.471x) | 18405 (0.465x) | 41478 (0.529x) | 53735 (1.53x) | 108564 (3.03x) | 196412 (6.55x) | 429496 (12.1x) | 777004 (38.9x) |
| F_(2^8) | 9711.6 (0.47x) | 18026 (0.457x) | 41088 (0.577x) | 55812 (1.62x) | 117131 (2.98x) | 200685 (7.05x) | 438306 (12.6x) | 792053 (41.6x) |
| F_(2^12) | 15006 (0.329x) | 32497 (0.294x) | 84369 (0.35x) | 101312 (1.08x) | 189765 (2.29x) | 342710 (5.18x) | 740484 (9.65x) | 1328956 (33.6x) |

**Single products:** lift ns per product (table / lift)

| Field | 2D = 4 | 2D = 8 | 2D = 16 | 2D = 32 | 2D = 64 | 2D = 128 | 2D = 256 | 2D = 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_2 | 54364 (0.883x) | 52658 (0.897x) | 53214 (0.916x) | 55778 (0.912x) | 58727 (1.02x) | 65537 (1.56x) | 72584 (3.93x) | 89657 (11.1x) |
| F_(2^2) | 94923 (1.64x) | 93229 (1.64x) | 94524 (1.74x) | 110694 (2x) | 114996 (3.45x) | 129938 (8.22x) | 157024 (23.8x) | 206892 (71.6x) |
| F_(2^4) | 93306 (1.6x) | 95085 (1.61x) | 103772 (1.62x) | 123183 (1.88x) | 133877 (3.09x) | 162192 (7.26x) | 223693 (19.8x) | 350117 (51.3x) |
| F_(2^6) | 96027 (1.58x) | 101718 (1.54x) | 121247 (1.42x) | 139777 (1.68x) | 160668 (2.81x) | 209405 (6.09x) | 314243 (15.5x) | 510199 (36.3x) |
| F_(2^7) | 265126 (0.581x) | 274321 (0.587x) | 295446 (0.598x) | 338815 (0.701x) | 383760 (1.18x) | 480005 (2.75x) | 687634 (7.76x) | 1013181 (19.7x) |
| F_(2^8) | 256464 (0.584x) | 263443 (0.581x) | 293796 (0.609x) | 349707 (0.684x) | 393091 (1.19x) | 500050 (2.93x) | 660016 (8.14x) | 1013752 (22.6x) |
| F_(2^12) | 267320 (0.562x) | 283639 (0.553x) | 339917 (0.535x) | 381842 (0.676x) | 463210 (1.14x) | 669123 (2.7x) | 985059 (7.1x) | 1562769 (20.4x) |

**Large loops.** The lift algorithm is the faster algorithm in large loops,
where its advantage reaches a factor of 149 in batch mode and 71.6 for single
products at dimension 512.

**Small loops.** In small loops the outcome depends on the degree of the field,
which sets the cost of a lifted product, and on the number of products in a
call. For single products the lift algorithm has the lower mean at every
dimension over F_4, F_16, and F_64, and from dimension 64 over the other fields,
while in batch mode it has the lower mean at every dimension over F_2, F_4, and
F_16, and from dimension 32 over the other fields.

## Central products

A central product adds a choice the other loops do not have, the method for its
signed components, so the `central` protocol times central products
`Q_(n_1) * Q_(n_2)` over R in two parts. The first compares the methods
`native`, `rows`, and `schedule` on `SplitArray` operands, where no transform is
involved, so that the tasks differ only in the method, and the second times the
full product, the split algorithm with the method that `"auto"` selects against
the table algorithm, on `ElementArray` operands, for the pairs where the table
algorithm is affordable. In the tables, `n_1 x n_2` gives the levels of the
factors, with the outer factor first, and `D` is the quotient order.

**Signed-component methods** (ns per product)

| n_1 x n_2 | D | pairs | native | rows | schedule | fastest mean | auto |
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

**Paired ratios of the signed-component methods,** where a ratio below 1 favors
the method named first:

| n_1 x n_2 | outer level | schedule / rows | native / rows |
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

**Which method wins.** `native` is the fastest method wherever it applies.
Through outer level 4, `rows` and `schedule` are within a factor of 1.3 of each
other, and which is faster depends on the inner level, while from outer level 5
`schedule` is faster in every measured pair, by a factor that reaches 6.1 at
`7 x 7`.

**Full product** (ns per product)

| n_1 x n_2 | 2D | pairs | split | table | table / split | fastest mean |
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

**The full product.** The split algorithm is faster than the table algorithm for
every measured central product, by a factor between 1.88 at `2 x 2`, the
smallest pair, and 293 at `3 x 7`.

## How the results set the defaults

Each of the three choices that `"auto"` makes follows a rule drawn from the
results above. Over R, C, and the finite fields of odd characteristic, it
selects the split algorithm for every loop and every number of products, because
it was the faster algorithm in every measured cell.

For a central product, `"auto"` selects the method for the signed components by
the level of the outer factor: `native` where it applies, and otherwise `rows`
through outer level 4 and `schedule` above. This rule matches the fastest mean
in every measured pair except `2 x 4` and `4 x 8`, where `schedule` was 6% and
15% faster than the `rows` that the rule selects.

In characteristic 2, `"auto"` chooses between the lift and the table algorithm
from the quotient order `D` and the number `R` of products in the call. It
selects the lift algorithm when

```text
D >= D_0     and     R D^2 >= W,
```

and the table algorithm otherwise, where the thresholds `D_0` and `W` depend on
how the field is lifted:

| Field | Lifted to | `D_0` | `W` | One product is lifted from | A large batch is lifted from |
| --- | --- | ---: | ---: | --- | --- |
| F_2 | F_p | 2 | `2^8` | `D = 16` | `D = 2` |
| F_(2^k), `2 <= k <= 6` | one extension of F_p, one product | 2 | 0 | `D = 2` | `D = 2` |
| F_(2^k), `7 <= k <= 12` | one extension of F_p, three products | 16 | `2^10` | `D = 32` | `D = 16` |

For the degrees 2 through 6 the second condition always holds, so every product
is lifted when `D >= 2`, and the algebra properties `default_algorithm` and
`default_batch_algorithm` report the choice for one product and for a large
batch. Measured against the second `domains` run, the rule selects the algorithm
with the lower mean, or one tied with it, in 108 of the 112 cells, and in the
others the algorithm it selects is slower by a factor of at most 1.33. The
largest differences are for batches over F_64 in loops of dimension at most 16,
where the rule selects the lift algorithm.

These rules reflect one run on one host, so they should be reviewed against the
results of any new run.

## Limitations

- All results come from one shared host, with one version of Python, NumPy, and
  pyfastcd, and timings made through the Python interface are sensitive to all
  three, especially at small dimensions, where the fixed cost of a call
  dominates.
- The intervals cover only the variation within one run in one process.
- The batches in characteristic 2 hold at most 64 products, and the rule of
  `"auto"` there was tested only on the fields and levels of the second
  `domains` run.
- Central products were measured over R only, with two factors.

## Running the benchmark

### All protocols

To install the package with its development extras and run all three protocols
one after another:

```sh
python -m pip install -e ".[dev]"
python benchmarks/protocols.py --protocol all --output results/run.json --cpu 2
```

The three protocols write `run.real.json`, `run.domains.json`, and
`run.central.json` and take about ten minutes together on the archive's host,
while the second `domains` run over the fields of characteristic 2 takes about
six minutes more. The README of the archive records the exact commands of the
archived run, whose `central` protocol was given a longer list of factor pairs
with `--factors`.

Each JSON file holds the raw observations, the corpus hashes, the hash of every
source file of the package, the versions of Python, NumPy, pyfastcd, and
pycdloop, and the state of the CPU, and a Markdown report of its results is
written beside it, while the corpus is written to `benchmark-corpora/` on first
use.

### Options of the drivers

Each protocol can also be run by its own driver, and the three drivers share
these options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--output` | required | JSON result path, with the Markdown report written beside it |
| `--pairs` | the protocol's own | operand pairs per level, as `LEVELS:COUNT` items or one count |
| `--blocks` | `24` | timing blocks, each of which takes one observation of every task |
| `--warmup` | `2` | untimed passes over the corpus before each observation |
| `--min-time-ms` | `20` | minimum duration of one observation |
| `--seed` | `20261111` | master seed of the corpus |
| `--corpus-dir` | `benchmark-corpora` | directory of the corpus files and their manifest |
| `--cpu` | not pinned | logical CPU to which the process is pinned, on Linux |

### The real protocol

The `real` protocol is run by:

```sh
python benchmarks/benchmark.py --output results/real.json --cpu 2
```

`--levels` selects the doubling levels, as integers and ranges, by default
`1-10`, with 256 pairs per level through level 6, 64 at levels 7 and 8, and 16
at levels 9 and 10, and `--tasks` times a comma-separated subset of the tasks.

### The domains protocol

The `domains` protocol is run by:

```sh
python benchmarks/domains.py --output results/domains.json --cpu 2
```

`--levels` defaults to `2,4,6,8`, with 64 pairs per level through level 4, 16 at
levels 5 and 6, and 4 at levels 7 and 8, and `--fields` takes the fields as
`char:deg` pairs, where C is `0:2`, by default the ten fields of the archived
run. In the reports the columns named `split` hold the timings of the lift
algorithm for the fields of characteristic 2. The second run, over F_(2^k)
alone, is:

```sh
python benchmarks/domains.py --output results/binary.json --cpu 2 \
    --fields 2:1,2:2,2:4,2:6,2:7,2:8,2:12 --levels 1,2,3,4,5,6,7,8
```

### The central protocol

The `central` protocol is run by:

```sh
python benchmarks/central.py --output results/central.json --cpu 2
```

`--factors` takes the pairs of factor levels as `n1xn2` items, by default eleven
pairs, to which the archived run added `2x4`, `3x5`, `4x6`, and `5x7`, and
`--max-table-level`, by default 10, bounds the total level of the pairs whose
full product is timed against the table algorithm.

### Testing the drivers

`benchmarks/test_benchmark.py` tests the drivers themselves: their argument
validation, the statistics, the independent reference, and a short run of every
protocol.
