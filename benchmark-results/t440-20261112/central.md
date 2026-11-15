# pycdloop multiplication benchmark: central products over R

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.4.0, pycdloop 0.1.0. Pinned to logical CPU 37; load average at start 3.02.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 1376 verified products agree with an independent evaluation of the definition.

A cell is labelled by the factor levels `n x m` of `Q_n * Q_m`. `D` is the dimension `2^(n+m)` of the tensor product of the two Cayley–Dickson algebras, and the loop algebra has dimension `N = 2D`.

## Signed-component methods (ns per product)

Products of two split arrays, which need no transform. `fastest mean` names the lowest mean, or a tie when the interval of the paired ratio of the two lowest means contains one. `auto` is the method that `"auto"` selects for the pair.

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

## Paired ratios of the signed-component methods

`schedule / rows` below 1 favours `schedule`; `native / rows` below 1 favours `native`.

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

## Full product (ns per product)

`split` on loop-basis arrays, with the method that `"auto"` selects, against `table`. `table / split` above 1 favours `split`.

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
