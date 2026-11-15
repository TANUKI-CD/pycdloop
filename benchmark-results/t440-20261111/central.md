# pycdloop multiplication benchmark: central products over R

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.3.1, pycdloop 0.1.0. Pinned to logical CPU 37; load average at start 3.12.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 1376 verified products agree with an independent evaluation of the definition.

A cell is labelled by the factor levels `n x m` of `Q_n * Q_m`. `D` is the dimension `2^(n+m)` of the tensor product of the two Cayley–Dickson algebras, and the loop algebra has dimension `N = 2D`.

## Signed-component methods (ns per product)

Products of two split arrays, which need no transform. `fastest mean` names the lowest mean, or a tie when the interval of the paired ratio of the two lowest means contains one. `auto` is the method that `"auto"` selects for the pair.

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

## Paired ratios of the signed-component methods

`schedule / rows` below 1 favours `schedule`; `native / rows` below 1 favours `native`.

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

## Full product (ns per product)

`split` on loop-basis arrays, with the method that `"auto"` selects, against `table`. `table / split` above 1 favours `split`.

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
