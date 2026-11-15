# pycdloop multiplication benchmark: central products over R

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.4.0, pycdloop 0.1.0. Pinned to logical CPU 36; load average at start 2.89.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 1376 verified products agree with an independent evaluation of the definition.

A cell is labelled by the factor levels `n x m` of `Q_n * Q_m`. `D` is the dimension `2^(n+m)` of the tensor product of the two Cayley–Dickson algebras, and the loop algebra has dimension `N = 2D`.

## Signed-component methods (ns per product)

Products of two split arrays, which need no transform. `fastest mean` names the lowest mean, or a tie when the interval of the paired ratio of the two lowest means contains one. `auto` is the method that `"auto"` selects for the pair.

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

## Paired ratios of the signed-component methods

`schedule / rows` below 1 favours `schedule`; `native / rows` below 1 favours `native`.

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

## Full product (ns per product)

`split` on loop-basis arrays, with the method that `"auto"` selects, against `table`. `table / split` above 1 favours `split`.

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
