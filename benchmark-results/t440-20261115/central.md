# pycdloop multiplication benchmark: central products over R

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.4.0, pycdloop 0.1.0. Pinned to logical CPU 36; load average at start 1.55.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 1376 verified products agree with an independent evaluation of the definition.

A cell is labelled by the factor levels `n x m` of `Q_n * Q_m`. `D` is the dimension `2^(n+m)` of the tensor product of the two Cayley–Dickson algebras, and the loop algebra has dimension `N = 2D`.

## Signed-component methods (ns per product)

Products of two split arrays, which need no transform. `fastest mean` names the lowest mean, or a tie when the interval of the paired ratio of the two lowest means contains one. `auto` is the method that `"auto"` selects for the pair.

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

## Paired ratios of the signed-component methods

`schedule / rows` below 1 favours `schedule`; `native / rows` below 1 favours `native`.

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

## Full product (ns per product)

`split` on loop-basis arrays, with the method that `"auto"` selects, against `table`. `table / split` above 1 favours `split`.

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
