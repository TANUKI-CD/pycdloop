# pycdloop multiplication benchmark: central products over R

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.4.0, pycdloop 0.1.0. Pinned to logical CPU 37; load average at start 3.37.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 1376 verified products agree with an independent evaluation of the definition.

A cell is labelled by the factor levels `n x m` of `Q_n * Q_m`. `D` is the dimension `2^(n+m)` of the tensor product of the two Cayley–Dickson algebras, and the loop algebra has dimension `N = 2D`.

## Signed-component methods (ns per product)

Products of two split arrays, which need no transform. `fastest mean` names the lowest mean, or a tie when the interval of the paired ratio of the two lowest means contains one. `auto` is the method that `"auto"` selects for the pair.

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

## Paired ratios of the signed-component methods

`schedule / rows` below 1 favours `schedule`; `native / rows` below 1 favours `native`.

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

## Full product (ns per product)

`split` on loop-basis arrays, with the method that `"auto"` selects, against `table`. `table / split` above 1 favours `split`.

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
