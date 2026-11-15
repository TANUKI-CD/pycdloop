# pycdloop multiplication benchmark: real coefficients

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.4.0, pycdloop 0.1.0. Pinned to logical CPU 36; load average at start 1.88.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 10176 verified products agree with an independent evaluation of the definition.

`split` multiplies through the decomposition of the loop algebra (three Walsh–Hadamard transforms, one coordinatewise product, and one Cayley–Dickson product); `table` evaluates the definition. `N` is the dimension of the loop algebra of `Q_n`, which is `2^(n+1)`. `fastest mean` names the lower mean of `split` and `table`, or a tie when the interval of their paired ratio contains one, and `default` is the algorithm that `"auto"` selects.

## Batch products (ns per product)

One call on arrays holding every pair of the corpus. `splitform` is the product of two split arrays, which needs no transform.

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

## Single products (ns per product)

One call per pair, on arrays of one record. `cached` is the product of two elements whose split forms are already cached.

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

## Paired ratios

`table / split` above 1 favours `split`. `split / splitform` and `split / cached` are the factors by which a product is slower when its operands and result are in the loop basis rather than in split form.

| N | n | table / split (batch) | table / split (single) | split / splitform (batch) | split / cached (single) |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 3.12 [3.04, 3.2] | 1.51 [1.48, 1.54] | 2.28 [2.21, 2.37] | 1.99 [1.94, 2.03] |
| 8 | 2 | 4.04 [3.95, 4.13] | 1.54 [1.51, 1.57] | 2.32 [2.23, 2.42] | 1.97 [1.92, 2.03] |
| 16 | 3 | 7.95 [7.76, 8.14] | 1.64 [1.61, 1.68] | 2.19 [2.13, 2.26] | 1.99 [1.94, 2.03] |
| 32 | 4 | 29.3 [28.5, 30.2] | 1.77 [1.71, 1.83] | 2.06 [2.01, 2.11] | 1.97 [1.92, 2.02] |
| 64 | 5 | 87.5 [85.3, 89.6] | 2.04 [2, 2.08] | 1.93 [1.88, 1.98] | 1.99 [1.96, 2.01] |
| 128 | 6 | 175 [170, 179] | 3.59 [3.51, 3.68] | 1.84 [1.8, 1.88] | 2.03 [1.98, 2.07] |
| 256 | 7 | 263 [255, 272] | 10.2 [9.9, 10.4] | 1.82 [1.75, 1.9] | 1.87 [1.82, 1.92] |
| 512 | 8 | 658 [638, 677] | 35.4 [34.6, 36.2] | 1.71 [1.63, 1.79] | 1.84 [1.77, 1.91] |
| 1024 | 9 | 1.1e+03 [1.08e+03, 1.12e+03] | 109 [107, 112] | 1.58 [1.54, 1.63] | 1.78 [1.72, 1.85] |
| 2048 | 10 | 1.56e+03 [1.51e+03, 1.6e+03] | 575 [562, 589] | 1.59 [1.54, 1.64] | 1.67 [1.61, 1.72] |

## Comparison with the Cayley–Dickson product

`pyfastcd` is the Cayley–Dickson product of the signed components alone, of dimension `N / 2`, timed through pyfastcd on its arrays (batch) and on its elements (single). A loop-algebra product contains one such product, so each ratio is the factor by which the rest of the work, and the Python layers around it, slow the product down.

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

## Share of the transforms

The fraction of a batch product spent converting the operands to split form and the result back, measured as `(forward + inverse) / (forward + inverse + splitform)` within each block.

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
