# pycdloop multiplication benchmark: real coefficients

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.4.0, pycdloop 0.1.0. Pinned to logical CPU 37; load average at start 2.57.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 10176 verified products agree with an independent evaluation of the definition.

`split` multiplies through the decomposition of the loop algebra (three Walsh–Hadamard transforms, one coordinatewise product and one Cayley–Dickson product); `table` evaluates the definition. `N` is the dimension of the loop algebra of `Q_n`, which is `2^(n+1)`. `fastest mean` names the lower mean of `split` and `table`, or a tie when the interval of their paired ratio contains one, and `default` is the algorithm that `"auto"` selects.

## Batch products (ns per product)

One call on arrays holding every pair of the corpus. `splitform` is the product of two split arrays, which needs no transform.

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

## Single products (ns per product)

One call per pair, on arrays of one record. `cached` is the product of two elements whose split forms are already cached.

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

## Paired ratios

`table / split` above 1 favours `split`. `split / splitform` and `split / cached` are the factors by which a product is slower when its operands and result are in the loop basis rather than in split form.

| N | n | table / split (batch) | table / split (single) | split / splitform (batch) | split / cached (single) |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 3.14 [3.03, 3.26] | 1.57 [1.54, 1.61] | 2.66 [2.49, 2.85] | 1.92 [1.87, 1.98] |
| 8 | 2 | 4.24 [4.16, 4.31] | 1.61 [1.54, 1.69] | 2.83 [2.72, 2.94] | 1.94 [1.8, 2.09] |
| 16 | 3 | 7.97 [7.77, 8.19] | 1.62 [1.6, 1.64] | 2.67 [2.58, 2.77] | 1.91 [1.87, 1.95] |
| 32 | 4 | 30.1 [29.6, 30.5] | 1.78 [1.75, 1.82] | 2.31 [2.26, 2.35] | 1.92 [1.88, 1.95] |
| 64 | 5 | 87.2 [85.5, 89] | 2.06 [2.03, 2.1] | 1.99 [1.93, 2.05] | 1.92 [1.88, 1.96] |
| 128 | 6 | 170 [167, 174] | 3.66 [3.53, 3.8] | 1.87 [1.82, 1.91] | 1.99 [1.93, 2.05] |
| 256 | 7 | 269 [263, 276] | 10.6 [10.4, 10.9] | 1.79 [1.73, 1.85] | 1.82 [1.76, 1.87] |
| 512 | 8 | 669 [655, 683] | 35.5 [34.1, 36.9] | 1.72 [1.66, 1.77] | 1.87 [1.8, 1.95] |
| 1024 | 9 | 1.14e+03 [1.12e+03, 1.16e+03] | 114 [112, 117] | 1.61 [1.56, 1.66] | 1.73 [1.68, 1.79] |
| 2048 | 10 | 1.57e+03 [1.55e+03, 1.59e+03] | 570 [558, 582] | 1.6 [1.57, 1.64] | 1.66 [1.6, 1.72] |

## Comparison with the Cayley–Dickson product

`pyfastcd` is the Cayley–Dickson product of the signed components alone, of dimension `N / 2`, timed through pyfastcd on its arrays (batch) and on its elements (single). A loop-algebra product contains one such product, so each ratio is the factor by which the rest of the work, and the Python layers around it, slow the product down.

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

## Share of the transforms

The fraction of a batch product spent converting the operands to split form and the result back, measured as `(forward + inverse) / (forward + inverse + splitform)` within each block.

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
