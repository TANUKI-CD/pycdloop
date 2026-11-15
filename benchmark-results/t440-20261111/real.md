# pycdloop multiplication benchmark: real coefficients

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.3.1, pycdloop 0.1.0. Pinned to logical CPU 37; load average at start 4.06.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 10176 verified products agree with an independent evaluation of the definition.

`split` multiplies through the decomposition of the loop algebra (three Walsh–Hadamard transforms, one coordinatewise product and one Cayley–Dickson product); `table` evaluates the definition. `N` is the dimension of the loop algebra of `Q_n`, which is `2^(n+1)`. `fastest mean` names the lower mean of `split` and `table`, or a tie when the interval of their paired ratio contains one, and `default` is the algorithm that `"auto"` selects.

## Batch products (ns per product)

One call on arrays holding every pair of the corpus. `splitform` is the product of two split arrays, which needs no transform.

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

## Single products (ns per product)

One call per pair, on arrays of one record. `cached` is the product of two elements whose split forms are already cached.

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

## Paired ratios

`table / split` above 1 favours `split`. `split / splitform` and `split / cached` are the factors by which a product is slower when its operands and result are in the loop basis rather than in split form.

| N | n | table / split (batch) | table / split (single) | split / splitform (batch) | split / cached (single) |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 1.58 [1.54, 1.62] | 1.5 [1.48, 1.52] | 1.46 [1.42, 1.51] | 1.92 [1.88, 1.96] |
| 8 | 2 | 2.19 [2.15, 2.23] | 1.49 [1.47, 1.51] | 1.52 [1.48, 1.56] | 1.95 [1.92, 1.99] |
| 16 | 3 | 4.23 [4.17, 4.29] | 1.62 [1.59, 1.66] | 1.54 [1.51, 1.57] | 1.9 [1.86, 1.94] |
| 32 | 4 | 18 [17.7, 18.4] | 1.71 [1.66, 1.76] | 1.55 [1.51, 1.58] | 1.91 [1.87, 1.95] |
| 64 | 5 | 61.1 [59.7, 62.4] | 1.97 [1.94, 2.01] | 1.67 [1.63, 1.71] | 1.92 [1.88, 1.95] |
| 128 | 6 | 140 [137, 143] | 3.57 [3.5, 3.64] | 1.64 [1.6, 1.67] | 1.91 [1.88, 1.94] |
| 256 | 7 | 195 [191, 198] | 9.38 [9.11, 9.66] | 1.59 [1.55, 1.64] | 1.87 [1.81, 1.93] |
| 512 | 8 | 584 [572, 596] | 33.5 [32.5, 34.5] | 1.58 [1.54, 1.62] | 1.85 [1.78, 1.92] |
| 1024 | 9 | 855 [843, 867] | 108 [106, 110] | 1.48 [1.45, 1.52] | 1.76 [1.73, 1.8] |
| 2048 | 10 | 1.4e+03 [1.37e+03, 1.42e+03] | 558 [546, 570] | 1.47 [1.44, 1.51] | 1.65 [1.6, 1.71] |

## Comparison with the Cayley–Dickson product

`pyfastcd` is the Cayley–Dickson product of the signed components alone, of dimension `N / 2`, timed through pyfastcd on its arrays (batch) and on its elements (single). A loop-algebra product contains one such product, so each ratio is the factor by which the rest of the work, and the Python layers around it, slow the product down.

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

## Share of the transforms

The fraction of a batch product spent converting the operands to split form and the result back, measured as `(forward + inverse) / (forward + inverse + splitform)` within each block.

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
