# pycdloop multiplication benchmark: real coefficients

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.4.0, pycdloop 0.1.0. Pinned to logical CPU 37; load average at start 4.14.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 10176 verified products agree with an independent evaluation of the definition.

`split` multiplies through the decomposition of the loop algebra (three Walsh–Hadamard transforms, one coordinatewise product and one Cayley–Dickson product); `table` evaluates the definition. `N` is the dimension of the loop algebra of `Q_n`, which is `2^(n+1)`. `fastest mean` names the lower mean of `split` and `table`, or a tie when the interval of their paired ratio contains one, and `default` is the algorithm that `"auto"` selects.

## Batch products (ns per product)

One call on arrays holding every pair of the corpus. `splitform` is the product of two split arrays, which needs no transform.

| N | n | pairs | split | table | splitform | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 183.6 [181.0, 186.2] | 593.8 [585.4, 602.1] | 66.59 [65.64, 67.55] | split | split |
| 8 | 2 | 256 | 205.0 [187.3, 222.7] | 832.8 [819.1, 846.6] | 71.03 [70.07, 71.98] | split | split |
| 16 | 3 | 256 | 212.1 [208.8, 215.3] | 1717.1 [1681.8, 1752.3] | 80.11 [78.90, 81.32] | split | split |
| 32 | 4 | 256 | 310.6 [295.9, 325.3] | 8855.4 [8693.9, 9016.9] | 133.5 [130.5, 136.4] | split | split |
| 64 | 5 | 256 | 546.8 [532.1, 561.5] | 48506 [46670, 50342] | 266.3 [262.8, 269.9] | split | split |
| 128 | 6 | 256 | 1082.6 [1063.3, 1101.8] | 183579 [181954, 185204] | 594.2 [578.4, 610.0] | split | split |
| 256 | 7 | 64 | 2699.3 [2668.2, 2730.3] | 713142 [706043, 720241] | 1473.0 [1450.9, 1495.0] | split | split |
| 512 | 8 | 64 | 5506.1 [5419.0, 5593.3] | 3785276 [3761010, 3809543] | 3287.2 [3205.3, 3369.1] | split | split |
| 1024 | 9 | 16 | 13436 [13222, 13650] | 14788687 [14680545, 14896830] | 8235.7 [8017.8, 8453.5] | split | split |
| 2048 | 10 | 16 | 28224 [27582, 28865] | 44572127 [44060960, 45083295] | 17334 [17031, 17637] | split | split |

## Single products (ns per product)

One call per pair, on arrays of one record. `cached` is the product of two elements whose split forms are already cached.

| N | n | pairs | split | table | cached | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 30021 [29816, 30227] | 48679 [48218, 49140] | 15515 [15325, 15704] | split | split |
| 8 | 2 | 256 | 30735 [28860, 32611] | 48257 [47815, 48699] | 15494 [15357, 15631] | split | split |
| 16 | 3 | 256 | 29982 [29730, 30233] | 50198 [49377, 51018] | 15586 [15373, 15798] | split | split |
| 32 | 4 | 256 | 31510 [30684, 32335] | 60510 [56985, 64036] | 16438 [16057, 16818] | split | split |
| 64 | 5 | 256 | 30551 [30116, 30985] | 66769 [61626, 71913] | 16076 [15915, 16236] | split | split |
| 128 | 6 | 256 | 31362 [30853, 31871] | 119885 [118010, 121760] | 16627 [16457, 16798] | split | split |
| 256 | 7 | 64 | 34640 [34065, 35215] | 366792 [360365, 373220] | 18836 [18504, 19169] | split | split |
| 512 | 8 | 64 | 38032 [37444, 38620] | 1382884 [1352335, 1413432] | 20727 [20181, 21272] | split | split |
| 1024 | 9 | 16 | 47142 [45063, 49222] | 5068743 [5001454, 5136032] | 26221 [25201, 27242] | split | split |
| 2048 | 10 | 16 | 59427 [57577, 61277] | 34466855 [34302721, 34630990] | 35674 [35007, 36341] | split | split |

## Paired ratios

`table / split` above 1 favours `split`. `split / splitform` and `split / cached` are the factors by which a product is slower when its operands and result are in the loop basis rather than in split form.

| N | n | table / split (batch) | table / split (single) | split / splitform (batch) | split / cached (single) |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 3.23 [3.18, 3.29] | 1.62 [1.6, 1.64] | 2.76 [2.71, 2.81] | 1.94 [1.91, 1.96] |
| 8 | 2 | 4.11 [3.88, 4.36] | 1.58 [1.5, 1.66] | 2.85 [2.67, 3.04] | 1.97 [1.87, 2.07] |
| 16 | 3 | 8.09 [7.96, 8.23] | 1.67 [1.65, 1.7] | 2.65 [2.6, 2.69] | 1.92 [1.89, 1.95] |
| 32 | 4 | 28.6 [27.7, 29.6] | 1.91 [1.82, 2] | 2.32 [2.22, 2.42] | 1.92 [1.86, 1.97] |
| 64 | 5 | 88.6 [86.2, 91] | 2.16 [2.04, 2.3] | 2.05 [2, 2.1] | 1.9 [1.87, 1.93] |
| 128 | 6 | 170 [167, 173] | 3.82 [3.75, 3.9] | 1.82 [1.76, 1.89] | 1.89 [1.85, 1.92] |
| 256 | 7 | 264 [260, 269] | 10.6 [10.4, 10.8] | 1.83 [1.8, 1.87] | 1.84 [1.79, 1.89] |
| 512 | 8 | 688 [677, 699] | 36.3 [35.3, 37.4] | 1.68 [1.63, 1.72] | 1.84 [1.79, 1.88] |
| 1024 | 9 | 1.1e+03 [1.08e+03, 1.12e+03] | 108 [104, 112] | 1.63 [1.59, 1.68] | 1.8 [1.72, 1.87] |
| 2048 | 10 | 1.58e+03 [1.54e+03, 1.62e+03] | 581 [565, 598] | 1.63 [1.59, 1.67] | 1.66 [1.6, 1.72] |

## Comparison with the Cayley–Dickson product

`pyfastcd` is the Cayley–Dickson product of the signed components alone, of dimension `N / 2`, timed through pyfastcd on its arrays (batch) and on its elements (single). A loop-algebra product contains one such product, so each ratio is the factor by which the rest of the work, and the Python layers around it, slow the product down.

| N | n | pyfastcd batch (ns) | splitform / pyfastcd | split / pyfastcd | pyfastcd single (ns) | cached / pyfastcd |
| ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 9.22 [9.15, 9.30] | 7.22 [7.1, 7.34] | 19.9 [19.6, 20.2] | 97.28 [96.23, 98.33] | 159 [157, 162] |
| 8 | 2 | 11.41 [11.26, 11.56] | 6.22 [6.12, 6.34] | 17.7 [16.6, 18.9] | 108.8 [98.40, 119.1] | 145 [136, 154] |
| 16 | 3 | 19.78 [19.57, 19.99] | 4.05 [3.98, 4.12] | 10.7 [10.5, 10.9] | 120.3 [119.1, 121.5] | 130 [128, 131] |
| 32 | 4 | 64.09 [62.39, 65.79] | 2.08 [2.02, 2.15] | 4.83 [4.66, 5.01] | 234.1 [230.6, 237.7] | 70.2 [68.8, 71.5] |
| 64 | 5 | 184.6 [183.0, 186.3] | 1.44 [1.42, 1.47] | 2.96 [2.88, 3.04] | 335.5 [331.7, 339.3] | 47.9 [47.4, 48.4] |
| 128 | 6 | 451.7 [447.3, 456.0] | 1.31 [1.28, 1.35] | 2.4 [2.35, 2.44] | 630.6 [623.5, 637.6] | 26.4 [26.2, 26.6] |
| 256 | 7 | 1130.9 [1120.0, 1141.9] | 1.3 [1.28, 1.33] | 2.39 [2.35, 2.42] | 1360.9 [1337.7, 1384.1] | 13.8 [13.6, 14.1] |
| 512 | 8 | 2606.1 [2571.7, 2640.4] | 1.26 [1.23, 1.29] | 2.11 [2.08, 2.15] | 2953.0 [2916.8, 2989.2] | 7.01 [6.86, 7.16] |
| 1024 | 9 | 6522.8 [6449.6, 6596.0] | 1.26 [1.23, 1.29] | 2.06 [2.02, 2.1] | 6716.9 [6640.5, 6793.3] | 3.89 [3.76, 4.02] |
| 2048 | 10 | 15042 [14681, 15403] | 1.15 [1.12, 1.18] | 1.88 [1.82, 1.94] | 14994 [14823, 15165] | 2.38 [2.34, 2.42] |

## Share of the transforms

The fraction of a batch product spent converting the operands to split form and the result back, measured as `(forward + inverse) / (forward + inverse + splitform)` within each block.

| N | n | forward (ns) | inverse (ns) | splitform (ns) | transform share |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 45.24 [44.56, 45.92] | 29.55 [29.05, 30.05] | 66.59 [65.64, 67.55] | 52.9% [52.5%, 53.3%] |
| 8 | 2 | 51.78 [51.16, 52.40] | 32.59 [32.25, 32.93] | 71.03 [70.07, 71.98] | 54.3% [53.9%, 54.6%] |
| 16 | 3 | 52.63 [52.10, 53.16] | 34.50 [34.05, 34.95] | 80.11 [78.90, 81.32] | 52.1% [51.8%, 52.5%] |
| 32 | 4 | 79.20 [77.88, 80.52] | 44.11 [43.03, 45.19] | 133.5 [130.5, 136.4] | 48.0% [47.6%, 48.5%] |
| 64 | 5 | 133.5 [122.5, 144.5] | 72.09 [70.78, 73.41] | 266.3 [262.8, 269.9] | 43.4% [42.2%, 44.6%] |
| 128 | 6 | 249.4 [244.8, 254.0] | 141.7 [138.0, 145.5] | 594.2 [578.4, 610.0] | 39.7% [39.4%, 40.0%] |
| 256 | 7 | 602.2 [594.0, 610.4] | 300.4 [294.1, 306.8] | 1473.0 [1450.9, 1495.0] | 38.0% [37.5%, 38.5%] |
| 512 | 8 | 1105.8 [1089.8, 1121.7] | 588.3 [577.2, 599.4] | 3287.2 [3205.3, 3369.1] | 34.0% [33.7%, 34.4%] |
| 1024 | 9 | 2660.3 [2588.2, 2732.3] | 1290.9 [1250.3, 1331.5] | 8235.7 [8017.8, 8453.5] | 32.4% [32.0%, 32.9%] |
| 2048 | 10 | 5363.9 [5233.1, 5494.7] | 2589.8 [2516.7, 2662.8] | 17334 [17031, 17637] | 31.5% [30.9%, 32.0%] |
