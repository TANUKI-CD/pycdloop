# pycdloop multiplication benchmark: real coefficients

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); Python 3.14.8, NumPy 2.5.4, pyfastcd 1.4.0, pycdloop 0.1.0. Pinned to logical CPU 36; load average at start 3.60.

24 blocks; each cell is the mean over blocks of nanoseconds per product with a two-sided 95% Student-t interval. Timing is in Python with the garbage collector disabled, after 2 untimed passes, for at least 20 ms per observation, and includes the call overhead of the package. A ratio is the geometric mean of per-block ratios with its 95% interval.

All 10176 verified products agree with an independent evaluation of the definition.

`split` multiplies through the decomposition of the loop algebra (three Walsh–Hadamard transforms, one coordinatewise product and one Cayley–Dickson product); `table` evaluates the definition. `N` is the dimension of the loop algebra of `Q_n`, which is `2^(n+1)`. `fastest mean` names the lower mean of `split` and `table`, or a tie when the interval of their paired ratio contains one, and `default` is the algorithm that `"auto"` selects.

## Batch products (ns per product)

One call on arrays holding every pair of the corpus. `splitform` is the product of two split arrays, which needs no transform.

| N | n | pairs | split | table | splitform | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 190.3 [185.1, 195.6] | 602.1 [578.3, 625.9] | 83.04 [80.40, 85.68] | split | split |
| 8 | 2 | 256 | 203.3 [196.5, 210.1] | 834.1 [813.1, 855.1] | 84.72 [82.52, 86.93] | split | split |
| 16 | 3 | 256 | 212.8 [207.8, 217.9] | 1740.6 [1698.4, 1782.7] | 95.78 [93.34, 98.21] | split | split |
| 32 | 4 | 256 | 307.7 [295.0, 320.3] | 8842.4 [8605.5, 9079.3] | 146.4 [142.4, 150.3] | split | split |
| 64 | 5 | 256 | 540.1 [529.5, 550.7] | 46956 [46522, 47390] | 279.7 [274.0, 285.4] | split | split |
| 128 | 6 | 256 | 1149.9 [1122.9, 1176.8] | 194279 [189335, 199223] | 624.9 [614.2, 635.5] | split | split |
| 256 | 7 | 64 | 2734.3 [2685.3, 2783.2] | 647863 [642389, 653337] | 1529.4 [1507.1, 1551.8] | split | split |
| 512 | 8 | 64 | 5503.7 [5380.8, 5626.6] | 3734639 [3719464, 3749814] | 3294.1 [3236.5, 3351.6] | split | split |
| 1024 | 9 | 16 | 13413 [13203, 13624] | 14744094 [14694059, 14794129] | 8490.9 [8217.8, 8764.0] | split | split |
| 2048 | 10 | 16 | 28270 [27469, 29071] | 43634747 [43487652, 43781841] | 18146 [17755, 18537] | split | split |

## Single products (ns per product)

One call per pair, on arrays of one record. `cached` is the product of two elements whose split forms are already cached.

| N | n | pairs | split | table | cached | fastest mean | default |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 256 | 30759 [30332, 31185] | 47298 [46560, 48037] | 15348 [15076, 15619] | split | split |
| 8 | 2 | 256 | 30530 [30088, 30972] | 47550 [46918, 48182] | 15609 [15284, 15934] | split | split |
| 16 | 3 | 256 | 30626 [30263, 30990] | 50165 [48857, 51472] | 15232 [14989, 15475] | split | split |
| 32 | 4 | 256 | 31086 [30677, 31495] | 54044 [52795, 55293] | 15697 [15403, 15990] | split | split |
| 64 | 5 | 256 | 31669 [31197, 32141] | 63259 [62080, 64438] | 15875 [15687, 16064] | split | split |
| 128 | 6 | 256 | 34926 [34322, 35530] | 124524 [121247, 127802] | 16781 [16473, 17089] | split | split |
| 256 | 7 | 64 | 36042 [35375, 36708] | 360453 [354995, 365912] | 18731 [18211, 19251] | split | split |
| 512 | 8 | 64 | 39011 [38032, 39989] | 1356510 [1337980, 1375040] | 20591 [19960, 21223] | split | split |
| 1024 | 9 | 16 | 45329 [44298, 46360] | 5092589 [5027865, 5157313] | 25600 [24968, 26232] | split | split |
| 2048 | 10 | 16 | 61153 [59247, 63059] | 34554684 [34368620, 34740749] | 35661 [34428, 36895] | split | split |

## Paired ratios

`table / split` above 1 favours `split`. `split / splitform` and `split / cached` are the factors by which a product is slower when its operands and result are in the loop basis rather than in split form.

| N | n | table / split (batch) | table / split (single) | split / splitform (batch) | split / cached (single) |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 3.16 [3.06, 3.26] | 1.54 [1.51, 1.57] | 2.29 [2.22, 2.37] | 2 [1.95, 2.06] |
| 8 | 2 | 4.11 [3.95, 4.27] | 1.56 [1.53, 1.59] | 2.4 [2.3, 2.5] | 1.96 [1.91, 2.01] |
| 16 | 3 | 8.18 [7.95, 8.41] | 1.64 [1.59, 1.68] | 2.22 [2.15, 2.3] | 2.01 [1.97, 2.05] |
| 32 | 4 | 28.8 [27.9, 29.7] | 1.74 [1.7, 1.77] | 2.1 [2.03, 2.17] | 1.98 [1.94, 2.02] |
| 64 | 5 | 87 [85.4, 88.7] | 2 [1.95, 2.05] | 1.93 [1.88, 1.99] | 1.99 [1.95, 2.04] |
| 128 | 6 | 169 [164, 174] | 3.56 [3.46, 3.67] | 1.84 [1.79, 1.89] | 2.08 [2.04, 2.13] |
| 256 | 7 | 237 [233, 241] | 10 [9.76, 10.3] | 1.79 [1.76, 1.82] | 1.93 [1.86, 2] |
| 512 | 8 | 679 [664, 695] | 34.8 [33.8, 35.9] | 1.67 [1.62, 1.72] | 1.9 [1.82, 1.98] |
| 1024 | 9 | 1.1e+03 [1.08e+03, 1.12e+03] | 112 [110, 115] | 1.58 [1.54, 1.63] | 1.77 [1.71, 1.83] |
| 2048 | 10 | 1.55e+03 [1.5e+03, 1.59e+03] | 566 [550, 583] | 1.56 [1.5, 1.62] | 1.72 [1.65, 1.78] |

## Comparison with the Cayley–Dickson product

`pyfastcd` is the Cayley–Dickson product of the signed components alone, of dimension `N / 2`, timed through pyfastcd on its arrays (batch) and on its elements (single). A loop-algebra product contains one such product, so each ratio is the factor by which the rest of the work, and the Python layers around it, slow the product down.

| N | n | pyfastcd batch (ns) | splitform / pyfastcd | split / pyfastcd | pyfastcd single (ns) | cached / pyfastcd |
| ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 1 | 9.58 [9.39, 9.78] | 8.65 [8.42, 8.9] | 19.8 [19.2, 20.5] | 98.27 [96.66, 99.88] | 156 [153, 159] |
| 8 | 2 | 11.42 [11.25, 11.58] | 7.41 [7.21, 7.62] | 17.8 [17.1, 18.4] | 106.9 [105.4, 108.4] | 146 [144, 148] |
| 16 | 3 | 19.53 [19.16, 19.90] | 4.9 [4.76, 5.05] | 10.9 [10.6, 11.2] | 119.8 [117.8, 121.8] | 127 [125, 129] |
| 32 | 4 | 62.73 [61.80, 63.67] | 2.33 [2.26, 2.41] | 4.89 [4.68, 5.1] | 222.7 [218.8, 226.6] | 70.5 [69, 72] |
| 64 | 5 | 182.6 [179.6, 185.6] | 1.53 [1.49, 1.57] | 2.96 [2.88, 3.03] | 331.8 [327.6, 336.0] | 47.8 [47.2, 48.5] |
| 128 | 6 | 451.4 [446.1, 456.7] | 1.38 [1.36, 1.4] | 2.54 [2.48, 2.61] | 647.6 [641.9, 653.2] | 25.9 [25.5, 26.3] |
| 256 | 7 | 1122.9 [1108.9, 1137.0] | 1.36 [1.34, 1.38] | 2.43 [2.38, 2.49] | 1348.4 [1333.2, 1363.6] | 13.9 [13.5, 14.2] |
| 512 | 8 | 2613.7 [2585.5, 2642.0] | 1.26 [1.23, 1.29] | 2.1 [2.05, 2.15] | 2953.3 [2902.8, 3003.8] | 6.96 [6.79, 7.13] |
| 1024 | 9 | 6599.5 [6516.5, 6682.4] | 1.28 [1.24, 1.33] | 2.03 [2, 2.07] | 6665.8 [6574.3, 6757.2] | 3.84 [3.76, 3.92] |
| 2048 | 10 | 14983 [14795, 15171] | 1.21 [1.18, 1.24] | 1.88 [1.83, 1.94] | 14940 [14736, 15144] | 2.38 [2.31, 2.45] |

## Share of the transforms

The fraction of a batch product spent converting the operands to split form and the result back, measured as `(forward + inverse) / (forward + inverse + splitform)` within each block.

| N | n | forward (ns) | inverse (ns) | splitform (ns) | transform share |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 1 | 46.96 [46.08, 47.83] | 30.85 [30.24, 31.46] | 83.04 [80.40, 85.68] | 48.4% [47.8%, 49.0%] |
| 8 | 2 | 54.39 [53.60, 55.18] | 33.52 [32.94, 34.09] | 84.72 [82.52, 86.93] | 51.0% [50.4%, 51.5%] |
| 16 | 3 | 55.26 [54.35, 56.16] | 36.21 [35.85, 36.56] | 95.78 [93.34, 98.21] | 48.9% [48.3%, 49.5%] |
| 32 | 4 | 78.67 [77.68, 79.66] | 45.79 [44.79, 46.79] | 146.4 [142.4, 150.3] | 46.0% [45.4%, 46.5%] |
| 64 | 5 | 129.5 [127.2, 131.8] | 72.90 [71.84, 73.96] | 279.7 [274.0, 285.4] | 42.0% [41.5%, 42.4%] |
| 128 | 6 | 254.6 [251.3, 257.9] | 144.9 [142.0, 147.8] | 624.9 [614.2, 635.5] | 39.0% [38.6%, 39.4%] |
| 256 | 7 | 600.4 [591.4, 609.3] | 313.0 [306.3, 319.7] | 1529.4 [1507.1, 1551.8] | 37.4% [36.9%, 37.9%] |
| 512 | 8 | 1201.9 [1182.1, 1221.7] | 595.4 [582.7, 608.1] | 3294.1 [3236.5, 3351.6] | 35.3% [34.9%, 35.8%] |
| 1024 | 9 | 2708.5 [2672.4, 2744.6] | 1346.5 [1313.6, 1379.4] | 8490.9 [8217.8, 8764.0] | 32.4% [31.8%, 33.0%] |
| 2048 | 10 | 5517.8 [5433.8, 5601.9] | 2666.3 [2592.7, 2739.8] | 18146 [17755, 18537] | 31.1% [30.6%, 31.6%] |
