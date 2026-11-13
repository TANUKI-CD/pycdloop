# Accuracy charts

The charts show the accuracy of the loop-algebra product for Cayley–Dickson loops, measured against an exact reference. S is the split algorithm and T is the table algorithm. In the chain charts, L keeps a left-nested product in split form and converts once at the end, and R returns to the loop basis after every product.

| Chart | PNG | SVG | PDF |
|:---|:---|:---|:---|
| Median relative error, real coefficients | [PNG](relative-error-real.png) | [SVG](relative-error-real.svg) | [PDF](relative-error-real.pdf) |
| Median normalized error eta, real coefficients | [PNG](eta-real.png) | [SVG](eta-real.svg) | [PDF](eta-real.pdf) |
| Median eta of chains of products, real coefficients | [PNG](chain-eta-real.png) | [SVG](chain-eta-real.svg) | [PDF](chain-eta-real.pdf) |
| Median relative error, complex coefficients | [PNG](relative-error-complex.png) | [SVG](relative-error-complex.svg) | [PDF](relative-error-complex.pdf) |
| Median normalized error eta, complex coefficients | [PNG](eta-complex.png) | [SVG](eta-complex.svg) | [PDF](eta-complex.pdf) |
| Median eta of chains of products, complex coefficients | [PNG](chain-eta-complex.png) | [SVG](chain-eta-complex.svg) | [PDF](chain-eta-complex.pdf) |

## Values

Each point of the relative-error and eta charts is the median over the Gaussian and uniform operand pairs of the study at one dimension; each point of the chain charts is the median over the chains of one length at one dimension. The error measures are defined in docs/stability.md. Every plotted value, with the number of samples behind it, is in plot-data.csv.

## Axes

N, the dimension of the loop algebra, is on a base-two logarithmic axis. The relative-error axis is logarithmic when every plotted median is positive; the eta axes are linear and start at zero. Medians do not describe worst-case errors; the maxima are in accuracy-tables.md.

## Reproduction

```sh
python3 benchmarks/stability_plots.py benchmark-results/stability-20261112
```
