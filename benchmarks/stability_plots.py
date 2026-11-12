#!/usr/bin/env python3
"""Draw the accuracy charts of a stability study from its CSV summary.

For each of R and C the script draws three charts, each as PNG, SVG and PDF:
the median relative error and the median normalized error eta of the split
algorithm (S) and the table algorithm (T) on dense inputs, and the median eta
of chains kept in split form (L) and returned to the loop basis after every
product (R). Every plotted value is also written to plot-data.csv, and
PLOTS.md lists the charts. Needs Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

FORMATS = ("png", "svg", "pdf")
DOMAINS = {"real": "real coefficients", "complex": "complex coefficients"}

# One colour for each method, kept across every chart: the first method of an
# experiment is blue and the second orange
COLORS = {"S": "#2a78d6", "T": "#eb6834", "L": "#2a78d6", "R": "#eb6834"}
MARKERS = {"S": "o", "T": "s", "L": "o", "R": "s"}
NAMES = {"S": "S, split algorithm", "T": "T, table algorithm",
         "L": "L, kept in split form", "R": "R, converted after every product"}
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

FIELDS = ("figure", "domain", "series", "chain_length", "dimension", "value", "samples")


def read_summary(path: Path) -> list[dict[str, Any]]:
    """Read the CSV summary, converting its numeric columns."""
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for key in ("level", "dimension", "samples"):
                row[key] = int(row[key])
            for key in ("median_eta", "max_eta", "median_relative"):
                row[key] = float(row[key]) if row[key] else None
            rows.append(row)
    return rows


def series(rows: list[dict[str, Any]], domain: str, experiment: str, family: str, method: str,
           key: str) -> list[tuple[int, float, int]]:
    """Return (dimension, value, samples) of one method, ordered by dimension."""
    points = [(r["dimension"], r[key], r["samples"]) for r in rows
              if r["domain"] == domain and r["experiment"] == experiment and r["family"] == family
              and r["method"] == method and r[key] is not None]
    return sorted(points)


def style(axis: Any, dimensions: list[int], top: float | None) -> None:
    """Apply the shared axis style: recessive grid and frame, base-two N axis.

    With *top* the value axis is linear from zero to a little above it;
    without it the axis is logarithmic.
    """
    axis.set_facecolor(SURFACE)
    axis.set_xscale("log", base=2)
    axis.set_xticks(dimensions)
    axis.set_xticklabels([str(n) for n in dimensions], rotation=45 if len(dimensions) > 7 else 0)
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if top is None:
        axis.set_yscale("log")
    else:
        axis.set_ylim(0, 1.15 * top if top > 0 else 1)
    axis.grid(True, axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(which="both", colors=MUTED, labelsize=9)
    axis.set_xlabel("N, dimension of the loop algebra", color=SECONDARY, fontsize=10)


def highest(lines: dict[str, list[tuple[int, float, int]]]) -> float:
    """Return the largest plotted value of a set of lines."""
    return max((p[1] for points in lines.values() for p in points), default=0.0)


def draw_lines(axis: Any, lines: dict[str, list[tuple[int, float, int]]], top: float | None) -> None:
    """Draw each method as a thin line with markers and label it at its last point."""
    ends = {method: points[-1][1] for method, points in lines.items() if points}
    span = top if top else max(ends.values(), default=1.0)
    close = len(ends) == 2 and abs(max(ends.values()) - min(ends.values())) < 0.06 * span
    upper = max(ends, key=lambda method: ends[method]) if ends else None
    for method, points in lines.items():
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        axis.plot(xs, ys, color=COLORS[method], marker=MARKERS[method], linewidth=2, markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, label=NAMES[method])
        if points:
            # Labels of two lines that end together are moved apart vertically
            shift = (7 if method == upper else -7) if close else 0
            axis.annotate(method, (xs[-1], ys[-1]), xytext=(8, shift), textcoords="offset points",
                          color=INK, fontsize=10, va="center")


def save(figure: Any, directory: Path, name: str) -> None:
    # Dates are left out of the vector files so that redrawing reproduces them
    metadata = {"png": None, "svg": {"Date": None}, "pdf": {"CreationDate": None}}
    for extension in FORMATS:
        figure.savefig(directory / f"{name}.{extension}", dpi=160, facecolor=SURFACE, metadata=metadata[extension])
    plt.close(figure)


def product_chart(rows: list[dict[str, Any]], directory: Path, domain: str, key: str, name: str,
                  ylabel: str, data: list[dict[str, Any]]) -> bool:
    """Draw S and T on dense inputs for one measure; return whether there was data."""
    lines = {m: series(rows, domain, "product", "dense", m, key) for m in ("S", "T")}
    if not any(lines.values()):
        return False
    dimensions = sorted({p[0] for points in lines.values() for p in points})
    log = key == "median_relative" and all(p[1] > 0 for points in lines.values() for p in points)
    top = None if log else highest(lines)
    figure, axis = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE, layout="constrained")
    draw_lines(axis, lines, top)
    for method, points in lines.items():
        data.extend({"figure": name, "domain": domain, "series": method, "chain_length": "", "dimension": n,
                     "value": repr(v), "samples": count} for n, v, count in points)
    style(axis, dimensions, top)
    axis.set_ylabel(ylabel, color=SECONDARY, fontsize=10)
    axis.set_title(f"{ylabel}, dense inputs, {DOMAINS[domain]}", color=INK, fontsize=11, loc="left")
    axis.legend(frameon=False, fontsize=9, labelcolor=SECONDARY, loc="upper left")
    save(figure, directory, name)
    return True


def chain_chart(rows: list[dict[str, Any]], directory: Path, domain: str, name: str,
                data: list[dict[str, Any]]) -> bool:
    """Draw L and R for every chain length as small multiples sharing one eta axis."""
    lengths = sorted({int(r["family"].split("_")[1]) for r in rows
                      if r["domain"] == domain and r["experiment"] == "chain"})
    if not lengths:
        return False
    figure, axes = plt.subplots(1, len(lengths), figsize=(3.6 * len(lengths), 4.2), sharey=True,
                                facecolor=SURFACE, layout="constrained", squeeze=False)
    panels = {length: {m: series(rows, domain, "chain", f"length_{length}", m, "median_eta") for m in ("L", "R")}
              for length in lengths}
    top = max(highest(lines) for lines in panels.values())
    for axis, (length, lines) in zip(axes[0], panels.items(), strict=True):
        draw_lines(axis, lines, top)
        for method, points in lines.items():
            data.extend({"figure": name, "domain": domain, "series": method, "chain_length": length, "dimension": n,
                         "value": repr(v), "samples": count} for n, v, count in points)
        style(axis, sorted({p[0] for points in lines.values() for p in points}), top)
        axis.set_title(f"k = {length} operands", color=INK, fontsize=10, loc="left")
    axes[0][0].set_ylabel("median eta", color=SECONDARY, fontsize=10)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=SECONDARY, loc="upper left")
    figure.suptitle(f"Median eta of left-nested chains, {DOMAINS[domain]}", color=INK, fontsize=11, x=0.01, ha="left")
    save(figure, directory, name)
    return True


def plots_document(figures: list[tuple[str, str]], command: str) -> str:
    """Return the Markdown description of the charts."""
    lines = [
        "# Accuracy charts", "",
        "The charts show the accuracy of the loop-algebra product for Cayley–Dickson loops, measured "
        "against an exact reference. S is the split algorithm and T is the table algorithm. In the chain "
        "charts, L keeps a left-nested product in split form and converts once at the end, and R returns "
        "to the loop basis after every product.", "",
        "| Chart | PNG | SVG | PDF |", "|:---|:---|:---|:---|",
    ]
    lines.extend(f"| {title} | [PNG]({name}.png) | [SVG]({name}.svg) | [PDF]({name}.pdf) |" for name, title in figures)
    lines.extend([
        "", "## Values", "",
        "Each point of the relative-error and eta charts is the median over the Gaussian and uniform "
        "operand pairs of the study at one dimension; each point of the chain charts is the median over "
        "the chains of one length at one dimension. The error measures are defined in docs/stability.md. "
        "Every plotted value, with the number of samples behind it, is in plot-data.csv.", "",
        "## Axes", "",
        "N, the dimension of the loop algebra, is on a base-two logarithmic axis. The relative-error axis "
        "is logarithmic when every plotted median is positive; the eta axes are linear and start at zero. "
        "Medians do not describe worst-case errors; the maxima are in accuracy-tables.md.", "",
        "## Reproduction", "", "```sh", command, "```", "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="study directory holding accuracy-summary.csv")
    args = parser.parse_args(argv)
    rows = read_summary(args.directory / "accuracy-summary.csv")
    data: list[dict[str, Any]] = []
    figures = []
    for domain, title in DOMAINS.items():
        charts = (
            (f"relative-error-{domain}", f"Median relative error, {title}",
             lambda name, d=domain: product_chart(rows, args.directory, d, "median_relative", name, "Median relative error", data)),
            (f"eta-{domain}", f"Median normalized error eta, {title}",
             lambda name, d=domain: product_chart(rows, args.directory, d, "median_eta", name, "Median eta", data)),
            (f"chain-eta-{domain}", f"Median eta of chains of products, {title}",
             lambda name, d=domain: chain_chart(rows, args.directory, d, name, data)),
        )
        for name, description, chart in charts:
            if chart(name):
                figures.append((name, description))
    with (args.directory / "plot-data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    command = f"python3 benchmarks/stability_plots.py benchmark-results/{args.directory.name}"
    (args.directory / "PLOTS.md").write_text(plots_document(figures, command))


if __name__ == "__main__":
    main()
