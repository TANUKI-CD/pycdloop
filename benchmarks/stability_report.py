#!/usr/bin/env python3
"""Render the accuracy tables, CSV summary, and key figures from stability.py results.

S is the split algorithm and T is the table algorithm of the loop-algebra
product. In the chain experiment, L keeps a chain of products in split form and
converts once at the end, and R returns to the loop basis after every product.
The same tables are written to the study directory and, with --update-docs,
into the delimited accuracy section of docs/stability.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN STABILITY ACCURACY -->"
END = "<!-- END STABILITY ACCURACY -->"
DENSE = ("gaussian", "uniform")
METHODS = ("S", "T")
CHAIN_METHODS = ("L", "R")
FOOTER = (
    "The archived CSV summary also gives, for every family and chain length, the count; the "
    "median, mean, p95, p99, and maximum of eta; the relative errors; and the paired comparison "
    "fractions. Groups with fewer than 100 samples have an empirical p99 close to their maximum "
    "and large sampling uncertainty."
)
FIELDS = ("experiment", "domain", "level", "dimension", "family", "method", "samples", "median_eta",
          "mean_eta", "p95_eta", "p99_eta", "max_eta", "median_relative", "max_relative",
          "zero_products", "fraction_first_better", "fraction_tied")


def fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3g}"


def values(rows: list[dict[str, Any]], method: str, key: str) -> list[float]:
    """Return one measure of one method over rows, leaving out undefined values."""
    return [r["methods"][method][key] for r in rows if r["methods"][method][key] is not None]


def med(rows: list[dict[str, Any]], method: str, key: str) -> float | None:
    data = values(rows, method, key)
    return float(np.median(data)) if data else None


def top(rows: list[dict[str, Any]], method: str, key: str) -> float | None:
    data = values(rows, method, key)
    return max(data) if data else None


def share(rows: list[dict[str, Any]], key: str) -> str:
    return f"{sum(r[key] for r in rows) / len(rows):.0%}"


def joined(rows: list[dict[str, Any]], methods: tuple[str, ...], operation: Any) -> str:
    """Format one statistic of eta for each method, separated by slashes."""
    return " / ".join(fmt(operation(values(rows, m, "normalized"))) for m in methods)


def section(text: str) -> str | None:
    """Return the delimited accuracy section of a document, or None without markers."""
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        return None
    return text[text.index(BEGIN) + len(BEGIN):text.index(END)].strip("\n")


def replace_section(text: str, tables: str) -> str | None:
    """Return a document with its accuracy section replaced, or None without markers."""
    if section(text) is None:
        return None
    return text[:text.index(BEGIN) + len(BEGIN)] + "\n\n" + tables.strip("\n") + "\n\n" + text[text.index(END):]


def summarize(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the statistics of every group of the product and chain experiments."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in document["samples"]:
        names = [row["family"], "dense"] if row["family"] in DENSE else [row["family"]]
        for name in names:
            groups.setdefault(("product", row["domain"], row["level"], name), []).append(row)
    for row in document["chains"]:
        groups.setdefault(("chain", row["domain"], row["level"], f"length_{row['length']}"), []).append(row)
    summaries = []
    for (experiment, domain, level, family), rows in groups.items():
        first = "s_better" if experiment == "product" else "l_at_least_as_accurate"
        for method in METHODS if experiment == "product" else CHAIN_METHODS:
            eta = values(rows, method, "normalized")
            relative = values(rows, method, "relative")
            summaries.append({
                "experiment": experiment, "domain": domain, "level": level, "dimension": 2 << level,
                "family": family, "method": method, "samples": len(rows),
                "median_eta": float(np.median(eta)), "mean_eta": float(np.mean(eta)),
                "p95_eta": float(np.percentile(eta, 95)), "p99_eta": float(np.percentile(eta, 99)),
                "max_eta": max(eta), "median_relative": float(np.median(relative)) if relative else None,
                "max_relative": max(relative) if relative else None,
                "zero_products": sum(r["methods"][method]["zero_product"] for r in rows),
                "fraction_first_better": sum(r[first] for r in rows) / len(rows),
                "fraction_tied": sum(r["tied"] for r in rows) / len(rows)})
    order = {"product": 0, "chain": 1}
    return sorted(summaries, key=lambda s: (order[s["experiment"]], s["domain"] != "real", s["level"], s["family"], s["method"]))


def render(document: dict[str, Any]) -> str:
    """Return the accuracy tables, without section markers."""
    if not document.get("completed"):
        raise ValueError("accuracy study did not finish; refusing a partial table")
    lines = [
        f"{len(document['samples']):,} operand pairs and {len(document['chains']):,} chains from the "
        f"{document['profile']} profile. S is the split algorithm and T is the table algorithm; both "
        "multiply the same stored operands. N is the dimension of the loop algebra, twice the "
        "dimension of the Cayley–Dickson algebra. Dense rows pool the Gaussian and uniform families, "
        "and \"S better / tied\" compares the exact errors of S and T. In the chain tables, L keeps a "
        "left-nested product of k operands in split form and converts once at the end, R returns to "
        "the loop basis after every product, and eta divides the error by the product of the operand "
        "norms. Empirical p99 values and maxima describe this corpus only.", "",
    ]
    for domain in ("real", "complex"):
        samples = [r for r in document["samples"] if r["domain"] == domain]
        dense = [r for r in samples if r["family"] in DENSE]
        if not dense:
            continue
        levels = sorted({r["level"] for r in dense})
        lines.extend([f"**{domain.capitalize()} dense inputs**", "",
                      "| N | pairs | median eta S / T | p99 eta S / T | max eta S / T | S better / tied |",
                      "|---:|---:|:---|:---|:---|:---|"])
        for level in levels:
            rows = [r for r in dense if r["level"] == level]
            lines.append(f"| {2 << level} | {len(rows)} | {joined(rows, METHODS, np.median)} | "
                         f"{joined(rows, METHODS, lambda a: np.percentile(a, 99))} | {joined(rows, METHODS, max)} | "
                         f"{share(rows, 's_better')} / {share(rows, 'tied')} |")
        lines.extend(["", f"**{domain.capitalize()} growth with dimension**, S on dense inputs", "",
                      "| N | median eta | eta / log2 N | median relative error | max relative error |",
                      "|---:|---:|---:|---:|---:|"])
        for level in levels:
            rows = [r for r in dense if r["level"] == level]
            eta = med(rows, "S", "normalized")
            lines.append(f"| {2 << level} | {fmt(eta)} | {fmt(eta / (level + 1))} | "
                         f"{fmt(med(rows, 'S', 'relative'))} | {fmt(top(rows, 'S', 'relative'))} |")
        level = min(8, max(levels))
        selected = [r for r in samples if r["level"] == level]
        lines.extend(["", f"**{domain.capitalize()} families at N={2 << level}**", "",
                      "| family | pairs | median eta S / T | max relative error S | S better / tied |",
                      "|:---|---:|:---|---:|:---|"])
        for family in dict.fromkeys(r["family"] for r in selected):
            rows = [r for r in selected if r["family"] == family]
            lines.append(f"| {family} | {len(rows)} | {joined(rows, METHODS, np.median)} | "
                         f"{fmt(top(rows, 'S', 'relative'))} | {share(rows, 's_better')} / {share(rows, 'tied')} |")
        chains = [r for r in document["chains"] if r["domain"] == domain]
        if chains:
            lines.extend(["", f"**{domain.capitalize()} chains**", "",
                          "| N | k | chains | median eta L / R | max eta L / R | L at least as accurate / tied |",
                          "|---:|---:|---:|:---|:---|:---|"])
            for level in sorted({r["level"] for r in chains}):
                for length in sorted({r["length"] for r in chains}):
                    rows = [r for r in chains if r["level"] == level and r["length"] == length]
                    lines.append(f"| {2 << level} | {length} | {len(rows)} | {joined(rows, CHAIN_METHODS, np.median)} | "
                                 f"{joined(rows, CHAIN_METHODS, max)} | "
                                 f"{share(rows, 'l_at_least_as_accurate')} / {share(rows, 'tied')} |")
        lines.append("")
    lines.append(FOOTER)
    return "\n".join(lines) + "\n"


def summary_csv(document: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summarize(document))
    return output.getvalue()


def key_figures(document: dict[str, Any]) -> dict[str, Any]:
    """Return the figures that the documentation quotes, computed from the results."""
    figures: dict[str, Any] = {"profile": document["profile"], "pairs": len(document["samples"]),
                               "chains": len(document["chains"]), "domains": {}}
    for domain in ("real", "complex"):
        samples = [r for r in document["samples"] if r["domain"] == domain]
        dense = [r for r in samples if r["family"] in DENSE]
        if not dense:
            continue
        levels = sorted({r["level"] for r in dense})
        per_level = {level: [r for r in dense if r["level"] == level] for level in levels}
        ratios = [med(rows, "S", "normalized") / med(rows, "T", "normalized") for rows in per_level.values()
                  if med(rows, "T", "normalized")]
        chains = [r for r in document["chains"] if r["domain"] == domain]
        record: dict[str, Any] = {
            "pairs": len(samples), "families": len({r["family"] for r in samples}),
            "levels": [levels[0], levels[-1]], "dimensions": [2 << levels[0], 2 << levels[-1]],
            "max_eta": {m: top(samples, m, "normalized") for m in METHODS},
            "dense_median_eta": {str(2 << level): {m: med(rows, m, "normalized") for m in METHODS}
                                 for level, rows in per_level.items()},
            "dense_median_ratio_S_over_T": [min(ratios), max(ratios)] if ratios else None,
            "dense_median_eta_over_log2N_S": [med(rows, "S", "normalized") / (level + 1)
                                              for level, rows in per_level.items()],
            "zero_product_pairs": sum(r["methods"]["S"]["zero_product"] for r in samples),
            "nonzero_outputs_for_zero_products": {m: sum(r["methods"][m]["zero_product"] and r["methods"][m]["absolute"] > 0
                                                       for r in samples) for m in METHODS},
        }
        if chains:
            lengths = sorted({r["length"] for r in chains})
            record["chains"] = {
                "count": len(chains), "lengths": lengths,
                "levels": [min(r["level"] for r in chains), max(r["level"] for r in chains)],
                "max_eta": {m: top(chains, m, "normalized") for m in CHAIN_METHODS},
                "L_at_least_as_accurate": sum(r["l_at_least_as_accurate"] for r in chains) / len(chains),
                "tied": sum(r["tied"] for r in chains) / len(chains),
                "median_eta_by_length": {str(k): {m: med([r for r in chains if r["length"] == k], m, "normalized")
                                                  for m in CHAIN_METHODS} for k in lengths},
            }
        figures["domains"][domain] = record
    assert all(math.isfinite(v) for d in figures["domains"].values() for v in d["max_eta"].values())
    return figures


def write(document: dict[str, Any], directory: Path) -> str:
    """Write the three report files into a study directory and return the tables."""
    tables = render(document)
    (directory / "accuracy-tables.md").write_text(tables)
    (directory / "accuracy-summary.csv").write_text(summary_csv(document))
    (directory / "key-figures.json").write_text(json.dumps(key_figures(document), indent=2, allow_nan=False) + "\n")
    return tables


def update_document(path: Path, tables: str) -> bool:
    """Rewrite the accuracy section of a document; do nothing without the file or markers."""
    if not path.is_file():
        return False
    updated = replace_section(path.read_text(), tables)
    if updated is None:
        return False
    path.write_text(updated)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="stability.py JSON results")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for the tables, CSV summary, and key figures")
    parser.add_argument("--update-docs", action="store_true", help="rewrite the accuracy section of the document")
    parser.add_argument("--document", type=Path, default=ROOT / "docs" / "stability.md")
    args = parser.parse_args(argv)
    tables = write(json.loads(args.results.read_text()), args.output_dir)
    if args.update_docs and not update_document(args.document, tables):
        print(f"{args.document} has no accuracy section; left unchanged")


if __name__ == "__main__":
    main()
