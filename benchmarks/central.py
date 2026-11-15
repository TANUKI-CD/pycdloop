#!/usr/bin/env python3
"""Time multiplication in loop algebras of central products ``Q_n * Q_m`` over R.

The loop algebra of a central product splits into a group algebra and the
tensor product of two Cayley--Dickson algebras, and pycdloop multiplies in
the tensor product by one of three methods: ``native`` (one factor of level
1, absorbed into complex coefficients), ``rows`` (every product of a row of
one operand with a row of the other) and ``schedule`` (the quasilinear
schedule of the smaller factor with coefficients in the larger). The script
measures two things for each pair of factor levels.

Signed methods
    The product of two split arrays with each available method. Operands in
    split form need no transform, so these tasks compare the methods
    themselves; every task also performs the coordinatewise product of the
    group components, which is the same for all of them.

Full product
    The ``split`` algorithm, with the method that ``"auto"`` selects,
    against the ``table`` algorithm on loop-basis arrays, for the pairs
    whose total level does not exceed ``--max-table-level``.

The corpus, blocks, statistics, and verification follow ``benchmark.py``; the
pair schedule is indexed by the total level ``n + m``.

Example:

    python3 benchmarks/central.py --output results/central.json --cpu 2
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The shared driver is imported first: it limits numerical libraries to one
# thread, which must happen before NumPy loads
try:
    from . import benchmark as common
except ImportError:
    import benchmark as common

import numpy as np  # noqa: E402

import pycdloop as cdl  # noqa: E402

DEFAULT_FACTORS = "1x4,1x8,2x2,3x3,4x4,5x5,6x6,7x7,2x6,3x7,4x8"
DEFAULT_PAIRS = "1-6:64,7-8:16,9-12:4,13-14:2"
METHODS = ("native", "rows", "schedule")
FULL = ("split/batch", "table/batch")
MAX_TOTAL_LEVEL = 14


def parse_factors(text: str) -> list[tuple[int, int]]:
    """Parse factor levels given as ``NxM`` items."""

    factors: list[tuple[int, int]] = []
    for item in text.split(","):
        try:
            n, m = (int(part) for part in item.lower().split("x"))
        except ValueError as error:
            raise ValueError(f"invalid factor specification: {item!r}") from error
        if n < 1 or m < 1 or n + m > MAX_TOTAL_LEVEL:
            raise ValueError(f"factor levels must be positive with total at most "
                             f"{MAX_TOTAL_LEVEL}: {item!r}")
        if (n, m) not in factors:
            factors.append((n, m))
    return factors


def parse_arguments(argv: Sequence[str] | None = None) -> Any:
    """Parse and validate the command line of the central protocol."""

    parser = common.base_parser(__doc__, None, DEFAULT_PAIRS)
    parser.add_argument("--factors", default=DEFAULT_FACTORS,
                        help="comma-separated NxM factor levels (default: %(default)s)")
    parser.add_argument("--max-table-level", type=int, default=10,
                        help="largest total level at which the table algorithm is timed")
    arguments = common.finish_arguments(parser, parser.parse_args(argv))
    if not 0 <= arguments.max_table_level <= common.MAX_LEVEL:
        parser.error(f"max-table-level must lie between 0 and {common.MAX_LEVEL}")
    try:
        arguments.factors = parse_factors(arguments.factors)
        for n, m in arguments.factors:
            common.pairs_for(arguments.pair_schedule, n + m)
    except ValueError as error:
        parser.error(str(error))
    return arguments


def central_tasks(algebra: Any, operands: np.ndarray, with_table: bool) -> list[common.Task]:
    """Build the signed-method tasks of one pair and, if affordable, the full product."""

    pairs = operands.shape[1]
    x, y = algebra.array(operands[0]), algebra.array(operands[1])
    s, t = x.split(), y.split()

    def signed(name: str) -> common.Task:
        """Return the task that multiplies the split arrays with one method."""

        return common.Task(
            f"{name}/signed", lambda: algebra.multiply(s, t, signed_algorithm=name), pairs,
            lambda: algebra.multiply(s, t, signed_algorithm=name).to_loop().coefficients)

    tasks = [signed(name) for name in METHODS if name in algebra.signed_algorithms]
    if with_table:
        tasks.append(common.Task("split/batch", lambda: x * y, pairs,
                                 lambda: (x * y).coefficients))
        tasks.append(common.Task(
            "table/batch", lambda: algebra.multiply(x, y, algorithm="table"), pairs,
            lambda: algebra.multiply(x, y, algorithm="table").coefficients))
    return tasks


def cell_ratios(results: dict[str, Any]) -> dict[str, Any]:
    """Return the paired ratios reported for one pair of factors."""

    return {
        "schedule_over_rows": common.task_ratio(results, "schedule/signed", "rows/signed"),
        "native_over_rows": common.task_ratio(results, "native/signed", "rows/signed"),
        "table_over_split": common.task_ratio(results, "table/batch", "split/batch"),
    }


def write_markdown(path: Path, document: dict[str, Any]) -> None:
    """Write the Markdown report of the central protocol."""

    lines = common.report_header(
        "pycdloop multiplication benchmark: central products over R", document)
    lines += [
        "A cell is labelled by the factor levels `n x m` of `Q_n * Q_m`. `D` is the "
        "dimension `2^(n+m)` of the tensor product of the two Cayley–Dickson algebras, "
        "and the loop algebra has dimension `N = 2D`.", "",
        "## Signed-component methods (ns per product)", "",
        "Products of two split arrays, which need no transform. `fastest mean` names the "
        "lowest mean, or a tie when the interval of the paired ratio of the two lowest "
        "means contains one. `auto` is the method that `\"auto\"` selects for the pair.", "",
        "| n x m | D | pairs | native | rows | schedule | fastest mean | auto |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for entry in document["results"]:
        tasks = entry["tasks"]
        cells = [common.format_ci(tasks[f"{name}/signed"]) if f"{name}/signed" in tasks
                 else "unavailable" for name in METHODS]
        lines.append(f"| {entry['label']} | {entry['quotient_order']} | {entry['pairs']} | "
                     + " | ".join(cells)
                     + f" | {entry['fastest_signed']} | {entry['default_signed_algorithm']} |")
    lines += ["", "## Paired ratios of the signed-component methods", "",
              "`schedule / rows` below 1 favours `schedule`; `native / rows` below 1 "
              "favours `native`.", "",
              "| n x m | outer level | schedule / rows | native / rows |",
              "| --- | ---: | --- | --- |"]
    for entry in document["results"]:
        ratios = entry["ratios"]
        native = (common.format_ratio(ratios["native_over_rows"])
                  if "native/signed" in entry["tasks"] else "unavailable")
        lines.append(f"| {entry['label']} | {min(entry['factors'])} | "
                     f"{common.format_ratio(ratios['schedule_over_rows'])} | {native} |")
    lines += ["", "## Full product (ns per product)", "",
              "`split` on loop-basis arrays, with the method that `\"auto\"` selects, against "
              "`table`. `table / split` above 1 favours `split`.", "",
              "| n x m | N | pairs | split | table | table / split | fastest mean |",
              "| --- | ---: | ---: | --- | --- | --- | --- |"]
    for entry in document["results"]:
        tasks = entry["tasks"]
        if "table/batch" not in tasks:
            continue
        lines.append(
            f"| {entry['label']} | {entry['dimension']} | {entry['pairs']} | "
            f"{common.format_ci(tasks['split/batch'])} | {common.format_ci(tasks['table/batch'])} | "
            f"{common.format_ratio(entry['ratios']['table_over_split'])} | "
            f"{entry['fastest_full']} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the central protocol and write its JSON document and Markdown report."""

    arguments = parse_arguments(argv)
    document = common.new_document(
        "central", arguments, argv, methods=list(METHODS),
        factors=[list(pair) for pair in arguments.factors],
        max_table_level=arguments.max_table_level)
    for n, m in arguments.factors:
        algebra = cdl.LoopAlgebra(cdl.CentralProduct(n, m))
        pairs = common.pairs_for(arguments.pair_schedule, n + m)
        shape = (2, pairs) + algebra.coefficient_shape
        operands, record = common.load_corpus(
            arguments.corpus_dir, f"central-n{n}-m{m}", "real", shape,
            [arguments.seed, n, m])
        document["corpus"][record["file"]] = record
        reference = common.reference_products((n, m), operands)
        tasks = central_tasks(algebra, operands, n + m <= arguments.max_table_level)
        results = common.run_cell(tasks, reference, operands, False, arguments)
        signed = [name for name in results if name.endswith("/signed")]
        document["results"].append({
            "label": f"{n} x {m}", "factors": [n, m], "level": n + m,
            "quotient_order": algebra.loop.quotient_order,
            "dimension": algebra.dimension, "pairs": pairs,
            "signed_algorithms": list(algebra.signed_algorithms),
            "default_signed_algorithm": algebra.default_signed_algorithm,
            "tasks": results, "ratios": cell_ratios(results),
            "fastest_signed": common.fastest_label(results, signed),
            "fastest_full": common.fastest_label(results, FULL),
        })
        print(f"{n} x {m}: " + ", ".join(
            f"{name} {common.format_ci(entry)}" for name, entry in results.items()),
            file=sys.stderr)
    document["verification"] = common.verification_totals(
        [entry["tasks"] for entry in document["results"]])
    common.write_json(arguments.output, document)
    write_markdown(arguments.output.with_suffix(".md"), document)
    return 1 if document["verification"]["failed_tasks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
