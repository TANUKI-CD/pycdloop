#!/usr/bin/env python3
"""Time loop-algebra multiplication over the complex numbers and finite fields.

For each coefficient field and doubling level the script times the ``split``
and ``table`` algorithms of the loop algebra of ``Q_n`` in batch and single
mode, with the corpus, blocks, and statistics of ``benchmark.py``. Fields are
given as ``char:deg`` pairs: ``0:2`` is C, ``p:1`` a prime field and ``p:d``
the extension of degree ``d`` in its default presentation. In characteristic
2 the algebra has no split form, and the tasks named ``split`` time the
``lift`` algorithm, which applies the split algorithm to integer lifts of the
coefficients.

Finite-field products are verified exactly against an independent schoolbook
implementation of the field arithmetic and of the definition of the loop
algebra product; complex products are verified against the definition with
the norm-scaled tolerance of the real protocol.

Example:

    python3 benchmarks/domains.py --output results/domains.json --cpu 2
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
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

DEFAULT_FIELDS = "0:2,2:1,2:2,2:8,3:1,13:1,65521:1,2147483647:1,3:5,13:5"
DEFAULT_LEVELS = "2,4,6,8"
DEFAULT_PAIRS = "1-4:64,5-6:16,7-8:4"
TASKS = ("split/batch", "table/batch", "split/single", "table/single")
MAX_LEVEL = 10


def parse_fields(text: str) -> list[tuple[int, int]]:
    """Parse ``char:deg`` pairs and check each field with pyfastcd's rules."""

    fields: list[tuple[int, int]] = []
    for item in text.split(","):
        try:
            char, deg = (int(part) for part in item.split(":"))
        except ValueError as error:
            raise ValueError(f"invalid field specification: {item!r}") from error
        if char == 0 and deg != 2:
            raise ValueError("characteristic 0 means C here, written 0:2; "
                             "benchmark.py covers real coefficients")
        try:
            cdl.LoopAlgebra(cdl.CDLoop(1), char=char, deg=deg)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported field {item!r}: {error}") from error
        if (char, deg) not in fields:
            fields.append((char, deg))
    return fields


def field_label(char: int, deg: int) -> str:
    """Name a coefficient field in the reports."""

    if char == 0:
        return "C"
    return f"F_{char}" if deg == 1 else f"F_({char}^{deg})"


def parse_arguments(argv: Sequence[str] | None = None) -> Any:
    """Parse and validate the command line of the domains protocol."""

    parser = common.base_parser(__doc__, DEFAULT_LEVELS, DEFAULT_PAIRS)
    parser.add_argument("--fields", default=DEFAULT_FIELDS,
                        help="comma-separated char:deg pairs; C is 0:2 (default: %(default)s)")
    arguments = common.finish_arguments(parser, parser.parse_args(argv), MAX_LEVEL)
    try:
        arguments.fields = parse_fields(arguments.fields)
    except ValueError as error:
        parser.error(str(error))
    return arguments


def field_multiply(char: int, modulus: Sequence[int]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return schoolbook multiplication in ``F_p[t] / (modulus)`` on residue arrays.

    Elements hold their polynomial coordinates on the last axis, constant
    coefficient first, and *modulus* is monic with its constant coefficient
    first. Products and reductions stay below ``2**63`` because every
    residue is below ``2**31``.
    """

    degree = len(modulus) - 1
    p = np.uint64(char)
    # Subtracting c * t^k is adding (p - c) * t^k
    folds = [np.uint64((char - int(c)) % char) for c in modulus[:degree]]

    def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two polynomials and reduce by the modulus."""

        left, right = a.astype(np.uint64), b.astype(np.uint64)
        shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        terms = np.zeros(shape + (2 * degree - 1,), dtype=np.uint64)
        for i in range(degree):
            for j in range(degree):
                terms[..., i + j] = (terms[..., i + j] + left[..., i] * right[..., j]) % p
        for k in range(2 * degree - 2, degree - 1, -1):
            for j in range(degree):
                place = k - degree + j
                terms[..., place] = (terms[..., place] + folds[j] * terms[..., k]) % p
        return terms[..., :degree]

    return multiply


def reference_arithmetic(algebra: Any) -> tuple[Callable[..., np.ndarray], int | None, int]:
    """Return the coefficient product, modulus, and budget of the reference."""

    field = algebra.field
    if field is None:
        return np.multiply, None, common.REFERENCE_BUDGET
    if field.deg == 1:
        p = np.uint64(field.char)
        return (lambda a, b: a.astype(np.uint64) * b % p), field.char, common.REFERENCE_BUDGET
    multiply = field_multiply(field.char, field.modulus)
    return multiply, field.char, common.REFERENCE_BUDGET // (2 * field.deg)


def ratios(results: dict[str, Any]) -> dict[str, Any]:
    """Return the paired table/split ratios of one cell."""

    return {mode: common.task_ratio(results, f"table/{mode}", f"split/{mode}")
            for mode in common.MODES}


def summary_table(document: dict[str, Any], mode: str) -> list[str]:
    """Render the split means of one mode with the table/split ratio in parentheses."""

    levels = document["protocol"]["levels"]
    heads = [f"N = {2 << level}" for level in levels]
    lines = ["| Field | " + " | ".join(heads) + " |", "| --- | " + " | ".join("---:" for _ in heads) + " |"]
    for char, deg in document["protocol"]["fields"]:
        row = [field_label(char, deg)]
        for level in levels:
            entry = next(e for e in document["results"]
                         if (e["char"], e["deg"], e["level"]) == (char, deg, level))
            split = entry["tasks"][f"split/{mode}"]
            ratio = entry["ratios"][mode]
            if split["status"] != "ok":
                row.append("FAILED")
            elif ratio is None:
                row.append(common.format_ns(split["summary"]["mean"]))
            else:
                row.append(f"{common.format_ns(split['summary']['mean'])} "
                           f"({ratio['geometric_mean']:.3g}x)")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def detail_table(document: dict[str, Any], mode: str) -> list[str]:
    """Render every cell of one mode with its intervals."""

    lines = ["| Field | N | n | pairs | split | table | table / split | fastest mean |",
             "| --- | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for entry in document["results"]:
        tasks = entry["tasks"]
        lines.append(
            f"| {entry['field']} | {entry['dimension']} | {entry['level']} | {entry['pairs']} | "
            f"{common.format_ci(tasks[f'split/{mode}'])} | "
            f"{common.format_ci(tasks[f'table/{mode}'])} | "
            f"{common.format_ratio(entry['ratios'][mode])} | {entry['fastest'][mode]} |")
    return lines


def write_markdown(path: Path, document: dict[str, Any]) -> None:
    """Write the Markdown report of the domains protocol."""

    lines = common.report_header(
        "pycdloop multiplication benchmark: complex and finite-field coefficients", document)
    lines += [
        "`split` multiplies through the decomposition of the loop algebra and `table` "
        "evaluates the definition. `N` is the dimension of the loop algebra of `Q_n`, "
        "which is `2^(n+1)`. Finite-field products are verified exactly against schoolbook "
        "field arithmetic; complex products are verified with a norm-scaled tolerance.", "",
    ]
    lines += [
        "Over the fields of characteristic 2 the algebra has no split form, and the "
        "`split` columns give the `lift` algorithm: the split algorithm applied to the "
        "coefficients read as integers over a large prime field, followed by reduction.", "",
    ]
    for mode in common.MODES:
        lines += [f"## {mode.capitalize()} products: split ns per product (table / split)", "",
                  "A ratio above 1 favours `split`.", ""]
        lines += summary_table(document, mode) + [""]
    for mode in common.MODES:
        lines += [f"## {mode.capitalize()} products with intervals (ns per product)", ""]
        lines += detail_table(document, mode) + [""]
    path.write_text("\n".join(lines).rstrip() + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the domains protocol and write its JSON document and Markdown report."""

    arguments = parse_arguments(argv)
    document = common.new_document("domains", arguments, argv, tasks=list(TASKS),
                                   fields=[list(field) for field in arguments.fields])
    for char, deg in arguments.fields:
        label = field_label(char, deg)
        for level in arguments.levels:
            algebra = cdl.LoopAlgebra(cdl.CDLoop(level), char=char, deg=deg)
            pairs = common.pairs_for(arguments.pair_schedule, level)
            shape = (2, pairs) + algebra.coefficient_shape
            kind = "complex" if char == 0 else "finite"
            operands, record = common.load_corpus(
                arguments.corpus_dir, f"domains-c{char}-d{deg}-n{level}", kind, shape,
                [arguments.seed, char, deg, level], char)
            document["corpus"][record["file"]] = record
            multiply, modulus, budget = reference_arithmetic(algebra)
            reference = common.reference_products((level,), operands, multiply, modulus, budget)
            tasks = common.loop_tasks(algebra, operands, TASKS)
            results = common.run_cell(tasks, reference, operands, char != 0, arguments)
            document["results"].append({
                "field": label, "char": char, "deg": deg,
                "modulus": None if algebra.field is None else list(algebra.field.modulus),
                "level": level, "dimension": algebra.dimension, "pairs": pairs,
                "tasks": results, "ratios": ratios(results),
                "fastest": {mode: common.fastest_label(
                    results, [f"{name}/{mode}" for name in common.ALGORITHMS])
                    for mode in common.MODES},
            })
            print(f"{label} level {level}: " + ", ".join(
                f"{name} {common.format_ci(entry)}" for name, entry in results.items()),
                file=sys.stderr)
    document["verification"] = common.verification_totals(
        [entry["tasks"] for entry in document["results"]])
    common.write_json(arguments.output, document)
    write_markdown(arguments.output.with_suffix(".md"), document)
    return 1 if document["verification"]["failed_tasks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
