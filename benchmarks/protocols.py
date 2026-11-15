#!/usr/bin/env python3
"""Run the benchmark protocols of pycdloop, alone or one after another.

``real`` times the loop algebra of ``Q_n`` over R (``benchmark.py``),
``domains`` times it over C and finite fields (``domains.py``), ``central``
times central products over R (``central.py``), and ``all`` runs the three in
that order, each in its own process with its own JSON document and Markdown
report, and writes an index of them. ``--levels``, ``--fields``, and
``--factors`` are passed to the protocols that define them and otherwise
keep each protocol's default.

Example:

    python3 benchmarks/protocols.py --protocol all --output results/run.json --cpu 2
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from . import benchmark as common
except ImportError:
    import benchmark as common

PROTOCOLS = ("real", "domains", "central")
SCRIPTS = {"real": "benchmark.py", "domains": "domains.py", "central": "central.py"}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the options shared by the protocols."""

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=(*PROTOCOLS, "all"), default="real",
                        help="benchmark protocol (default: real)")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON result path; with all, the index of the three reports")
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-time-ms", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--corpus-dir", type=Path, default=Path("benchmark-corpora"))
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--levels", help="levels for real and domains (default: each protocol's own)")
    parser.add_argument("--fields", help="char:deg pairs for domains")
    parser.add_argument("--factors", help="NxM factor levels for central")
    arguments = parser.parse_args(argv)
    if arguments.blocks < 1 or arguments.warmup < 0:
        parser.error("blocks must be positive and warmup must be nonnegative")
    if not math.isfinite(arguments.min_time_ms) or arguments.min_time_ms < 0:
        parser.error("min-time-ms must be finite and nonnegative")
    if arguments.output.suffix.lower() == ".md":
        parser.error("output must not use a .md suffix; JSON and Markdown "
                     "paths must differ")
    return arguments


def commands(arguments: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Return one command per selected protocol, with distinct output files."""

    directory = Path(__file__).resolve().parent
    selected = PROTOCOLS if arguments.protocol == "all" else (arguments.protocol,)
    result = []
    for protocol in selected:
        output = (arguments.output.with_name(f"{arguments.output.stem}.{protocol}.json")
                  if arguments.protocol == "all" else arguments.output)
        command = [
            sys.executable, str(directory / SCRIPTS[protocol]),
            "--output", str(output),
            "--blocks", str(arguments.blocks),
            "--warmup", str(arguments.warmup),
            "--min-time-ms", f"{arguments.min_time_ms:g}",
            "--seed", str(arguments.seed),
            "--corpus-dir", str(arguments.corpus_dir),
        ]
        if arguments.cpu is not None:
            command += ["--cpu", str(arguments.cpu)]
        if arguments.levels is not None and protocol != "central":
            command += ["--levels", arguments.levels]
        if arguments.fields is not None and protocol == "domains":
            command += ["--fields", arguments.fields]
        if arguments.factors is not None and protocol == "central":
            command += ["--factors", arguments.factors]
        result.append((protocol, command))
    return result


def preflight(arguments: argparse.Namespace, selected: list[tuple[str, list[str]]]) -> None:
    """Refuse to overwrite earlier results before a long run starts."""

    paths = [arguments.output, arguments.output.with_suffix(".md")]
    for _, command in selected:
        output = Path(command[command.index("--output") + 1])
        paths += [output, output.with_suffix(".md")]
    for path in paths:
        if arguments.protocol == "all" and path.exists():
            raise ValueError(f"refusing to overwrite {path}")


def run(arguments: argparse.Namespace) -> int:
    """Run the selected protocols and, for all, write the index of their reports."""

    selected = commands(arguments)
    preflight(arguments, selected)
    document: dict[str, Any] = {"schema": "pycdloop-benchmark-suite-1", "status": "incomplete",
                                "protocol": arguments.protocol, "reports": []}
    for protocol, command in selected:
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
        output = Path(command[command.index("--output") + 1])
        document["reports"].append({
            "protocol": protocol, "json": output.name,
            "markdown": output.with_suffix(".md").name,
            "sha256": common.sha256_file(output), "command": command,
        })
    if arguments.protocol == "all":
        document["status"] = "ok"
        common.write_json(arguments.output, document)
        lines = [
            "# Benchmark protocols", "",
            "Each protocol has its own JSON observations and Markdown report.", "",
            "The real protocol times the loop algebra of a Cayley–Dickson loop over R, "
            "domains times it over C and finite fields, and central times the loop "
            "algebras of central products over R.", "",
            "| Protocol | Results | Raw data |", "|---|---|---|",
        ]
        for entry in document["reports"]:
            lines.append(f"| {entry['protocol']} | [Report]({entry['markdown']}) | "
                         f"[JSON]({entry['json']}) |")
        arguments.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the selected protocols."""

    arguments = parse_arguments(argv)
    try:
        return run(arguments)
    except ValueError as error:
        print(f"protocols.py: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
