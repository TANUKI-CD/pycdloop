#!/usr/bin/env python3
"""Run the stability study end to end and write a self-describing study directory.

The study measures the accuracy of the loop-algebra product against an exact
reference (stability.py), tabulates the results (stability_report.py) and draws
the charts (stability_plots.py). Published files sit at the top of the study
directory and are distributed with the repository; the raw results and the
operand corpus sit under raw/ and stay on the machine that ran the study.
SHA256SUMS lists the published files, and study.json records the hashes of the
raw files and of the measured sources.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from . import stability, stability_plots, stability_report
except ImportError:
    import stability
    import stability_plots
    import stability_report

ROOT = Path(__file__).resolve().parents[1]
RAW = "raw"
PUBLISHED = {
    "README.md": "This description, generated from the results.",
    "SHA256SUMS": "SHA-256 of every published file of the study.",
    "study.json": "The command, the host and software, the hashes of the measured sources and of the raw files, and every step with its duration.",
    "key-figures.json": "The figures quoted in docs/stability.md, computed from the results.",
    "accuracy-tables.md": "The accuracy tables shown in docs/stability.md.",
    "accuracy-summary.csv": "Error statistics for every domain, dimension, family, chain length and method.",
    "PLOTS.md": "Description of the charts and the command that drew them.",
    "plot-data.csv": "Every plotted value in tabular form.",
}
STEPS = {
    "accuracy": "Measure every error against the exact product.",
    "tables": "Tabulate the accuracy results.",
    "plots": "Draw the charts.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured_package() -> Path:
    """Return the directory of the package that the study imports and measures."""
    import pycdloop

    return Path(pycdloop.__file__).resolve().parent


def source_hashes(package: Path) -> dict[str, str]:
    """Return the SHA-256 of every file of the measured package.

    The files are named as in a checkout, under ``src/pycdloop``, wherever
    the package was imported from.
    """
    files = sorted(p for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {f"src/pycdloop/{p.relative_to(package).as_posix()}": sha256(p) for p in files}


def source_record() -> dict[str, Any]:
    """Describe the measured package: where it was imported from, and its files."""
    package = measured_package()
    origin = "checkout" if package == ROOT / "src" / "pycdloop" else "installed"
    return {"package": origin, "files_sha256": source_hashes(package)}


def published_files(directory: Path) -> list[Path]:
    """Return the files of a study directory that are distributed."""
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.name != "SHA256SUMS" and RAW not in p.relative_to(directory).parts)


def write_checksums(directory: Path) -> None:
    lines = [f"{sha256(p)}  {p.relative_to(directory).as_posix()}" for p in published_files(directory)]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def command_line(args: argparse.Namespace, name: str) -> str:
    """Return the command that reproduces the study, relative to the repository root."""
    parts = ["python3", "benchmarks/stability_study.py", "--profile", args.profile, "--seed", str(args.seed),
             "--jobs", str(args.jobs), "--output", f"benchmark-results/{name}"]
    defaults = stability.parse_arguments(["--output", "unused"])
    for option in ("real_levels", "complex_levels", "real_chain_levels", "complex_chain_levels", "precision"):
        if getattr(args, option) != getattr(defaults, option):
            parts.extend([f"--{option.replace('_', '-')}", str(getattr(args, option))])
    if args.update_docs:
        parts.append("--update-docs")
    return " ".join(parts)


def readme(study: dict[str, Any], figures: dict[str, Any], files: list[str]) -> str:
    """Return the description of a study directory."""
    environment = study["environment"]
    date = study["date"]
    lines = [
        f"# Stability study, {date}", "",
        "This directory holds the measurements behind [docs/stability.md](../../docs/stability.md), "
        "which defines the error measures and interprets the results.", "",
        "## What was measured", "",
        "The floating-point accuracy of the product of the loop algebra of a Cayley–Dickson loop, "
        "over the real and the complex numbers, against the exact product of the stored operands.", "",
        "- **S, the split algorithm:** three Walsh–Hadamard transforms, one coordinatewise product and one "
        "Cayley–Dickson product.",
        "- **T, the table algorithm:** the direct evaluation of the definition in the loop basis.",
        "- **L and R, chains of products:** a left-nested product of 2, 4 or 8 operands, kept in split form "
        "with one conversion at the end (L), or returned to the loop basis after every product (R).", "",
        f"The {study['profile']} profile measured {figures['pairs']:,} operand pairs and {figures['chains']:,} chains. "
        "The reference converts every binary64 coefficient to an integer times a power of two, sums the "
        "product from its definition in arbitrary-precision integers, and evaluates norms and errors with "
        f"MPFR at {study['precision_bits']} bits.", "",
        "## Host and software", "", "| | |", "|:---|:---|",
        f"| CPU | {environment['cpu_model']} |",
        f"| Operating system | {environment['platform']} |",
        f"| Python | {environment['python'].split()[0]} |",
        f"| NumPy, gmpy2, MPFR | {environment['numpy']}, {environment['gmpy2']}, {environment['mpfr']} |",
        f"| pyfastcd | {environment['pyfastcd']} |",
        f"| pycdloop | {environment['pycdloop']} |",
        f"| Matplotlib | {environment['matplotlib']} |",
        f"| Worker processes | {study['jobs']} |", "",
        "## How it was produced", "", "```sh", study["command"], "```", "",
        "| Step | What it does | Duration |", "|:---|:---|---:|",
    ]
    lines.extend(f"| {name} | {STEPS[name]} | {seconds:.0f} s |" for name, seconds in study["steps"].items())
    lines.extend(["", f"The operands derive from the root seed {study['seed']}. study.json records the SHA-256 of "
                  "every file of the measured package.", "", "## Files", "", "| File | Contents |", "|:---|:---|"])
    lines.extend(f"| {name} | {text} |" for name, text in PUBLISHED.items())
    charts = sorted({name.rsplit(".", 1)[0] for name in files if name.endswith(".png")})
    lines.append(f"| {', '.join(charts)} | The charts, each as PNG, SVG and PDF. |")
    lines.extend(["", f"The raw per-case results and the operand corpus are written to `{RAW}/`. They are not "
                  "distributed; study.json records their SHA-256, and `tools/check_archives.py --full` "
                  "checks them where they are present.", ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    """Run every step of the study and return the study directory."""
    import matplotlib

    directory = args.output.resolve()
    raw = directory / RAW
    raw.mkdir(parents=True, exist_ok=True)
    steps: dict[str, float] = {}

    start = time.perf_counter()
    results = raw / "accuracy.json"
    arguments = ["--output", str(results), "--profile", args.profile, "--seed", str(args.seed), "--jobs", str(args.jobs),
                 "--precision", str(args.precision), "--real-levels", str(args.real_levels),
                 "--complex-levels", str(args.complex_levels), "--real-chain-levels", str(args.real_chain_levels),
                 "--complex-chain-levels", str(args.complex_chain_levels)]
    document = stability.run(stability.parse_arguments(arguments))
    steps["accuracy"] = time.perf_counter() - start

    start = time.perf_counter()
    tables = stability_report.write(document, directory)
    figures = stability_report.key_figures(document)
    if args.update_docs and not stability_report.update_document(args.document, tables):
        print(f"{args.document} has no accuracy section; left unchanged")
    steps["tables"] = time.perf_counter() - start

    start = time.perf_counter()
    stability_plots.main([str(directory)])
    steps["plots"] = time.perf_counter() - start

    study = {
        "schema": "pycdloop-stability-study-v1",
        "date": args.date, "profile": document["profile"], "seed": document["seed"],
        "precision_bits": document["precision_bits"], "jobs": document["jobs"], "levels": document["levels"],
        "command": command_line(args, directory.name),
        "environment": {**document["environment"], "matplotlib": matplotlib.__version__},
        "signed_algorithms": document["signed_algorithms"],
        "oracle_validation": document["oracle_validation"], "precision_checks": document["precision_checks"],
        "corpus": document["corpus"], "steps": steps,
        "source": source_record(),
        "tools_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("stability*.py"))},
        "raw_sha256": {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(raw.iterdir()) if p.is_file()},
    }
    (directory / "study.json").write_text(json.dumps(study, indent=2, allow_nan=False) + "\n")
    names = [p.relative_to(directory).as_posix() for p in published_files(directory)]
    (directory / "README.md").write_text(readme(study, figures, names))
    write_checksums(directory)
    return directory


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    today = datetime.date.today()
    defaults = stability.parse_arguments(["--output", "unused"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results" / f"stability-{today:%Y%m%d}",
                        help="study directory")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--jobs", type=int, default=defaults.jobs, help="worker processes of the accuracy step")
    parser.add_argument("--precision", type=int, default=defaults.precision)
    parser.add_argument("--real-levels", type=int, default=defaults.real_levels)
    parser.add_argument("--complex-levels", type=int, default=defaults.complex_levels)
    parser.add_argument("--real-chain-levels", type=int, default=defaults.real_chain_levels)
    parser.add_argument("--complex-chain-levels", type=int, default=defaults.complex_chain_levels)
    parser.add_argument("--update-docs", action="store_true", help="rewrite the accuracy section of the document")
    parser.add_argument("--document", type=Path, default=ROOT / "docs" / "stability.md")
    args = parser.parse_args(argv)
    args.date = today.isoformat()
    return args


if __name__ == "__main__":
    print(run(parse_arguments(sys.argv[1:])))
