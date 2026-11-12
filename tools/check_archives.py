#!/usr/bin/env python3
"""Check the stability studies and the tables shown in docs/stability.md.

For every stability-*/ directory under benchmark-results/ with a SHA256SUMS
file:

- every published file matches its recorded SHA-256, and every published file
  present is recorded; files under raw/ are not distributed and may be absent;
- study.json is present. Differences between the working tree and the sources
  it records are reported but are not failures.

When docs/stability.md has a delimited accuracy section, the section must equal
the accuracy tables of the newest study. With --full, the raw files must be
present and match the hashes in study.json, and the tables are regenerated
from the raw results (this needs NumPy). The default mode uses only the
standard library. Nothing is ever written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = "raw"
BEGIN = "<!-- BEGIN STABILITY ACCURACY -->"
END = "<!-- END STABILITY ACCURACY -->"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checksums(directory: Path) -> dict[str, str]:
    """Return the recorded hashes of a study, keyed by relative path."""
    entries = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.partition("  ")
        require(
            len(digest) == 64 and bool(name),
            f"malformed line in {directory}/SHA256SUMS",
        )
        entries[name] = digest
    return entries


def section(text: str) -> str | None:
    """Return the delimited accuracy section of a document, or None without markers."""
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        return None
    if text.index(BEGIN) > text.index(END):
        return None
    return text[text.index(BEGIN) + len(BEGIN) : text.index(END)].strip("\n")


def source_differences(root: Path, study: dict[str, Any]) -> list[str]:
    """List the package files that differ from the ones a study measured."""
    recorded: dict[str, str] = study["source"]["files_sha256"]
    package = root / "src" / "pycdloop"
    working = {
        p.relative_to(root).as_posix(): p
        for p in package.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    return sorted(
        name
        for name in set(working) | set(recorded)
        if name not in working
        or name not in recorded
        or sha256(working[name]) != recorded[name]
    )


def regenerated_tables(root: Path, directory: Path) -> str:
    """Render the accuracy tables again from the raw results."""
    sys.path.insert(0, str(root / "benchmarks"))
    try:
        import stability_report
    finally:
        sys.path.pop(0)
    document = json.loads((directory / RAW / "accuracy.json").read_text())
    tables: str = stability_report.render(document)
    return tables


def check_archive(root: Path, directory: Path, full: bool) -> dict[str, Any]:
    entries = read_checksums(directory)
    present = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file()
        and p.name != "SHA256SUMS"
        and RAW not in p.relative_to(directory).parts
    }
    unrecorded = sorted(present - set(entries))
    require(
        not unrecorded, f"files not recorded in {directory}/SHA256SUMS: {unrecorded}"
    )
    for name, digest in entries.items():
        path = directory / name
        require(path.is_file(), f"missing file: {path}")
        require(sha256(path) == digest, f"checksum differs: {path}")
    require("study.json" in entries, f"{directory} has no study.json")
    require("accuracy-tables.md" in entries, f"{directory} has no accuracy-tables.md")
    study = json.loads((directory / "study.json").read_text())
    raw: dict[str, str] = study["raw_sha256"]
    present_raw = [name for name in raw if (directory / name).is_file()]
    if full:
        require(len(present_raw) == len(raw), f"raw files are missing from {directory}")
    for name in present_raw:
        require(
            sha256(directory / name) == raw[name],
            f"checksum differs: {directory / name}",
        )
    report = {
        "directory": str(directory),
        "published_files": len(entries),
        "raw_files_present": len(present_raw),
        "raw_files_absent": len(raw) - len(present_raw),
        "profile": study["profile"],
        "date": study["date"],
        "working_tree_differs_from_measured_sources": source_differences(root, study),
    }
    if full:
        tables = (directory / "accuracy-tables.md").read_text()
        require(
            regenerated_tables(root, directory) == tables,
            f"the tables of {directory} differ from the ones its raw results give",
        )
        report["tables_regenerated"] = True
    return report


def check(root: Path, results: Path, document: Path, full: bool) -> dict[str, Any]:
    """Check every study under a results directory and the documented tables."""
    studies = sorted(
        p
        for p in results.glob("stability-*")
        if p.is_dir() and (p / "SHA256SUMS").is_file()
    )
    report: dict[str, Any] = {
        "studies": [check_archive(root, p, full) for p in studies]
    }
    if not studies:
        report["note"] = f"no stability studies under {results}"
        return report
    newest = studies[-1]
    text = document.read_text() if document.is_file() else None
    shown = None if text is None else section(text)
    if shown is None:
        report["document"] = f"{document} has no delimited accuracy section"
        return report
    tables = (newest / "accuracy-tables.md").read_text().strip("\n")
    require(
        shown == tables,
        f"the accuracy section of {document} differs from the tables of {newest}",
    )
    report["document"] = f"{document} matches {newest.name}"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "benchmark-results",
        help="directory holding the stability-*/ studies",
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=ROOT / "docs" / "stability.md",
        help="document whose accuracy section is compared",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="also check the raw files and regenerate the tables",
    )
    args = parser.parse_args(argv)
    try:
        report = check(ROOT, args.results, args.document, args.full)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
