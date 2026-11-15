#!/usr/bin/env python3
"""Time loop-algebra multiplication in pycdloop over the real numbers.

The script generates (or reuses) one deterministic corpus of random operand
pairs per doubling level, then times the product of the loop algebra of the
Cayley--Dickson loop ``Q_n`` as the package ships it. Timing happens in
Python, with the garbage collector disabled inside each timed region, so the
call overhead of the package is part of every measurement.

Two algorithms are compared. ``split`` multiplies through the decomposition
of the algebra: Walsh--Hadamard transforms, one coordinatewise product, and
one Cayley--Dickson product. ``table`` evaluates the definition of the
product with a quadratic number of multiplications. Both are timed in two
modes over the same pairs: ``batch`` (one call on arrays holding the whole
corpus) and ``single`` (one call per pair). Two further tasks time the
product on operands that are already in split form, where no transform is
needed: ``splitform/batch`` on split arrays, and ``cached/single`` on
elements whose split forms are cached. The tasks ``forward/batch`` and
``inverse/batch`` time the conversions to and from split form, which gives
the share of a ``split/batch`` product spent in the transforms. For scale,
``pyfastcd/batch`` and ``pyfastcd/single`` time the Cayley--Dickson product
of the signed components alone, through pyfastcd, on its arrays and on its
elements.

Each block takes one observation of every task in a rotated order, so every
task occupies every position equally often when the block count is a
multiple of the task count. Block observations are summarized with means,
two-sided 95% Student-t intervals, and paired ratios. Every product of every
task is verified against an independent evaluation of the definition before
any timing; a failed check is reported as a failure, never as a timing.

Example:

    python3 benchmarks/benchmark.py --output results/real.json --cpu 2

``benchmarks/protocols.py`` runs this protocol together with the protocols
for other coefficient fields (``domains``) and central products
(``central``).
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# Numerical libraries are limited to one thread before NumPy loads, so that no
# idle worker threads of a linear-algebra backend compete with the timed code
for _variable in THREAD_VARIABLES:
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402
import pyfastcd  # noqa: E402

import pycdloop as cdl  # noqa: E402

SCHEMA = "pycdloop-benchmark-1"
ALGORITHMS = ("split", "table")
MODES = ("batch", "single")
TASKS = (
    "split/batch",
    "table/batch",
    "splitform/batch",
    "forward/batch",
    "inverse/batch",
    "pyfastcd/batch",
    "split/single",
    "table/single",
    "cached/single",
    "pyfastcd/single",
)
DEFAULT_LEVELS = "1-10"
# Pairs per level: fewer at the large levels, where one table product is slow
DEFAULT_PAIRS = "1-6:256,7-8:64,9-10:16"
MAX_LEVEL = 12
# Coefficients that the reference product may hold in one intermediate array
REFERENCE_BUDGET = 1 << 22

# Two-sided 95% Student-t critical values t(0.975, df) for df = 1..30, with the
# df = 30 value used for larger degrees of freedom, which is slightly conservative
T95 = (
    float("nan"), 12.7062047364, 4.30265272975, 3.18244630528, 2.77644510520,
    2.57058183564, 2.44691185114, 2.36462425101, 2.30600413503, 2.26215716285,
    2.22813885196, 2.20098516008, 2.17881282966, 2.16036865646, 2.14478668792,
    2.13144954556, 2.11990529922, 2.10981557783, 2.10092204024, 2.09302405441,
    2.08596344727, 2.07961384473, 2.07387306790, 2.06865761042, 2.06389856163,
    2.05953855275, 2.05552943864, 2.05183051648, 2.04840714180, 2.04522964213,
    2.04227245630,
)  # fmt: skip


@dataclass
class Task:
    """One timed operation over a corpus.

    ``run`` makes one pass over the corpus, which evaluates ``products``
    products. ``result`` returns the loop-basis coefficients of every
    product for verification, or is ``None`` for a stage that is not a
    product.
    """

    name: str
    run: Callable[[], Any]
    products: int
    result: Callable[[], np.ndarray] | None = None


# Arguments


def parse_levels(text: str, maximum: int = MAX_LEVEL) -> list[int]:
    """Parse levels given as a comma-separated list of integers and ranges."""

    levels: list[int] = []
    for item in text.split(","):
        low, dash, high = item.partition("-")
        try:
            first = int(low)
            last = int(high) if dash else first
        except ValueError as error:
            raise ValueError(f"invalid level specification: {item!r}") from error
        if first > last:
            raise ValueError(f"invalid level range: {item!r}")
        levels.extend(range(first, last + 1))
    if not levels or min(levels) < 1 or max(levels) > maximum:
        raise ValueError(f"levels must lie between 1 and {maximum}")
    return sorted(set(levels))


def parse_pairs(text: str) -> list[tuple[int, int, int]]:
    """Parse a pair schedule such as ``1-6:256,7-8:64`` or a single count."""

    schedule: list[tuple[int, int, int]] = []
    for item in text.split(","):
        span, colon, count = item.rpartition(":")
        try:
            pairs = int(count)
            if colon:
                low, dash, high = span.partition("-")
                first = int(low)
                last = int(high) if dash else first
            else:
                first, last = 0, 1 << 30
        except ValueError as error:
            raise ValueError(f"invalid pair schedule: {item!r}") from error
        if pairs < 1 or first > last:
            raise ValueError(f"invalid pair schedule: {item!r}")
        schedule.append((first, last, pairs))
    return schedule


def pairs_for(schedule: Sequence[tuple[int, int, int]], level: int) -> int:
    """Return the number of operand pairs that a schedule assigns to a level."""

    for first, last, pairs in schedule:
        if first <= level <= last:
            return pairs
    raise ValueError(f"the pair schedule does not cover level {level}")


def base_parser(description: str | None, levels: str | None,
                pairs: str) -> argparse.ArgumentParser:
    """Create a parser with the options shared by every protocol.

    A protocol whose cells are not indexed by one level passes ``None`` for
    *levels* and supplies its own option.
    """

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON result path; a Markdown report is written next to it")
    if levels is not None:
        parser.add_argument("--levels", default=levels,
                            help="doubling levels, as integers and ranges "
                                 "(default: %(default)s)")
    parser.add_argument("--pairs", default=pairs,
                        help="operand pairs per level, as LEVELS:COUNT items or one count "
                             "(default: %(default)s)")
    parser.add_argument("--blocks", type=int, default=24,
                        help="timing blocks; each yields one observation per task")
    parser.add_argument("--warmup", type=int, default=2,
                        help="untimed corpus passes before each observation")
    parser.add_argument("--min-time-ms", type=float, default=20.0,
                        help="minimum duration of one observation")
    parser.add_argument("--seed", type=int, default=20261111, help="master corpus seed")
    parser.add_argument("--corpus-dir", type=Path, default=Path("benchmark-corpora"),
                        help="directory holding the corpus files and manifest.json")
    parser.add_argument("--cpu", type=int, default=None,
                        help="pin the process to one logical CPU (Linux; default: not pinned)")
    return parser


def finish_arguments(parser: argparse.ArgumentParser, arguments: argparse.Namespace,
                     maximum: int = MAX_LEVEL) -> argparse.Namespace:
    """Validate the shared options before any work is done."""

    if arguments.blocks < 1 or arguments.warmup < 0:
        parser.error("blocks must be positive and warmup must be nonnegative")
    if not math.isfinite(arguments.min_time_ms) or arguments.min_time_ms < 0:
        parser.error("min-time-ms must be finite and nonnegative")
    if arguments.seed < 0:
        parser.error("seed must be nonnegative")
    if arguments.output.suffix.lower() == ".md":
        parser.error("output must not use a .md suffix; JSON and Markdown "
                     "paths must differ")
    try:
        arguments.pair_schedule = parse_pairs(arguments.pairs)
        if hasattr(arguments, "levels"):
            arguments.levels = parse_levels(arguments.levels, maximum)
            for level in arguments.levels:
                pairs_for(arguments.pair_schedule, level)
    except ValueError as error:
        parser.error(str(error))
    return arguments


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command line of the real protocol."""

    parser = base_parser(__doc__, DEFAULT_LEVELS, DEFAULT_PAIRS)
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help="comma-separated subset of the tasks (default: all)")
    arguments = finish_arguments(parser, parser.parse_args(argv))
    names = arguments.tasks.split(",")
    if not names or any(name not in TASKS for name in names):
        parser.error("tasks must be a comma-separated subset of " + ",".join(TASKS))
    arguments.tasks = tuple(name for name in TASKS if name in names)
    return arguments


# Provenance


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_state(cpu: int | None) -> dict[str, Any]:
    """Record frequency-scaling and SMT settings that affect timing, when exposed."""

    base = Path("/sys/devices/system/cpu")
    files: dict[str, Path | None] = {
        "scaling_governor": (base / f"cpu{cpu}" / "cpufreq" / "scaling_governor"
                             if cpu is not None else None),
        "intel_pstate_no_turbo": base / "intel_pstate" / "no_turbo",
        "cpufreq_boost": base / "cpufreq" / "boost",
        "smt_control": base / "smt" / "control",
    }
    state: dict[str, Any] = {}
    for key, path in files.items():
        try:
            state[key] = None if path is None else path.read_text().strip()
        except OSError:
            state[key] = None
    return state


def cpu_name() -> str:
    """Return the CPU model name, using /proc/cpuinfo on Linux."""

    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def git_state(root: Path) -> dict[str, Any]:
    """Return the commit and dirty flag of a checkout, or ``None`` values outside one."""

    def git(*options: str) -> str | None:
        """Run one git query, returning ``None`` when it fails."""

        try:
            result = subprocess.run(["git", "-C", str(root), *options], check=False,
                                    capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(git("status", "--porcelain"))}


def source_hashes() -> dict[str, str]:
    """Return the SHA-256 digest of every file of the installed package."""

    # The compiled module and its source are part of what is measured
    root = Path(cdl.__file__).resolve().parent
    files = sorted(path for path in root.iterdir() if path.is_file())
    return {path.name: sha256_file(path) for path in files}


def pin_cpu(cpu: int | None) -> list[int] | None:
    """Pin the process to one logical CPU where supported, and return its affinity."""

    if not hasattr(os, "sched_setaffinity"):
        return None
    if cpu is not None:
        try:
            os.sched_setaffinity(0, {cpu})
        except OSError as error:
            raise SystemExit(f"cannot pin to CPU {cpu}: {error}") from error
    return sorted(os.sched_getaffinity(0))


def environment(arguments: argparse.Namespace) -> dict[str, Any]:
    """Describe the interpreter, libraries, and machine of a run."""

    affinity = pin_cpu(arguments.cpu)
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "pyfastcd": pyfastcd.__version__,
        "pycdloop": cdl.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "cpu": arguments.cpu,
        "affinity": affinity,
        "load_average": load,
        "thread_limits": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "cpu_state": cpu_state(arguments.cpu),
        "git": git_state(Path(__file__).resolve().parents[1]),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def new_document(protocol: str, arguments: argparse.Namespace,
                 argv: Sequence[str] | None, **parameters: Any) -> dict[str, Any]:
    """Start a result document with the protocol parameters and provenance."""

    command = [Path(sys.argv[0]).name, *(sys.argv[1:] if argv is None else argv)]
    return {
        "schema": SCHEMA,
        "protocol": {
            "name": protocol,
            "levels": getattr(arguments, "levels", None),
            "pairs": arguments.pairs,
            "blocks": arguments.blocks,
            "warmup": arguments.warmup,
            "min_time_ms": arguments.min_time_ms,
            "seed": arguments.seed,
            "timer": "time.perf_counter_ns, garbage collector disabled while timing",
            **parameters,
        },
        "command": command,
        "environment": environment(arguments),
        "sources_sha256": source_hashes(),
        "corpus": {},
        "results": [],
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Write a result document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


# Corpus


def generate_operands(kind: str, shape: tuple[int, ...], seed: Sequence[int],
                      char: int = 0) -> np.ndarray:
    """Draw the operands of one corpus file from a recorded seed."""

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(seed))))
    if kind == "real":
        return rng.standard_normal(shape)
    if kind == "complex":
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.integers(0, char, size=shape, dtype=np.uint32)


def load_corpus(directory: Path, name: str, kind: str, shape: tuple[int, ...],
                seed: Sequence[int], char: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the operands of one corpus file and its manifest record.

    The array has shape ``(2, pairs) + coefficient_shape``: index 0 of the
    first axis holds the left operands and index 1 the right operands. A
    file is reused when its manifest record has the requested parameters
    and its recorded hash; a file whose hash differs from its record is an
    error, so a run never times inputs other than the ones it reports.
    """

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    path = directory / f"{name}.npy"
    wanted = {"file": path.name, "kind": kind, "shape": list(shape), "seed": list(seed),
              "char": char, "rng": "NumPy Generator(PCG64(SeedSequence(seed)))",
              "layout": "[operand=2][pair] + coefficient_shape, C order"}
    record = manifest.get(name)
    if record is not None and path.exists():
        if all(record.get(key) == value for key, value in wanted.items()):
            if sha256_file(path) != record.get("sha256"):
                raise SystemExit(f"corpus file {path} does not match its manifest record; "
                                 "delete it to regenerate the corpus")
            return np.load(path), record
    operands = generate_operands(kind, shape, seed, char)
    np.save(path, operands)
    record = {**wanted, "dtype": str(operands.dtype), "sha256": sha256_file(path),
              "bytes": path.stat().st_size}
    manifest[name] = record
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return operands, record


# Independent reference


@functools.cache
def sign_table(level: int) -> np.ndarray:
    """Derive the basis signs of one Cayley--Dickson algebra from the doubling formula.

    With (a, b)(c, d) = (ac - conj(d) b, da + b conj(c)), the table of
    dimension 2M is built from the table ``s`` of dimension M in four
    blocks; the off-diagonal blocks reverse the product, and conjugation
    changes the sign of every column but the first.
    """

    if level == 0:
        return np.ones((1, 1), dtype=np.int8)
    s = sign_table(level - 1)
    conjugation = -np.ones(s.shape[0], dtype=np.int8)
    conjugation[0] = 1
    return np.block([[s, s.T], [s * conjugation, -s.T * conjugation]])


def twist(levels: Sequence[int], p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the signs of products of basis elements of a central product.

    A flat index concatenates the binary digits of one index for each
    factor, first factor first, and the sign is the product of the factors'
    signs.
    """

    signs = np.ones(np.broadcast_shapes(p.shape, q.shape), dtype=np.int8)
    shift = sum(levels)
    for level in levels:
        shift -= level
        mask = (1 << level) - 1
        signs = signs * sign_table(level)[(p >> shift) & mask, (q >> shift) & mask]
    return signs


def reference_multiply(levels: Sequence[int], x: np.ndarray, y: np.ndarray,
                       multiply: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.multiply,
                       modulus: int | None = None,
                       budget: int = REFERENCE_BUDGET) -> np.ndarray:
    """Multiply two loop-algebra elements straight from the definition.

    ``x`` and ``y`` have shape ``(2, D) + tail``: row 0 holds the
    coefficients of the basis elements ``[e_t]`` and row 1 those of
    ``[-e_t]``. The product of ``[s e_p]`` and ``[t e_q]`` is the basis
    element ``[s t sigma(p, q) e_(p ^ q)]``, so each pair of coefficients is
    multiplied and added to the coefficient of that basis element. No
    pycdloop arithmetic is used. *multiply* is the coefficient product, with
    *modulus* the sums are reduced, and *budget* bounds the number of
    coefficient products formed at once.
    """

    size = 1 << sum(levels)
    indices = np.arange(size)
    out = np.zeros(x.shape, dtype=x.dtype if modulus is None else np.uint64)
    step = max(1, budget // size)
    for start in range(0, size, step):
        p = indices[start:start + step]
        # Column k of q holds the index with p ^ q = k
        q = p[:, None] ^ indices[None, :]
        positive = twist(levels, p[:, None], q) > 0
        positive = positive.reshape(positive.shape + (1,) * (x.ndim - 2))
        left_plus, left_minus = x[0][p][:, None], x[1][p][:, None]
        right_plus, right_minus = y[0][q], y[1][q]
        equal = multiply(left_plus, right_plus) + multiply(left_minus, right_minus)
        opposite = multiply(left_plus, right_minus) + multiply(left_minus, right_plus)
        out[0] += np.where(positive, equal, opposite).sum(axis=0)
        out[1] += np.where(positive, opposite, equal).sum(axis=0)
        if modulus is not None:
            out %= modulus
    return out.astype(x.dtype)


def reference_products(levels: Sequence[int], operands: np.ndarray,
                       multiply: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.multiply,
                       modulus: int | None = None,
                       budget: int = REFERENCE_BUDGET) -> np.ndarray:
    """Return the reference product of every operand pair of a corpus."""

    pairs = operands.shape[1]
    return np.stack([reference_multiply(levels, operands[0, i], operands[1, i],
                                        multiply, modulus, budget) for i in range(pairs)])


def check_products(computed: np.ndarray, reference: np.ndarray, operands: np.ndarray,
                   exact: bool) -> dict[str, Any]:
    """Compare computed products with the reference.

    Exact fields require equality. Over R and C every coordinate must
    satisfy ``|computed - reference| <= 128 * eps * D * ||x||_2 * ||y||_2``;
    the largest ratio of an error to its bound is recorded.
    """

    pairs = int(reference.shape[0])
    if computed.shape != reference.shape:
        return {"pairs": pairs, "passed": False, "reason": "shape mismatch"}
    if exact:
        return {"pairs": pairs, "passed": bool(np.array_equal(computed, reference))}
    flat = operands.reshape(2, pairs, -1)
    norms = np.linalg.norm(flat[0], axis=1) * np.linalg.norm(flat[1], axis=1)
    size = reference.shape[2]
    bound = 128.0 * np.finfo(np.float64).eps * size * norms
    error = np.abs(computed - reference).reshape(pairs, -1).max(axis=1)
    finite = bool(np.isfinite(computed).all())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bound > 0, error / bound, np.where(error > 0, np.inf, 0.0))
    return {"pairs": pairs, "passed": finite and bool((error <= bound).all()),
            "worst_error_over_bound": float(ratio.max()) if ratio.size else 0.0}


# Timing


def timed_passes(task: Task, passes: int) -> int:
    """Return the nanoseconds taken by *passes* passes, without garbage collection."""

    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        for _ in range(passes):
            task.run()
        return time.perf_counter_ns() - start
    finally:
        gc.enable()


def calibrate(task: Task, minimum_ns: float) -> int:
    """Return a number of passes whose timed duration reaches the minimum."""

    passes = 1
    while True:
        elapsed = timed_passes(task, passes)
        if elapsed >= minimum_ns or passes >= 1 << 24:
            return passes
        scale = 1.25 * minimum_ns / max(elapsed, 1)
        passes = max(passes + 1, min(passes * 16, math.ceil(passes * scale)))


def observe(task: Task, passes: int, warmup: int) -> float:
    """Return one observation of a task, in nanoseconds per product."""

    for _ in range(warmup):
        task.run()
    return timed_passes(task, passes) / (passes * task.products)


def run_cell(tasks: Sequence[Task], reference: np.ndarray, operands: np.ndarray,
             exact: bool, arguments: argparse.Namespace) -> dict[str, Any]:
    """Verify and time every task of one cell.

    Verification comes first and is outside every timer. A task that fails
    it is recorded as failed and is not timed. The remaining tasks are then
    observed once per block, in an order rotated from block to block.
    """

    results: dict[str, Any] = {}
    timed: list[Task] = []
    for task in tasks:
        entry: dict[str, Any] = {"task": task.name, "products_per_pass": task.products}
        if task.result is None:
            entry["verification"] = None
        else:
            entry["verification"] = check_products(task.result(), reference, operands, exact)
        if entry["verification"] is not None and not entry["verification"]["passed"]:
            entry["status"] = "failed"
        else:
            entry["status"] = "ok"
            entry["observations_ns_per_product"] = []
            timed.append(task)
        results[task.name] = entry
    minimum_ns = arguments.min_time_ms * 1e6
    for task in timed:
        for _ in range(arguments.warmup):
            task.run()
        results[task.name]["passes"] = calibrate(task, minimum_ns)
    for block in range(arguments.blocks):
        offset = block % len(timed) if timed else 0
        for task in timed[offset:] + timed[:offset]:
            entry = results[task.name]
            entry["observations_ns_per_product"].append(
                observe(task, entry["passes"], arguments.warmup))
    for task in timed:
        entry = results[task.name]
        entry["summary"] = summarize(entry["observations_ns_per_product"])
    return results


# Statistics


def summarize(values: list[float]) -> dict[str, Any]:
    """Return the mean, spread, and 95% Student-t interval of block observations."""

    count = len(values)
    mean = math.fsum(values) / count
    if count > 1:
        variance = math.fsum((v - mean) ** 2 for v in values) / (count - 1)
        sd = math.sqrt(variance)
        se = sd / math.sqrt(count)
        half = T95[min(count - 1, 30)] * se
        ci = [mean - half, mean + half]
    else:
        sd = se = None
        ci = [None, None]
    ordered = sorted(values)
    return {"count": count, "mean": mean, "sd": sd, "se": se, "ci95": ci,
            "median": ordered[count // 2] if count % 2 else
            0.5 * (ordered[count // 2 - 1] + ordered[count // 2]),
            "min": ordered[0], "max": ordered[-1]}


def paired_ratio(competitor: list[float], baseline: list[float]) -> dict[str, Any]:
    """Return the geometric mean of per-block ratios with its 95% interval."""

    logs = [math.log(c / b) for c, b in zip(competitor, baseline, strict=True)]
    stats = summarize(logs)
    wins = sum(1 for c, b in zip(competitor, baseline, strict=True) if c < b)
    losses = sum(1 for c, b in zip(competitor, baseline, strict=True) if c > b)
    return {"definition": "competitor_ns / baseline_ns per block, geometric mean",
            "paired_blocks": stats["count"],
            "geometric_mean": math.exp(stats["mean"]),
            "ci95": [None if v is None else math.exp(v) for v in stats["ci95"]],
            "competitor_faster_blocks": wins, "competitor_slower_blocks": losses}


def task_ratio(results: dict[str, Any], competitor: str, baseline: str) -> dict[str, Any] | None:
    """Return the paired ratio of two tasks, or ``None`` unless both were timed."""

    entries = [results.get(competitor), results.get(baseline)]
    if any(entry is None or entry["status"] != "ok" for entry in entries):
        return None
    return paired_ratio(entries[0]["observations_ns_per_product"],
                        entries[1]["observations_ns_per_product"])


def fastest_label(results: dict[str, Any], names: Sequence[str]) -> str:
    """Name the lowest verified mean, or a tie when the runner-up is not separable.

    Two tasks tie when the 95% interval of their paired per-block ratio
    contains one, so the label never claims a winner the data cannot
    support. Labels drop the mode suffix of the task names.
    """

    verified = sorted((results[name] for name in names
                       if name in results and results[name]["status"] == "ok"),
                      key=lambda entry: entry["summary"]["mean"])
    labels = [entry["task"].split("/")[0] for entry in verified]
    if not verified:
        return "none verified"
    if len(verified) == 1:
        return labels[0]
    ratio = paired_ratio(verified[0]["observations_ns_per_product"],
                         verified[1]["observations_ns_per_product"])
    lo, hi = ratio["ci95"]
    if lo is not None and lo <= 1.0 <= hi:
        return f"tie ({labels[0]}, {labels[1]})"
    return labels[0]


def format_ns(value: float) -> str:
    """Format nanoseconds with three or four significant digits."""

    return f"{value:.2f}" if value < 100 else f"{value:.1f}" if value < 10000 else f"{value:.0f}"


def format_ci(entry: dict[str, Any] | None) -> str:
    """Render one timing cell as ``mean [lo, hi]``, or the reason it is absent."""

    if entry is None:
        return "not timed"
    if entry["status"] != "ok":
        return "FAILED"
    summary = entry["summary"]
    lo, hi = summary["ci95"]
    if lo is None:
        return format_ns(summary["mean"])
    return f"{format_ns(summary['mean'])} [{format_ns(lo)}, {format_ns(hi)}]"


def format_ratio(ratio: dict[str, Any] | None) -> str:
    """Render one paired ratio as ``mean [lo, hi]``."""

    if ratio is None:
        return "not timed"
    lo, hi = ratio["ci95"]
    interval = "" if lo is None else f" [{lo:.3g}, {hi:.3g}]"
    return f"{ratio['geometric_mean']:.3g}{interval}"


def verification_totals(cells: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count the verified products and failed tasks of a run."""

    checked = failed = 0
    for cell in cells:
        for entry in cell.values():
            record = entry.get("verification")
            if record is None:
                continue
            checked += record["pairs"]
            failed += 0 if record["passed"] else 1
    return {"products_checked": checked, "failed_tasks": failed}


def report_header(title: str, document: dict[str, Any]) -> list[str]:
    """Render the heading, host line, and verification line of a report."""

    host = document["environment"]
    protocol = document["protocol"]
    totals = document["verification"]
    load = host["load_average"]
    lines = [
        f"# {title}", "",
        f"Host: {host['processor']} ({host['machine']}); Python {host['python']}, "
        f"NumPy {host['numpy']}, pyfastcd {host['pyfastcd']}, pycdloop {host['pycdloop']}. "
        + ("Pinned to logical CPU " + str(host["cpu"]) if host["cpu"] is not None
           else "Not pinned to a CPU")
        + ("" if load is None else f"; load average at start {load[0]:.2f}") + ".", "",
        f"{protocol['blocks']} blocks; each cell is the mean over blocks of nanoseconds "
        "per product with a two-sided 95% Student-t interval. Timing is in Python "
        "with the garbage collector disabled, after "
        f"{protocol['warmup']} untimed passes, for at least {protocol['min_time_ms']:g} ms "
        "per observation, and includes the call overhead of the package. A ratio is the "
        "geometric mean of per-block ratios with its 95% interval.", "",
    ]
    if totals["failed_tasks"]:
        lines += [f"**Verification failed for {totals['failed_tasks']} task(s); "
                  "their cells read FAILED and carry no timing.**", ""]
    else:
        lines += [f"All {totals['products_checked']} verified products agree with an "
                  "independent evaluation of the definition.", ""]
    return lines


# The real protocol


def loop_tasks(algebra: Any, operands: np.ndarray, names: Sequence[str]) -> list[Task]:
    """Build the selected tasks of one loop algebra, over any coefficient field."""

    pairs = operands.shape[1]
    x, y = algebra.array(operands[0]), algebra.array(operands[1])
    singles = [(algebra.array(operands[0, i:i + 1]), algebra.array(operands[1, i:i + 1]))
               for i in range(pairs)]
    # In characteristic 2 the algebra has no split form: the tasks named
    # "split" time the "lift" algorithm there, which is the split algorithm
    # applied to integer lifts of the coefficients
    binary = algebra.char == 2
    fast = "lift" if binary else "split"

    def single(algorithm: str) -> Callable[[], Any]:
        """Return a pass that makes one multiply call per pair."""

        def run() -> None:
            """Multiply every pair with one call each."""

            for a, b in singles:
                algebra.multiply(a, b, algorithm=algorithm)

        return run

    def single_result(algorithm: str) -> Callable[[], np.ndarray]:
        """Return the products of the single-call pass for verification."""

        return lambda: np.concatenate(
            [algebra.multiply(a, b, algorithm=algorithm).coefficients for a, b in singles])

    available = {
        "split/batch": Task("split/batch", lambda: algebra.multiply(x, y, algorithm=fast),
                            pairs, lambda: algebra.multiply(x, y, algorithm=fast).coefficients),
        "table/batch": Task("table/batch", lambda: algebra.multiply(x, y, algorithm="table"),
                            pairs, lambda: algebra.multiply(x, y, algorithm="table").coefficients),
        "split/single": Task("split/single", single(fast), pairs, single_result(fast)),
        "table/single": Task("table/single", single("table"), pairs, single_result("table")),
    }
    if binary:
        return [available[name] for name in names if name in available]

    s, t = x.split(), y.split()
    product = s * t
    elements = [(algebra(operands[0, i]), algebra(operands[1, i])) for i in range(pairs)]
    for a, b in elements:
        # Reading the spectrum derives and keeps the split form of an element
        a.spectrum()
        b.spectrum()

    def cached() -> None:
        """Multiply every pair of elements whose split forms are cached."""

        for a, b in elements:
            a * b

    def forward() -> None:
        """Convert both operand arrays to split form."""

        x.split()
        y.split()

    cd = algebra.cd_algebra
    if cd is not None:
        signed_x, signed_y = s.signed, t.signed
        natives = [(cd(signed_x.coefficients[i]), cd(signed_y.coefficients[i]))
                   for i in range(pairs)]

    def native_single() -> None:
        """Multiply the signed components of every pair as pyfastcd elements."""

        for a, b in natives:
            a * b

    available.update({
        "splitform/batch": Task("splitform/batch", lambda: s * t, pairs,
                                lambda: (s * t).to_loop().coefficients),
        "forward/batch": Task("forward/batch", forward, pairs),
        "inverse/batch": Task("inverse/batch", product.to_loop, pairs),
        "cached/single": Task("cached/single", cached, pairs,
                              lambda: np.stack([(a * b).coefficients for a, b in elements])),
    })
    if cd is not None:
        # Products of signed components only, so they are timed but not verified
        available["pyfastcd/batch"] = Task("pyfastcd/batch", lambda: signed_x * signed_y, pairs)
        available["pyfastcd/single"] = Task("pyfastcd/single", native_single, pairs)
    return [available[name] for name in names if name in available]


def transform_share(results: dict[str, Any]) -> dict[str, Any] | None:
    """Return the share of a split/batch product spent converting to and from split form."""

    names = ("forward/batch", "inverse/batch", "splitform/batch")
    if any(name not in results or results[name]["status"] != "ok" for name in names):
        return None
    forward, inverse, product = (results[name]["observations_ns_per_product"] for name in names)
    shares = [(f + i) / (f + i + p) for f, i, p in zip(forward, inverse, product, strict=True)]
    return {"definition": "(forward + inverse) / (forward + inverse + splitform), per block",
            "summary": summarize(shares)}


def real_ratios(results: dict[str, Any]) -> dict[str, Any]:
    """Return the paired ratios reported for one level."""

    return {
        "table_over_split_batch": task_ratio(results, "table/batch", "split/batch"),
        "table_over_split_single": task_ratio(results, "table/single", "split/single"),
        "split_over_splitform_batch": task_ratio(results, "split/batch", "splitform/batch"),
        "split_over_cached_single": task_ratio(results, "split/single", "cached/single"),
        "single_over_batch_split": task_ratio(results, "split/single", "split/batch"),
        "splitform_over_pyfastcd_batch": task_ratio(results, "splitform/batch", "pyfastcd/batch"),
        "split_over_pyfastcd_batch": task_ratio(results, "split/batch", "pyfastcd/batch"),
        "cached_over_pyfastcd_single": task_ratio(results, "cached/single", "pyfastcd/single"),
    }


def write_markdown(path: Path, document: dict[str, Any]) -> None:
    """Write the Markdown report of the real protocol."""

    lines = report_header("pycdloop multiplication benchmark: real coefficients", document)
    lines += [
        "`split` multiplies through the decomposition of the loop algebra (three "
        "Walsh–Hadamard transforms, one coordinatewise product, and one Cayley–Dickson "
        "product); `table` evaluates the definition. `N` is the dimension of the loop "
        "algebra of `Q_n`, which is `2^(n+1)`. `fastest mean` names the lower mean of "
        "`split` and `table`, or a tie when the interval of their paired ratio contains "
        "one, and `default` is the algorithm that `\"auto\"` selects.", "",
    ]
    columns = {"batch": ("split/batch", "table/batch", "splitform/batch"),
               "single": ("split/single", "table/single", "cached/single")}
    notes = {
        "batch": "One call on arrays holding every pair of the corpus. `splitform` is the "
                 "product of two split arrays, which needs no transform.",
        "single": "One call per pair, on arrays of one record. `cached` is the product of "
                  "two elements whose split forms are already cached.",
    }
    for mode in MODES:
        heads = [name.split("/")[0] for name in columns[mode]]
        lines += [f"## {mode.capitalize()} products (ns per product)", "", notes[mode], "",
                  "| N | n | pairs | " + " | ".join(heads) + " | fastest mean | default |",
                  "| ---: | ---: | ---: | " + " | ".join("---" for _ in heads) + " | --- | --- |"]
        for entry in document["results"]:
            tasks = entry["tasks"]
            cells = [format_ci(tasks.get(name)) for name in columns[mode]]
            lines.append(f"| {entry['dimension']} | {entry['level']} | {entry['pairs']} | "
                         + " | ".join(cells) + f" | {entry['fastest'][mode]} | "
                         f"{entry['default_algorithm']} |")
        lines.append("")
    lines += ["## Paired ratios", "",
              "`table / split` above 1 favours `split`. `split / splitform` and "
              "`split / cached` are the factors by which a product is slower when its "
              "operands and result are in the loop basis rather than in split form.", "",
              "| N | n | table / split (batch) | table / split (single) | "
              "split / splitform (batch) | split / cached (single) |",
              "| ---: | ---: | --- | --- | --- | --- |"]
    keys = ("table_over_split_batch", "table_over_split_single",
            "split_over_splitform_batch", "split_over_cached_single")
    for entry in document["results"]:
        cells = [format_ratio(entry["ratios"][key]) for key in keys]
        lines.append(f"| {entry['dimension']} | {entry['level']} | " + " | ".join(cells) + " |")
    lines += ["", "## Comparison with the Cayley–Dickson product", "",
              "`pyfastcd` is the Cayley–Dickson product of the signed components alone, "
              "of dimension `N / 2`, timed through pyfastcd on its arrays (batch) and on "
              "its elements (single). A loop-algebra product contains one such product, so "
              "each ratio is the factor by which the rest of the work, and the Python "
              "layers around it, slow the product down.", "",
              "| N | n | pyfastcd batch (ns) | splitform / pyfastcd | split / pyfastcd | "
              "pyfastcd single (ns) | cached / pyfastcd |",
              "| ---: | ---: | --- | --- | --- | --- | --- |"]
    for entry in document["results"]:
        tasks, ratios = entry["tasks"], entry["ratios"]
        cells = [format_ci(tasks.get("pyfastcd/batch")),
                 format_ratio(ratios.get("splitform_over_pyfastcd_batch")),
                 format_ratio(ratios.get("split_over_pyfastcd_batch")),
                 format_ci(tasks.get("pyfastcd/single")),
                 format_ratio(ratios.get("cached_over_pyfastcd_single"))]
        lines.append(f"| {entry['dimension']} | {entry['level']} | " + " | ".join(cells) + " |")
    lines += ["", "## Share of the transforms", "",
              "The fraction of a batch product spent converting the operands to split form "
              "and the result back, measured as `(forward + inverse) / (forward + inverse + "
              "splitform)` within each block.", "",
              "| N | n | forward (ns) | inverse (ns) | splitform (ns) | transform share |",
              "| ---: | ---: | --- | --- | --- | --- |"]
    for entry in document["results"]:
        tasks = entry["tasks"]
        share = entry["transform_share"]
        if share is None:
            cell = "not timed"
        else:
            lo, hi = share["summary"]["ci95"]
            cell = f"{100 * share['summary']['mean']:.1f}%"
            if lo is not None:
                cell += f" [{100 * lo:.1f}%, {100 * hi:.1f}%]"
        cells = [format_ci(tasks.get(name))
                 for name in ("forward/batch", "inverse/batch", "splitform/batch")]
        lines.append(f"| {entry['dimension']} | {entry['level']} | " + " | ".join(cells)
                     + f" | {cell} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real protocol and write its JSON document and Markdown report."""

    arguments = parse_arguments(argv)
    document = new_document("real", arguments, argv, tasks=list(arguments.tasks))
    for level in arguments.levels:
        algebra = cdl.LoopAlgebra(cdl.CDLoop(level))
        pairs = pairs_for(arguments.pair_schedule, level)
        shape = (2, pairs) + algebra.coefficient_shape
        operands, record = load_corpus(arguments.corpus_dir, f"real-n{level}", "real",
                                       shape, [arguments.seed, level])
        document["corpus"][record["file"]] = record
        reference = reference_products((level,), operands)
        tasks = loop_tasks(algebra, operands, arguments.tasks)
        results = run_cell(tasks, reference, operands, False, arguments)
        document["results"].append({
            "level": level,
            "dimension": algebra.dimension,
            "quotient_order": algebra.loop.quotient_order,
            "pairs": pairs,
            "default_algorithm": algebra.default_algorithm,
            "tasks": results,
            "ratios": real_ratios(results),
            "fastest": {mode: fastest_label(results, [f"{name}/{mode}" for name in ALGORITHMS])
                        for mode in MODES},
            "transform_share": transform_share(results),
        })
        print(f"level {level}: " + ", ".join(
            f"{name} {format_ci(entry)}" for name, entry in results.items()), file=sys.stderr)
    document["verification"] = verification_totals(
        [entry["tasks"] for entry in document["results"]])
    write_json(arguments.output, document)
    write_markdown(arguments.output.with_suffix(".md"), document)
    return 1 if document["verification"]["failed_tasks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
