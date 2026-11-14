# Contributing to pycdloop

The canonical repository is
[git.tanuki-cd.com/algebraity/pycdloop](https://git.tanuki-cd.com/algebraity/pycdloop).

## Building and testing

```sh
python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m mypy
python -m mypy tests/typing_checks.py
python -m ruff check src tests benchmarks tools scripts setup.py
python -m ruff format --check src tests tools scripts setup.py
```

Building needs a C11 compiler and the Python development headers, for the
extension module `pycdloop._core`. After changing `src/pycdloop/_core.c`,
run `python -m pip install -e .` again to rebuild it. The tests under
`benchmarks/` cover the benchmark and stability tooling and need the
`benchmark` extras (gmpy2 and matplotlib), which `dev` includes.

## Layout

- `src/pycdloop/_core.c`: the compiled kernels, which are the
  Walsh–Hadamard transform and the change of coordinates between the loop
  basis and the split form, for doubles and for residues.
- `src/pycdloop/_field.py`, `_transform.py`, `_kernel.py`: arithmetic on raw
  coefficient arrays. Every routine here is written once and serves
  elements and both kinds of array. `_transform.py` also holds the
  reference evaluation of the transform, which accepts any element type.
- `src/pycdloop/_binary.py`, `_lift.py`: the fields of characteristic 2,
  and the products over them that are lifted to a prime field or an
  extension of it.
- `src/pycdloop/_signed.py`, `_schedule.py`: the product of signed
  components, delegated to pyfastcd for a Cayley–Dickson loop and evaluated
  by the tensor-product methods for a central product.
- `src/pycdloop/_structure.py`: zero divisors, annihilators, minimal
  polynomials, the center and multiplication matrices.
- `src/pycdloop/_loop.py`, `_algebra.py`, `_element.py`, `_array.py`: the
  public classes.
- `tests/reference.py`: an oracle that imports neither pycdloop nor
  pyfastcd and derives everything from the doubling formula in exact
  arithmetic. Tests of arithmetic compare against it.

## Conventions

- In `src/pycdloop`, every function has a docstring that begins with a
  one-line summary, and public classes and functions have NumPy-style
  docstrings. Examples in docstrings must run. Comments have no
  closing punctuation, except multi-line explanations of an algorithm,
  which are written as full sentences where the algorithm is implemented.
- The C source follows the conventions of fastCD: four-space block
  indentation, eight-space continuations, pointers attached to their types,
  and a short purpose comment before each function. It must build without
  warnings under GCC, Clang and MSVC, with no floating-point contraction or
  value-changing optimization, and it allocates only the scratch of one
  record.
- A compiled routine must perform the arithmetic of its reference
  evaluation in the same order on each coordinate, so that results stay
  bit-identical; `tests/test_kernel.py` checks this. Every buffer a routine
  receives is checked for its element type, contiguity, alignment and
  length before it is read or written.
- Error messages used by more than one module are defined in
  `src/pycdloop/_validation.py`.
- Operations on an `Element` must not depend on which of its two forms it
  has cached: choose the coordinates of an operation from how the operands
  were created, as `Element._mode` does.
- A method that needs the decomposition raises `TypeError` in
  characteristic 2 through `LoopAlgebra._require_split`, and a method that
  needs powers free of bracketing checks `LoopAlgebra._has_powers`. Every
  public method has a defined behavior in characteristic 2, documented in
  `docs/api.md`.
- Every public behavior change needs a test, a documentation update, and a
  line in `CHANGELOG.md`.

## Arithmetic checks

For changes to the transform, the splitting or the schedule, run
`python tools/check_operation_counts.py`. It runs the schedule and the
reference evaluations of the transform and of the change of coordinates on
counting scalars, checks the number of products of the schedule and the
cost formulas of both algorithms, and prints the operation-count tables of
`docs/mathematics.md` for comparison.

Over finite fields every result is exact, so tests compare with the oracle
exactly; in characteristic 2 the lift and table algorithms must also agree
with each other. Over R and C, tests use small integer coefficients
wherever the arithmetic is then exact, and tolerances only for quantities
that are genuinely rounded.

## Algorithm defaults

The automatic choices are `LoopAlgebra.default_algorithm`; for central
products, `ROWS_MAX_LEVEL` in `src/pycdloop/_signed.py`; and in
characteristic 2, the thresholds `LIFT_THRESHOLDS` in
`src/pycdloop/_algebra.py`. All are pinned by tests. Change them only on
the basis of a new benchmark run (see `docs/benchmark.md`), and update
`docs/mathematics.md` and `docs/api.md` in the same commit.

## Benchmark archives

Never edit the archived results under `benchmark-results/`, including their
checksum records. `python tools/check_archives.py` checks each stability
study against its `SHA256SUMS` and checks that the tables in
`docs/stability.md` match the newest study. Raw study data under `raw/` is
not distributed. It is checked where it is present; `--full` requires it
and regenerates the tables from it.

## Release checks

Update the version in `src/pycdloop/_version.py`, the README heading, and
the version and release date (`date-released`) in `CITATION.cff`. Move the
pending changelog entries into a dated release section.

Build and check the distributions in a development environment:

```sh
python -m build
python -m twine check --strict dist/*
```

CI builds the source distribution and a wheel from it, installs the wheel,
and runs the tests from the extracted source distribution outside the
checkout, so that the distributed files are what is tested. The release
workflow builds the wheels for every supported platform with cibuildwheel,
whose configuration is in `pyproject.toml`.

To publish a release, tag its commit with `v` and the version, such as
`v0.1.0`, and push the tag to the canonical repository. On the GitHub
mirror, the tag starts the release workflow, which checks that the tag names
the packaged version and publishes the source distribution and the wheels
to PyPI through trusted publishing. A manual run of the workflow publishes
to TestPyPI instead. Neither index accepts a file name twice, even after the
file is deleted, so a correction is published under a new version.
