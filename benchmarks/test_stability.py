"""Tests of the exact reference, the operand families, and the study tooling."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from itertools import product
from pathlib import Path
from unittest import mock

import numpy as np

import pycdloop as cdl

try:
    from . import stability, stability_report, stability_study
except ImportError:  # Support unittest discovery from this directory.
    import stability
    import stability_report
    import stability_study

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_archives.py"
UNIT = 2.0**-53


def doubling(x: list[Fraction], y: list[Fraction]) -> list[Fraction]:
    """Multiply by the defining recursion (a,b)(c,d) = (ac - d* b, da + b c*)."""

    if len(x) == 1:
        return [x[0] * y[0]]
    h = len(x) // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    conj_d = d[:1] + [-v for v in d[1:]]
    conj_c = c[:1] + [-v for v in c[1:]]
    upper = [u - v for u, v in zip(doubling(a, c), doubling(conj_d, b), strict=True)]
    lower = [u + v for u, v in zip(doubling(d, a), doubling(b, conj_c), strict=True)]
    return upper + lower


def brute_force(x: np.ndarray, y: np.ndarray) -> list[list[tuple[Fraction, Fraction]]]:
    """Multiply two loop-algebra elements pair by pair over the Gaussian rationals."""

    size = x.shape[1]
    out = [[(Fraction(0), Fraction(0)) for _ in range(size)] for _ in range(2)]
    for p, q in product(range(size), repeat=2):
        left, right = [Fraction(0)] * size, [Fraction(0)] * size
        left[p], right[q] = Fraction(1), Fraction(1)
        sign = doubling(left, right)[p ^ q]
        for s, t in product(range(2), repeat=2):
            a, b = complex(x[s, p]), complex(y[t, q])
            real = Fraction(a.real) * Fraction(b.real) - Fraction(a.imag) * Fraction(b.imag)
            imag = Fraction(a.real) * Fraction(b.imag) + Fraction(a.imag) * Fraction(b.real)
            row = (s + t + (sign < 0)) % 2
            out[row][p ^ q] = (out[row][p ^ q][0] + real, out[row][p ^ q][1] + imag)
    return out


def as_fractions(element: stability.Exact) -> list[list[tuple[Fraction, Fraction]]]:
    """Return an exact element as Gaussian rationals."""

    real, imag, exponent = element
    scale = 1 << exponent
    zero = np.zeros(real.shape, dtype=object)
    imag = zero if imag is None else imag
    return [[(Fraction(int(a), scale), Fraction(int(b), scale)) for a, b in zip(row_r, row_i, strict=True)]
            for row_r, row_i in zip(real, imag, strict=True)]


def is_zero(element: stability.Exact) -> bool:
    """Return whether an exact element vanishes."""

    real, imag, _ = element
    return not any(real.ravel()) and (imag is None or not any(imag.ravel()))


class ReferenceTests(unittest.TestCase):
    def test_oracle_validation_passes(self) -> None:
        """Check signs and small products against rational arithmetic."""

        report = stability.validate_oracle()
        self.assertEqual(report["basis_cases_against_rational_doubling"], 341)
        self.assertEqual(report["loop_algebra_levels_against_rational_definition"], [0, 1, 2, 3])

    def test_encode_is_exact(self) -> None:
        """Convert binary64 values, including a subnormal, without rounding."""

        values = np.array([0.1, -3.5, 5e-324, 1e300, 0.0, 2.0**-40, -1.0 / 3.0])
        integers, exponent = stability.encode(values)
        for value, integer in zip(values, integers, strict=True):
            self.assertEqual(Fraction(float(value)), Fraction(int(integer), 1 << exponent))

    def test_exact_product_matches_brute_force(self) -> None:
        """Compare the exact product with a pair-by-pair rational evaluation."""

        rng = np.random.default_rng(7)
        for domain, level in product(("real", "complex"), (0, 1, 2, 3)):
            with self.subTest(domain=domain, level=level):
                shape = (2, 1 << level)
                x, y = rng.standard_normal(shape), rng.standard_normal(shape)
                if domain == "complex":
                    x = x + 1j * rng.standard_normal(shape)
                    y = y + 1j * rng.standard_normal(shape)
                reference = stability.exact_product(stability.exact(x), stability.exact(y))
                self.assertEqual(as_fractions(reference), brute_force(x, y))

    def test_exact_chain_of_two_is_the_product(self) -> None:
        """Reduce a chain of two operands to one exact product."""

        operands = stability.make_chain("real", 2, 2, 5)
        chain = stability.exact_chain(list(operands))
        pair = stability.exact_product(stability.exact(operands[0]), stability.exact(operands[1]))
        self.assertEqual(as_fractions(chain), as_fractions(pair))

    def test_chain_is_left_nested(self) -> None:
        """Bracket a chain of three octonion-loop operands from the left."""

        x, y, z = stability.make_chain("real", 3, 3, 9)
        left = stability.exact_product(stability.exact_product(stability.exact(x), stability.exact(y)), stability.exact(z))
        right = stability.exact_product(stability.exact(x), stability.exact_product(stability.exact(y), stability.exact(z)))
        chain = as_fractions(stability.exact_chain([x, y, z]))
        self.assertEqual(chain, as_fractions(left))
        self.assertNotEqual(chain, as_fractions(right))


class FamilyTests(unittest.TestCase):
    def inputs(self, domain: str, level: int, family: str, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
        return stability.make_inputs(domain, level, family, seed)

    def test_every_family_is_deterministic_and_well_formed(self) -> None:
        """Generate finite operands of the right shape and type from a seed."""

        for domain, families in (("real", stability.REAL_FAMILIES), ("complex", stability.COMPLEX_FAMILIES)):
            for family, level in product(families, (1, 4, 5)):
                if not stability.available(domain, level, family):
                    continue
                with self.subTest(domain=domain, family=family, level=level):
                    x, y = self.inputs(domain, level, family)
                    again = self.inputs(domain, level, family)
                    self.assertEqual(x.shape, (2, 1 << level))
                    self.assertEqual(x.dtype, np.complex128 if domain == "complex" else np.float64)
                    self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
                    self.assertEqual((x.tobytes(), y.tobytes()), (again[0].tobytes(), again[1].tobytes()))

    def test_availability(self) -> None:
        """Offer zero divisors over R from level 4 and isotropic operands over C."""

        self.assertFalse(stability.available("real", 3, "zero_divisor"))
        self.assertTrue(stability.available("real", 4, "near_zero_divisor_40"))
        self.assertFalse(stability.available("complex", 5, "zero_divisor"))
        self.assertFalse(stability.available("real", 5, "isotropic"))
        self.assertTrue(stability.available("complex", 1, "near_isotropic"))

    def test_dense_and_sparse_families(self) -> None:
        """Normalize Gaussian operands and limit sparse supports."""

        x, _ = self.inputs("real", 4, "gaussian")
        self.assertAlmostEqual(stability.euclidean_norm(x), 1.0, places=14)
        x, y = self.inputs("complex", 4, "rademacher")
        self.assertEqual(set(np.abs(x.real).ravel()) | set(np.abs(y.imag).ravel()), {2.0**-3})
        for family, support in (("sparse_one", 1), ("sparse_four", 4), ("sparse_eighth", 2)):
            x, y = self.inputs("real", 4, family)
            self.assertEqual((np.count_nonzero(x), np.count_nonzero(y)), (support, support))
        x, y = self.inputs("real", 3, "repeated")
        self.assertTrue(np.array_equal(x, y))
        x, _ = self.inputs("real", 3, "alternating_magnitude")
        self.assertTrue(all(math.frexp(abs(v))[0] == 0.5 for v in x.ravel()))
        x, y = self.inputs("complex", 3, "scalar_only")
        self.assertEqual((np.count_nonzero(x), np.count_nonzero(y)), (1, 1))
        self.assertTrue(x[0, 0] != 0 and y[0, 0] != 0)

    def test_ideal_families(self) -> None:
        """Place operands exactly in the group ideal or the signed ideal."""

        for domain in ("real", "complex"):
            x, y = self.inputs(domain, 3, "group_ideal")
            self.assertTrue(np.array_equal(x[0], x[1]) and np.array_equal(y[0], y[1]))
            x, y = self.inputs(domain, 3, "signed_ideal")
            self.assertTrue(np.array_equal(x[0], -x[1]) and np.array_equal(y[0], -y[1]))
            x, y = self.inputs(domain, 3, "cross_ideal")
            self.assertTrue(np.array_equal(x[0], x[1]) and np.array_equal(y[0], -y[1]))
            self.assertTrue(x.any() and y.any())
            self.assertTrue(is_zero(stability.exact_product(stability.exact(x), stability.exact(y))))

    def test_cancellation_families(self) -> None:
        """Leave one component of the decomposition near 2**-40 of the other."""

        x, _ = self.inputs("real", 4, "signed_cancel")
        self.assertLess(np.abs(x[0] - x[1]).max(), 2.0**-38)
        self.assertGreater(np.abs(x[0] - x[1]).max(), 0)
        x, _ = self.inputs("real", 4, "group_cancel")
        self.assertLess(np.abs(x[0] + x[1]).max(), 2.0**-38)
        self.assertGreater(np.abs(x[0] + x[1]).max(), 0)

    def test_idempotent_and_character_families(self) -> None:
        """Build exact idempotents and elements with a one-coordinate spectrum."""

        for level in (1, 3, 5):
            x, _ = self.inputs("real", level, "idempotent")
            element = stability.exact(x)
            self.assertEqual(as_fractions(stability.exact_product(element, element)), as_fractions(element))
            for domain in ("real", "complex"):
                x, _ = self.inputs(domain, level, "character")
                self.assertTrue(np.array_equal(x[0], x[1]))
                # The transform is taken in exact integers, part by part
                matrix = stability.characters(1 << level).astype(int).astype(object)
                real, imag, _ = stability.exact(x.astype(np.complex128))
                spectrum = [matrix.dot(part[0] + part[1]) for part in (real, imag)]
                nonzero = [bool(a) or bool(b) for a, b in zip(*spectrum, strict=True)]
                self.assertEqual(sum(nonzero), 1)

    def test_zero_divisor_families(self) -> None:
        """Lift an exact sedenion zero divisor to the signed ideal."""

        a, b = stability.zero_divisor()
        self.assertEqual((np.flatnonzero(a).tolist(), np.flatnonzero(b).tolist()), ([1, 10], [4, 15]))
        for level in (4, 6):
            x, y = self.inputs("real", level, "zero_divisor")
            self.assertTrue(x.any() and y.any())
            self.assertTrue(np.array_equal(x[0], -x[1]))
            self.assertTrue(is_zero(stability.exact_product(stability.exact(x), stability.exact(y))))
            near_x, near_y = self.inputs("real", level, "near_zero_divisor_20")
            self.assertLess(np.abs(near_x - x).max(), 2.0**-20)
            self.assertFalse(is_zero(stability.exact_product(stability.exact(near_x), stability.exact(near_y))))
        with self.assertRaises(ValueError):
            self.inputs("real", 3, "zero_divisor")

    def test_isotropic_families(self) -> None:
        """Give the signed component a vanishing quadratic form."""

        x, _ = self.inputs("complex", 3, "isotropic")
        signed = x[0] - x[1]
        self.assertTrue(np.array_equal(x[0], -x[1]))
        self.assertEqual(complex((signed * signed).sum()), 0)
        self.assertTrue(signed.any())
        near, _ = self.inputs("complex", 3, "near_isotropic")
        self.assertLess(np.abs(near - x).max(), 2.0**-40)
        with self.assertRaises(ValueError):
            self.inputs("real", 3, "isotropic")


class MeasureTests(unittest.TestCase):
    def test_constructed_errors(self) -> None:
        """Report an exact output as error zero and a perturbed one by its perturbation."""

        y = np.array([[0.5, 0.25], [1.0, -2.0]])
        x = np.zeros((2, 2))
        x[0, 0] = 3.0
        ex, ey = stability.exact(x), stability.exact(y)
        reference = stability.exact_product(ex, ey)
        norms = [stability.squared_norm(ex), stability.squared_norm(ey)]
        good = 3.0 * y
        bad = good.copy()
        bad[1, 1] += 2.0**-40
        results, totals = stability.measures({"good": good, "bad": bad}, reference, norms, 256)
        self.assertEqual(results["good"]["absolute"], 0.0)
        self.assertEqual(results["bad"]["absolute"], 2.0**-40)
        norm_y = math.sqrt(0.25 + 0.0625 + 1.0 + 4.0)
        self.assertAlmostEqual(results["bad"]["normalized"], 2.0**-40 / (UNIT * 3.0 * norm_y), delta=1e-6)
        self.assertAlmostEqual(results["bad"]["relative"], 2.0**-40 / (3.0 * norm_y), delta=1e-24)
        self.assertAlmostEqual(results["bad"]["chi"], 1.0, places=14)
        self.assertLess(totals["good"], totals["bad"])
        self.assertFalse(results["good"]["zero_product"])

    def test_zero_product_has_no_relative_error(self) -> None:
        """Leave the relative error and chi undefined when the exact product is zero."""

        x, y = stability.make_inputs("real", 2, "cross_ideal", 1)
        ex, ey = stability.exact(x), stability.exact(y)
        reference = stability.exact_product(ex, ey)
        output = np.zeros((2, 4))
        output[0, 1] = 2.0**-60
        results, _ = stability.measures({"S": output}, reference, [stability.squared_norm(ex), stability.squared_norm(ey)], 256)
        self.assertTrue(results["S"]["zero_product"])
        self.assertIsNone(results["S"]["relative"])
        self.assertIsNone(results["S"]["chi"])
        self.assertEqual(results["S"]["absolute"], 2.0**-60)

    def test_precision_does_not_change_the_measures(self) -> None:
        """Agree between 256 and 512 bits to binary64 accuracy."""

        row = stability.run_products("complex", 3, "gaussian", "smoke", 11, 256)
        for check in row["checks"]:
            self.assertEqual(check["maximum_reference_discrepancy"], 0.0)
        etas = {check["precision_bits"]: check["normalized_error_S"] for check in row["checks"]}
        self.assertEqual(set(etas), {256, 512})
        self.assertEqual(etas[256], etas[512])

    def test_measured_methods_are_accurate(self) -> None:
        """Keep every method within a small multiple of the unit roundoff."""

        for domain in ("real", "complex"):
            rows = stability.run_products(domain, 4, "gaussian", "smoke", 2, 256)["samples"]
            for row in rows:
                self.assertEqual(row["dimension"], 32)
                for method in ("S", "T"):
                    self.assertLess(row["methods"][method]["normalized"], 50)
            chains = stability.run_chains(domain, 3, 4, "smoke", 2, 256)["chains"]
            for row in chains:
                for method in ("L", "R"):
                    self.assertLess(row["methods"][method]["normalized"], 200)

    def test_seeds_depend_on_every_label(self) -> None:
        """Derive distinct seeds for distinct cases."""

        seeds = {stability.case_seed(1, *labels) for labels in product(range(2), range(2), range(3), range(3))}
        self.assertEqual(len(seeds), 36)
        self.assertEqual(stability.case_seed(1, 0, 1, 2), stability.case_seed(1, 0, 1, 2))


class DocumentSectionTests(unittest.TestCase):
    def test_section_round_trip(self) -> None:
        """Replace only the text between the markers."""

        text = f"intro\n\n{stability_report.BEGIN}\n\nold\n\n{stability_report.END}\n\noutro\n"
        updated = stability_report.replace_section(text, "new table\n")
        assert updated is not None
        self.assertEqual(stability_report.section(updated), "new table")
        self.assertTrue(updated.startswith("intro\n") and updated.endswith("\noutro\n"))
        self.assertIsNone(stability_report.section("no markers"))
        self.assertIsNone(stability_report.replace_section("no markers", "x"))

    def test_update_document_without_file_or_markers(self) -> None:
        """Leave a missing or unmarked document alone."""

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.md"
            self.assertFalse(stability_report.update_document(missing, "table"))
            self.assertFalse(missing.exists())
            plain = Path(directory) / "plain.md"
            plain.write_text("text\n")
            self.assertFalse(stability_report.update_document(plain, "table"))
            self.assertEqual(plain.read_text(), "text\n")


class ProvenanceTests(unittest.TestCase):
    def test_source_record_describes_the_imported_package(self) -> None:
        """Hash the files of whichever package the study would measure."""

        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "site" / "pycdloop"
            (package / "__pycache__").mkdir(parents=True)
            (package / "__init__.py").write_bytes(b"one\n")
            (package / "_core.so").write_bytes(b"\x00binary")
            (package / "__pycache__" / "skipped.pyc").write_bytes(b"cache")
            with mock.patch.object(stability_study, "measured_package", lambda: package):
                record = stability_study.source_record()
        self.assertEqual(record["package"], "installed")
        self.assertEqual(
            record["files_sha256"],
            {
                "src/pycdloop/__init__.py": hashlib.sha256(b"one\n").hexdigest(),
                "src/pycdloop/_core.so": hashlib.sha256(b"\x00binary").hexdigest(),
            },
        )
        checkout = stability_study.ROOT / "src" / "pycdloop"
        with mock.patch.object(stability_study, "measured_package", lambda: checkout):
            self.assertEqual(stability_study.source_record()["package"], "checkout")


class SmokeStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.results = Path(cls.temporary.name) / "results"
        cls.document = Path(cls.temporary.name) / "stability.md"
        cls.document.write_text(f"# Title\n\n{stability_report.BEGIN}\n{stability_report.END}\n\nEnd.\n")
        arguments = ["--profile", "smoke", "--jobs", "2", "--output", str(cls.results / "stability-20260101"),
                     "--real-levels", "4", "--complex-levels", "2", "--real-chain-levels", "2",
                     "--complex-chain-levels", "1", "--update-docs", "--document", str(cls.document)]
        cls.directory = stability_study.run(stability_study.parse_arguments(arguments))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def check(self, *options: str) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(CHECKER), "--results", str(self.results), "--document", str(self.document), *options]
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def test_published_files(self) -> None:
        """Write the tables, summary, figures, charts, and descriptions."""

        names = {p.name for p in self.directory.iterdir()}
        expected = {"README.md", "SHA256SUMS", "study.json", "key-figures.json", "accuracy-tables.md",
                    "accuracy-summary.csv", "PLOTS.md", "plot-data.csv", "raw"}
        charts = {f"{kind}-{domain}.{extension}" for kind in ("relative-error", "eta", "chain-eta")
                  for domain in ("real", "complex") for extension in ("png", "svg", "pdf")}
        self.assertEqual(names, expected | charts)
        self.assertEqual({p.name for p in (self.directory / "raw").iterdir()}, {"accuracy.json", "accuracy-corpus.npz"})

    def test_checksums_cover_the_published_files(self) -> None:
        """Record exactly the published files, with their hashes."""

        lines = (self.directory / "SHA256SUMS").read_text().splitlines()
        recorded = dict(reversed(line.split("  ", 1)) for line in lines)
        published = {p.name for p in self.directory.iterdir() if p.is_file() and p.name != "SHA256SUMS"}
        self.assertEqual(set(recorded), published)
        for name, digest in recorded.items():
            self.assertEqual(hashlib.sha256((self.directory / name).read_bytes()).hexdigest(), digest)

    def test_study_record(self) -> None:
        """Record the profile, seed, versions, source hashes, and raw hashes."""

        study = json.loads((self.directory / "study.json").read_text())
        self.assertEqual((study["profile"], study["seed"]), ("smoke", 20261111))
        for key in ("python", "numpy", "gmpy2", "pyfastcd", "pycdloop", "platform", "cpu_model"):
            self.assertIn(key, study["environment"])
        recorded = study["source"]["files_sha256"]
        self.assertIn("src/pycdloop/_kernel.py", recorded)
        # The hashes are those of the package that was imported and measured
        package = stability_study.measured_package()
        self.assertEqual(package, Path(cdl.__file__).resolve().parent)
        present = {
            f"src/pycdloop/{p.relative_to(package).as_posix()}"
            for p in package.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        self.assertEqual(set(recorded), present)
        for name, digest in recorded.items():
            path = package / name.removeprefix("src/pycdloop/")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        checkout = stability_study.ROOT / "src" / "pycdloop"
        origin = "checkout" if package == checkout else "installed"
        self.assertEqual(study["source"]["package"], origin)
        self.assertEqual(set(study["raw_sha256"]), {"raw/accuracy.json", "raw/accuracy-corpus.npz"})
        self.assertNotIn(str(self.results), study["command"])
        self.assertIn("benchmark-results/stability-20260101", study["command"])

    def test_raw_results(self) -> None:
        """Store every case with its seed, operand hash, and measures."""

        document = json.loads((self.directory / "raw" / "accuracy.json").read_text())
        self.assertTrue(document["completed"])
        corpus = np.load(self.directory / "raw" / "accuracy-corpus.npz")
        for row in document["samples"]:
            x, y = corpus[row["key"] + "_x"], corpus[row["key"] + "_y"]
            self.assertEqual(hashlib.sha256(stability.payload(x, y)).hexdigest(), row["operands_sha256"])
            self.assertEqual(set(row["methods"]), {"S", "T"})
        self.assertEqual({row["length"] for row in document["chains"]}, {2, 4, 8})
        zero = [row for row in document["samples"] if row["family"] in ("cross_ideal", "zero_divisor")]
        self.assertTrue(zero and all(row["methods"]["S"]["zero_product"] for row in zero))
        figures = json.loads((self.directory / "key-figures.json").read_text())
        self.assertEqual(figures["pairs"], len(document["samples"]))
        self.assertEqual(stability_report.render(document), (self.directory / "accuracy-tables.md").read_text())

    def test_descriptions_are_self_contained(self) -> None:
        """Describe the study without paths of the machine that ran it."""

        for name in ("README.md", "PLOTS.md", "accuracy-tables.md"):
            text = (self.directory / name).read_text()
            self.assertNotIn(self.temporary.name, text)
            self.assertNotIn(str(ROOT), text)
        self.assertIn("stability-20260101", (self.directory / "PLOTS.md").read_text())

    def test_checker_accepts_the_study(self) -> None:
        """Accept the study and the document section it wrote."""

        for options in ((), ("--full",)):
            result = self.check(*options)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(len(report["studies"]), 1)
            self.assertIn("matches stability-20260101", report["document"])
        tables = (self.directory / "accuracy-tables.md").read_text().strip("\n")
        self.assertEqual(stability_report.section(self.document.read_text()), tables)

    def test_checker_rejects_changes(self) -> None:
        """Fail on a changed file, an unrecorded file, and a stale document."""

        target = self.directory / "plot-data.csv"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n")
            self.assertEqual(self.check().returncode, 1)
        finally:
            target.write_bytes(original)
        extra = self.directory / "extra.txt"
        try:
            extra.write_text("unrecorded\n")
            self.assertEqual(self.check().returncode, 1)
        finally:
            extra.unlink()
        text = self.document.read_text()
        try:
            self.document.write_text(text.replace("operand pairs", "operand pears"))
            result = self.check()
            self.assertEqual(result.returncode, 1)
            self.assertIn("differs", result.stderr)
        finally:
            self.document.write_text(text)
        self.assertEqual(self.check().returncode, 0)

    def test_checker_without_studies(self) -> None:
        """Succeed with a note when there is nothing to check."""

        with tempfile.TemporaryDirectory() as empty:
            command = [sys.executable, str(CHECKER), "--results", empty]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no stability studies", json.loads(result.stdout)["note"])


if __name__ == "__main__":
    unittest.main()
