"""Tests of the benchmark drivers: arguments, statistics, reference, and smoke runs."""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from . import benchmark as driver
    from . import central, domains, protocols
except ImportError:  # Support unittest discovery from this directory.
    import benchmark as driver
    import central
    import domains
    import protocols


def rejected(parse: object, *arguments: str) -> str:
    """Return the error printed when a parser rejects a command line."""

    error = io.StringIO()
    with contextlib.redirect_stderr(error):
        try:
            parse(list(arguments))  # type: ignore[operator]
        except SystemExit as stop:
            assert stop.code == 2
            return error.getvalue()
    raise AssertionError("the command line was accepted")


class BenchmarkArgumentsTests(unittest.TestCase):
    def test_levels_accept_integers_and_ranges(self) -> None:
        self.assertEqual(driver.parse_levels("1-3,5"), [1, 2, 3, 5])
        self.assertEqual(driver.parse_levels("4,2,2"), [2, 4])
        for text in ("0", "13", "3-1", "a", "", "1-", "2,,3"):
            with self.subTest(levels=text), self.assertRaises(ValueError):
                driver.parse_levels(text)

    def test_pair_schedule_covers_levels_or_is_rejected(self) -> None:
        schedule = driver.parse_pairs("1-6:256,7-8:64,9:16")
        self.assertEqual([driver.pairs_for(schedule, n) for n in (1, 6, 7, 9)],
                         [256, 256, 64, 16])
        self.assertEqual(driver.pairs_for(driver.parse_pairs("32"), 11), 32)
        with self.assertRaises(ValueError):
            driver.pairs_for(schedule, 10)
        for text in ("1-2:0", "x", "2-1:4", "1:2:3x"):
            with self.subTest(pairs=text), self.assertRaises(ValueError):
                driver.parse_pairs(text)
        self.assertIn("level 3", rejected(driver.parse_arguments, "--output", "run.json",
                                          "--levels", "1-3", "--pairs", "1-2:8"))

    def test_tasks_keep_canonical_order_and_reject_unknown_names(self) -> None:
        arguments = driver.parse_arguments(
            ["--output", "run.json", "--tasks", "table/single,split/batch,split/batch"])
        self.assertEqual(arguments.tasks, ("split/batch", "table/single"))
        self.assertEqual(driver.parse_arguments(["--output", "run.json"]).tasks, driver.TASKS)
        for tasks in ("split", "split/batch,", "", "direct/batch"):
            with self.subTest(tasks=tasks):
                self.assertIn("tasks", rejected(driver.parse_arguments, "--output",
                                                "run.json", "--tasks", tasks))

    def test_numeric_options_are_validated(self) -> None:
        for option, value in (("--blocks", "0"), ("--warmup", "-1"), ("--min-time-ms", "nan"),
                              ("--min-time-ms", "-1"), ("--seed", "-5"), ("--levels", "0")):
            with self.subTest(option=option, value=value):
                rejected(driver.parse_arguments, "--output", "run.json", option, value)

    def test_markdown_output_is_rejected_before_any_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for module in (driver, domains, central, protocols):
                for name in ("run.md", "run.MD"):
                    with self.subTest(module=module.__name__, output=name):
                        error = rejected(module.main, "--output", str(Path(directory) / name),
                                         "--corpus-dir", str(Path(directory) / "corpora"))
                        self.assertIn(".md", error)
                        self.assertEqual(list(Path(directory).iterdir()), [])

    def test_fields_and_factors_are_validated(self) -> None:
        self.assertEqual(domains.parse_fields("0:2,3:1,13:5,3:1"), [(0, 2), (3, 1), (13, 5)])
        self.assertEqual(domains.parse_fields("2:1,2:8"), [(2, 1), (2, 8)])
        for text in ("0:1", "4:1", "2:13", "3", "x:y", "3:13"):
            with self.subTest(fields=text), self.assertRaises(ValueError):
                domains.parse_fields(text)
        fields = ((0, 2), (3, 1), (13, 5), (2, 1), (2, 8))
        self.assertEqual([domains.field_label(*field) for field in fields],
                         ["C", "F_3", "F_(13^5)", "F_2", "F_(2^8)"])
        self.assertEqual(central.parse_factors("1x4,2X2,1x4"), [(1, 4), (2, 2)])
        for text in ("0x1", "8x8", "3", "2x2x2", "axb"):
            with self.subTest(factors=text), self.assertRaises(ValueError):
                central.parse_factors(text)
        self.assertIn("level 5", rejected(central.parse_arguments, "--output", "run.json",
                                          "--factors", "2x3", "--pairs", "1-4:8"))

    def test_protocol_commands_have_distinct_outputs(self) -> None:
        arguments = protocols.parse_arguments(
            ["--protocol", "all", "--output", "out/run.json", "--levels", "1-2",
             "--fields", "0:2", "--factors", "1x1", "--cpu", "3"])
        commands = dict(protocols.commands(arguments))
        self.assertEqual(tuple(commands), protocols.PROTOCOLS)
        outputs = [command[command.index("--output") + 1] for command in commands.values()]
        self.assertEqual([Path(output).name for output in outputs],
                         ["run.real.json", "run.domains.json", "run.central.json"])
        self.assertIn("--levels", commands["real"])
        self.assertNotIn("--levels", commands["central"])
        self.assertIn("--fields", commands["domains"])
        self.assertNotIn("--fields", commands["real"])
        self.assertIn("--factors", commands["central"])
        self.assertTrue(all("--cpu" in command for command in commands.values()))
        single = protocols.commands(protocols.parse_arguments(
            ["--protocol", "domains", "--output", "out/run.json"]))
        self.assertEqual(Path(single[0][1][single[0][1].index("--output") + 1]),
                         Path("out/run.json"))


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_summarize_reports_moments_and_interval(self) -> None:
        summary = driver.summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertEqual(summary["count"], 8)
        self.assertAlmostEqual(summary["mean"], 5.0)
        self.assertAlmostEqual(summary["median"], 4.5)
        self.assertAlmostEqual(summary["sd"], 2.138089935299395)
        half = driver.T95[7] * summary["sd"] / 8 ** 0.5
        self.assertAlmostEqual(summary["ci95"][0], 5.0 - half)
        self.assertAlmostEqual(summary["ci95"][1], 5.0 + half)
        self.assertEqual((summary["min"], summary["max"]), (2.0, 9.0))
        self.assertEqual(driver.summarize([3.0])["ci95"], [None, None])

    def test_t95_table_is_decreasing_and_ends_near_normal(self) -> None:
        values = driver.T95[1:]
        self.assertEqual(len(values), 30)
        self.assertTrue(all(a > b for a, b in itertools.pairwise(values)))
        self.assertAlmostEqual(values[0], 12.7062047364)
        self.assertAlmostEqual(values[-1], 2.0422724563)

    def test_paired_ratio_is_a_geometric_mean_with_win_counts(self) -> None:
        ratio = driver.paired_ratio([2.0, 2.0, 8.0], [1.0, 1.0, 2.0])
        self.assertAlmostEqual(ratio["geometric_mean"], (2.0 * 2.0 * 4.0) ** (1 / 3))
        self.assertEqual(ratio["paired_blocks"], 3)
        self.assertEqual(ratio["competitor_faster_blocks"], 0)
        self.assertEqual(ratio["competitor_slower_blocks"], 3)

    def test_fastest_label_marks_inseparable_means_as_ties(self) -> None:
        def result(name: str, observations: list[float]) -> dict[str, object]:
            return {"task": name, "status": "ok", "summary": driver.summarize(observations),
                    "observations_ns_per_product": observations}

        names = ["split/batch", "table/batch"]
        clear = {"table/batch": result("table/batch", [10.0, 10.1, 9.9, 10.0]),
                 "split/batch": result("split/batch", [5.0, 5.1, 4.9, 5.0])}
        self.assertEqual(driver.fastest_label(clear, names), "split")
        tied = {"table/batch": result("table/batch", [5.0, 5.2, 4.8, 5.1]),
                "split/batch": result("split/batch", [5.1, 4.9, 5.2, 4.9])}
        self.assertTrue(driver.fastest_label(tied, names).startswith("tie ("))
        failed = dict(clear, **{"split/batch": {"task": "split/batch", "status": "failed"}})
        self.assertEqual(driver.fastest_label(failed, names), "table")
        self.assertEqual(driver.fastest_label({}, names), "none verified")
        self.assertIsNone(driver.task_ratio(failed, "table/batch", "split/batch"))
        ratio = driver.task_ratio(clear, "table/batch", "split/batch")
        self.assertAlmostEqual(ratio["geometric_mean"], 2.0, places=2)

    def test_cells_render_means_intervals_and_failures(self) -> None:
        entry = {"status": "ok", "summary": driver.summarize([1.0, 3.0])}
        self.assertTrue(driver.format_ci(entry).startswith("2.00 ["))
        self.assertEqual(driver.format_ci({"status": "failed"}), "FAILED")
        self.assertEqual(driver.format_ci(None), "not timed")
        self.assertEqual(driver.format_ratio(None), "not timed")
        self.assertEqual([driver.format_ns(v) for v in (1.234, 123.44, 12345.6)],
                         ["1.23", "123.4", "12346"])

    def test_cpu_state_reports_every_key(self) -> None:
        state = driver.cpu_state(None)
        self.assertEqual(set(state), {"scaling_governor", "intel_pstate_no_turbo",
                                      "cpufreq_boost", "smt_control"})
        self.assertIsNone(state["scaling_governor"])


class ReferenceTests(unittest.TestCase):
    def test_sign_table_fixes_the_orientation(self) -> None:
        quaternions = driver.sign_table(2)
        self.assertTrue((quaternions[0] == 1).all() and (quaternions[:, 0] == 1).all())
        self.assertTrue((np.diag(quaternions)[1:] == -1).all())
        self.assertEqual([quaternions[1, 2], quaternions[2, 3], quaternions[3, 1]], [1, 1, 1])
        self.assertEqual([quaternions[2, 1], quaternions[3, 2], quaternions[1, 3]], [-1, -1, -1])
        octonions = driver.sign_table(3)
        self.assertEqual([octonions[1, 4], octonions[2, 4], octonions[3, 4]], [1, 1, 1])
        sedenions = driver.sign_table(4)
        off = ~np.eye(16, dtype=bool)
        off[0] = off[:, 0] = False
        self.assertTrue((sedenions[off] == -sedenions.T[off]).all())

    def test_twist_of_a_central_product_multiplies_the_factor_signs(self) -> None:
        p, q = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
        signs = driver.twist((1, 2), p, q)
        expected = np.kron(driver.sign_table(1), driver.sign_table(2))
        np.testing.assert_array_equal(signs, expected)
        # The basis element with index (1, 1) of Q_1 * Q_1 squares to +1
        self.assertEqual(int(driver.twist((1, 1), np.array(3), np.array(3))), 1)

    def test_reference_multiplies_basis_elements_by_the_loop_rule(self) -> None:
        def basis(row: int, index: int) -> np.ndarray:
            element = np.zeros((2, 4))
            element[row, index] = 1.0
            return element

        # [e_1][e_2] = [e_3], [e_2][e_1] = [-e_3], [-e_1][e_1] = [e_0]
        cases = (((0, 1), (0, 2), (0, 3)), ((0, 2), (0, 1), (1, 3)),
                 ((1, 1), (0, 1), (0, 0)), ((1, 1), (1, 1), (1, 0)))
        for left, right, product in cases:
            with self.subTest(left=left, right=right):
                np.testing.assert_array_equal(
                    driver.reference_multiply((2,), basis(*left), basis(*right)),
                    basis(*product))

    def test_reference_agrees_with_a_term_by_term_sum(self) -> None:
        rng = np.random.default_rng(5)
        for levels in ((2,), (3,), (1, 2)):
            size = 1 << sum(levels)
            x = rng.integers(-4, 5, (2, size)).astype(float)
            y = rng.integers(-4, 5, (2, size)).astype(float)
            expected = np.zeros((2, size))
            for (s, p), (t, q) in itertools.product(
                    itertools.product(range(2), range(size)), repeat=2):
                sign = (-1) ** (s + t) * int(driver.twist(levels, np.array(p), np.array(q)))
                expected[0 if sign > 0 else 1, p ^ q] += x[s, p] * y[t, q]
            with self.subTest(levels=levels):
                np.testing.assert_array_equal(driver.reference_multiply(levels, x, y), expected)
                np.testing.assert_array_equal(
                    driver.reference_multiply(levels, x, y, budget=size), expected)

    def test_schoolbook_field_arithmetic(self) -> None:
        # In F_7[t] / (t^2 + 1): (2 + 3t)(4 + 5t) = 8 - 15 + 22t = 0 + t
        multiply = domains.field_multiply(7, (1, 0, 1))
        np.testing.assert_array_equal(
            multiply(np.array([2, 3], dtype=np.uint32), np.array([4, 5], dtype=np.uint32)),
            [0, 1])
        # In F_3[t] / (t^3 + 2t + 1): t * t^2 = t^3 = -2t - 1 = t + 2
        cube = domains.field_multiply(3, (1, 2, 0, 1))
        np.testing.assert_array_equal(
            cube(np.array([0, 1, 0], dtype=np.uint32), np.array([0, 0, 1], dtype=np.uint32)),
            [2, 1, 0])
        largest = np.uint32(2147483646)
        prime = domains.field_multiply(2147483647, (0, 1))
        np.testing.assert_array_equal(prime(np.array([largest]), np.array([largest])), [1])

    def test_check_products_applies_the_norm_scaled_bound(self) -> None:
        operands = np.ones((2, 1, 2, 4))
        reference = np.ones((1, 2, 4))
        bound = 128 * np.finfo(float).eps * 4 * 8.0
        good = driver.check_products(reference + 0.5 * bound, reference, operands, False)
        bad = driver.check_products(reference + 2.0 * bound, reference, operands, False)
        self.assertTrue(good["passed"] and good["worst_error_over_bound"] < 1)
        self.assertFalse(bad["passed"])
        self.assertFalse(driver.check_products(reference * np.nan, reference, operands,
                                               False)["passed"])
        self.assertFalse(driver.check_products(reference[:, :1], reference, operands,
                                               False)["passed"])
        exact = np.ones((1, 2, 4), dtype=np.uint32)
        self.assertTrue(driver.check_products(exact, exact, operands, True)["passed"])
        self.assertFalse(driver.check_products(exact + 1, exact, operands, True)["passed"])


class CorpusTests(unittest.TestCase):
    def test_corpus_is_deterministic_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            shape = (2, 3, 2, 4)
            a, record_a = driver.load_corpus(Path(first), "real-n2", "real", shape, [7, 2])
            b, record_b = driver.load_corpus(Path(second), "real-n2", "real", shape, [7, 2])
            np.testing.assert_array_equal(a, b)
            self.assertEqual(record_a["sha256"], record_b["sha256"])
            self.assertEqual(a.shape, shape)
            stamp = (Path(first) / "real-n2.npy").stat().st_mtime_ns
            again, record = driver.load_corpus(Path(first), "real-n2", "real", shape, [7, 2])
            np.testing.assert_array_equal(again, a)
            self.assertEqual(record, record_a)
            self.assertEqual((Path(first) / "real-n2.npy").stat().st_mtime_ns, stamp)
            other, changed = driver.load_corpus(Path(first), "real-n2", "real", shape, [8, 2])
            self.assertFalse(np.array_equal(other, a))
            self.assertNotEqual(changed["sha256"], record_a["sha256"])
            manifest = json.loads((Path(first) / "manifest.json").read_text())
            self.assertEqual(manifest["real-n2"]["seed"], [8, 2])

    def test_corpus_kinds_have_their_storage_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            finite, _ = driver.load_corpus(Path(directory), "f", "finite", (2, 4, 2, 2), [1], 13)
            complex_, _ = driver.load_corpus(Path(directory), "c", "complex", (2, 4, 2, 2), [1])
            self.assertEqual(finite.dtype, np.uint32)
            self.assertTrue(int(finite.max()) < 13)
            self.assertEqual(complex_.dtype, np.complex128)

    def test_a_file_that_differs_from_its_manifest_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shape = (2, 2, 2, 2)
            operands, _ = driver.load_corpus(Path(directory), "real-n1", "real", shape, [1, 1])
            np.save(Path(directory) / "real-n1.npy", operands + 1.0)
            with self.assertRaises(SystemExit) as raised:
                driver.load_corpus(Path(directory), "real-n1", "real", shape, [1, 1])
            self.assertIn("manifest", str(raised.exception))


class SmokeTests(unittest.TestCase):
    QUICK = ("--blocks", "2", "--min-time-ms", "0.5", "--warmup", "1")

    def run_protocol(self, module: object, *options: str) -> tuple[dict, str]:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            command = ["--output", str(output), "--corpus-dir", str(Path(directory) / "corpora"),
                       *self.QUICK, *options]
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(module.main(command), 0)  # type: ignore[attr-defined]
            return json.loads(output.read_text()), output.with_suffix(".md").read_text()

    def check_document(self, document: dict, name: str) -> None:
        self.assertEqual(document["schema"], driver.SCHEMA)
        self.assertEqual(document["protocol"]["name"], name)
        self.assertEqual(document["protocol"]["blocks"], 2)
        self.assertEqual(document["verification"]["failed_tasks"], 0)
        self.assertGreater(document["verification"]["products_checked"], 0)
        self.assertIn("_kernel.py", document["sources_sha256"])
        for key in ("python", "numpy", "pyfastcd", "pycdloop", "processor", "cpu_state"):
            self.assertIn(key, document["environment"])
        for record in document["corpus"].values():
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
        for entry in document["results"]:
            for task in entry["tasks"].values():
                self.assertEqual(task["status"], "ok")
                self.assertEqual(len(task["observations_ns_per_product"]), 2)
                self.assertGreater(task["summary"]["mean"], 0)
                self.assertGreaterEqual(task["passes"], 1)

    def test_real_protocol(self) -> None:
        document, report = self.run_protocol(driver, "--levels", "1-2")
        self.check_document(document, "real")
        self.assertEqual([entry["dimension"] for entry in document["results"]], [4, 8])
        self.assertEqual(set(document["results"][0]["tasks"]), set(driver.TASKS))
        share = document["results"][1]["transform_share"]["summary"]["mean"]
        self.assertTrue(0 < share < 1)
        self.assertIn("## Batch products", report)
        self.assertIn("## Share of the transforms", report)
        self.assertNotIn("FAILED", report)

    def test_real_protocol_with_a_task_subset(self) -> None:
        document, report = self.run_protocol(driver, "--levels", "1", "--tasks",
                                             "split/batch,table/batch")
        self.assertEqual(set(document["results"][0]["tasks"]), {"split/batch", "table/batch"})
        self.assertIsNone(document["results"][0]["transform_share"])
        self.assertIn("not timed", report)

    def test_domains_protocol(self) -> None:
        document, report = self.run_protocol(domains, "--levels", "1-2", "--fields",
                                             "0:2,3:1,3:2")
        self.check_document(document, "domains")
        self.assertEqual(len(document["results"]), 6)
        self.assertEqual({entry["field"] for entry in document["results"]},
                         {"C", "F_3", "F_(3^2)"})
        self.assertIn("| F_(3^2) |", report)

    def test_central_protocol(self) -> None:
        document, report = self.run_protocol(central, "--factors", "1x2,2x2",
                                             "--max-table-level", "3")
        self.check_document(document, "central")
        first, second = document["results"]
        self.assertEqual(set(first["tasks"]), {"native/signed", "rows/signed",
                                               "schedule/signed", "split/batch", "table/batch"})
        self.assertEqual(set(second["tasks"]), {"rows/signed", "schedule/signed"})
        self.assertEqual(first["default_signed_algorithm"], "native")
        self.assertIn("unavailable", report)
        self.assertIn("## Full product", report)

    def test_a_failed_verification_is_reported_and_not_timed(self) -> None:
        operands = np.ones((2, 2, 2, 2))
        reference = driver.reference_products((1,), operands)
        tasks = [driver.Task("split/batch", lambda: None, 2, lambda: reference + 1.0),
                 driver.Task("table/batch", lambda: None, 2, lambda: reference)]
        arguments = driver.parse_arguments(["--output", "run.json", *self.QUICK])
        results = driver.run_cell(tasks, reference, operands, False, arguments)
        self.assertEqual(results["split/batch"]["status"], "failed")
        self.assertNotIn("observations_ns_per_product", results["split/batch"])
        self.assertEqual(results["table/batch"]["status"], "ok")
        self.assertEqual(driver.verification_totals([results]),
                         {"products_checked": 4, "failed_tasks": 1})
        self.assertEqual(driver.format_ci(results["split/batch"]), "FAILED")

    def test_all_protocols_write_an_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            command = ["--protocol", "all", "--output", str(output),
                       "--corpus-dir", str(Path(directory) / "corpora"), *self.QUICK,
                       "--levels", "1", "--fields", "3:1", "--factors", "1x1"]
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(protocols.main(command), 0)
            index = json.loads(output.read_text())
            self.assertEqual(index["status"], "ok")
            self.assertEqual([entry["protocol"] for entry in index["reports"]],
                             list(protocols.PROTOCOLS))
            for entry in index["reports"]:
                self.assertTrue((Path(directory) / entry["json"]).is_file())
                self.assertTrue((Path(directory) / entry["markdown"]).is_file())
            self.assertIn("| central |", output.with_suffix(".md").read_text())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(protocols.main(command), 2)


if __name__ == "__main__":
    unittest.main()
