from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).with_name("bench_mlx_tuning.py")
SPEC = importlib.util.spec_from_file_location("bench_mlx_tuning", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bench_mlx_tuning = importlib.util.module_from_spec(SPEC)
sys.modules["bench_mlx_tuning"] = bench_mlx_tuning
SPEC.loader.exec_module(bench_mlx_tuning)


def result_fixture(
    weighted_ttft_ms: float,
    weighted_decode_tps: float,
    qa_accuracy: float,
    long_accuracy: float,
    structured_accuracy: float,
    cache_passed: bool,
    cache_speedup: float,
) -> dict:
    return {
        "prompt_sweep": {
            "weighted_ttft_ms": weighted_ttft_ms,
            "weighted_decode_tps": weighted_decode_tps,
        },
        "qa": {"accuracy": qa_accuracy},
        "long_context": {"accuracy": long_accuracy},
        "structured_output": {"accuracy": structured_accuracy},
        "prefix_cache": {
            "passed": cache_passed,
            "speedup": cache_speedup,
            "accuracy": 1.0 if cache_passed else 0.0,
        },
    }


class BenchMlxTuningTests(unittest.TestCase):
    def test_normalize_text(self) -> None:
        text = "  Leading\tand\ntrailing  MIXED  Case "
        normalized = bench_mlx_tuning.normalize_text(text)
        self.assertEqual(normalized, "leading and trailing mixed case")

    def test_cache_speedup_is_capped(self) -> None:
        self.assertEqual(bench_mlx_tuning.clamp_cache_speedup(96.0), 32.0)
        self.assertEqual(bench_mlx_tuning.clamp_cache_speedup(12.0), 12.0)

    def test_compute_iteration_score_applies_formula(self) -> None:
        result = result_fixture(
            weighted_ttft_ms=100.0,
            weighted_decode_tps=100.0,
            qa_accuracy=1.0,
            long_accuracy=0.5,
            structured_accuracy=0.0,
            cache_passed=True,
            cache_speedup=96.0,
        )

        score = bench_mlx_tuning.compute_iteration_score(
            result,
            best_ttft=100.0,
            best_decode=50.0,
            best_cache=16.0,
        )
        expected_accuracy = (1.0 * 0.45) + (0.5 * 0.25) + (0.0 * 0.15) + (1.0 * 0.15)
        expected_speed = (100.0 / 100.0) * 0.55 + (100.0 / 50.0) * 0.45
        expected_cache = (32.0 / 16.0)
        expected_composite = 100.0 * (
            (expected_accuracy * 0.45) + (expected_speed * 0.45) + (expected_cache * 0.10)
        )

        self.assertAlmostEqual(score["accuracy"], expected_accuracy, places=9)
        self.assertAlmostEqual(score["speed"], expected_speed, places=9)
        self.assertAlmostEqual(score["cache"], expected_cache, places=9)
        self.assertAlmostEqual(score["composite"], expected_composite, places=9)

    def test_score_results_marks_all_results(self) -> None:
        results = [
            {
                "prompt_sweep": {
                    "weighted_ttft_ms": 200.0,
                    "weighted_decode_tps": 80.0,
                },
                "qa": {"accuracy": 0.4},
                "long_context": {"accuracy": 0.0},
                "structured_output": {"accuracy": 1.0},
                "prefix_cache": {"passed": False, "accuracy": 0.0, "speedup": 12.0},
            },
            {
                "prompt_sweep": {
                    "weighted_ttft_ms": 100.0,
                    "weighted_decode_tps": 160.0,
                },
                "qa": {"accuracy": 0.8},
                "long_context": {"accuracy": 1.0},
                "structured_output": {"accuracy": 1.0},
                "prefix_cache": {"passed": True, "accuracy": 1.0, "speedup": 96.0},
            },
        ]

        bench_mlx_tuning.score_results(results)
        self.assertIn("score", results[0])
        self.assertIn("score", results[1])
        self.assertNotEqual(results[0]["score"]["composite"], results[1]["score"]["composite"])

    def test_rank_results_by_score(self) -> None:
        results = [
            {"score": {"composite": 33.0}},
            {"score": {"composite": 72.0}},
            {"score": {"composite": 51.0}},
        ]
        ranked = bench_mlx_tuning.rank_results_by_score(results)
        self.assertEqual(ranked[0]["score"]["composite"], 72.0)


if __name__ == "__main__":
    unittest.main()
