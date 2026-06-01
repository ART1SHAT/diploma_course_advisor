"""Юнит-тесты offline_evaluator (без запуска recommender)."""
import unittest

from tests.evaluation.offline_evaluator import (
    bootstrap_ci,
    coverage,
    evaluate,
    fairness_gap,
    mrr,
    ndcg_at_k,
    precision_at_k,
)


class TestOfflineMetrics(unittest.TestCase):
    def setUp(self):
        self.preds = [
            ["a", "b", "c", "d", "e"],
            ["x", "b", "y", "z", "w"],
            ["b", "a", "c", "d", "e"],
        ]
        self.gt = [
            ["a", "b", "c"],
            ["b"],
            ["b", "c"],
        ]
        self.k = 5

    def test_precision_at_k(self):
        p = precision_at_k(self.preds, self.gt, self.k)
        self.assertGreater(p, 0)
        self.assertLessEqual(p, 1)

    def test_ndcg_and_mrr(self):
        self.assertGreater(ndcg_at_k(self.preds, self.gt, self.k), 0)
        self.assertGreater(mrr(self.preds, self.gt, self.k), 0)

    def test_coverage(self):
        cov = coverage(self.preds, catalog_size=100, k=3)
        self.assertGreater(cov, 0)

    def test_fairness_gap(self):
        groups = ["g1", "g1", "g2"]
        gap = fairness_gap(self.preds, self.gt, groups, self.k)
        self.assertGreaterEqual(gap, 0)

    def test_evaluate_with_bootstrap(self):
        result = evaluate(
            self.preds,
            self.gt,
            k=3,
            groups=["g1", "g1", "g2"],
            catalog_size=50,
            n_bootstrap=200,
        )
        self.assertIn("precision_at_k", result)
        self.assertIn("ci_lower", result["precision_at_k"])
        self.assertLessEqual(
            result["precision_at_k"]["ci_lower"],
            result["precision_at_k"]["ci_upper"],
        )


class TestBootstrapCI(unittest.TestCase):
    def test_ci_bounds(self):
        scores = [0.2, 0.4, 0.6, 0.8]
        mean, lo, hi = bootstrap_ci(scores, n_bootstrap=500, seed=1)
        self.assertAlmostEqual(mean, 0.5, places=5)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)


if __name__ == "__main__":
    unittest.main()
