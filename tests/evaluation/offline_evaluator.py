"""
Offline-метрики качества рекомендаций (§3.4).
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

MetricPerQuery = List[float]


def precision_at_k(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int,
) -> float:
    """Precision@k: доля релевантных среди top-k усреднённая по запросам."""
    if not predictions:
        return 0.0
    scores: MetricPerQuery = []
    for pred, gt in zip(predictions, ground_truth):
        pred_k = pred[:k]
        if not pred_k:
            scores.append(0.0)
            continue
        gt_set = set(gt)
        hits = sum(1 for item in pred_k if item in gt_set)
        scores.append(hits / len(pred_k))
    return sum(scores) / len(scores)


def _per_query_precision(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int,
) -> MetricPerQuery:
    scores: MetricPerQuery = []
    for pred, gt in zip(predictions, ground_truth):
        pred_k = pred[:k]
        if not pred_k:
            scores.append(0.0)
            continue
        gt_set = set(gt)
        hits = sum(1 for item in pred_k if item in gt_set)
        scores.append(hits / len(pred_k))
    return scores


def ndcg_at_k(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int,
) -> float:
    """NDCG@k с бинарной релевантностью."""
    if not predictions:
        return 0.0
    scores: MetricPerQuery = []
    for pred, gt in zip(predictions, ground_truth):
        gt_set = set(gt)
        dcg = 0.0
        for i, item in enumerate(pred[:k]):
            rel = 1.0 if item in gt_set else 0.0
            if rel > 0:
                dcg += rel / math.log2(i + 2)
        ideal_hits = min(len(gt_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores)


def _per_query_ndcg(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int,
) -> MetricPerQuery:
    scores: MetricPerQuery = []
    for pred, gt in zip(predictions, ground_truth):
        gt_set = set(gt)
        dcg = 0.0
        for i, item in enumerate(pred[:k]):
            if item in gt_set:
                dcg += 1.0 / math.log2(i + 2)
        ideal_hits = min(len(gt_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return scores


def mrr(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: Optional[int] = None,
) -> float:
    """Mean Reciprocal Rank первого релевантного документа."""
    if not predictions:
        return 0.0
    scores: MetricPerQuery = []
    for pred, gt in zip(predictions, ground_truth):
        gt_set = set(gt)
        ranked = pred[:k] if k else pred
        rr = 0.0
        for i, item in enumerate(ranked):
            if item in gt_set:
                rr = 1.0 / (i + 1)
                break
        scores.append(rr)
    return sum(scores) / len(scores)


def _per_query_mrr(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: Optional[int] = None,
) -> MetricPerQuery:
    scores: MetricPerQuery = []
    for pred, gt in zip(predictions, ground_truth):
        gt_set = set(gt)
        ranked = pred[:k] if k else pred
        rr = 0.0
        for i, item in enumerate(ranked):
            if item in gt_set:
                rr = 1.0 / (i + 1)
                break
        scores.append(rr)
    return scores


def coverage(
    predictions: List[List[str]],
    catalog_size: int,
    k: int,
) -> float:
    """Доля каталога, попавшая хотя бы в один top-k."""
    if catalog_size <= 0:
        return 0.0
    recommended: set[str] = set()
    for pred in predictions:
        recommended.update(pred[:k])
    return len(recommended) / catalog_size


def fairness_gap(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    groups: Sequence[str],
    k: int,
) -> float:
    """
    Разрыв Precision@k между группами пользователей (max − min).
  """
    if not groups or len(set(groups)) < 2:
        return 0.0
    by_group: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)

    group_prec: List[float] = []
    for indices in by_group.values():
        preds = [predictions[i] for i in indices]
        gts = [ground_truth[i] for i in indices]
        group_prec.append(precision_at_k(preds, gts, k))

    return max(group_prec) - min(group_prec)


def bootstrap_ci(
    per_query_scores: MetricPerQuery,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Бутстрэп 95% ДИ для среднего по запросам.
    Возвращает (mean, ci_lower, ci_upper).
    """
    if not per_query_scores:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    n = len(per_query_scores)
    means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [per_query_scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    point = sum(per_query_scores) / n
    return point, means[lo_idx], means[hi_idx]


def _metric_with_ci(
    name: str,
    per_query: MetricPerQuery,
    n_bootstrap: int,
) -> Dict[str, Any]:
    value, lo, hi = bootstrap_ci(per_query, n_bootstrap=n_bootstrap)
    return {
        "value": round(value, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
    }


def evaluate(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int,
    *,
    groups: Optional[Sequence[str]] = None,
    catalog_size: Optional[int] = None,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Полный отчёт метрик с 95% ДИ (бутстрэп, n=1000).

    Args:
        predictions: список ранжированных id курсов на запрос
        ground_truth: список релевантных id (экспертная разметка)
        k: cutoff
        groups: метки групп для fairness_gap
        catalog_size: размер каталога для coverage
        n_bootstrap: число итераций бутстрэпа
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions и ground_truth должны иметь одинаковую длину")

    per_p = _per_query_precision(predictions, ground_truth, k)
    per_n = _per_query_ndcg(predictions, ground_truth, k)
    per_m = _per_query_mrr(predictions, ground_truth, k)

    cov_value = coverage(
        predictions,
        catalog_size if catalog_size is not None else _estimate_catalog(predictions),
        k,
    )

    fgap = fairness_gap(
        predictions, ground_truth, groups or [], k
    )

    return {
        "k": k,
        "n_queries": len(predictions),
        "precision_at_k": _metric_with_ci("precision_at_k", per_p, n_bootstrap),
        "ndcg_at_k": _metric_with_ci("ndcg_at_k", per_n, n_bootstrap),
        "mrr": _metric_with_ci("mrr", per_m, n_bootstrap),
        "coverage": {
            "value": round(cov_value, 4),
            "ci_lower": None,
            "ci_upper": None,
        },
        "fairness_gap": {
            "value": round(fgap, 4),
            "ci_lower": None,
            "ci_upper": None,
        },
        "n_bootstrap": n_bootstrap,
    }


def _estimate_catalog(predictions: List[List[str]]) -> int:
    items: set[str] = set()
    for pred in predictions:
        items.update(pred)
    return max(len(items), 1)
