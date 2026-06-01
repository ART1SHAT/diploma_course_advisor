"""
Запуск offline-оценки: python -m tests.evaluation.run
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Корень проекта в PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.recommender import HybridRecommender
from app.user_profile import UserProfile
from tests.evaluation.offline_evaluator import evaluate

SCENARIOS_PATH = PROJECT_ROOT / "tests" / "data" / "expert_scenarios.json"
REPORT_PATH = PROJECT_ROOT / "tests" / "evaluation" / "report.md"
COURSES_PATH = PROJECT_ROOT / "data" / "unified_courses.json"


def load_scenarios() -> dict:
    with SCENARIOS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def profile_from_dict(data: dict) -> UserProfile:
    return UserProfile(
        budget=data.get("budget"),
        knowledge_level=data.get("knowledge_level"),
        time_availability=data.get("time_availability"),
        career_focus=data.get("career_focus"),
        interests=data.get("interests") or [],
        goals=data.get("goals") or "",
    )


def run_predictions(
    recommender: HybridRecommender,
    scenarios: list,
    k: int,
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    predictions: list[list[str]] = []
    ground_truth: list[list[str]] = []
    groups: list[str] = []

    for sc in scenarios:
        profile = profile_from_dict(sc["profile"])
        recs = recommender.recommend(profile, top_k=k)
        predictions.append([str(c["id"]) for c in recs])
        ground_truth.append(list(sc["expected_top3"]))
        groups.append(sc.get("group", "unknown"))

    return predictions, ground_truth, groups


def format_metric_row(name: str, m: dict) -> str:
    if m.get("ci_lower") is not None:
        return (
            f"| {name} | {m['value']:.4f} | "
            f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}] |"
        )
    return f"| {name} | {m['value']:.4f} | — |"


def build_report(metrics: dict, n_scenarios: int, k: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    p = metrics["precision_at_k"]
    n = metrics["ndcg_at_k"]
    m = metrics["mrr"]
    c = metrics["coverage"]
    f = metrics["fairness_gap"]

    slide = (
        f"На выборке из {n_scenarios} экспертных сценариев при k={k} система CourseAdvisor "
        f"достигла Precision@{k} = {p['value']:.2f} "
        f"(95% ДИ: {p['ci_lower']:.2f}–{p['ci_upper']:.2f}), "
        f"NDCG@{k} = {n['value']:.2f}, MRR = {m['value']:.2f}. "
        f"Покрытие каталога — {c['value']:.1%}; разрыв качества между группами "
        f"(career vs personal) — {f['value']:.3f}, что свидетельствует о "
        f"{'приемлемой' if f['value'] < 0.15 else 'требующей внимания'} справедливости выдачи."
    )

    return f"""# Offline-оценка CourseAdvisor (§3.4)

**Дата прогона:** {ts}  
**Сценариев:** {n_scenarios}  
**k:** {k}  
**Бутстрэп:** n={metrics['n_bootstrap']} (95% ДИ для Precision, NDCG, MRR)

## Таблица метрик

| Метрика | Значение | 95% ДИ |
|---------|----------|--------|
{format_metric_row(f"Precision@{k}", p)}
{format_metric_row(f"NDCG@{k}", n)}
{format_metric_row("MRR", m)}
{format_metric_row("Coverage", c)}
{format_metric_row("Fairness gap", f)}

## Интерпретация

- **Precision@{k}** — доля релевантных курсов в top-{k} относительно экспертного `expected_top3`.
- **NDCG@{k}** — качество ранжирования с учётом позиции релевантных курсов.
- **MRR** — средний обратный ранг первого релевантного курса.
- **Coverage** — доля каталога, представленная в рекомендациях хотя бы раз.
- **Fairness gap** — разница Precision@{k} между группами `career` и `personal`.

## Текст для слайда презентации

> {slide}

## Воспроизведение

```bash
python -m tests.evaluation.run
```

Данные: `tests/data/expert_scenarios.json`  
Каталог: `data/unified_courses.json`
"""


def main() -> int:
    print("Загрузка экспертных сценариев…")
    data = load_scenarios()
    scenarios = data["scenarios"]
    k = int(data.get("k_default", 5))

    print(f"Инициализация рекомендателя ({len(scenarios)} сценариев, k={k})…")
    print("(первый запуск может занять 1–2 мин: семантический индекс)")
    recommender = HybridRecommender(json_path=str(COURSES_PATH))

    catalog_size = len(recommender.courses)
    predictions, ground_truth, groups = run_predictions(
        recommender, scenarios, k
    )

    print("Расчёт метрик и бутстрэпа…")
    metrics = evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        k=k,
        groups=groups,
        catalog_size=catalog_size,
        n_bootstrap=1000,
    )

    report = build_report(metrics, len(scenarios), k)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"\nОтчёт сохранён: {REPORT_PATH}")
    print(
        f"Precision@{k}={metrics['precision_at_k']['value']:.4f} "
        f"[{metrics['precision_at_k']['ci_lower']:.4f}, {metrics['precision_at_k']['ci_upper']:.4f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
