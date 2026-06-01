"""Аудит трассировки объяснений (§2.3): JSONL с timestamp, profile_hash, правилами."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "explanation_audit.jsonl"


def profile_hash(profile_data: Dict[str, Any]) -> str:
    """Стабильный хеш профиля для аудита без хранения PII целиком."""
    normalized = json.dumps(profile_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def log_explanation_trace(
    *,
    course_id: str,
    profile_data: Dict[str, Any],
    fuzzy_trace: List[Dict[str, Any]],
    confidence: float,
    reason_types: Optional[List[str]] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Записывает строку JSON в audit log.
    Возвращает созданную запись (для тестов).
    """
    activated = [
        t["rule_id"]
        for t in fuzzy_trace
        if float(t.get("activation", 0)) > 0.1
    ]

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile_hash": profile_hash(profile_data),
        "course_id": str(course_id),
        "activated_rules": activated,
        "confidence": round(confidence, 4),
        "reason_types": reason_types or [],
    }

    path = log_path or DEFAULT_LOG_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("Не удалось записать audit log: %s", e)

    return entry
