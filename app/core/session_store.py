"""In-memory хранилище диалоговых сессий с TTL."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.services.belief_state import BeliefState

DEFAULT_TTL_SECONDS = 30 * 60  # 30 минут


@dataclass
class SessionRecord:
    belief: BeliefState
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    """Потокобезопасное для MVP in-memory хранилище сессий."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._sessions: Dict[str, SessionRecord] = {}

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, rec in self._sessions.items()
            if now - rec.updated_at > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]

    def create_session(self, belief: Optional[BeliefState] = None) -> str:
        self._purge_expired()
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionRecord(belief=belief or BeliefState())
        return session_id

    def get(self, session_id: str) -> Optional[SessionRecord]:
        self._purge_expired()
        rec = self._sessions.get(session_id)
        if rec is None:
            return None
        if time.time() - rec.updated_at > self._ttl:
            del self._sessions[session_id]
            return None
        return rec

    def save(self, session_id: str, belief: BeliefState) -> bool:
        self._purge_expired()
        rec = self._sessions.get(session_id)
        if rec is None:
            return False
        rec.belief = belief
        rec.updated_at = time.time()
        return True

    def get_or_create(
        self, session_id: Optional[str], prev_belief: Optional[dict] = None
    ) -> tuple[str, BeliefState]:
        """Возвращает (session_id, belief). Создаёт сессию при отсутствии."""
        if prev_belief is not None:
            belief = BeliefState.from_dict(prev_belief)
        else:
            belief = BeliefState()

        if session_id:
            rec = self.get(session_id)
            if rec is not None:
                return session_id, rec.belief
            self._purge_expired()
            self._sessions[session_id] = SessionRecord(belief=belief)
            return session_id, belief

        new_id = self.create_session(belief)
        return new_id, belief


# Singleton для API
session_store = SessionStore()
