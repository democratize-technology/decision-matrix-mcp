# MIT License
#
# Copyright (c) 2025 Democratize Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Session management for Decision Matrix MCP"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from .constants import ValidationLimits
from .exceptions import ResourceLimitError
from .models import DecisionSession

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(
        self,
        max_sessions: int = 50,
        session_ttl_hours: int = 24,
        cleanup_interval_minutes: int = 30,
    ):
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)

        self.sessions: dict[str, DecisionSession] = {}
        self.last_cleanup = datetime.now(timezone.utc)

        self.stats = {
            "sessions_created": 0,
            "sessions_expired": 0,
            "sessions_cleaned": 0,
            "max_concurrent": 0,
        }

    def create_session(
        self, topic: str, initial_options: list[str] | None = None, temperature: float = 0.1
    ) -> DecisionSession:
        self._cleanup_if_needed()

        if len(self.sessions) >= self.max_sessions:
            self._cleanup_expired_sessions()

            if len(self.sessions) >= self.max_sessions:
                raise ResourceLimitError(
                    f"Session limit of {self.max_sessions} exceeded",
                    f"Maximum number of active sessions ({self.max_sessions}) reached. Please try again later.",
                )

        session_id = str(uuid4())
        session = DecisionSession(
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
            topic=topic,
            default_temperature=temperature,
        )

        if initial_options:
            for option_name in initial_options:
                session.add_option(option_name)

        self.sessions[session_id] = session

        self.stats["sessions_created"] += 1
        self.stats["max_concurrent"] = max(self.stats["max_concurrent"], len(self.sessions))

        logger.info(f"Created session {session_id[:8]} for topic: {topic}")
        return session

    def get_session(self, session_id: str) -> DecisionSession | None:
        self._cleanup_if_needed()

        session = self.sessions.get(session_id)
        if session and self._is_session_expired(session):
            self._remove_session(session_id)
            return None

        return session

    def remove_session(self, session_id: str) -> bool:
        return self._remove_session(session_id)

    def list_active_sessions(self) -> dict[str, DecisionSession]:
        self._cleanup_if_needed()
        return self.sessions.copy()

    def clear_all_sessions(self) -> None:
        self.sessions.clear()

    def get_stats(self) -> dict[str, Any]:
        return {
            **self.stats,
            "active_sessions": len(self.sessions),
            "last_cleanup": self.last_cleanup.isoformat(),
        }

    def get_current_session(self) -> DecisionSession | None:
        self._cleanup_if_needed()

        if not self.sessions:
            return None

        sorted_sessions = sorted(self.sessions.items(), key=lambda x: x[1].created_at, reverse=True)

        return sorted_sessions[0][1]

    def _cleanup_if_needed(self) -> None:
        if datetime.now(timezone.utc) - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired_sessions()

    def _cleanup_expired_sessions(self) -> None:
        now = datetime.now(timezone.utc)
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self._remove_session(session_id)

        self.last_cleanup = now

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def _is_session_expired(self, session: DecisionSession) -> bool:
        return datetime.now(timezone.utc) - session.created_at > self.session_ttl

    def _remove_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.stats["sessions_cleaned"] += 1
            logger.debug(f"Removed session {session_id[:8]}")
            return True
        return False


class SessionValidator:
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        if not session_id or not isinstance(session_id, str):
            return False
        if len(session_id) > ValidationLimits.MAX_SESSION_ID_LENGTH:
            return False
        return True

    @staticmethod
    def validate_topic(topic: str) -> bool:
        if not topic or not isinstance(topic, str):
            return False
        if len(topic.strip()) == 0 or len(topic) > ValidationLimits.MAX_TOPIC_LENGTH:
            return False
        return True

    @staticmethod
    def validate_option_name(option_name: str) -> bool:
        if not option_name or not isinstance(option_name, str):
            return False
        if (
            len(option_name.strip()) == 0
            or len(option_name) > ValidationLimits.MAX_OPTION_NAME_LENGTH
        ):
            return False
        return True

    @staticmethod
    def validate_criterion_name(criterion_name: str) -> bool:
        if not criterion_name or not isinstance(criterion_name, str):
            return False
        if (
            len(criterion_name.strip()) == 0
            or len(criterion_name) > ValidationLimits.MAX_CRITERION_NAME_LENGTH
        ):
            return False
        return True

    @staticmethod
    def validate_weight(weight: float) -> bool:
        if not isinstance(weight, (int, float)):
            return False
        return (
            ValidationLimits.MIN_CRITERION_WEIGHT <= weight <= ValidationLimits.MAX_CRITERION_WEIGHT
        )

    @staticmethod
    def validate_description(description: str) -> bool:
        if not description or not isinstance(description, str):
            return False
        if (
            len(description.strip()) == 0
            or len(description) > ValidationLimits.MAX_DESCRIPTION_LENGTH
        ):
            return False
        return True
