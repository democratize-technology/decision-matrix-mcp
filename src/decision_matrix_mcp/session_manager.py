"""Session management for Decision Matrix MCP"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from .exceptions import ResourceLimitError
from .models import DecisionSession

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages multiple decision analysis sessions with automatic cleanup"""

    def __init__(
        self,
        max_sessions: int = 50,
        session_ttl_hours: int = 24,
        cleanup_interval_minutes: int = 30,
    ):
        """Initialize session manager

        Args:
            max_sessions: Maximum number of concurrent sessions
            session_ttl_hours: Hours before sessions expire
            cleanup_interval_minutes: Minutes between cleanup runs
        """
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

    def create_session(self, topic: str, initial_options: list | None = None) -> DecisionSession:
        """Create a new decision analysis session"""
        self._cleanup_if_needed()

        if len(self.sessions) >= self.max_sessions:
            self._cleanup_expired_sessions()

            if len(self.sessions) >= self.max_sessions:
                raise ResourceLimitError(
                    f"Session limit of {self.max_sessions} exceeded",
                    f"Maximum number of active sessions ({self.max_sessions}) reached. Please try again later."
                )

        session_id = str(uuid4())
        session = DecisionSession(
            session_id=session_id, created_at=datetime.now(timezone.utc), topic=topic
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
        """Get a session by ID, None if not found or expired"""
        self._cleanup_if_needed()

        session = self.sessions.get(session_id)
        if session and self._is_session_expired(session):
            self._remove_session(session_id)
            return None

        return session

    def remove_session(self, session_id: str) -> bool:
        """Manually remove a session"""
        return self._remove_session(session_id)

    def list_active_sessions(self) -> dict[str, DecisionSession]:
        """List all active (non-expired) sessions"""
        self._cleanup_if_needed()
        return self.sessions.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get session manager statistics"""
        return {
            **self.stats,
            "active_sessions": len(self.sessions),
            "last_cleanup": self.last_cleanup.isoformat(),
        }

    def _cleanup_if_needed(self) -> None:
        """Run cleanup if enough time has passed"""
        if datetime.now(timezone.utc) - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired_sessions()

    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions"""
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
        """Check if a session has expired"""
        return datetime.now(timezone.utc) - session.created_at > self.session_ttl

    def _remove_session(self, session_id: str) -> bool:
        """Remove a session and update stats"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.stats["sessions_cleaned"] += 1
            logger.debug(f"Removed session {session_id[:8]}")
            return True
        return False


class SessionValidator:
    """Validates session operations and data"""

    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """Validate session ID format"""
        if not session_id or not isinstance(session_id, str):
            return False
        if len(session_id) > 100:  # Reasonable limit
            return False
        return True

    @staticmethod
    def validate_topic(topic: str) -> bool:
        """Validate topic string"""
        if not topic or not isinstance(topic, str):
            return False
        if len(topic.strip()) == 0 or len(topic) > 500:  # Reasonable limits
            return False
        return True

    @staticmethod
    def validate_option_name(option_name: str) -> bool:
        """Validate option name"""
        if not option_name or not isinstance(option_name, str):
            return False
        if len(option_name.strip()) == 0 or len(option_name) > 200:
            return False
        return True

    @staticmethod
    def validate_criterion_name(criterion_name: str) -> bool:
        """Validate criterion name"""
        if not criterion_name or not isinstance(criterion_name, str):
            return False
        if len(criterion_name.strip()) == 0 or len(criterion_name) > 100:
            return False
        return True

    @staticmethod
    def validate_weight(weight: float) -> bool:
        """Validate criterion weight"""
        if not isinstance(weight, (int, float)):
            return False
        return 0.1 <= weight <= 10.0  # Reasonable weight range

    @staticmethod
    def validate_description(description: str) -> bool:
        """Validate description text"""
        if not description or not isinstance(description, str):
            return False
        if len(description.strip()) == 0 or len(description) > 1000:
            return False
        return True


session_manager = SessionManager()
