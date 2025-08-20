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

"""Session management for Decision Matrix MCP."""

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import logging
from typing import Any
from uuid import uuid4

from .constants import SessionLimits, ValidationLimits
from .exceptions import ResourceLimitError
from .models import DecisionSession

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages decision analysis sessions with TTL cleanup and LRU eviction.

    Provides thread-safe session management with automatic cleanup of expired
    sessions and LRU-based eviction when maximum session limits are reached.
    """

    def __init__(
        self,
        max_sessions: int = 10,  # type: ignore[assignment]
        session_ttl_hours: int = 24,  # type: ignore[assignment]
        cleanup_interval_minutes: int = 60,  # type: ignore[assignment]
    ) -> None:
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)

        # Use OrderedDict for LRU tracking - maintains insertion/access order
        self.sessions: OrderedDict[str, DecisionSession] = OrderedDict()
        self.last_cleanup = datetime.now(timezone.utc)

        self.stats = {
            "sessions_created": 0,
            "sessions_expired": 0,
            "sessions_cleaned": 0,
            "max_concurrent": 0,
        }

    def create_session(
        self,
        topic: str,
        initial_options: list[str] | None = None,
        temperature: float = 0.1,
    ) -> DecisionSession:
        """Create a new decision analysis session with the given topic and options."""
        # Always run cleanup first (includes both TTL and LRU eviction)
        self._cleanup_if_needed()

        # Check if we're at the creation limit (max_sessions)
        if len(self.sessions) >= self.max_sessions:
            # Try TTL cleanup first
            self._cleanup_expired_sessions()

            # If still at creation limit, try LRU eviction to make room
            if len(self.sessions) >= self.max_sessions:
                self._evict_lru_sessions()

                # If still at creation limit after all cleanup attempts, reject
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

        logger.info("Created session %s for topic: %s", session_id[:8], topic)
        return session

    def get_session(self, session_id: str) -> DecisionSession | None:
        """Retrieve an active session by ID, updating access time."""
        self._cleanup_if_needed()

        session = self.sessions.get(session_id)
        if session and self._is_session_expired(session):
            self._remove_session(session_id)
            return None

        # Track access for LRU - move to end (most recently used)
        if session:
            self._touch_session(session_id)

        return session

    def remove_session(self, session_id: str) -> bool:
        """Remove a session from active management."""
        return self._remove_session(session_id)

    def list_active_sessions(self) -> dict[str, DecisionSession]:
        """Get dictionary of all active sessions."""
        self._cleanup_if_needed()
        return self.sessions.copy()

    def clear_all_sessions(self) -> None:
        """Clear all sessions and reset statistics."""
        self.sessions.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get session management statistics and current state."""
        return {
            **self.stats,
            "active_sessions": len(self.sessions),
            "last_cleanup": self.last_cleanup.isoformat(),
        }

    def get_current_session(self) -> DecisionSession | None:
        """Get the most recently accessed session."""
        self._cleanup_if_needed()

        if not self.sessions:
            return None

        sorted_sessions = sorted(self.sessions.items(), key=lambda x: x[1].created_at, reverse=True)

        return sorted_sessions[0][1]

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
            logger.info("Cleaned up %d expired sessions", len(expired_sessions))

    def _is_session_expired(self, session: DecisionSession) -> bool:
        """Check if session has expired, handling both timezone-aware and naive datetimes."""
        now = datetime.now(timezone.utc)
        session_time = session.created_at

        # Ensure session time is timezone-aware
        if session_time.tzinfo is None:
            # Assume UTC for naive datetime (legacy sessions)
            session_time = session_time.replace(tzinfo=timezone.utc)
            logger.debug("Session %s has naive datetime, assuming UTC", session.session_id[:8])

        return now - session_time > self.session_ttl

    def _touch_session(self, session_id: str) -> None:
        """Update LRU order by moving session to end (most recently used)."""
        if session_id in self.sessions:
            self.sessions.move_to_end(session_id)

    def _evict_lru_sessions(self) -> None:
        """Evict least recently used sessions when approaching memory limits."""
        current_count = len(self.sessions)

        # Only evict if we're approaching the memory limit (95% threshold)
        memory_threshold = int(SessionLimits.MAX_ACTIVE_SESSIONS * 0.95)
        if current_count < memory_threshold:
            return

        # Calculate target final size: try to get down to a safe level below the limit
        # Aim for the smaller of: max_sessions limit, or 90% of MAX_ACTIVE_SESSIONS
        target_final_size = min(self.max_sessions, int(SessionLimits.MAX_ACTIVE_SESSIONS * 0.9))

        # Calculate how many to evict
        if current_count > target_final_size:
            target_evictions = current_count - target_final_size
            # Cap at batch size to avoid massive cleanups
            target_evictions = min(target_evictions, SessionLimits.LRU_EVICTION_BATCH_SIZE)
        else:
            # If we're not that far over, just evict the batch size
            target_evictions = SessionLimits.LRU_EVICTION_BATCH_SIZE

        # Ensure we evict at least 1 but never more than available
        target_evictions = max(1, min(target_evictions, current_count - 1))

        sessions_to_evict = []

        # Collect LRU sessions for eviction (first items in OrderedDict are oldest/LRU)
        for session_id in list(self.sessions.keys()):
            if len(sessions_to_evict) >= target_evictions:
                break
            sessions_to_evict.append(session_id)

        # Perform actual eviction
        for session_id in sessions_to_evict:
            self._remove_session(session_id)

        if sessions_to_evict:
            logger.info(
                "Evicted %d LRU sessions to maintain memory bounds (was %d, now %d)",
                len(sessions_to_evict),
                current_count,
                len(self.sessions),
            )

    def _cleanup_if_needed(self) -> None:
        """More aggressive cleanup including both TTL and LRU-based eviction."""
        now = datetime.now(timezone.utc)

        # Check if cleanup interval has passed
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired_sessions()

        # Always check if we're approaching memory limits - be more proactive
        # Start LRU eviction when we get close to the limit, not just at it
        memory_threshold = int(SessionLimits.MAX_ACTIVE_SESSIONS * 0.95)  # 95% of limit
        if len(self.sessions) >= memory_threshold:
            self._evict_lru_sessions()

    def _remove_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.stats["sessions_cleaned"] += 1
            logger.debug("Removed session %s", session_id[:8])
            return True
        return False


class SessionValidator:
    """Validation utilities for session-related inputs."""

    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """Validate session ID format and length."""
        if not session_id or not isinstance(session_id, str):
            return False
        return not len(session_id) > ValidationLimits.MAX_SESSION_ID_LENGTH

    @staticmethod
    def validate_topic(topic: str) -> bool:
        """Validate decision topic format and length."""
        if not topic or not isinstance(topic, str):
            return False
        return not (len(topic.strip()) == 0 or len(topic) > ValidationLimits.MAX_TOPIC_LENGTH)

    @staticmethod
    def validate_option_name(option_name: str) -> bool:
        """Validate option name format and length."""
        if not option_name or not isinstance(option_name, str):
            return False
        return not (
            len(option_name.strip()) == 0
            or len(option_name) > ValidationLimits.MAX_OPTION_NAME_LENGTH
        )

    @staticmethod
    def validate_criterion_name(criterion_name: str) -> bool:
        """Validate criterion name format and length."""
        if not criterion_name or not isinstance(criterion_name, str):
            return False
        return not (
            len(criterion_name.strip()) == 0
            or len(criterion_name) > ValidationLimits.MAX_CRITERION_NAME_LENGTH
        )

    @staticmethod
    def validate_weight(weight: float) -> bool:
        """Validate criterion weight is within allowed range."""
        if not isinstance(weight, (int, float)):
            return False
        return (
            ValidationLimits.MIN_CRITERION_WEIGHT <= weight <= ValidationLimits.MAX_CRITERION_WEIGHT
        )

    @staticmethod
    def validate_description(description: str) -> bool:
        """Validate description format and length."""
        if not description or not isinstance(description, str):
            return False
        return not (
            len(description.strip()) == 0
            or len(description) > ValidationLimits.MAX_DESCRIPTION_LENGTH
        )
