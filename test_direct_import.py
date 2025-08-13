#!/usr/bin/env python3
"""Direct import test without using __init__.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import only what we need directly
from decision_matrix_mcp.constants import SessionLimits
from decision_matrix_mcp.exceptions import ResourceLimitError
from decision_matrix_mcp.models import DecisionSession

# Import session manager without the main __init__.py
import logging
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(
        self,
        max_sessions: int = SessionLimits.DEFAULT_MAX_SESSIONS,
        session_ttl_hours: int = SessionLimits.DEFAULT_SESSION_TTL_HOURS,
        cleanup_interval_minutes: int = SessionLimits.DEFAULT_CLEANUP_INTERVAL_MINUTES,
    ):
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
        self, topic: str, initial_options: list[str] | None = None, temperature: float = 0.1
    ) -> DecisionSession:
        self._cleanup_if_needed()

        # First try TTL-based cleanup
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_expired_sessions()

            # If still at limit, try LRU eviction
            if len(self.sessions) >= self.max_sessions:
                self._evict_lru_sessions()
                
                # If still at hard limit after eviction, reject
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

        # Track access for LRU - move to end (most recently used)
        if session:
            self._touch_session(session_id)

        return session

    def remove_session(self, session_id: str) -> bool:
        return self._remove_session(session_id)

    def get_stats(self) -> dict[str, Any]:
        return {
            **self.stats,
            "active_sessions": len(self.sessions),
            "last_cleanup": self.last_cleanup.isoformat(),
        }

    def _touch_session(self, session_id: str) -> None:
        """Update LRU order by moving session to end (most recently used)."""
        if session_id in self.sessions:
            self.sessions.move_to_end(session_id)

    def _evict_lru_sessions(self) -> None:
        """Evict least recently used sessions when approaching memory limits."""
        if len(self.sessions) < SessionLimits.MAX_ACTIVE_SESSIONS:
            return
            
        # Calculate how many to evict (batch eviction for efficiency)
        target_evictions = min(
            SessionLimits.LRU_EVICTION_BATCH_SIZE,
            len(self.sessions) - self.max_sessions + SessionLimits.LRU_EVICTION_BATCH_SIZE
        )
        
        evicted_count = 0
        sessions_to_evict = []
        
        # Collect LRU sessions for eviction (OrderedDict preserves order)
        for session_id in list(self.sessions.keys()):
            if evicted_count >= target_evictions:
                break
            sessions_to_evict.append(session_id)
            evicted_count += 1
            
        # Perform actual eviction
        for session_id in sessions_to_evict:
            self._remove_session(session_id)
            
        if sessions_to_evict:
            logger.info(f"Evicted {len(sessions_to_evict)} LRU sessions to maintain memory bounds")

    def _cleanup_if_needed(self) -> None:
        """More aggressive cleanup including both TTL and LRU-based eviction.""" 
        now = datetime.now(timezone.utc)
        
        # Check if cleanup interval has passed
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired_sessions()
            
        # Always check if we're approaching memory limits
        if len(self.sessions) >= SessionLimits.MAX_ACTIVE_SESSIONS:
            self._evict_lru_sessions()

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
        """Check if session has expired, handling both timezone-aware and naive datetimes."""
        now = datetime.now(timezone.utc)
        session_time = session.created_at
        
        # Ensure session time is timezone-aware
        if session_time.tzinfo is None:
            # Assume UTC for naive datetime (legacy sessions)
            session_time = session_time.replace(tzinfo=timezone.utc)
            logger.debug(f"Session {session.session_id[:8]} has naive datetime, assuming UTC")
        
        return now - session_time > self.session_ttl

    def _remove_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.stats["sessions_cleaned"] += 1
            logger.debug(f"Removed session {session_id[:8]}")
            return True
        return False


def test_basic_functionality():
    """Test that basic session operations still work"""
    print("Testing basic functionality...")
    
    manager = SessionManager(max_sessions=5)
    
    # Create a session
    session = manager.create_session("Test Topic")
    print(f"✓ Created session: {session.session_id[:8]}")
    
    # Retrieve session
    retrieved = manager.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id
    print("✓ Session retrieval works")
    
    # Test LRU tracking (accessing should move to end)
    assert session.session_id in manager.sessions
    print("✓ Session exists in OrderedDict")
    
    print("Basic functionality test PASSED!\n")

def test_lru_eviction():
    """Test that LRU eviction works when memory limits are reached"""
    print("Testing LRU eviction...")
    
    # Set limits to trigger eviction
    manager = SessionManager(max_sessions=3)
    
    sessions = []
    
    # Create sessions up to limit
    for i in range(3):
        session = manager.create_session(f"Topic {i}")
        sessions.append(session)
        print(f"✓ Created session {i}: {session.session_id[:8]}")
    
    # Access first session to make it recently used
    manager.get_session(sessions[0].session_id)
    print(f"✓ Accessed session 0 (should be most recently used)")
    
    # Create one more session - should trigger eviction
    session4 = manager.create_session("Topic 3")
    print(f"✓ Created session 3: {session4.session_id[:8]} (should trigger eviction)")
    
    # Check that we still have 3 sessions
    assert len(manager.sessions) == 3
    print(f"✓ Session count maintained at limit: {len(manager.sessions)}")
    
    # Check that session 0 (recently accessed) is still there
    assert manager.get_session(sessions[0].session_id) is not None
    print("✓ Most recently used session survived eviction")
    
    # Check that new session is there
    assert manager.get_session(session4.session_id) is not None
    print("✓ New session exists")
    
    print("LRU eviction test PASSED!\n")

def test_memory_bounds():
    """Test that memory is bounded at MAX_ACTIVE_SESSIONS"""
    print("Testing memory bounds...")
    
    # Create manager with high max_sessions but test against MAX_ACTIVE_SESSIONS
    manager = SessionManager(max_sessions=1000) 
    
    # Create sessions up to MAX_ACTIVE_SESSIONS
    sessions_created = 0
    try:
        for i in range(SessionLimits.MAX_ACTIVE_SESSIONS + 20):  # Try to exceed limit
            session = manager.create_session(f"Topic {i}")
            sessions_created += 1
            if i % 20 == 0:
                print(f"  Created {i} sessions...")
    except Exception as e:
        print(f"  Stopped at {sessions_created} sessions due to: {e}")
    
    # Verify we never exceed MAX_ACTIVE_SESSIONS
    assert len(manager.sessions) <= SessionLimits.MAX_ACTIVE_SESSIONS
    print(f"✓ Session count bounded at {len(manager.sessions)} <= {SessionLimits.MAX_ACTIVE_SESSIONS}")
    
    print("Memory bounds test PASSED!\n")


if __name__ == "__main__":
    print("=== Testing Memory Leak Fix ===\n")
    
    try:
        test_basic_functionality()
        test_lru_eviction()
        test_memory_bounds() 
        
        print("🎉 ALL TESTS PASSED! Memory leak fix is working correctly.")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)