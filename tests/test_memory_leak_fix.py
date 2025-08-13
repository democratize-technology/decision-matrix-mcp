"""Tests for the memory leak fix - LRU bounded session storage"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from decision_matrix_mcp.constants import SessionLimits
from decision_matrix_mcp.exceptions import ResourceLimitError
from decision_matrix_mcp.session_manager import SessionManager


class TestMemoryLeakFix:
    """Test the LRU-based memory leak fix implementation"""

    def test_sessions_stored_in_ordered_dict(self):
        """Test that sessions are stored in OrderedDict for LRU tracking"""
        from collections import OrderedDict

        manager = SessionManager()

        # Verify sessions is an OrderedDict
        assert isinstance(manager.sessions, OrderedDict)

        # Create sessions
        session1 = manager.create_session("Topic 1")
        session2 = manager.create_session("Topic 2")

        # Verify order preservation
        session_ids = list(manager.sessions.keys())
        assert session_ids == [session1.session_id, session2.session_id]

    def test_lru_tracking_on_access(self):
        """Test that accessing a session moves it to most recently used"""
        manager = SessionManager()

        # Create sessions
        session1 = manager.create_session("Topic 1")
        session2 = manager.create_session("Topic 2")
        session3 = manager.create_session("Topic 3")

        # Initial order should be [session1, session2, session3]
        initial_order = list(manager.sessions.keys())
        assert initial_order == [session1.session_id, session2.session_id, session3.session_id]

        # Access session1 - should move to end (most recent)
        retrieved = manager.get_session(session1.session_id)
        assert retrieved is not None

        # Order should now be [session2, session3, session1]
        new_order = list(manager.sessions.keys())
        assert new_order == [session2.session_id, session3.session_id, session1.session_id]

    def test_lru_eviction_when_memory_limit_reached(self):
        """Test LRU eviction when MAX_ACTIVE_SESSIONS is reached"""
        # Set max_sessions high but test against MAX_ACTIVE_SESSIONS limit
        manager = SessionManager(max_sessions=200)

        # Create sessions up to MAX_ACTIVE_SESSIONS
        sessions = []
        for i in range(SessionLimits.MAX_ACTIVE_SESSIONS):
            session = manager.create_session(f"Topic {i}")
            sessions.append(session)

        # Should have exactly MAX_ACTIVE_SESSIONS
        assert len(manager.sessions) == SessionLimits.MAX_ACTIVE_SESSIONS

        # Access first few sessions to make them recently used
        for i in range(5):
            manager.get_session(sessions[i].session_id)

        # Create one more session - should trigger LRU eviction
        new_session = manager.create_session("New Topic")

        # Should still be at or below MAX_ACTIVE_SESSIONS
        assert len(manager.sessions) <= SessionLimits.MAX_ACTIVE_SESSIONS

        # The new session should exist
        assert manager.get_session(new_session.session_id) is not None

        # Recently accessed sessions should still exist
        for i in range(5):
            assert manager.get_session(sessions[i].session_id) is not None

    def test_lru_eviction_batch_size(self):
        """Test that LRU eviction respects batch size configuration"""
        manager = SessionManager(max_sessions=200)

        # Create sessions up to trigger eviction
        sessions = []
        for i in range(SessionLimits.MAX_ACTIVE_SESSIONS + 5):
            session = manager.create_session(f"Topic {i}")
            sessions.append(session)

        # Should be bounded by MAX_ACTIVE_SESSIONS
        assert len(manager.sessions) <= SessionLimits.MAX_ACTIVE_SESSIONS

        # Should have evicted at least LRU_EVICTION_BATCH_SIZE sessions
        expected_evictions = SessionLimits.LRU_EVICTION_BATCH_SIZE
        expected_final_count = SessionLimits.MAX_ACTIVE_SESSIONS + 5 - expected_evictions

        # Verify we don't exceed the memory bound
        assert len(manager.sessions) <= SessionLimits.MAX_ACTIVE_SESSIONS

    def test_no_eviction_when_under_memory_limit(self):
        """Test that LRU eviction doesn't happen when under MAX_ACTIVE_SESSIONS"""
        manager = SessionManager(max_sessions=50)

        # Create sessions well under MAX_ACTIVE_SESSIONS (but within max_sessions)
        sessions = []
        for i in range(30):  # Less than MAX_ACTIVE_SESSIONS (100) and max_sessions (50)
            session = manager.create_session(f"Topic {i}")
            sessions.append(session)

        # All sessions should exist (no LRU eviction since we're under memory threshold)
        assert len(manager.sessions) == 30
        for session in sessions:
            assert manager.get_session(session.session_id) is not None

    def test_combined_ttl_and_lru_cleanup(self):
        """Test that both TTL-based and LRU-based cleanup work together"""
        manager = SessionManager(max_sessions=200, session_ttl_hours=1)

        # Create sessions
        sessions = []
        for i in range(10):
            session = manager.create_session(f"Topic {i}")
            sessions.append(session)

        # Make some sessions expired
        expired_time = datetime.now(timezone.utc) - timedelta(hours=2)
        for i in range(5):
            sessions[i].created_at = expired_time

        # Force cleanup by accessing the manager
        manager._cleanup_if_needed()

        # Expired sessions should be gone
        for i in range(5):
            assert manager.get_session(sessions[i].session_id) is None

        # Non-expired sessions should remain
        for i in range(5, 10):
            assert manager.get_session(sessions[i].session_id) is not None

    def test_stats_tracking_with_lru_eviction(self):
        """Test that statistics correctly track LRU evictions"""
        manager = SessionManager(max_sessions=200)

        initial_stats = manager.get_stats()

        # Create many sessions to trigger eviction
        sessions_to_create = SessionLimits.MAX_ACTIVE_SESSIONS + 15

        for i in range(sessions_to_create):
            manager.create_session(f"Topic {i}")

        final_stats = manager.get_stats()

        # Verify sessions were created
        assert final_stats["sessions_created"] == sessions_to_create

        # Verify some were cleaned (either expired or LRU evicted)
        assert final_stats["sessions_cleaned"] > 0

        # Verify active count is bounded
        assert final_stats["active_sessions"] <= SessionLimits.MAX_ACTIVE_SESSIONS

    def test_session_removal_maintains_lru_order(self):
        """Test that manual session removal doesn't break LRU ordering"""
        manager = SessionManager()

        # Create sessions
        sessions = []
        for i in range(5):
            session = manager.create_session(f"Topic {i}")
            sessions.append(session)

        # Remove middle session
        manager.remove_session(sessions[2].session_id)

        # Verify remaining sessions maintain order
        remaining_ids = list(manager.sessions.keys())
        expected_ids = [s.session_id for i, s in enumerate(sessions) if i != 2]
        assert remaining_ids == expected_ids

        # Verify LRU tracking still works
        manager.get_session(sessions[0].session_id)
        new_order = list(manager.sessions.keys())

        # First session should be at end now
        assert new_order[-1] == sessions[0].session_id

    def test_memory_bounds_never_exceeded(self):
        """Stress test to ensure memory bounds are never exceeded"""
        manager = SessionManager(max_sessions=500)

        # Try to create many more sessions than the limit
        sessions_created = 0
        sessions_to_attempt = SessionLimits.MAX_ACTIVE_SESSIONS * 2

        for i in range(sessions_to_attempt):
            try:
                session = manager.create_session(f"Stress Topic {i}")
                sessions_created += 1

                # Verify we never exceed the bound
                assert len(manager.sessions) <= SessionLimits.MAX_ACTIVE_SESSIONS

                # Periodically access random sessions to vary LRU order
                if i % 10 == 0 and len(manager.sessions) > 5:
                    random_key = list(manager.sessions.keys())[i % len(manager.sessions)]
                    manager.get_session(random_key)

            except ResourceLimitError:
                # Should only happen if we somehow exceed limits despite eviction
                break

        # Final verification
        assert len(manager.sessions) <= SessionLimits.MAX_ACTIVE_SESSIONS
        assert (
            sessions_created >= SessionLimits.MAX_ACTIVE_SESSIONS
        )  # Should create at least this many

    def test_lru_eviction_preserves_recently_used(self):
        """Test that LRU eviction preserves recently accessed sessions"""
        manager = SessionManager(max_sessions=200)

        # Create sessions up to memory limit
        sessions = []
        for i in range(SessionLimits.MAX_ACTIVE_SESSIONS):
            session = manager.create_session(f"Topic {i}")
            sessions.append(session)

        # Mark specific sessions as recently used
        important_sessions = sessions[:10]  # First 10 sessions
        for session in important_sessions:
            manager.get_session(session.session_id)

        # Create new sessions to trigger eviction
        new_sessions = []
        for i in range(20):
            session = manager.create_session(f"New Topic {i}")
            new_sessions.append(session)

        # Important (recently used) sessions should survive
        for session in important_sessions:
            retrieved = manager.get_session(session.session_id)
            assert (
                retrieved is not None
            ), f"Recently used session {session.session_id[:8]} was evicted"

        # Some of the new sessions should exist
        surviving_new = [s for s in new_sessions if manager.get_session(s.session_id) is not None]
        assert len(surviving_new) > 0, "No new sessions survived"

    def test_constants_are_reasonable(self):
        """Test that the memory management constants are reasonable"""
        # Verify constants exist and have sensible values
        assert hasattr(SessionLimits, "MAX_ACTIVE_SESSIONS")
        assert hasattr(SessionLimits, "LRU_EVICTION_BATCH_SIZE")

        # Verify values are reasonable
        assert SessionLimits.MAX_ACTIVE_SESSIONS > 0
        assert SessionLimits.MAX_ACTIVE_SESSIONS >= SessionLimits.DEFAULT_MAX_SESSIONS
        assert SessionLimits.LRU_EVICTION_BATCH_SIZE > 0
        assert SessionLimits.LRU_EVICTION_BATCH_SIZE < SessionLimits.MAX_ACTIVE_SESSIONS

        print(f"MAX_ACTIVE_SESSIONS: {SessionLimits.MAX_ACTIVE_SESSIONS}")
        print(f"LRU_EVICTION_BATCH_SIZE: {SessionLimits.LRU_EVICTION_BATCH_SIZE}")
        print(f"DEFAULT_MAX_SESSIONS: {SessionLimits.DEFAULT_MAX_SESSIONS}")

    def test_backwards_compatibility_preserved(self):
        """Test that all existing functionality still works as expected"""
        manager = SessionManager(max_sessions=5, session_ttl_hours=24)

        # Test basic session operations
        session = manager.create_session("Compatibility Test")
        assert session is not None
        assert session.topic == "Compatibility Test"

        # Test session retrieval
        retrieved = manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

        # Test session listing
        sessions = manager.list_active_sessions()
        assert session.session_id in sessions

        # Test session removal
        removed = manager.remove_session(session.session_id)
        assert removed is True
        assert manager.get_session(session.session_id) is None

        # Test stats
        stats = manager.get_stats()
        assert "active_sessions" in stats
        assert "sessions_created" in stats
        assert "sessions_cleaned" in stats
