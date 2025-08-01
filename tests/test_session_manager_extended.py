"""Extended tests for session manager edge cases and uncovered code"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from decision_matrix_mcp.exceptions import ResourceLimitError
from decision_matrix_mcp.session_manager import SessionManager, SessionValidator


class TestSessionManagerEdgeCases:
    """Test edge cases and uncovered functionality in SessionManager"""

    def test_session_expiration(self):
        """Test session expiration logic"""
        manager = SessionManager(max_sessions=5, session_ttl_hours=1)

        # Create a session
        session = manager.create_session("Test topic")
        session_id = session.session_id

        # Session should be retrievable immediately
        assert manager.get_session(session_id) is not None

        # Mock time to make session expired
        expired_time = datetime.now(timezone.utc) + timedelta(hours=2)
        with patch("decision_matrix_mcp.session_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = expired_time
            mock_datetime.timezone = timezone

            # Should return None for expired session
            assert manager.get_session(session_id) is None

            # Session should be removed from manager
            assert session_id not in manager.sessions

    def test_cleanup_interval(self):
        """Test automatic cleanup based on interval"""
        manager = SessionManager(max_sessions=10, session_ttl_hours=1, cleanup_interval_minutes=30)

        # Create sessions
        manager.create_session("Topic 1")
        manager.create_session("Topic 2")

        # Initially, cleanup should not run (too soon)
        initial_cleanup = manager.last_cleanup
        manager._cleanup_if_needed()
        assert manager.last_cleanup == initial_cleanup

        # Mock time to trigger cleanup
        future_time = datetime.now(timezone.utc) + timedelta(minutes=31)
        with patch("decision_matrix_mcp.session_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = future_time
            mock_datetime.timezone = timezone

            manager._cleanup_if_needed()
            assert manager.last_cleanup > initial_cleanup

    def test_cleanup_expired_sessions_during_create(self):
        """Test that expired sessions are cleaned up when hitting limit"""
        manager = SessionManager(max_sessions=2, session_ttl_hours=1)

        # Create max sessions
        session1 = manager.create_session("Topic 1")
        manager.create_session("Topic 2")

        # Make first session expired
        session1.created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Creating third session should succeed after cleanup
        session3 = manager.create_session("Topic 3")
        assert session3 is not None

        # First session should be gone
        assert manager.get_session(session1.session_id) is None
        assert len(manager.sessions) == 2

    def test_session_stats_tracking(self):
        """Test statistics tracking"""
        manager = SessionManager(max_sessions=5)

        # Initial stats
        stats = manager.get_stats()
        assert stats["sessions_created"] == 0
        assert stats["active_sessions"] == 0

        # Create sessions
        session1 = manager.create_session("Topic 1")
        manager.create_session("Topic 2")

        stats = manager.get_stats()
        assert stats["sessions_created"] == 2
        assert stats["active_sessions"] == 2
        assert stats["max_concurrent"] == 2

        # Remove a session
        manager.remove_session(session1.session_id)

        stats = manager.get_stats()
        assert stats["sessions_cleaned"] == 1
        assert stats["active_sessions"] == 1

        # Create another session
        manager.create_session("Topic 3")

        stats = manager.get_stats()
        assert stats["sessions_created"] == 3
        assert stats["max_concurrent"] == 2  # Still 2, not 3

    def test_remove_nonexistent_session(self):
        """Test removing a session that doesn't exist"""
        manager = SessionManager()

        # Should return False for non-existent session
        assert manager.remove_session("fake-session-id") is False

        # Stats should not change
        stats = manager.get_stats()
        assert stats["sessions_cleaned"] == 0

    def test_list_active_sessions_with_cleanup(self):
        """Test listing sessions triggers cleanup"""
        manager = SessionManager(cleanup_interval_minutes=30)

        # Create sessions
        session1 = manager.create_session("Topic 1")
        session2 = manager.create_session("Topic 2")

        # Make one expired
        session1.created_at = datetime.now(timezone.utc) - timedelta(days=2)

        # Mock time to trigger cleanup
        future_time = datetime.now(timezone.utc) + timedelta(minutes=31)
        with patch("decision_matrix_mcp.session_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = future_time
            mock_datetime.timezone = timezone

            # List should trigger cleanup
            active = manager.list_active_sessions()

            # Only non-expired session should be listed
            assert len(active) == 1
            assert session2.session_id in active

    def test_session_manager_isolation(self):
        """Test that session manager returns copies, not references"""
        manager = SessionManager()

        manager.create_session("Test")

        # Get active sessions
        active = manager.list_active_sessions()

        # Modify the returned dict
        active.clear()

        # Original should be unchanged
        assert len(manager.sessions) == 1

    def test_max_concurrent_tracking_with_removals(self):
        """Test max_concurrent stat with session removals"""
        manager = SessionManager()

        # Create 3 sessions
        sessions = []
        for i in range(3):
            sessions.append(manager.create_session(f"Topic {i}"))

        assert manager.get_stats()["max_concurrent"] == 3

        # Remove 2 sessions
        manager.remove_session(sessions[0].session_id)
        manager.remove_session(sessions[1].session_id)

        # Create 1 more
        manager.create_session("Topic 4")

        # Max concurrent should still be 3
        assert manager.get_stats()["max_concurrent"] == 3

    def test_cleanup_logging(self):
        """Test that cleanup logs correctly"""
        manager = SessionManager(session_ttl_hours=0)  # Immediate expiration

        # Create sessions
        manager.create_session("Topic 1")
        manager.create_session("Topic 2")

        # Force cleanup
        with patch("decision_matrix_mcp.session_manager.logger") as mock_logger:
            manager._cleanup_expired_sessions()

            # Should log cleanup
            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            assert "2 expired sessions" in call_args


class TestSessionValidatorEdgeCases:
    """Test edge cases for SessionValidator"""

    def test_validate_session_id_edge_cases(self):
        """Test session ID validation edge cases"""
        # Valid cases
        assert SessionValidator.validate_session_id("valid-id")
        assert SessionValidator.validate_session_id("a" * 100)  # Max length

        # Invalid cases
        assert not SessionValidator.validate_session_id("")
        assert not SessionValidator.validate_session_id(None)
        assert not SessionValidator.validate_session_id(123)  # Not a string
        assert not SessionValidator.validate_session_id("a" * 101)  # Too long

    def test_validate_topic_edge_cases(self):
        """Test topic validation edge cases"""
        # Valid cases
        assert SessionValidator.validate_topic("a")  # Single char
        assert SessionValidator.validate_topic("Valid topic with spaces")
        assert SessionValidator.validate_topic("x" * 500)  # Max length

        # Invalid cases
        assert not SessionValidator.validate_topic(123)  # Not a string
        assert not SessionValidator.validate_topic([])  # Not a string

    def test_validate_option_name_edge_cases(self):
        """Test option name validation edge cases"""
        # Valid cases
        assert SessionValidator.validate_option_name("a")  # Single char
        assert SessionValidator.validate_option_name("Option-123_v2.0")
        assert SessionValidator.validate_option_name("x" * 200)  # Max length

        # Invalid cases
        assert not SessionValidator.validate_option_name(123)  # Not a string
        assert not SessionValidator.validate_option_name({})  # Not a string

    def test_validate_criterion_name_edge_cases(self):
        """Test criterion name validation edge cases"""
        # Valid cases
        assert SessionValidator.validate_criterion_name("a")  # Single char
        assert SessionValidator.validate_criterion_name("Cost-Benefit Analysis")
        assert SessionValidator.validate_criterion_name("x" * 100)  # Max length

        # Invalid cases
        assert not SessionValidator.validate_criterion_name("")
        assert not SessionValidator.validate_criterion_name("   ")
        assert not SessionValidator.validate_criterion_name("x" * 101)  # Too long
        assert not SessionValidator.validate_criterion_name(None)
        assert not SessionValidator.validate_criterion_name(123)  # Not a string

    def test_validate_weight_edge_cases(self):
        """Test weight validation edge cases"""
        # Valid cases at boundaries
        assert SessionValidator.validate_weight(0.1)  # Min
        assert SessionValidator.validate_weight(10.0)  # Max
        assert SessionValidator.validate_weight(5)  # Integer

        # Invalid cases
        assert not SessionValidator.validate_weight(0.09)  # Below min
        assert not SessionValidator.validate_weight(10.01)  # Above max
        assert not SessionValidator.validate_weight("5")  # String
        assert not SessionValidator.validate_weight(None)

    def test_validate_description_edge_cases(self):
        """Test description validation edge cases"""
        # Valid cases
        assert SessionValidator.validate_description("a")  # Single char
        assert SessionValidator.validate_description("x" * 1000)  # Max length
        assert SessionValidator.validate_description("Multi\nline\ndescription")

        # Invalid cases
        assert not SessionValidator.validate_description("")
        assert not SessionValidator.validate_description("   ")
        assert not SessionValidator.validate_description("x" * 1001)  # Too long
        assert not SessionValidator.validate_description(None)
        assert not SessionValidator.validate_description(123)  # Not a string
        assert not SessionValidator.validate_description([])  # Not a string


class TestSessionManagerConcurrency:
    """Test concurrent session operations"""

    def test_session_creation_near_limit(self):
        """Test session creation when near the limit"""
        manager = SessionManager(max_sessions=3)

        # Create sessions up to limit - 1
        manager.create_session("Topic 1")
        manager.create_session("Topic 2")

        # Should still be able to create one more
        session3 = manager.create_session("Topic 3")
        assert session3 is not None

        # Now at limit, next should fail
        with pytest.raises(ResourceLimitError):
            manager.create_session("Topic 4")

    def test_expired_session_cleanup_under_load(self):
        """Test cleanup behavior with many expired sessions"""
        manager = SessionManager(max_sessions=5, session_ttl_hours=1)

        # Create max sessions
        sessions = []
        for i in range(5):
            sessions.append(manager.create_session(f"Topic {i}"))

        # Make all but last expired
        expired_time = datetime.now(timezone.utc) - timedelta(hours=2)
        for session in sessions[:-1]:
            session.created_at = expired_time

        # Try to create new session - should succeed after cleanup
        new_session = manager.create_session("New Topic")
        assert new_session is not None

        # Should have 2 sessions now (1 old + 1 new)
        assert len(manager.sessions) == 2
        assert manager.stats["sessions_cleaned"] == 4

    def test_get_stats_with_empty_manager(self):
        """Test getting stats from empty manager"""
        manager = SessionManager()

        stats = manager.get_stats()
        assert stats["active_sessions"] == 0
        assert stats["sessions_created"] == 0
        assert stats["sessions_expired"] == 0
        assert stats["sessions_cleaned"] == 0
        assert stats["max_concurrent"] == 0
        assert "last_cleanup" in stats

    def test_session_id_generation_uniqueness(self):
        """Test that session IDs are unique across many generations"""
        manager = SessionManager(max_sessions=1100)  # Increase limit for this test

        session_ids = set()
        # Test with 1000 sessions to ensure UUID generation is truly unique
        for i in range(1000):
            session = manager.create_session(f"Topic {i}")
            assert (
                session.session_id not in session_ids
            ), f"Duplicate session ID found at iteration {i}"
            session_ids.add(session.session_id)

        # Verify we created exactly 1000 unique session IDs
        assert len(session_ids) == 1000


class TestGetCurrentSession:
    """Test the get_current_session functionality"""

    def test_current_session_empty_manager(self):
        """Test getting current session when no sessions exist"""
        manager = SessionManager()
        
        current = manager.get_current_session()
        assert current is None
        
    def test_current_session_single_session(self):
        """Test getting current session with only one session"""
        manager = SessionManager()
        
        session = manager.create_session("Test Topic")
        current = manager.get_current_session()
        
        assert current is not None
        assert current.session_id == session.session_id
        assert current.topic == "Test Topic"
        
    def test_current_session_multiple_sessions(self):
        """Test getting most recent session with multiple sessions"""
        manager = SessionManager()
        
        # Create sessions with small delays to ensure different timestamps
        session1 = manager.create_session("First Topic")
        import time
        time.sleep(0.01)  # Small delay to ensure different timestamps
        session2 = manager.create_session("Second Topic")
        time.sleep(0.01)
        session3 = manager.create_session("Third Topic")
        
        current = manager.get_current_session()
        
        assert current is not None
        assert current.session_id == session3.session_id
        assert current.topic == "Third Topic"
        
    def test_current_session_after_removal(self):
        """Test getting current session after removing the most recent one"""
        manager = SessionManager()
        
        session1 = manager.create_session("First Topic")
        import time
        time.sleep(0.01)
        session2 = manager.create_session("Second Topic")
        time.sleep(0.01)
        session3 = manager.create_session("Third Topic")
        
        # Remove the most recent session
        manager.remove_session(session3.session_id)
        
        current = manager.get_current_session()
        assert current is not None
        assert current.session_id == session2.session_id
        assert current.topic == "Second Topic"
        
    def test_current_session_expired_cleanup(self):
        """Test that current session triggers cleanup of expired sessions"""
        manager = SessionManager(session_ttl_hours=1, cleanup_interval_minutes=0)  # Set interval to 0 for immediate cleanup
        
        # Create an old session
        session1 = manager.create_session("Old Topic")
        
        # Make it expired
        expired_time = datetime.now(timezone.utc) - timedelta(hours=2)
        session1.created_at = expired_time
        
        # Create a new session
        session2 = manager.create_session("New Topic")
        
        # Get current should trigger cleanup and return the new session
        current = manager.get_current_session()
        
        assert current is not None
        assert current.session_id == session2.session_id
        assert current.topic == "New Topic"
        
        # Old session should be cleaned up
        assert session1.session_id not in manager.sessions
        
    def test_current_session_all_expired(self):
        """Test getting current session when all sessions are expired"""
        manager = SessionManager(session_ttl_hours=1)
        
        # Create sessions
        session1 = manager.create_session("Topic 1")
        session2 = manager.create_session("Topic 2")
        
        # Make all expired
        expired_time = datetime.now(timezone.utc) - timedelta(hours=2)
        session1.created_at = expired_time
        session2.created_at = expired_time
        
        # Force cleanup to occur
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        with patch("decision_matrix_mcp.session_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = future_time
            mock_datetime.timezone = timezone
            
            current = manager.get_current_session()
            assert current is None
            
            # All sessions should be cleaned up
            assert len(manager.sessions) == 0
            
    def test_current_session_ordering_with_same_timestamp(self):
        """Test behavior when sessions have the same timestamp"""
        manager = SessionManager()
        
        # Create sessions and manually set same timestamp
        fixed_time = datetime.now(timezone.utc)
        
        session1 = manager.create_session("Topic 1")
        session1.created_at = fixed_time
        
        session2 = manager.create_session("Topic 2")
        session2.created_at = fixed_time
        
        session3 = manager.create_session("Topic 3")
        session3.created_at = fixed_time
        
        # Get current - when timestamps are equal, it should still return one
        current = manager.get_current_session()
        assert current is not None
        assert current.topic in ["Topic 1", "Topic 2", "Topic 3"]
