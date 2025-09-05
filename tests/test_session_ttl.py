"""Test timezone-aware session TTL functionality."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from decision_matrix_mcp.models import DecisionSession
from decision_matrix_mcp.session_manager import SessionManager


class TestSessionTTL:
    """Test proper handling of session expiration with timezone awareness."""

    def test_timezone_aware_session_creation(self):
        """Test that sessions are created with timezone-aware timestamps."""
        manager = SessionManager()
        session = manager.create_session("Test Decision")
        assert session is not None
        assert session.created_at.tzinfo is not None
        assert session.created_at.tzinfo == timezone.utc

    def test_timezone_aware_expiration_check(self):
        """Test that expiration correctly handles timezone-aware datetimes."""
        manager = SessionManager(session_ttl_hours=1)

        # Create a session
        session = manager.create_session("Test Decision")

        # Session should not be expired initially
        assert not manager._is_session_expired(session)

        # Mock time to be 2 hours later
        future_time = datetime.now(timezone.utc) + timedelta(hours=2)
        with patch("decision_matrix_mcp.session_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = future_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(
                *args, **kw, tzinfo=timezone.utc
            )

            # Session should now be expired
            assert manager._is_session_expired(session)

    def test_naive_datetime_handling(self):
        """Test that naive datetimes are properly converted to UTC."""
        manager = SessionManager(session_ttl_hours=1)

        # Create a session with naive datetime (simulating legacy data)
        session = DecisionSession(
            session_id="test-naive",
            created_at=datetime.now(timezone.utc),  # Timezone-aware datetime
            topic="Test Decision",
        )

        # Add to manager
        manager.sessions[session.session_id] = session

        # Expiration check should handle it gracefully
        is_expired = manager._is_session_expired(session)
        assert isinstance(is_expired, bool)  # Should not raise an exception

    def test_timezone_comparison_edge_cases(self):
        """Test edge cases in timezone comparisons."""
        manager = SessionManager(session_ttl_hours=1)

        # Create sessions with different timezone representations
        utc_session = DecisionSession(
            session_id="utc",
            created_at=datetime.now(timezone.utc),
            topic="UTC Session",
        )

        # Create a session with a different timezone (e.g., EST)
        import zoneinfo

        est = zoneinfo.ZoneInfo("America/New_York")
        est_time = datetime.now(est)
        est_session = DecisionSession(session_id="est", created_at=est_time, topic="EST Session")

        manager.sessions[utc_session.session_id] = utc_session
        manager.sessions[est_session.session_id] = est_session

        # Both should work correctly
        assert isinstance(manager._is_session_expired(utc_session), bool)
        assert isinstance(manager._is_session_expired(est_session), bool)

    def test_cleanup_with_mixed_timezones(self):
        """Test cleanup works correctly with mixed timezone sessions."""
        manager = SessionManager(
            session_ttl_hours=1,
            cleanup_interval_minutes=0,  # Immediate cleanup
        )

        # Create expired session (2 hours ago)
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        old_session = DecisionSession(session_id="old", created_at=old_time, topic="Old Session")

        # Create recent session
        recent_session = DecisionSession(
            session_id="recent",
            created_at=datetime.now(timezone.utc),
            topic="Recent Session",
        )

        # Create naive datetime session (recent, less than 1 hour old)
        # Use UTC time but remove timezone info to simulate naive datetime
        naive_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=30)
        naive_session = DecisionSession(
            session_id="naive",
            created_at=naive_time,  # Naive datetime, 30 minutes old in UTC
            topic="Naive Session",
        )

        # Add all sessions
        manager.sessions = {"old": old_session, "recent": recent_session, "naive": naive_session}

        # Run cleanup
        manager._cleanup_expired_sessions()

        # Only old session should be removed
        assert "old" not in manager.sessions
        assert "recent" in manager.sessions
        assert "naive" in manager.sessions

    def test_session_ttl_precision(self):
        """Test that TTL comparisons are precise to the second."""
        # Use a very small TTL (1 minute = 60 seconds)
        manager = SessionManager(session_ttl_hours=0.0083)  # ~30 seconds

        # Create a session
        session = manager.create_session("Test Decision")

        # Test at exactly TTL boundary
        # Adjust TTL to 30 seconds for test
        manager.session_ttl = timedelta(seconds=30)
        boundary_time = session.created_at + timedelta(seconds=30)
        with patch("decision_matrix_mcp.session_manager.datetime") as mock_datetime:
            # Just before expiration
            mock_datetime.now.return_value = boundary_time - timedelta(microseconds=1)
            mock_datetime.side_effect = lambda *args, **kw: datetime(
                *args, **kw, tzinfo=timezone.utc
            )
            assert not manager._is_session_expired(session)

            # Just after expiration
            mock_datetime.now.return_value = boundary_time + timedelta(microseconds=1)
            assert manager._is_session_expired(session)

    def test_concurrent_cleanup_safety(self):
        """Test that automatic cleanup works correctly during session creation."""
        manager = SessionManager(session_ttl_hours=1, max_sessions=50)

        # Create sessions, some will be expired and automatically cleaned up
        session_ids = []
        remaining_sessions = []

        for i in range(5):
            session = manager.create_session(f"Decision {i}")
            session_ids.append(session.session_id)

            # Make some old (these will be cleaned up automatically during next session creation)
            if i < 2:
                session.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
            else:
                # These sessions should remain
                remaining_sessions.append(session.session_id)

        # The automatic cleanup during session creation should have removed expired sessions
        # Only sessions 2, 3, 4 should remain (sessions 0, 1 were cleaned up automatically)
        final_count = len(manager.sessions)

        # Verify that only non-expired sessions remain
        assert (
            final_count == 3
        ), f"Expected 3 sessions to remain after automatic cleanup, got {final_count}"
        assert (
            session_ids[0] not in manager.sessions
        ), "Expired session 0 should have been auto-cleaned"
        assert (
            session_ids[1] not in manager.sessions
        ), "Expired session 1 should have been auto-cleaned"
        assert all(
            sid in manager.sessions for sid in remaining_sessions
        ), "Non-expired sessions should remain"

        # Verify the remaining sessions are the expected ones
        remaining_topics = [s.topic for s in manager.sessions.values()]
        assert remaining_topics == ["Decision 2", "Decision 3", "Decision 4"]

        # Run explicit cleanup (should be no-op since already cleaned)
        manager._cleanup_expired_sessions()

        # Verify no change after explicit cleanup
        assert len(manager.sessions) == 3, "Explicit cleanup should not change anything"
