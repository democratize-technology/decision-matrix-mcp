"""Additional session manager tests for 100% coverage"""

from datetime import datetime, timedelta, timezone

from decision_matrix_mcp.session_manager import SessionManager


class TestSessionManagerFullCoverage:
    """Additional tests to achieve 100% coverage"""

    def test_get_session_with_expired_cleanup(self):
        """Test get_session removes expired sessions"""
        manager = SessionManager(session_ttl_hours=1)

        # Create a session
        session = manager.create_session("Test topic")
        session_id = session.session_id

        # Manually set creation time to expired
        session.created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Get session should return None and remove it
        result = manager.get_session(session_id)
        assert result is None
        assert session_id not in manager.sessions

        # Stats should reflect cleanup
        stats = manager.get_stats()
        assert stats["sessions_cleaned"] == 1
