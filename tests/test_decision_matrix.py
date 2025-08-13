"""Tests for Decision Matrix MCP"""

from datetime import datetime, timezone

import pytest

from decision_matrix_mcp.exceptions import ResourceLimitError
from decision_matrix_mcp.models import Criterion, DecisionSession, Option, Score
from decision_matrix_mcp.session_manager import SessionManager, SessionValidator


class TestModels:
    """Test core data models"""

    def test_score_creation(self):
        """Test Score model creation and properties"""
        score = Score(
            criterion_name="performance",
            option_name="option_a",
            score=8.5,
            justification="Good performance characteristics",
        )

        assert score.criterion_name == "performance"
        assert score.option_name == "option_a"
        assert score.score == 8.5
        assert not score.abstained

        # Test abstained score
        abstained_score = Score(
            criterion_name="cost",
            option_name="option_b",
            score=None,
            justification="Not applicable",
        )

        assert abstained_score.abstained

    def test_criterion_creation(self):
        """Test Criterion model creation and prompt generation"""
        criterion = Criterion(
            name="scalability",
            description="How well does this scale with growth?",
            weight=2.0,
        )

        assert criterion.name == "scalability"
        assert criterion.weight == 2.0
        assert "scalability" in criterion.system_prompt.lower()
        assert "2.0" in criterion.system_prompt
        # Test new default parameters
        assert criterion.temperature == 0.0
        assert criterion.max_tokens == 1024

    def test_criterion_with_custom_llm_params(self):
        """Test Criterion creation with custom LLM parameters"""
        criterion = Criterion(
            name="performance",
            description="Performance evaluation",
            weight=1.5,
            temperature=0.7,
            max_tokens=2048,
        )

        assert criterion.temperature == 0.7
        assert criterion.max_tokens == 2048

    def test_option_weighted_total(self):
        """Test Option weighted total calculation"""
        option = Option(name="test_option")

        # Create criteria
        criteria = {
            "performance": Criterion("performance", "Performance test", weight=2.0),
            "cost": Criterion("cost", "Cost test", weight=1.0),
            "usability": Criterion("usability", "Usability test", weight=1.5),
        }

        # Add scores
        option.add_score(Score("performance", "test_option", 8.0, "Good"))
        option.add_score(Score("cost", "test_option", 6.0, "Moderate"))
        option.add_score(Score("usability", "test_option", None, "Not applicable"))  # Abstained

        weighted_total = option.get_weighted_total(criteria)

        # Should be (8.0 * 2.0 + 6.0 * 1.0) / (2.0 + 1.0) = 22.0 / 3.0 ≈ 7.33
        expected = (8.0 * 2.0 + 6.0 * 1.0) / (2.0 + 1.0)
        assert abs(weighted_total - expected) < 0.01

    def test_decision_session(self):
        """Test DecisionSession functionality"""
        session = DecisionSession(
            session_id="test-123",
            created_at=datetime.now(timezone.utc),
            topic="Test decision",
        )

        # Add options
        session.add_option("option_a", "First option")
        session.add_option("option_b", "Second option")

        # Add criteria
        criterion = Criterion("test_criterion", "Test description", weight=1.0)
        session.add_criterion(criterion)

        assert len(session.options) == 2
        assert len(session.criteria) == 1
        assert "option_a" in session.options
        assert "test_criterion" in session.criteria
        # Test default parameters
        assert session.default_temperature == 0.1

    def test_decision_session_with_custom_defaults(self):
        """Test DecisionSession with custom default parameters"""
        session = DecisionSession(
            session_id="test-456",
            created_at=datetime.now(timezone.utc),
            topic="Test decision",
            default_temperature=0.5,
        )

        assert session.default_temperature == 0.5

        # Test parameter inheritance when adding criterion
        criterion = Criterion("test", "Test criterion", weight=1.0)
        session.add_criterion(criterion)

        # Criterion should inherit session defaults
        assert session.criteria["test"].temperature == 0.5


class TestSessionManager:
    """Test session management functionality"""

    def test_session_creation(self):
        """Test creating and retrieving sessions"""
        manager = SessionManager(max_sessions=5, session_ttl_hours=1)

        # Create session
        session = manager.create_session("Test topic", ["option1", "option2"])

        assert session.topic == "Test topic"
        assert len(session.options) == 2

        # Retrieve session
        retrieved = manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_session_creation_with_llm_params(self):
        """Test creating session with custom LLM parameters"""
        manager = SessionManager(max_sessions=5, session_ttl_hours=1)

        # Create session with custom parameters
        session = manager.create_session("Test topic", ["option1", "option2"], temperature=0.7)

        assert session.default_temperature == 0.7

    def test_session_limit(self):
        """Test session limit enforcement"""
        manager = SessionManager(max_sessions=2, session_ttl_hours=1)

        # Create max sessions
        manager.create_session("Topic 1")
        manager.create_session("Topic 2")

        # Third session should fail
        with pytest.raises(ResourceLimitError) as exc_info:
            manager.create_session("Topic 3")
        assert "Maximum number of active sessions" in str(exc_info.value.user_message)


class TestSessionValidator:
    """Test input validation"""

    def test_topic_validation(self):
        """Test topic string validation"""
        assert SessionValidator.validate_topic("Valid topic")
        assert not SessionValidator.validate_topic("")
        assert not SessionValidator.validate_topic("   ")
        assert not SessionValidator.validate_topic("x" * 600)  # Too long
        assert not SessionValidator.validate_topic(None)

    def test_weight_validation(self):
        """Test weight validation"""
        assert SessionValidator.validate_weight(1.0)
        assert SessionValidator.validate_weight(0.5)
        assert SessionValidator.validate_weight(5.0)
        assert not SessionValidator.validate_weight(0.05)  # Too small
        assert not SessionValidator.validate_weight(15.0)  # Too large
        assert not SessionValidator.validate_weight("invalid")

    def test_option_name_validation(self):
        """Test option name validation"""
        assert SessionValidator.validate_option_name("Valid Option")
        assert not SessionValidator.validate_option_name("")
        assert not SessionValidator.validate_option_name("x" * 250)  # Too long
        assert not SessionValidator.validate_option_name(None)


@pytest.mark.asyncio()
class TestIntegration:
    """Integration tests for the complete system"""

    async def test_basic_workflow(self):
        """Test a complete decision analysis workflow"""
        # This would test the full MCP workflow but requires
        # more complex mocking of LLM backends


if __name__ == "__main__":
    pytest.main([__file__])
