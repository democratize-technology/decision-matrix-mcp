"""Extended tests for model methods not covered in basic tests"""

from datetime import datetime, timezone
from unittest.mock import patch

from decision_matrix_mcp.models import (
    Criterion,
    CriterionThread,
    DecisionSession,
    ModelBackend,
    Option,
    Score,
)


class TestOptionExtended:
    """Test extended Option functionality"""

    def test_get_score_breakdown_complete(self):
        """Test complete score breakdown with multiple criteria"""
        option = Option(name="Option A", description="Test option")

        # Add criteria
        criteria = {
            "Performance": Criterion(name="Performance", description="Speed", weight=2.0),
            "Cost": Criterion(name="Cost", description="Price", weight=1.5),
            "Reliability": Criterion(name="Reliability", description="Uptime", weight=1.0),
        }

        # Add scores
        score1 = Score(
            criterion_name="Performance",
            option_name="Option A",
            score=8.0,
            justification="Very fast",
        )
        score2 = Score(
            criterion_name="Cost",
            option_name="Option A",
            score=6.0,
            justification="Moderate price",
        )
        score3 = Score(
            criterion_name="Reliability",
            option_name="Option A",
            score=None,  # Abstained
            justification="Cannot evaluate reliability",
        )

        option.add_score(score1)
        option.add_score(score2)
        option.add_score(score3)

        # Get breakdown
        breakdown = option.get_score_breakdown(criteria)

        assert len(breakdown) == 3

        # Check each criterion
        perf_breakdown = next(b for b in breakdown if b["criterion"] == "Performance")
        assert perf_breakdown["weight"] == 2.0
        assert perf_breakdown["raw_score"] == 8.0
        assert perf_breakdown["weighted_score"] == 16.0
        assert perf_breakdown["justification"] == "Very fast"
        assert perf_breakdown["abstained"] is False

        cost_breakdown = next(b for b in breakdown if b["criterion"] == "Cost")
        assert cost_breakdown["weight"] == 1.5
        assert cost_breakdown["raw_score"] == 6.0
        assert cost_breakdown["weighted_score"] == 9.0
        assert cost_breakdown["justification"] == "Moderate price"
        assert cost_breakdown["abstained"] is False

        reliability_breakdown = next(b for b in breakdown if b["criterion"] == "Reliability")
        assert reliability_breakdown["weight"] == 1.0
        assert reliability_breakdown["raw_score"] is None
        assert reliability_breakdown["weighted_score"] is None
        assert reliability_breakdown["justification"] == "Cannot evaluate reliability"
        assert reliability_breakdown["abstained"] is True

    def test_get_score_breakdown_empty(self):
        """Test score breakdown with no scores"""
        option = Option(name="Option A")
        criteria = {
            "Performance": Criterion(name="Performance", description="Speed", weight=2.0),
        }

        breakdown = option.get_score_breakdown(criteria)
        assert breakdown == []

    def test_get_score_breakdown_partial_criteria(self):
        """Test score breakdown when scores exist for criteria not in the criteria dict"""
        option = Option(name="Option A")

        # Add score for a criterion
        score = Score(
            criterion_name="Performance",
            option_name="Option A",
            score=8.0,
            justification="Good performance",
        )
        option.add_score(score)

        # Provide different criteria
        criteria = {
            "Cost": Criterion(name="Cost", description="Price", weight=1.5),
        }

        # Should only return breakdown for criteria in the dict
        breakdown = option.get_score_breakdown(criteria)
        assert len(breakdown) == 0


class TestCriterionThreadExtended:
    """Test extended CriterionThread functionality"""

    def test_conversation_history_timestamps(self):
        """Test that conversation history includes timestamps"""
        criterion = Criterion(name="Performance", description="Speed", weight=2.0)
        thread = CriterionThread(id="thread-1", criterion=criterion)

        # Mock datetime to control timestamps
        mock_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        with patch("decision_matrix_mcp.models.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time
            mock_datetime.timezone = timezone

            thread.add_message("user", "Evaluate this option")
            thread.add_message("assistant", "I'll evaluate it now")

        assert len(thread.conversation_history) == 2

        # Check first message
        assert thread.conversation_history[0]["role"] == "user"
        assert thread.conversation_history[0]["content"] == "Evaluate this option"
        assert thread.conversation_history[0]["timestamp"] == "2024-01-01T12:00:00+00:00"

        # Check second message
        assert thread.conversation_history[1]["role"] == "assistant"
        assert thread.conversation_history[1]["content"] == "I'll evaluate it now"
        assert thread.conversation_history[1]["timestamp"] == "2024-01-01T12:00:00+00:00"

    def test_multiple_conversations(self):
        """Test managing multiple message exchanges"""
        criterion = Criterion(name="Cost", description="Price analysis", weight=1.5)
        thread = CriterionThread(id="thread-2", criterion=criterion)

        # Simulate a conversation
        thread.add_message("user", "What's the cost of Option A?")
        thread.add_message("assistant", "Let me analyze the cost...")
        thread.add_message("user", "Consider long-term costs too")
        thread.add_message("assistant", "Including maintenance costs...")

        assert len(thread.conversation_history) == 4

        # Verify conversation order
        roles = [msg["role"] for msg in thread.conversation_history]
        assert roles == ["user", "assistant", "user", "assistant"]


class TestDecisionSessionExtended:
    """Test extended DecisionSession functionality"""

    def test_get_decision_matrix_no_evaluations(self):
        """Test matrix generation with no evaluations"""
        session = DecisionSession(
            session_id="test-1",
            created_at=datetime.now(timezone.utc),
            topic="Test Decision",
        )

        # No options or criteria
        result = session.get_decision_matrix()
        assert result == {"error": "Need both options and criteria to generate matrix"}

        # Add option but no criteria
        session.add_option("Option A")
        result = session.get_decision_matrix()
        assert result == {"error": "Need both options and criteria to generate matrix"}

        # Add criteria
        criterion = Criterion(name="Performance", description="Speed", weight=2.0)
        session.add_criterion(criterion)

        # Now should generate empty matrix
        result = session.get_decision_matrix()
        assert "matrix" in result
        assert "rankings" in result

    def test_get_decision_matrix_with_abstentions(self):
        """Test matrix generation with abstained scores"""
        session = DecisionSession(
            session_id="test-2",
            created_at=datetime.now(timezone.utc),
            topic="Choose Database",
        )

        # Add options
        session.add_option("PostgreSQL")
        session.add_option("MongoDB")

        # Add criteria
        criterion1 = Criterion(
            name="ACID Compliance", description="Transaction support", weight=2.0
        )
        criterion2 = Criterion(name="Scalability", description="Horizontal scaling", weight=1.5)
        session.add_criterion(criterion1)
        session.add_criterion(criterion2)

        # Add scores with abstention
        session.options["PostgreSQL"].add_score(
            Score(
                criterion_name="ACID Compliance",
                option_name="PostgreSQL",
                score=10.0,
                justification="Full ACID support",
            )
        )
        session.options["PostgreSQL"].add_score(
            Score(
                criterion_name="Scalability",
                option_name="PostgreSQL",
                score=7.0,
                justification="Good vertical scaling",
            )
        )

        session.options["MongoDB"].add_score(
            Score(
                criterion_name="ACID Compliance",
                option_name="MongoDB",
                score=None,
                justification="Not applicable for NoSQL",
            )  # Abstained
        )
        session.options["MongoDB"].add_score(
            Score(
                criterion_name="Scalability",
                option_name="MongoDB",
                score=9.0,
                justification="Excellent horizontal scaling",
            )
        )

        # Generate matrix
        result = session.get_decision_matrix()

        # Check matrix structure
        assert result["matrix"]["PostgreSQL"]["ACID Compliance"]["raw_score"] == 10.0
        assert result["matrix"]["PostgreSQL"]["ACID Compliance"]["weighted_score"] == 20.0

        assert result["matrix"]["MongoDB"]["ACID Compliance"]["raw_score"] is None
        assert result["matrix"]["MongoDB"]["ACID Compliance"]["weighted_score"] is None
        assert "Abstained" in result["matrix"]["MongoDB"]["ACID Compliance"]["justification"]

    def test_get_decision_matrix_close_race(self):
        """Test recommendation for close race scenarios"""
        session = DecisionSession(
            session_id="test-3",
            created_at=datetime.now(timezone.utc),
            topic="Close Race Test",
        )

        # Add options
        session.add_option("Option A")
        session.add_option("Option B")

        # Add criterion
        criterion = Criterion(name="Overall", description="Overall score", weight=1.0)
        session.add_criterion(criterion)

        # Add very close scores
        session.options["Option A"].add_score(
            Score(
                criterion_name="Overall",
                option_name="Option A",
                score=8.0,
                justification="Good option",
            )
        )
        session.options["Option B"].add_score(
            Score(
                criterion_name="Overall",
                option_name="Option B",
                score=8.0,
                justification="Also good",
            )  # Same score
        )

        result = session.get_decision_matrix()

        # Should indicate close race
        assert "Close race" in result["recommendation"]

    def test_get_decision_matrix_clear_winner(self):
        """Test recommendation for clear winner"""
        session = DecisionSession(
            session_id="test-4",
            created_at=datetime.now(timezone.utc),
            topic="Clear Winner Test",
        )

        # Add options
        session.add_option("Winner")
        session.add_option("Loser")

        # Add criterion
        criterion = Criterion(name="Quality", description="Overall quality", weight=2.0)
        session.add_criterion(criterion)

        # Add scores with clear winner
        session.options["Winner"].add_score(
            Score(
                criterion_name="Quality",
                option_name="Winner",
                score=9.0,
                justification="Excellent quality",
            )
        )
        session.options["Loser"].add_score(
            Score(
                criterion_name="Quality",
                option_name="Loser",
                score=4.0,
                justification="Poor quality",
            )
        )

        result = session.get_decision_matrix()

        # Should indicate clear winner
        assert "clear winner" in result["recommendation"]
        assert "Winner" in result["recommendation"]

    def test_get_decision_matrix_single_option(self):
        """Test matrix with only one option"""
        session = DecisionSession(
            session_id="test-5",
            created_at=datetime.now(timezone.utc),
            topic="Single Option Test",
        )

        session.add_option("Only Option")
        criterion = Criterion(name="Test", description="Test criterion", weight=1.0)
        session.add_criterion(criterion)

        session.options["Only Option"].add_score(
            Score(
                criterion_name="Test",
                option_name="Only Option",
                score=7.0,
                justification="Decent score",
            )
        )

        result = session.get_decision_matrix()

        # Should still generate valid matrix
        assert len(result["rankings"]) == 1
        assert result["rankings"][0]["option"] == "Only Option"

    def test_record_evaluation(self):
        """Test recording evaluation history"""
        session = DecisionSession(
            session_id="test-6",
            created_at=datetime.now(timezone.utc),
            topic="Evaluation History Test",
        )

        # Record first evaluation
        eval1 = {
            "evaluation_results": {"criterion1": {"option1": (8.0, "Good")}},
            "total_scores": 1,
            "abstentions": 0,
            "errors": 0,
        }

        with patch("decision_matrix_mcp.models.datetime") as mock_datetime:
            mock_time1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_time1
            mock_datetime.timezone = timezone

            session.record_evaluation(eval1)

        # Record second evaluation
        eval2 = {
            "evaluation_results": {"criterion2": {"option1": (7.0, "OK")}},
            "total_scores": 1,
            "abstentions": 0,
            "errors": 0,
        }

        with patch("decision_matrix_mcp.models.datetime") as mock_datetime:
            mock_time2 = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_time2
            mock_datetime.timezone = timezone

            session.record_evaluation(eval2)

        # Check evaluations
        assert len(session.evaluations) == 2

        assert session.evaluations[0]["timestamp"] == "2024-01-01T10:00:00+00:00"
        assert session.evaluations[0]["results"] == eval1

        assert session.evaluations[1]["timestamp"] == "2024-01-01T11:00:00+00:00"
        assert session.evaluations[1]["results"] == eval2

    def test_decision_matrix_timestamp(self):
        """Test that decision matrix includes current timestamp"""
        session = DecisionSession(
            session_id="test-7",
            created_at=datetime.now(timezone.utc),
            topic="Timestamp Test",
        )

        session.add_option("Option A")
        criterion = Criterion(name="Test", description="Test", weight=1.0)
        session.add_criterion(criterion)

        mock_time = datetime(2024, 1, 1, 15, 30, 0, tzinfo=timezone.utc)

        with patch("decision_matrix_mcp.models.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time
            mock_datetime.timezone = timezone

            result = session.get_decision_matrix()

        assert result["evaluation_timestamp"] == "2024-01-01T15:30:00+00:00"

    def test_criteria_weights_in_matrix(self):
        """Test that matrix includes criteria weights"""
        session = DecisionSession(
            session_id="test-8",
            created_at=datetime.now(timezone.utc),
            topic="Weights Test",
        )

        session.add_option("Option A")

        # Add multiple criteria with different weights
        criterion1 = Criterion(name="Speed", description="Performance", weight=2.5)
        criterion2 = Criterion(name="Cost", description="Price", weight=1.0)
        criterion3 = Criterion(name="Reliability", description="Uptime", weight=3.0)

        session.add_criterion(criterion1)
        session.add_criterion(criterion2)
        session.add_criterion(criterion3)

        result = session.get_decision_matrix()

        assert result["criteria_weights"]["Speed"] == 2.5
        assert result["criteria_weights"]["Cost"] == 1.0
        assert result["criteria_weights"]["Reliability"] == 3.0

    def test_thread_creation_on_criterion_add(self):
        """Test that adding criterion creates associated thread"""
        session = DecisionSession(
            session_id="test-9",
            created_at=datetime.now(timezone.utc),
            topic="Thread Creation Test",
        )

        # Initially no threads
        assert len(session.threads) == 0

        # Add criterion
        criterion = Criterion(
            name="Performance",
            description="Speed evaluation",
            weight=2.0,
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-4",
        )
        session.add_criterion(criterion)

        # Should create thread
        assert len(session.threads) == 1
        assert "Performance" in session.threads

        thread = session.threads["Performance"]
        assert thread.criterion == criterion
        assert len(thread.id) > 0
        assert thread.conversation_history == []
