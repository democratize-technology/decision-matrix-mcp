"""Comprehensive tests for DecisionFormatter class"""

from datetime import datetime, timezone

import pytest

from decision_matrix_mcp.formatting import DecisionFormatter


class TestDecisionFormatter:
    """Test the DecisionFormatter class for LLM-optimized output"""

    def test_formatter_initialization(self):
        """Test formatter initialization with different verbosity levels"""
        # Default is progressive
        formatter = DecisionFormatter()
        assert formatter.verbosity == DecisionFormatter.PROGRESSIVE

        # Test explicit verbosity
        formatter_concise = DecisionFormatter(verbosity=DecisionFormatter.CONCISE)
        assert formatter_concise.verbosity == DecisionFormatter.CONCISE

        formatter_detailed = DecisionFormatter(verbosity=DecisionFormatter.DETAILED)
        assert formatter_detailed.verbosity == DecisionFormatter.DETAILED

    def test_format_session_created(self):
        """Test formatting session creation response"""
        formatter = DecisionFormatter()

        # Basic session data
        session_data = {
            "session_id": "test-123",
            "topic": "Choose a database",
            "options": ["PostgreSQL", "MongoDB", "DynamoDB"],
            "criteria_added": [],
            "model_backend": "bedrock",
            "model_name": None,
        }

        output = formatter.format_session_created(session_data)

        # Check key elements are present
        assert "# 🎯 Decision Analysis Session Created" in output
        assert "**Topic**: Choose a database" in output
        assert "**Session ID**: `test-123`" in output
        assert "1. **PostgreSQL**" in output
        assert "2. **MongoDB**" in output
        assert "3. **DynamoDB**" in output
        assert "*Model: bedrock*" in output
        assert "## 🎬 Next Steps" in output

        # Test with initial criteria
        session_data["criteria_added"] = ["Performance", "Cost"]
        output = formatter.format_session_created(session_data)
        assert "## ⚖️ Initial Criteria" in output
        assert "- Performance" in output
        assert "- Cost" in output

    def test_format_criterion_added(self):
        """Test formatting criterion addition response"""
        formatter = DecisionFormatter()

        criterion_data = {
            "criterion_added": "Scalability",
            "description": "How well the solution scales with growth",
            "weight": 2.5,
            "total_criteria": 3,
            "all_criteria": ["Performance", "Cost", "Scalability"],
        }

        # Test progressive (default) verbosity
        output = formatter.format_criterion_added(criterion_data)
        assert "## ✅ Added Criterion: **Scalability**" in output
        assert "**Description**: How well the solution scales with growth" in output
        assert "**Weight**: 2.5x importance" in output
        assert "### 📈 Progress: 3 criteria defined" in output
        assert "**All criteria**:" in output
        assert "- Performance" in output

        # Test concise verbosity
        formatter.set_verbosity(DecisionFormatter.CONCISE)
        output = formatter.format_criterion_added(criterion_data)
        assert "**All criteria**:" not in output
        assert "💡 **Ready to evaluate**" in output

    def test_format_evaluation_complete(self):
        """Test formatting evaluation completion response"""
        formatter = DecisionFormatter()

        eval_data = {
            "summary": {
                "options_evaluated": 3,
                "criteria_used": 4,
                "total_evaluations": 12,
                "successful_scores": 10,
                "abstentions": 1,
                "errors": 1,
            },
            "errors": ["Performance→MongoDB: Error: Timeout"],
        }

        output = formatter.format_evaluation_complete(eval_data)
        assert "# ✨ Evaluation Complete!" in output
        assert "- **Options evaluated**: 3" in output
        assert "- ✅ **Successful scores**: 10" in output
        assert "- 🤔 **Abstentions**: 1 *(criteria not applicable)*" in output
        assert "- ❌ **Errors**: 1" in output
        assert "### ⚠️ Evaluation Errors" in output
        assert "- Performance→MongoDB: Error: Timeout" in output

        # Test with many errors
        eval_data["errors"] = [f"Error {i}" for i in range(10)]
        output = formatter.format_evaluation_complete(eval_data)
        assert "- Error 0" in output
        assert "- Error 4" in output
        assert "- *...and 5 more*" in output

        # Test concise mode - no error details
        formatter.set_verbosity(DecisionFormatter.CONCISE)
        output = formatter.format_evaluation_complete(eval_data)
        assert "### ⚠️ Evaluation Errors" not in output

    def test_format_decision_matrix(self):
        """Test formatting the complete decision matrix"""
        formatter = DecisionFormatter()

        matrix_data = {
            "session_id": "test-123",
            "topic": "Choose a programming language",
            "rankings": [
                {
                    "option": "Python",
                    "weighted_total": 8.5,
                    "breakdown": [
                        {
                            "criterion": "Ease of Use",
                            "weight": 2.0,
                            "raw_score": 9.0,
                            "weighted_score": 18.0,
                            "justification": "Very beginner friendly",
                            "abstained": False,
                        },
                        {
                            "criterion": "Performance",
                            "weight": 1.5,
                            "raw_score": 6.0,
                            "weighted_score": 9.0,
                            "justification": "Adequate for most tasks",
                            "abstained": False,
                        },
                    ],
                },
                {
                    "option": "Rust",
                    "weighted_total": 8.2,
                    "breakdown": [
                        {
                            "criterion": "Ease of Use",
                            "weight": 2.0,
                            "raw_score": 5.0,
                            "weighted_score": 10.0,
                            "justification": "Steep learning curve",
                            "abstained": False,
                        },
                        {
                            "criterion": "Performance",
                            "weight": 1.5,
                            "raw_score": 10.0,
                            "weighted_score": 15.0,
                            "justification": "Excellent performance",
                            "abstained": False,
                        },
                    ],
                },
            ],
            "recommendation": "Python edges ahead with 8.5 points",
            "criteria_weights": {"Ease of Use": 2.0, "Performance": 1.5},
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Test progressive mode
        output = formatter.format_decision_matrix(matrix_data)
        assert "# 🎯 Decision Matrix: Choose a programming language" in output
        assert "### 🥇 **Winner: Python**" in output
        assert "**Score**: 8.5 points" in output
        assert "🥇 **Python** - 8.5 pts" in output
        assert "🥈 **Rust** - 8.2 pts" in output
        assert "[████████████████████]" in output  # Progress bar
        assert "```" in output  # Code block for breakdown
        assert "Ease of Use: 9.0 x 2.0 = 18.0" in output
        assert "## ⚖️ Criteria Weights" in output
        assert "## 💡 Key Insights" in output
        assert "- 🔍 **Very close decision**" in output  # Within 1 point

        # Test concise mode
        formatter.set_verbosity(DecisionFormatter.CONCISE)
        output = formatter.format_decision_matrix(matrix_data)
        assert "```" not in output  # No breakdown in concise mode
        assert "## ⚖️ Criteria Weights" not in output
        assert "---" not in output  # No session metadata

        # Test detailed mode with abstentions
        formatter.set_verbosity(DecisionFormatter.DETAILED)
        matrix_data["rankings"][0]["breakdown"].append(
            {
                "criterion": "Community",
                "weight": 1.0,
                "raw_score": None,
                "weighted_score": None,
                "justification": "Not applicable",
                "abstained": True,
            },
        )
        output = formatter.format_decision_matrix(matrix_data)
        assert "Very beginner friendly" in output  # Justifications shown in detailed
        assert "- 🤔 **1 abstentions**" in output

    def test_format_option_added(self):
        """Test formatting option addition response"""
        formatter = DecisionFormatter()

        option_data = {
            "option_added": "Redis",
            "description": "In-memory data store",
            "total_options": 4,
            "all_options": ["PostgreSQL", "MongoDB", "DynamoDB", "Redis"],
        }

        output = formatter.format_option_added(option_data)
        assert "## ✅ Added Option: **Redis**" in output
        assert "*In-memory data store*" in output
        assert "Total options now: **4**" in output
        assert "### 📋 All Options" in output
        assert "- Redis" in output
        assert "⚡ **Action Required**: Re-run evaluation" in output

        # Test without description
        option_data["description"] = None
        output = formatter.format_option_added(option_data)
        assert "*In-memory data store*" not in output

        # Test concise mode
        formatter.set_verbosity(DecisionFormatter.CONCISE)
        output = formatter.format_option_added(option_data)
        assert "### 📋 All Options" not in output

    def test_format_sessions_list(self):
        """Test formatting sessions list"""
        formatter = DecisionFormatter()

        # Test with active sessions
        sessions_data = {
            "sessions": [
                {
                    "session_id": "abc-123",
                    "topic": "Choose a database",
                    "created_at": "2025-01-01T12:00:00",
                    "options": ["PostgreSQL", "MongoDB", "DynamoDB", "Redis", "Cassandra"],
                    "criteria": ["Performance", "Cost", "Scalability"],
                    "evaluations_run": 2,
                    "status": "evaluated",
                },
                {
                    "session_id": "def-456",
                    "topic": "Select cloud provider",
                    "created_at": "2025-01-01T13:00:00",
                    "options": ["AWS", "GCP"],
                    "criteria": ["Cost"],
                    "evaluations_run": 0,
                    "status": "setup",
                },
            ],
            "total_active": 2,
            "stats": {
                "total_created": 10,
                "active_sessions": 2,
                "total_removed": 8,
            },
        }

        output = formatter.format_sessions_list(sessions_data)
        assert "# 📋 Active Decision Sessions (2)" in output
        assert "## ✅ Choose a database" in output
        assert "## 🔄 Select cloud provider" in output
        assert "- Options: PostgreSQL, MongoDB, DynamoDB..." in output  # Truncated
        assert "- Criteria: 3" in output
        assert "### 📊 Session Stats" in output
        assert "- Total created: 10" in output

        # Test empty sessions
        empty_data = {"sessions": [], "total_active": 0, "stats": {}}
        output = formatter.format_sessions_list(empty_data)
        assert "## 📭 No Active Sessions" in output
        assert "Start a new decision analysis to begin!" in output

    def test_format_error(self):
        """Test error formatting with different contexts"""
        formatter = DecisionFormatter()

        # Session not found error
        output = formatter.format_error("Session abc-123 not found or expired")
        assert "## ❌ Error Encountered" in output
        assert "**Issue**: Session abc-123 not found or expired" in output
        assert "💡 **Suggestions**:" in output
        assert "- Check the session ID is correct" in output
        assert "- Session may have expired (30 min timeout)" in output

        # No options error
        output = formatter.format_error("No options to evaluate. Add options first.")
        assert "💡 **Next step**: Add options to evaluate first" in output

        # No criteria error
        output = formatter.format_error("No criteria defined. Add criteria first.")
        assert "💡 **Next step**: Add evaluation criteria first" in output

        # Error with context
        output = formatter.format_error("Connection timeout", "LLM Backend Error")
        assert "**Context**: LLM Backend Error" in output

        # Generic error without suggestions
        output = formatter.format_error("Unknown error occurred")
        assert "💡 **Suggestions**:" not in output
        assert "💡 **Next step**:" not in output

    def test_create_score_bar(self):
        """Test visual progress bar creation"""
        formatter = DecisionFormatter()

        # Test normal cases
        assert formatter._create_score_bar(10, 10) == "[████████████████████]"
        assert formatter._create_score_bar(5, 10) == "[██████████░░░░░░░░░░]"
        assert formatter._create_score_bar(0, 10) == "[░░░░░░░░░░░░░░░░░░░░]"
        assert formatter._create_score_bar(7.5, 10) == "[███████████████░░░░░]"

        # Test edge cases
        assert formatter._create_score_bar(0, 0) == ""  # Division by zero
        assert formatter._create_score_bar(10, 10, width=10) == "[██████████]"
        assert formatter._create_score_bar(3, 10, width=5) == "[█░░░░]"

    def test_set_verbosity(self):
        """Test changing verbosity levels"""
        formatter = DecisionFormatter()

        # Valid levels
        formatter.set_verbosity(DecisionFormatter.CONCISE)
        assert formatter.verbosity == DecisionFormatter.CONCISE

        formatter.set_verbosity(DecisionFormatter.DETAILED)
        assert formatter.verbosity == DecisionFormatter.DETAILED

        formatter.set_verbosity(DecisionFormatter.PROGRESSIVE)
        assert formatter.verbosity == DecisionFormatter.PROGRESSIVE

        # Invalid level
        with pytest.raises(ValueError) as exc_info:
            formatter.set_verbosity("invalid")
        assert "Invalid verbosity level: invalid" in str(exc_info.value)

    def test_edge_cases(self):
        """Test edge cases and robustness"""
        formatter = DecisionFormatter()

        # Empty rankings
        matrix_data = {
            "topic": "Test",
            "rankings": [],
            "recommendation": "No recommendation",
            "criteria_weights": {},
            "session_id": "test",
        }
        output = formatter.format_decision_matrix(matrix_data)
        assert "# 🎯 Decision Matrix: Test" in output

        # Single option
        matrix_data["rankings"] = [{"option": "Only", "weighted_total": 5.0, "breakdown": []}]
        output = formatter.format_decision_matrix(matrix_data)
        assert "🥇 **Only** - 5.0 pts" in output

        # Missing fields
        session_data = {
            "session_id": "test",
            "topic": "Test",
            "options": [],
            "model_backend": None,  # Missing
        }
        output = formatter.format_session_created(session_data)
        assert "*Model: default*" in output  # Default value used

        # Very long option names
        option_data = {
            "option_added": "A" * 100,
            "total_options": 1,
            "all_options": ["A" * 100],
        }
        output = formatter.format_option_added(option_data)
        assert "A" * 100 in output  # Should handle long strings

    def test_markdown_formatting(self):
        """Test that markdown formatting is correct"""
        formatter = DecisionFormatter()

        # Check proper markdown elements
        session_data = {
            "session_id": "test",
            "topic": "Test **bold** and *italic*",
            "options": ["Option `code`", "Option with [link]"],
            "criteria_added": [],
            "model_backend": "test",
        }
        output = formatter.format_session_created(session_data)

        # Markdown should be preserved
        assert "**bold**" in output
        assert "*italic*" in output
        assert "`code`" in output
        assert "[link]" in output

        # Headers should have proper levels
        assert output.count("# ") >= 1  # At least one h1
        assert output.count("## ") >= 2  # Multiple h2s

    def test_unicode_and_emoji_handling(self):
        """Test that unicode and emojis are handled correctly"""
        formatter = DecisionFormatter()

        # Test with unicode characters
        criterion_data = {
            "criterion_added": "Coût €",
            "description": "Cost in euros: €100-€500",
            "weight": 1.5,
            "total_criteria": 1,
            "all_criteria": ["Coût €"],
        }
        output = formatter.format_criterion_added(criterion_data)
        assert "Coût €" in output
        assert "€100-€500" in output

        # Emojis should be preserved
        assert "✅" in output
        assert "📈" in output
        assert "💡" in output
