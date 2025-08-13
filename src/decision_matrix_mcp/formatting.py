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

"""LLM-optimized output formatting for Decision Matrix MCP."""

from typing import Any

from .models import ModelBackend


class DecisionFormatter:
    """Format decision matrix outputs for optimal LLM comprehension."""

    # Verbosity levels
    CONCISE = "concise"
    DETAILED = "detailed"
    PROGRESSIVE = "progressive"

    def __init__(self, verbosity: str = PROGRESSIVE) -> None:
        """Initialize formatter with verbosity level."""
        self.verbosity = verbosity

    def format_session_created(self, session_data: dict[str, Any]) -> str:
        """Format session creation response."""
        lines = [
            "# 🎯 Decision Analysis Session Created",
            "",
            f"**Topic**: {session_data['topic']}",
            f"**Session ID**: `{session_data['session_id']}`",
            "",
            "## 📊 Options to Evaluate",
        ]

        for i, option in enumerate(session_data["options"], 1):
            lines.append(f"{i}. **{option}**")

        if session_data.get("criteria_added"):
            lines.extend(
                [
                    "",
                    "## ⚖️ Initial Criteria",
                ],
            )
            for criterion in session_data["criteria_added"]:
                lines.append(f"- {criterion}")

        lines.extend(
            [
                "",
                "## 🎬 Next Steps",
                "1. **Add criteria** → Define what matters for this decision",
                "2. **Evaluate options** → Run the analysis",
                "3. **Get results** → See the decision matrix",
                "",
                f"*Model: {session_data.get('model_backend') or 'default'}*",
            ],
        )

        return "\n".join(lines)

    def format_criterion_added(self, criterion_data: dict[str, Any]) -> str:
        """Format criterion addition response."""
        lines = [
            f"## ✅ Added Criterion: **{criterion_data['criterion_added']}**",
            "",
            f"**Description**: {criterion_data['description']}",
            f"**Weight**: {criterion_data['weight']}x importance",
            "",
            f"### 📈 Progress: {criterion_data['total_criteria']} criteria defined",
        ]

        if self.verbosity != self.CONCISE:
            lines.extend(
                [
                    "",
                    "**All criteria**:",
                ],
            )
            for criterion in criterion_data["all_criteria"]:
                lines.append(f"- {criterion}")

        lines.extend(["", "💡 **Ready to evaluate** once you've added all relevant criteria"])

        return "\n".join(lines)

    def format_evaluation_complete(self, eval_data: dict[str, Any]) -> str:
        """Format evaluation completion response."""
        summary = eval_data["summary"]

        lines = [
            "# ✨ Evaluation Complete!",
            "",
            "## 📊 Summary",
            f"- **Options evaluated**: {summary['options_evaluated']}",
            f"- **Criteria applied**: {summary['criteria_used']}",
            f"- **Total evaluations**: {summary['total_evaluations']}",
            "",
            "### 🎯 Results Breakdown",
            f"- ✅ **Successful scores**: {summary['successful_scores']}",
            f"- 🤔 **Abstentions**: {summary['abstentions']} *(criteria not applicable)*",
            f"- ❌ **Errors**: {summary['errors']}",
        ]

        if eval_data.get("errors") and self.verbosity != self.CONCISE:
            lines.extend(
                [
                    "",
                    "### ⚠️ Evaluation Errors",
                ],
            )
            for error in eval_data["errors"][:5]:  # Show first 5
                lines.append(f"- {error}")
            if len(eval_data["errors"]) > 5:
                lines.append(f"- *...and {len(eval_data['errors']) - 5} more*")

        lines.extend(
            [
                "",
                "## 🎬 Next Step",
                "→ **Get the decision matrix** to see rankings and recommendations",
            ],
        )

        return "\n".join(lines)

    def format_decision_matrix(self, matrix_data: dict[str, Any]) -> str:
        """Format the complete decision matrix for optimal LLM parsing."""
        rankings = matrix_data["rankings"]

        lines = [
            f"# 🎯 Decision Matrix: {matrix_data['topic']}",
            "",
            "## 🏆 Rankings & Recommendations",
            "",
        ]

        if rankings:
            lines.extend(
                [
                    f"### 🥇 **Winner: {rankings[0]['option']}**",
                    f"**Score**: {rankings[0]['weighted_total']:.1f} points",
                    "",
                ],
            )

        lines.extend(
            [
                matrix_data["recommendation"],
                "",
            ],
        )

        if self.verbosity != self.CONCISE:
            lines.extend(
                [
                    "---",
                    "",
                ],
            )

        lines.extend(
            [
                "## 📊 Complete Rankings",
            ],
        )

        # Rankings with visual indicators
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

        for i, rank in enumerate(rankings):
            medal = medals[i] if i < len(medals) else f"{i + 1}."
            score_bar = self._create_score_bar(
                rank["weighted_total"],
                max(r["weighted_total"] for r in rankings),
            )

            lines.append(
                f"{medal} **{rank['option']}** - {rank['weighted_total']:.1f} pts {score_bar}",
            )

            if self.verbosity == self.DETAILED or (self.verbosity == self.PROGRESSIVE and i < 3):
                # Show breakdown for top options
                lines.append("   ```")
                for item in rank["breakdown"]:
                    if not item["abstained"]:
                        score_str = f"{item['raw_score']:.1f}" if item["raw_score"] else "N/A"
                        weighted_str = (
                            f"{item['weighted_score']:.1f}" if item["weighted_score"] else "N/A"
                        )
                        lines.append(
                            f"   {item['criterion']}: {score_str} × {item['weight']} = {weighted_str}",
                        )
                        if self.verbosity == self.DETAILED:
                            lines.append(f"     → {item['justification'][:80]}...")
                lines.append("   ```")

            lines.append("")

        # Criteria weights summary
        if self.verbosity != self.CONCISE:
            lines.extend(
                [
                    "## ⚖️ Criteria Weights",
                ],
            )
            for name, weight in matrix_data["criteria_weights"].items():
                lines.append(f"- **{name}**: {weight}x")

        # Key insights
        lines.extend(
            [
                "",
                "## 💡 Key Insights",
            ],
        )

        # Analyze score patterns
        if rankings:
            winner = rankings[0]
            runner_up = rankings[1] if len(rankings) > 1 else None

            if runner_up and winner["weighted_total"] - runner_up["weighted_total"] < 1.0:
                lines.append("- 🔍 **Very close decision** - top options within 1 point")
            elif winner["weighted_total"] > 8.0:
                lines.append("- 🌟 **Strong winner** - excellent scores across criteria")

        # Check for abstentions
        total_abstentions = sum(
            1 for rank in rankings for item in rank["breakdown"] if item["abstained"]
        )
        if total_abstentions > 0:
            lines.append(
                f"- 🤔 **{total_abstentions} abstentions** - some criteria didn't apply to certain options",
            )

        # Session metadata
        if self.verbosity != self.CONCISE:
            lines.extend(
                [
                    "",
                    "---",
                    f"*Analysis completed at {matrix_data.get('evaluation_timestamp', 'unknown')}*",
                    f"*Session: {matrix_data.get('session_id', 'unknown')}*",
                ],
            )

        return "\n".join(lines)

    def format_option_added(self, option_data: dict[str, Any]) -> str:
        """Format option addition response."""
        lines = [
            f"## ✅ Added Option: **{option_data['option_added']}**",
            "",
            f"Total options now: **{option_data['total_options']}**",
        ]

        if option_data.get("description"):
            lines.insert(2, f"*{option_data['description']}*")
            lines.insert(3, "")

        if self.verbosity != self.CONCISE:
            lines.extend(
                [
                    "",
                    "### 📋 All Options",
                ],
            )
            for option in option_data["all_options"]:
                lines.append(f"- {option}")

        lines.extend(["", "⚡ **Action Required**: Re-run evaluation to score the new option"])

        return "\n".join(lines)

    def format_sessions_list(self, sessions_data: dict[str, Any]) -> str:
        """Format active sessions list."""
        sessions = sessions_data["sessions"]

        if not sessions:
            return "## 📭 No Active Sessions\n\nStart a new decision analysis to begin!"

        lines = [
            f"# 📋 Active Decision Sessions ({len(sessions)})",
            "",
        ]

        for session in sessions:
            status_icon = "✅" if session["status"] == "evaluated" else "🔄"
            lines.extend(
                [
                    f"## {status_icon} {session['topic']}",
                    f"**ID**: `{session['session_id']}`",
                    f"**Created**: {session['created_at']}",
                    f"**Status**: {session['status']}",
                    f"- Options: {', '.join(session['options'][:3])}{'...' if len(session['options']) > 3 else ''}",
                    f"- Criteria: {len(session['criteria'])}",
                    f"- Evaluations: {session['evaluations_run']}",
                    "",
                ],
            )

        stats = sessions_data.get("stats", {})
        if stats:
            lines.extend(
                [
                    "---",
                    "### 📊 Session Stats",
                    f"- Total created: {stats.get('total_created', 0)}",
                    f"- Currently active: {stats.get('active_sessions', 0)}",
                    f"- Total removed: {stats.get('total_removed', 0)}",
                ],
            )

        return "\n".join(lines)

    def format_error(self, error_msg: str, context: str | None = None) -> str:
        """Format error messages for clarity."""
        lines = [
            "## ❌ Error Encountered",
            "",
            f"**Issue**: {error_msg}",
        ]

        if context:
            lines.extend(
                [
                    "",
                    f"**Context**: {context}",
                ],
            )

        # Add helpful suggestions based on error type
        if "not found" in error_msg.lower():
            lines.extend(
                [
                    "",
                    "💡 **Suggestions**:",
                    "- Check the session ID is correct",
                    "- List active sessions to find valid IDs",
                    "- Session may have expired (30 min timeout)",
                ],
            )
        elif "no options" in error_msg.lower():
            lines.extend(
                [
                    "",
                    "💡 **Next step**: Add options to evaluate first",
                ],
            )
        elif "no criteria" in error_msg.lower():
            lines.extend(
                [
                    "",
                    "💡 **Next step**: Add evaluation criteria first",
                ],
            )

        return "\n".join(lines)

    def format_session_summary(self, session: Any) -> str:
        """Format a session summary for the current_session response."""
        lines = [
            "# 📊 Current Decision Analysis Session",
            "",
            f"**Topic**: {session.topic}",
            f"**Session ID**: `{session.session_id}`",
            f"**Created**: {session.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Status**: {'✅ Evaluated' if len(session.evaluations) > 0 else '⏳ Pending evaluation'}",
            "",
        ]

        if session.options:
            lines.extend(
                [
                    f"## 📋 Options ({len(session.options)})",
                    *[f"- {opt.name}" for opt in session.options.values()],
                    "",
                ],
            )
        else:
            lines.extend(
                [
                    "## 📋 Options",
                    "*(No options defined yet)*",
                    "",
                ],
            )

        if session.criteria:
            lines.extend(
                [
                    f"## ⚖️ Criteria ({len(session.criteria)})",
                    *[
                        f"- **{crit.name}** (weight: {crit.weight}x)"
                        for crit in session.criteria.values()
                    ],
                    "",
                ],
            )
        else:
            lines.extend(
                [
                    "## ⚖️ Criteria",
                    "*(No criteria defined yet)*",
                    "",
                ],
            )

        if len(session.evaluations) > 0:
            lines.extend(
                [
                    "## 📊 Analysis Summary",
                    f"- Evaluations completed: {len(session.evaluations)}",
                    f"- Model backend: {getattr(session, 'model_backend', ModelBackend.BEDROCK).value}",
                    "",
                    "💡 **Next step**: Use `get_decision_matrix` to see results",
                ],
            )
        else:
            next_steps = []
            if not session.options:
                next_steps.append("`add_option` - Add options to evaluate")
            if not session.criteria:
                next_steps.append("`add_criterion` - Add evaluation criteria")
            if session.options and session.criteria:
                next_steps.append("`evaluate_options` - Run the analysis")

            if next_steps:
                lines.extend(
                    [
                        "## 🎬 Next Steps",
                        *[f"- {step}" for step in next_steps],
                    ],
                )

        return "\n".join(lines)

    def _create_score_bar(self, score: float, max_score: float, width: int = 20) -> str:
        """Create a visual progress bar for scores."""
        if max_score == 0:
            return ""

        filled = int((score / max_score) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def set_verbosity(self, level: str) -> None:
        """Change verbosity level."""
        if level in [self.CONCISE, self.DETAILED, self.PROGRESSIVE]:
            self.verbosity = level
        else:
            raise ValueError(f"Invalid verbosity level: {level}")
