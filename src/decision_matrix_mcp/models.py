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

"""Data models for Decision Matrix MCP"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class ModelBackend(str, Enum):
    """Supported model backends"""

    BEDROCK = "bedrock"
    LITELLM = "litellm"
    OLLAMA = "ollama"


@dataclass
class Score:
    """Represents a score given by a criterion to an option"""

    criterion_name: str
    option_name: str
    score: float | None  # 1-10 scale, None if abstained
    justification: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def abstained(self) -> bool:
        """True if this criterion abstained from scoring this option"""
        return self.score is None


@dataclass
class Criterion:
    """Represents an evaluation criterion"""

    name: str
    description: str
    weight: float = 1.0
    system_prompt: str = ""
    model_backend: ModelBackend = ModelBackend.BEDROCK
    model_name: str | None = None

    def __post_init__(self) -> None:
        """Generate system prompt if not provided"""
        if not self.system_prompt:
            self.system_prompt = f"""You are evaluating options based on the '{self.name}' criterion: {self.description}

SCORING INSTRUCTIONS:
1. Score each option from 1-10 (10 = excellent, 1 = poor)
2. Provide clear justification for your score
3. If this criterion doesn't apply to an option, respond with [NO_RESPONSE]

RESPONSE FORMAT:
SCORE: [number 1-10 or NO_RESPONSE]
JUSTIFICATION: [detailed explanation]

Focus on: {self.description}
Weight in decision: {self.weight}x importance"""


@dataclass
class Option:
    """Represents a decision option"""

    name: str
    description: str | None = None
    scores: dict[str, Score] = field(default_factory=dict)  # criterion_name -> Score

    def add_score(self, score: Score) -> None:
        """Add a score from a criterion"""
        self.scores[score.criterion_name] = score

    def get_weighted_total(self, criteria: dict[str, Criterion]) -> float:
        """Calculate weighted total score for this option"""
        total = 0.0
        total_weight = 0.0

        for criterion_name, score in self.scores.items():
            if not score.abstained and criterion_name in criteria:
                criterion = criteria[criterion_name]
                # Type safety: Check score.score is not None before multiplication
                if score.score is not None:
                    total += score.score * criterion.weight
                    total_weight += criterion.weight

        return total / total_weight if total_weight > 0 else 0.0

    def get_score_breakdown(self, criteria: dict[str, Criterion]) -> list[dict[str, Any]]:
        """Get detailed breakdown of scores"""
        breakdown = []
        for criterion_name, criterion in criteria.items():
            score = self.scores.get(criterion_name)
            if score:
                breakdown.append(
                    {
                        "criterion": criterion_name,
                        "weight": criterion.weight,
                        "raw_score": score.score,
                        "weighted_score": score.score * criterion.weight if score.score is not None else None,
                        "justification": score.justification,
                        "abstained": score.abstained,
                    }
                )
        return breakdown


@dataclass
class CriterionThread:
    """Represents a criterion evaluation thread"""

    id: str
    criterion: Criterion
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history"""
        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.now(timezone.utc).isoformat()}
        )


@dataclass
class DecisionSession:
    """Manages a complete decision analysis session"""

    session_id: str
    created_at: datetime
    topic: str
    options: dict[str, Option] = field(default_factory=dict)  # option_name -> Option
    criteria: dict[str, Criterion] = field(default_factory=dict)  # criterion_name -> Criterion
    threads: dict[str, CriterionThread] = field(default_factory=dict)  # criterion_name -> Thread
    evaluations: list[dict[str, Any]] = field(default_factory=list)  # History of evaluations

    def add_option(self, name: str, description: str | None = None) -> None:
        """Add a new option to evaluate"""
        if name not in self.options:
            self.options[name] = Option(name=name, description=description)

    def add_criterion(self, criterion: Criterion) -> None:
        """Add a new evaluation criterion"""
        self.criteria[criterion.name] = criterion

        # Create thread for this criterion
        thread = CriterionThread(id=str(uuid4()), criterion=criterion)
        self.threads[criterion.name] = thread

    def get_decision_matrix(self) -> dict[str, Any]:
        """Generate complete decision matrix with scores and recommendations"""
        if not self.options or not self.criteria:
            return {"error": "Need both options and criteria to generate matrix"}

        # Calculate scores matrix
        matrix: dict[str, dict[str, dict[str, Any]]] = {}
        for option_name, option in self.options.items():
            matrix[option_name] = {}
            for criterion_name in self.criteria.keys():
                score = option.scores.get(criterion_name)
                if score and not score.abstained:
                    # Type safety: Check score.score is not None before multiplication
                    if score.score is not None:
                        matrix[option_name][criterion_name] = {
                            "raw_score": score.score,
                            "weighted_score": score.score * self.criteria[criterion_name].weight,
                            "justification": score.justification,
                        }
                    else:
                        matrix[option_name][criterion_name] = {
                            "raw_score": None,
                            "weighted_score": None,
                            "justification": "Score not available",
                        }
                else:
                    matrix[option_name][criterion_name] = {
                        "raw_score": None,
                        "weighted_score": None,
                        "justification": "Abstained - criterion not applicable",
                    }

        # Calculate totals and rankings
        rankings: list[dict[str, Any]] = []
        for option_name, option in self.options.items():
            weighted_total = option.get_weighted_total(self.criteria)
            rankings.append(
                {
                    "option": option_name,
                    "weighted_total": weighted_total,
                    "breakdown": option.get_score_breakdown(self.criteria),
                }
            )

        # Sort by weighted total (descending)
        rankings.sort(key=lambda x: x["weighted_total"], reverse=True)

        # Generate recommendation
        if rankings:
            winner = rankings[0]
            if len(rankings) > 1 and winner["weighted_total"] > rankings[1]["weighted_total"]:
                recommendation = f"{winner['option']} is the clear winner with {winner['weighted_total']:.1f} points"
            else:
                recommendation = f"Close race, but {winner['option']} edges ahead with {winner['weighted_total']:.1f} points"
        else:
            recommendation = "No clear recommendation available"

        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "matrix": matrix,
            "rankings": rankings,
            "recommendation": recommendation,
            "criteria_weights": {name: crit.weight for name, crit in self.criteria.items()},
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def record_evaluation(self, evaluation_results: dict[str, Any]) -> None:
        """Record an evaluation for history"""
        self.evaluations.append(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "results": evaluation_results}
        )
