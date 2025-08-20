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

"""Variance-based convergence detection for iterative score calculations.

This module provides convergence detection for the decision matrix tool's
iterative scoring process, specifically during Chain of Thought reasoning.
Uses variance-based analysis to detect when scores have stabilized.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceTracker:
    """Tracks score convergence using variance-based analysis.

    Monitors intermediate scores during iterative evaluation to detect
    when the scoring process has stabilized. Uses variance analysis
    rather than semantic similarity for numerical score stability.

    Attributes:
        convergence_threshold: Variance threshold below which scores are considered converged
        min_samples: Minimum number of scores needed before checking convergence
        window_size: Maximum number of recent scores to consider
        scores: Recent score history for variance calculation
        iteration_count: Total number of scores recorded
        converged: Whether convergence has been detected
        convergence_iteration: Iteration where convergence was first detected
    """

    convergence_threshold: float = 0.1
    min_samples: int = 3
    window_size: int = 5
    scores: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    iteration_count: int = 0
    converged: bool = False
    convergence_iteration: int | None = None

    def add_score(self, score: float) -> bool:
        """Add a new score and check for convergence.

        Args:
            score: New score value (should be in 1-10 range)

        Returns:
            True if convergence detected, False otherwise

        Raises:
            ValueError: If score is not a valid number
        """
        if not isinstance(score, (int, float)) or score < 0:
            msg = f"Invalid score: {score}. Must be a positive number."
            raise ValueError(msg)

        # Normalize score to 1-10 range if needed
        if score > 10:
            logger.warning("Score %s above expected range, clamping to 10", score)
            score = 10.0
        elif score < 1:
            logger.warning("Score %s below expected range, clamping to 1", score)
            score = 1.0

        self.scores.append(float(score))
        self.iteration_count += 1

        logger.debug(
            "Score iteration %s: %.2f (window: %s)",
            self.iteration_count,
            score,
            list(self.scores),
        )

        # Check convergence if we have enough samples
        if (
            not self.converged
            and len(self.scores) >= self.min_samples
            and self._check_convergence()
        ):
            self.converged = True
            self.convergence_iteration = self.iteration_count
            self._log_convergence()
            return True

        return False

    def _check_convergence(self) -> bool:
        """Check if scores have converged based on variance.

        Returns:
            True if variance is below threshold, False otherwise
        """
        if len(self.scores) < self.min_samples:
            return False

        try:
            # Calculate variance of recent scores
            variance = statistics.variance(self.scores)

            logger.debug(
                "Convergence check: variance=%.4f, threshold=%s, samples=%s",
                variance,
                self.convergence_threshold,
                len(self.scores),
            )

        except statistics.StatisticsError as e:
            logger.warning("Error calculating variance: %s", e)
            return False
        else:
            return variance < self.convergence_threshold

    def _log_convergence(self) -> None:
        """Log convergence detection with detailed metrics."""
        try:
            mean_score = statistics.mean(self.scores)
            variance = statistics.variance(self.scores)
            std_dev = statistics.stdev(self.scores)

            logger.info(
                "🎯 Score convergence detected at iteration %s! "
                "Variance: %.4f < %s "
                "(mean: %.2f, std: %.3f, window: %s)",
                self.convergence_iteration,
                variance,
                self.convergence_threshold,
                mean_score,
                std_dev,
                list(self.scores),
            )

        except statistics.StatisticsError as e:
            logger.info(
                "🎯 Score convergence detected at iteration %s! "
                "Variance below threshold %s "
                "(error calculating detailed stats: %s)",
                self.convergence_iteration,
                self.convergence_threshold,
                e,
            )

    def get_convergence_metrics(self) -> dict[str, Any]:
        """Get detailed convergence analysis metrics.

        Returns:
            Dictionary with convergence statistics and analysis
        """
        if len(self.scores) == 0:
            return {
                "converged": self.converged,
                "iteration_count": self.iteration_count,
                "scores_collected": 0,
                "error": "No scores collected",
            }

        try:
            scores_list = list(self.scores)
            variance = statistics.variance(self.scores) if len(self.scores) > 1 else 0.0

            return {
                "converged": self.converged,
                "convergence_iteration": self.convergence_iteration,
                "iteration_count": self.iteration_count,
                "scores_collected": len(self.scores),
                "current_variance": variance,
                "convergence_threshold": self.convergence_threshold,
                "mean_score": statistics.mean(self.scores),
                "std_deviation": statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0,
                "score_range": {"min": min(scores_list), "max": max(scores_list)},
                "recent_scores": scores_list,
                "stability_ratio": (
                    (self.convergence_threshold - variance) / self.convergence_threshold
                    if variance <= self.convergence_threshold
                    else 0.0
                ),
            }

        except statistics.StatisticsError as e:
            return {
                "converged": self.converged,
                "iteration_count": self.iteration_count,
                "scores_collected": len(self.scores),
                "error": f"Statistics calculation error: {e}",
            }

    def reset(self) -> None:
        """Reset tracker for new evaluation."""
        self.scores.clear()
        self.iteration_count = 0
        self.converged = False
        self.convergence_iteration = None
        logger.debug("Convergence tracker reset")


class ConvergenceManager:
    """Manages convergence tracking for multiple evaluations.

    Provides centralized convergence detection across different
    criterion-option pairs during parallel evaluation.
    """

    def __init__(
        self,
        convergence_threshold: float = 0.1,
        min_samples: int = 3,
        window_size: int = 5,
    ) -> None:
        """Initialize convergence manager.

        Args:
            convergence_threshold: Variance threshold for convergence detection
            min_samples: Minimum scores needed before checking convergence
            window_size: Maximum scores to keep in sliding window
        """
        self.convergence_threshold = convergence_threshold
        self.min_samples = min_samples
        self.window_size = window_size
        self.trackers: dict[str, ConvergenceTracker] = {}

        logger.info(
            "ConvergenceManager initialized: threshold=%s, min_samples=%s, window_size=%s",
            convergence_threshold,
            min_samples,
            window_size,
        )

    def get_tracker(self, evaluation_id: str) -> ConvergenceTracker:
        """Get or create a convergence tracker for an evaluation.

        Args:
            evaluation_id: Unique identifier for the evaluation

        Returns:
            ConvergenceTracker instance for this evaluation
        """
        if evaluation_id not in self.trackers:
            self.trackers[evaluation_id] = ConvergenceTracker(
                convergence_threshold=self.convergence_threshold,
                min_samples=self.min_samples,
                window_size=self.window_size,
            )
            logger.debug("Created new convergence tracker for %s", evaluation_id)

        return self.trackers[evaluation_id]

    def record_score(self, evaluation_id: str, score: float) -> bool:
        """Record a score and check for convergence.

        Args:
            evaluation_id: Unique identifier for the evaluation
            score: Score value to record

        Returns:
            True if convergence detected, False otherwise
        """
        tracker = self.get_tracker(evaluation_id)
        converged = tracker.add_score(score)

        if converged:
            logger.info(
                "Evaluation '%s' converged after %s iterations",
                evaluation_id,
                tracker.iteration_count,
            )

        return converged

    def is_converged(self, evaluation_id: str) -> bool:
        """Check if an evaluation has converged.

        Args:
            evaluation_id: Unique identifier for the evaluation

        Returns:
            True if evaluation has converged, False otherwise
        """
        if evaluation_id not in self.trackers:
            return False
        return self.trackers[evaluation_id].converged

    def get_convergence_summary(self) -> dict[str, Any]:
        """Get summary of all tracked evaluations.

        Returns:
            Dictionary with convergence statistics for all evaluations
        """
        total_evaluations = len(self.trackers)
        converged_count = sum(1 for tracker in self.trackers.values() if tracker.converged)

        summary = {
            "total_evaluations": total_evaluations,
            "converged_evaluations": converged_count,
            "convergence_rate": (
                converged_count / total_evaluations if total_evaluations > 0 else 0.0
            ),
            "configuration": {
                "convergence_threshold": self.convergence_threshold,
                "min_samples": self.min_samples,
                "window_size": self.window_size,
            },
            "evaluations": {},
        }

        for eval_id, tracker in self.trackers.items():
            summary["evaluations"][eval_id] = tracker.get_convergence_metrics()

        return summary

    def cleanup_tracker(self, evaluation_id: str) -> None:
        """Remove a tracker to free memory.

        Args:
            evaluation_id: Unique identifier for the evaluation to remove
        """
        if evaluation_id in self.trackers:
            del self.trackers[evaluation_id]
            logger.debug("Cleaned up convergence tracker for %s", evaluation_id)

    def cleanup_all(self) -> None:
        """Clean up all trackers."""
        tracker_count = len(self.trackers)
        self.trackers.clear()
        logger.info("Cleaned up %s convergence trackers", tracker_count)


def create_evaluation_id(criterion_name: str, option_name: str) -> str:
    """Create a unique evaluation ID for criterion-option pair.

    Args:
        criterion_name: Name of the evaluation criterion
        option_name: Name of the option being evaluated

    Returns:
        Unique identifier string for this evaluation
    """
    return f"{criterion_name}::{option_name}"
