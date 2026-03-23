"""Tests for deferred holdout evaluation timing.

Holdout evaluations should run after gradient updates (on rollout start),
not immediately during advance_level (which fires mid-rollout before
weights are updated).
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestHoldoutTiming:
    """Tests for the pending holdout queue mechanism."""

    def _make_trainer(self, tmpdir: str):
        """Create a CurriculumTrainer with minimal config for testing."""
        from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer

        trainer = CurriculumTrainer(
            algorithm="ppo",
            output_dir=tmpdir,
            override_episodes_per_level=50,
            verbose=False,
        )
        return trainer

    def test_trainer_has_pending_holdout_list(self):
        """The trainer should initialize with an empty pending holdout list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(tmpdir)
            assert hasattr(trainer, "_pending_holdout_labels")
            assert trainer._pending_holdout_labels == []

    def test_advance_level_queues_label(self):
        """advance_level should append a label to _pending_holdout_labels.

        We verify this by inspecting the source code path rather than calling
        the full advance_level (which has many dependencies).  The queue
        mechanism itself is tested separately via flush tests.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(tmpdir)

            # Directly verify that the queueing line exists in advance_level
            import inspect

            source = inspect.getsource(trainer.advance_level)
            assert "_pending_holdout_labels.append" in source
            assert "_record_holdout_evaluation(checkpoint_label)" not in source

    def test_flush_runs_all_queued_evaluations(self):
        """_flush_pending_holdouts should run holdout for each queued label."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(tmpdir)

            # Queue some labels
            trainer._pending_holdout_labels = ["label_1", "label_2", "label_3"]

            # Mock the actual holdout evaluation
            trainer._record_holdout_evaluation = MagicMock()

            trainer._flush_pending_holdouts()

            # Should have run holdout for each label
            assert trainer._record_holdout_evaluation.call_count == 3
            calls = [c.args[0] for c in trainer._record_holdout_evaluation.call_args_list]
            assert calls == ["label_1", "label_2", "label_3"]

    def test_flush_clears_the_queue(self):
        """After flushing, the pending list should be empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(tmpdir)

            trainer._pending_holdout_labels = ["label_1", "label_2"]
            trainer._record_holdout_evaluation = MagicMock()

            trainer._flush_pending_holdouts()

            assert trainer._pending_holdout_labels == []

    def test_flush_with_empty_queue_is_noop(self):
        """Flushing an empty queue should not call holdout evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(tmpdir)

            trainer._record_holdout_evaluation = MagicMock()
            trainer._flush_pending_holdouts()

            trainer._record_holdout_evaluation.assert_not_called()
            assert trainer._pending_holdout_labels == []

    def test_multiple_advance_levels_queue_multiple_labels(self):
        """Multiple level advancements should accumulate labels in the queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(tmpdir)

            # Simulate queueing (directly, since advance_level has many dependencies)
            trainer._pending_holdout_labels.append("level_1_fundamentals_graduated")
            trainer._pending_holdout_labels.append("level_2_basic_skills_graduated")
            trainer._pending_holdout_labels.append("level_3_intermediate_graduated")

            assert len(trainer._pending_holdout_labels) == 3

            # Flush should process all
            trainer._record_holdout_evaluation = MagicMock()
            trainer._flush_pending_holdouts()

            assert trainer._record_holdout_evaluation.call_count == 3
            assert trainer._pending_holdout_labels == []
