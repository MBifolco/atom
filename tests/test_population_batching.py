#!/usr/bin/env python3
"""
Test population training batching to ensure memory-safe parallel execution.
"""

import pytest
from unittest.mock import MagicMock, patch, call, Mock
import time
import threading
from concurrent.futures import Future
import logging


class TestPopulationBatching:
    """Test that population training properly batches fighter training."""

    def test_batching_limits_concurrent_execution(self):
        """Test that only n_parallel_fighters run concurrently."""
        # Track concurrent executions
        concurrent_count = {'max': 0, 'current': 0}
        lock = threading.Lock()
        execution_log = []

        class MockExecutor:
            """Mock executor that tracks concurrent execution."""

            def __init__(self, max_workers=None, mp_context=None):
                self.max_workers = max_workers
                self.submitted_count = 0

            def submit(self, func, *args):
                """Track task submission and simulate execution."""
                fighter_name = args[0]  # First arg is fighter name
                self.submitted_count += 1

                future = Future()

                def run_task():
                    with lock:
                        concurrent_count['current'] += 1
                        concurrent_count['max'] = max(concurrent_count['max'], concurrent_count['current'])
                        execution_log.append(('start', fighter_name, time.time(), concurrent_count['current']))

                    # Simulate training time
                    time.sleep(0.05)  # Shorter time for faster test

                    with lock:
                        concurrent_count['current'] -= 1
                        execution_log.append(('end', fighter_name, time.time(), concurrent_count['current']))

                    # Set result on future
                    result = {
                        'fighter_name': fighter_name,
                        'episodes': 100,
                        'mean_reward': 50.0,
                        'final_model_path': f"/tmp/{fighter_name}.zip"
                    }
                    future.set_result(result)

                # Start task in background thread
                thread = threading.Thread(target=run_task)
                thread.daemon = True
                thread.start()

                return future

            def shutdown(self, wait=True, cancel_futures=False):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        # Patch ProcessPoolExecutor and file operations
        with patch('src.training.trainers.population.population_trainer.ProcessPoolExecutor', MockExecutor):
            with patch('pathlib.Path.mkdir'):
                with patch('pathlib.Path.unlink'):
                    with patch('stable_baselines3.PPO.load', return_value=Mock()):
                        from src.training.trainers.population.population_trainer import PopulationTrainer, PopulationFighter

                        trainer = PopulationTrainer(
                            population_size=6,  # 6 fighters total
                            algorithm='ppo',
                            output_dir='/tmp/test_batching',
                            max_ticks=100,
                            verbose=False,
                            n_parallel_fighters=2,  # Only 2 at a time
                            use_vmap=False  # CPU mode for testing
                        )

                        # Initialize population with mock models
                        trainer.population = []
                        for i in range(6):
                            fighter = PopulationFighter(
                                name=f"fighter_{i}",
                                model=Mock(),  # Simple mock
                                mass=70.0
                            )
                            # Mock the save method to create a valid zip file
                            def mock_save(path, fighter_name=f"fighter_{i}"):
                                # Create a valid empty zip file
                                import zipfile
                                import json
                                with zipfile.ZipFile(path, 'w') as zf:
                                    # Add minimal data to make it a valid SB3 model file
                                    data = {"policy_class": "MlpPolicy", "verbose": 0}
                                    zf.writestr("data", json.dumps(data))

                            fighter.model.save = Mock(side_effect=lambda path, fname=fighter.name: mock_save(path, fname))
                            trainer.population.append(fighter)

                        # Create fighter-opponent pairs
                        fighter_opponent_pairs = []
                        for fighter in trainer.population:
                            opponents = [f for f in trainer.population if f != fighter][:2]
                            fighter_opponent_pairs.append((fighter, opponents))

                        # Run training
                        results = trainer.train_fighters_parallel(
                            fighter_opponent_pairs,
                            episodes_per_fighter=100
                        )

                        # Give a bit more time for all threads to complete
                        time.sleep(0.5)

                        # Verify batching
                        print(f"\nMax concurrent executions: {concurrent_count['max']}")
                        assert concurrent_count['max'] <= 2, \
                            f"Too many concurrent processes: {concurrent_count['max']} (expected max 2)"

                        # Check that all 6 fighters trained
                        assert len(results) == 6, f"Not all fighters trained: {len(results)} of 6"

                        # Verify execution pattern shows batching
                        print(f"\nExecution log (showing batching behavior):")
                        for event, name, timestamp, concurrent in execution_log[:20]:  # Show first 20 events
                            print(f"  {event:5s}: {name:10s} at {timestamp:7.3f}s (concurrent: {concurrent})")

                        # Verify that we never had more than 2 concurrent
                        max_concurrent_from_log = max(c for _, _, _, c in [e for e in execution_log if e[0] == 'start'])
                        assert max_concurrent_from_log <= 2, \
                            f"Log shows {max_concurrent_from_log} concurrent executions, expected max 2"

    def test_gpu_batching_defaults_to_small_batch(self):
        """Test that GPU mode defaults to small batch size."""
        from src.training.trainers.population.population_trainer import PopulationTrainer

        # Create a proper mock for FileHandler
        mock_handler = Mock()
        mock_handler.level = logging.INFO
        mock_handler.setFormatter = Mock()

        with patch('pathlib.Path.mkdir'):
            with patch('logging.FileHandler', return_value=mock_handler):  # Mock file handler to avoid log file creation
                trainer = PopulationTrainer(
                    algorithm='ppo',
                    output_dir='/tmp/test_gpu_batching',
                    max_ticks=100,
                    verbose=False,
                    n_parallel_fighters=None,  # Should auto-detect
                    use_vmap=True  # GPU mode
                )

                # In GPU mode with n_parallel_fighters=None, should default to 2
                assert trainer.n_parallel_fighters == 2, \
                    f"GPU mode should default to 2 parallel fighters, got {trainer.n_parallel_fighters}"

    def test_cpu_batching_uses_multiple_cores(self):
        """Test that CPU mode defaults to using multiple cores."""
        from src.training.trainers.population.population_trainer import PopulationTrainer
        import multiprocessing

        # Create a proper mock for FileHandler
        mock_handler = Mock()
        mock_handler.level = logging.INFO
        mock_handler.setFormatter = Mock()

        with patch('pathlib.Path.mkdir'):
            with patch('logging.FileHandler', return_value=mock_handler):  # Mock file handler to avoid log file creation
                trainer = PopulationTrainer(
                    algorithm='ppo',
                    output_dir='/tmp/test_cpu_batching',
                    max_ticks=100,
                    verbose=False,
                    n_parallel_fighters=None,  # Should auto-detect
                    use_vmap=False  # CPU mode
                )

                expected = max(1, multiprocessing.cpu_count() - 1)
                assert trainer.n_parallel_fighters == expected, \
                    f"CPU mode should default to {expected} parallel fighters, got {trainer.n_parallel_fighters}"

    def test_explicit_parallel_count_is_respected(self):
        """Test that explicit n_parallel_fighters is respected."""
        from src.training.trainers.population.population_trainer import PopulationTrainer

        # Create a proper mock for FileHandler
        mock_handler = Mock()
        mock_handler.level = logging.INFO
        mock_handler.setFormatter = Mock()

        with patch('pathlib.Path.mkdir'):
            with patch('logging.FileHandler', return_value=mock_handler):  # Mock file handler to avoid log file creation
                trainer = PopulationTrainer(
                    algorithm='ppo',
                    output_dir='/tmp/test_explicit_batching',
                    max_ticks=100,
                    verbose=False,
                    n_parallel_fighters=4,  # Explicit setting
                    use_vmap=True  # Even in GPU mode
                )

                assert trainer.n_parallel_fighters == 4, \
                    f"Explicit setting should be respected, got {trainer.n_parallel_fighters}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])