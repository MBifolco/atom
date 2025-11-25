"""
NaN Detection and Debugging Utilities for Training

Helps identify the source of NaN values during training by logging
observations, actions, rewards, and model outputs at each step.
"""

import numpy as np
import torch
import logging
from typing import Any, Dict, Optional, Union, List
import traceback
from pathlib import Path
import json
from datetime import datetime


class NaNDetector:
    """Detects and logs NaN occurrences during training."""

    def __init__(self, log_dir: str = "outputs/nan_debug", verbose: bool = True):
        """
        Initialize NaN detector.

        Args:
            log_dir: Directory to save debug logs
            verbose: Whether to print warnings immediately
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Setup logging
        log_file = self.log_dir / f"nan_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = logging.getLogger("NaNDetector")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Console handler for warnings
        if verbose:
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.step_count = 0
        self.episode_count = 0
        self.nan_count = 0
        self.last_observations = None
        self.last_actions = None
        self.last_rewards = None

    def check_observations(self, obs: Union[np.ndarray, torch.Tensor], step: Optional[int] = None) -> bool:
        """
        Check observations for NaN or inf values.

        Returns:
            True if NaN/inf detected
        """
        self.step_count += 1
        step = step or self.step_count

        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = obs

        has_nan = np.isnan(obs_np).any()
        has_inf = np.isinf(obs_np).any()

        if has_nan or has_inf:
            self.nan_count += 1
            self.logger.error(f"Step {step}: NaN/Inf in observations!")
            self.logger.error(f"  Shape: {obs_np.shape}")
            self.logger.error(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
            self.logger.error(f"  Min: {np.nanmin(obs_np)}, Max: {np.nanmax(obs_np)}")
            self.logger.error(f"  Mean: {np.nanmean(obs_np)}, Std: {np.nanstd(obs_np)}")

            # Log the actual values
            if obs_np.size < 100:  # Only log if not too large
                self.logger.error(f"  Values: {obs_np}")

            # Save problematic observation
            np.save(self.log_dir / f"nan_obs_step_{step}.npy", obs_np)

            # Log previous step's data if available
            if self.last_observations is not None:
                self.logger.error("Previous step data:")
                self.logger.error(f"  Last obs min: {np.nanmin(self.last_observations)}, max: {np.nanmax(self.last_observations)}")
                if self.last_actions is not None:
                    self.logger.error(f"  Last action: {self.last_actions}")
                if self.last_rewards is not None:
                    self.logger.error(f"  Last reward: {self.last_rewards}")

            return True

        # Store for next step comparison
        self.last_observations = obs_np.copy()

        # Log statistics periodically (every 1000 steps)
        if step % 1000 == 0:
            self.logger.info(f"Step {step}: Observations OK - Min: {np.min(obs_np):.3f}, Max: {np.max(obs_np):.3f}, Mean: {np.mean(obs_np):.3f}")

        return False

    def check_actions(self, actions: Union[np.ndarray, torch.Tensor], step: Optional[int] = None) -> bool:
        """Check actions for NaN or inf values."""
        step = step or self.step_count

        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions

        has_nan = np.isnan(actions_np).any()
        has_inf = np.isinf(actions_np).any()

        if has_nan or has_inf:
            self.nan_count += 1
            self.logger.error(f"Step {step}: NaN/Inf in actions!")
            self.logger.error(f"  Shape: {actions_np.shape}")
            self.logger.error(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
            self.logger.error(f"  Values: {actions_np}")

            # Save problematic actions
            np.save(self.log_dir / f"nan_actions_step_{step}.npy", actions_np)
            return True

        self.last_actions = actions_np.copy()
        return False

    def check_rewards(self, rewards: Union[np.ndarray, float, List], step: Optional[int] = None) -> bool:
        """Check rewards for NaN or inf values."""
        step = step or self.step_count

        if isinstance(rewards, (list, tuple)):
            rewards_np = np.array(rewards)
        elif isinstance(rewards, torch.Tensor):
            rewards_np = rewards.detach().cpu().numpy()
        elif isinstance(rewards, (int, float)):
            rewards_np = np.array([rewards])
        else:
            rewards_np = rewards

        has_nan = np.isnan(rewards_np).any()
        has_inf = np.isinf(rewards_np).any()

        if has_nan or has_inf:
            self.nan_count += 1
            self.logger.error(f"Step {step}: NaN/Inf in rewards!")
            self.logger.error(f"  Values: {rewards_np}")
            return True

        # Check for extremely large rewards
        if np.abs(rewards_np).max() > 10000:
            self.logger.warning(f"Step {step}: Large reward detected: {rewards_np}")

        self.last_rewards = rewards_np.copy()

        # Log reward statistics periodically
        if step % 1000 == 0:
            self.logger.info(f"Step {step}: Rewards OK - Min: {np.min(rewards_np):.1f}, Max: {np.max(rewards_np):.1f}, Mean: {np.mean(rewards_np):.1f}")

        return False

    def check_model_output(self, output: torch.Tensor, name: str = "output", step: Optional[int] = None) -> bool:
        """Check model outputs for NaN or inf values."""
        step = step or self.step_count

        if output is None:
            return False

        output_np = output.detach().cpu().numpy()
        has_nan = np.isnan(output_np).any()
        has_inf = np.isinf(output_np).any()

        if has_nan or has_inf:
            self.nan_count += 1
            self.logger.error(f"Step {step}: NaN/Inf in model {name}!")
            self.logger.error(f"  Shape: {output_np.shape}")
            self.logger.error(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")

            # Find where the NaN/Inf values are
            if has_nan:
                nan_indices = np.where(np.isnan(output_np))
                self.logger.error(f"  NaN indices: {nan_indices}")
            if has_inf:
                inf_indices = np.where(np.isinf(output_np))
                self.logger.error(f"  Inf indices: {inf_indices}")

            # Save the problematic output
            np.save(self.log_dir / f"nan_{name}_step_{step}.npy", output_np)

            # Get stack trace
            self.logger.error("Stack trace:")
            for line in traceback.format_stack():
                self.logger.error(line.strip())

            return True

        return False

    def check_gradients(self, model: torch.nn.Module, step: Optional[int] = None) -> bool:
        """Check model gradients for NaN or inf values."""
        step = step or self.step_count

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_np = param.grad.detach().cpu().numpy()
                has_nan = np.isnan(grad_np).any()
                has_inf = np.isinf(grad_np).any()

                if has_nan or has_inf:
                    self.nan_count += 1
                    self.logger.error(f"Step {step}: NaN/Inf in gradients of {name}!")
                    self.logger.error(f"  Shape: {grad_np.shape}")
                    self.logger.error(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
                    self.logger.error(f"  Grad norm: {np.linalg.norm(grad_np)}")
                    return True

                # Check for exploding gradients
                grad_norm = np.linalg.norm(grad_np)
                if grad_norm > 100:
                    self.logger.warning(f"Step {step}: Large gradient in {name}: norm={grad_norm:.2f}")

        return False

    def log_episode_end(self, episode_info: Dict[str, Any]):
        """Log episode information."""
        self.episode_count += 1

        # Log episode stats
        self.logger.info(f"Episode {self.episode_count} completed:")
        self.logger.info(f"  Length: {episode_info.get('l', 'unknown')}")
        self.logger.info(f"  Reward: {episode_info.get('r', 'unknown')}")

        if self.episode_count % 100 == 0:
            self.logger.info(f"Total NaN occurrences so far: {self.nan_count}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_steps": self.step_count,
            "total_episodes": self.episode_count,
            "nan_count": self.nan_count,
            "nan_rate": self.nan_count / max(1, self.step_count)
        }