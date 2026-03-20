#!/usr/bin/env python3
"""
Test gradient stability to prevent NaN issues during training.
"""

import pytest
import numpy as np
import torch
from typing import List


class TestRewardStability:
    """Test gradient clipping and stability measures."""

    def test_gradient_clipping(self):
        """Test that gradient clipping prevents explosion."""
        # Create a simple network
        net = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

        # Create extreme input that could cause gradient explosion
        x = torch.randn(32, 10) * 1000  # Very large inputs
        target = torch.randn(32, 2)

        # Forward pass
        output = net(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients before clipping
        total_norm_before = 0
        for p in net.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.data.norm(2).item() ** 2
        total_norm_before = total_norm_before ** 0.5

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)

        # Check gradients after clipping
        total_norm_after = 0
        for p in net.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5

        # Should be clipped to max norm
        assert total_norm_after <= 10.01  # Small tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])