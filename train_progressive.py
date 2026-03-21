#!/usr/bin/env python3
"""Compatibility wrapper for progressive training."""

from __future__ import annotations

from apps.training.train_progressive import main
from src.atom.training.pipelines import ProgressiveTrainer

__all__ = ["ProgressiveTrainer", "main"]


if __name__ == "__main__":
    main()
