"""
Basic tests for opponents_jax module.
"""

import pytest


class TestOpponentsJax:
    """Test opponents JAX module."""

    def test_import_opponents_jax(self):
        """Test opponents_jax module imports."""
        from src.training import opponents_jax

        # Module should load
        assert opponents_jax is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
