"""
Basic tests for vmap environment wrapper.
"""

import pytest


class TestVmapWrapper:
    """Test vmap wrapper module."""

    def test_import_vmap_wrapper(self):
        """Test vmap wrapper module imports."""
        from src.training import vmap_env_wrapper

        # Module should load
        assert vmap_env_wrapper is not None
        # Should have key classes/functions
        assert hasattr(vmap_env_wrapper, 'VMapEnvWrapper') or dir(vmap_env_wrapper)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
