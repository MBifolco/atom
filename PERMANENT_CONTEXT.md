## Basics
- Become familiar with the Atom Combat concept by reading: /home/biff/eng/atom/docs/original_vision/concept.md
- Do not add documentation to the root directory - only update the readme if relevant otherwise add documentation to the readme of the correct module/directory or in the docs directory
- Do not add scripts to the root directory unless they are to be run frequently as part of expected use of the application
- ALWAYS write small methods/functions
- ALWAYS refactor for reusability rather than duplicate code

## Testing
- Write tests after you fix a bug to ensure its covered
- After fixing a bug - think about why that bug wasn't covered by tests and then apply that to whether or not other features/functions should have additional tests
- Write comprehensive tests for new features (see tests/test_discrete_hits.py as example)
- Include edge cases, physics validation, and integration tests
- Run full test suite with `pytest tests/ -v` after changes
- Tests should cover: normal operation, edge cases, error conditions
- Add tests to the tests/ directory for any edit or addition
- Always run tests after making changes and fix any issues

## JAX & GPU
- Always include CPU fallback for GPU operations
- Use `ATOM_FORCE_CPU` environment variable support
- Test on both CPU and GPU when possible
- Handle JAX import errors gracefully
- Force CPU mode for compatibility: `jax.config.update('jax_platform_name', 'cpu')`

## Fighter Files
- Must have `decide(state)` function (not `get_action`)
- State format: `{"you": {...}, "opponent": {...}, "arena": {...}}`
- Return format: `{"acceleration": float, "stance": str}`
- 3 stances only: "neutral", "extended", "defending"
- Include docstring describing strategy

## Code Style
- Use type hints for function signatures
- Include docstrings for all public functions
- Prefer pure functions for JAX compatibility (no side effects)
- Import organization: stdlib, third-party, local
- Use descriptive variable names

## Breaking Changes
- Document breaking changes clearly in docs/
- No need for backward compatibility unless explicitly needed
- Clean breaks preferred over legacy support
- Update all affected code in same commit
