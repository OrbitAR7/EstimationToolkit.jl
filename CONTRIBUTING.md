# Contributing to EstimationToolkit.jl

Thank you for your interest in contributing! This package aims to provide a comprehensive, well-tested suite of estimation algorithms for the Julia community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Run tests to ensure everything works
6. Submit a pull request

## Development Setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

## Code Style

We follow standard Julia conventions:

- Use 4 spaces for indentation (no tabs)
- Keep lines under 92 characters when practical
- Use descriptive variable names
- Add docstrings to all exported functions
- Include references to algorithms in docstrings

### Example Docstring Format

```julia
"""
    MyFilter(x0, P0, α=1.0)

Brief one-line description.

Extended description explaining what the filter does,
when to use it, and any important considerations.

# Arguments
- `x0::AbstractVector`: Initial state estimate
- `P0::AbstractMatrix`: Initial covariance matrix
- `α::Real=1.0`: Tuning parameter (optional)

# Returns
- `MyFilter`: Initialized filter instance

# References
- Author, A. (Year). "Title". Journal, Volume(Issue), pages.

# Example
```julia
x0 = [0.0, 0.0]
P0 = Matrix(1.0I, 2, 2)
filter = MyFilter(x0, P0)
```
"""
```

## Testing

All new filters must include comprehensive tests:

- Basic functionality (predict/update cycles)
- Edge cases (zero covariance, degenerate measurements)
- Consistency checks (NIS/NEES tests when applicable)
- Comparison with known solutions (linear systems)

Place tests in `test/` directory and follow existing naming conventions.

## Documentation

For new filters, please provide:

1. **Algorithm description** in the docstring
2. **Mathematical foundations** (even if brief)
3. **References** to original papers
4. **Usage example** showing typical application
5. **When to use** this filter vs alternatives

## Pull Request Process

1. **Update tests**: Add tests for new functionality
2. **Update README**: If adding a new filter, update the filter list
3. **Run full test suite**: Ensure `Pkg.test()` passes
4. **Write clear commit messages**: Explain what and why
5. **Keep PRs focused**: One feature or fix per PR when possible

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line is a concise summary (50 chars or less)
- Detailed explanation follows after a blank line if needed

Example:
```
Add Ensemble Square-Root Filter implementation

Implements the deterministic EnKF variant using square-root
updates to maintain covariance structure. Includes comprehensive
tests and comparison with standard EnKF.
```

## Types of Contributions

### Bug Reports

When filing a bug, please include:

- Julia version
- EstimationToolkit.jl version
- Minimal reproducible example
- Expected vs actual behavior
- Error messages (full stack trace)

### New Filter Implementations

We welcome implementations of additional estimation algorithms! Please:

1. Check if it's already implemented
2. Provide references to the original paper/textbook
3. Include thorough tests
4. Add performance benchmarks if relevant
5. Explain when this filter is preferable to existing ones

### Examples

Additional examples are always welcome, especially for:

- Aerospace applications (orbit determination, atmospheric reentry)
- Real-world sensor fusion scenarios
- Comparative studies showing filter strengths/weaknesses
- Tutorial-style walkthroughs for beginners

### Documentation

Help improve documentation by:

- Fixing typos or unclear explanations
- Adding mathematical derivations
- Creating tutorials
- Improving docstrings
- Adding comments to complex code sections

## Code Review Process

All submissions require review. We'll check for:

- Correctness (algorithm matches references)
- Code quality (readable, well-structured)
- Test coverage (comprehensive tests included)
- Documentation (clear docstrings and comments)
- Performance (no unnecessary allocations or inefficiencies)

## Questions?

Feel free to open an issue for:

- Implementation questions
- Algorithm clarifications
- Feature requests
- General discussion

We're here to help make your contribution successful!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
