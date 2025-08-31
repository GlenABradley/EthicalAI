# Contributing

## Development Setup

1. Clone the repository
2. Install Python 3.11
3. Create virtual environment: `python -m venv venv`
4. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
5. Install dependencies: `pip install -r requirements.txt`
6. Install pre-commit: `pip install pre-commit`
7. Install hooks: `pre-commit install`

## How to Contribute

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Add tests for new functionality
5. Run pre-commit: `pre-commit run --all-files`
6. Run tests: `pytest`
7. Update documentation if needed
8. Submit a pull request

## Code Style

- **Linting**: Use `ruff` for fast Python linting
- **Formatting**: Use `black` for consistent formatting
- **Type checking**: Use `mypy` for static type analysis
- **Pre-commit**: All hooks must pass before committing

## Testing

- Write unit tests for new functions
- Write integration tests for API endpoints
- Ensure all tests pass with `pytest`
- Aim for good test coverage

## Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Provide detailed steps to reproduce
- Include relevant logs and system information
