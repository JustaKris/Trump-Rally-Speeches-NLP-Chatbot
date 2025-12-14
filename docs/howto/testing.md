# Testing & Development Guide

## Running Tests

### Using uv (recommended)

This repository uses uv to manage virtual environments and run commands in a reproducible project environment. If you've already been using `uv` in this project, the examples below will work as-is.

```powershell
# Install project dependencies (including dev groups defined in pyproject)
uv sync               # sync all default dependencies
uv sync --group dev   # sync dev dependencies group

# Run a command inside the project's environment
uv run <command>   # e.g. `uv run pytest` or `uv run ruff format src/`
```

If you prefer to use Poetry directly, the original Poetry commands are still valid and left as alternatives in this document.

### Install Development Dependencies (alternative: Poetry)

```powershell
# With Poetry (alternative)
poetry install --with dev

# Or activate Poetry shell and run commands directly
poetry shell
```

### Run All Tests

```powershell
# Run all tests with coverage
uv run pytest

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_preprocessing.py

# Run test with details on warnings
pytest -W default::Warning

# Run tests with enforced warnings
pytest -W error::DeprecationWarning
```

### Code Coverage

```powershell
# Generate coverage report
uv run pytest --cov=src --cov-report=html

# Open coverage report
start htmlcov/index.html  # Windows
```

## Code Quality Tools

### Formatting with Ruff

```powershell
# Format code with Ruff
uv run ruff format src/ scripts/ tests/

# Check formatting without changes (as used in CI)
uv run ruff format --check src/ scripts/ tests/
```

### Linting with Ruff

```powershell
# Run Ruff linter (as used in CI)
uv run ruff check src/ scripts/ tests/

# Auto-fix issues
uv run ruff check --fix src/ scripts/ tests/

# Show detailed statistics
uv run ruff check src/ scripts/ tests/ --statistics
```

### Type Checking

```powershell
# Run mypy type checker (as used in CI)
uv run mypy src/ scripts/ tests/ --ignore-missing-imports
```

### Security Scanning

```powershell
# Run Bandit security scan (as used in CI)
uv run bandit -r src/ scripts/ -c pyproject.toml

# Generate JSON report for CI
uv run bandit -r src/ scripts/ -c pyproject.toml -f json -o bandit-report.json

# Check for dependency vulnerabilities
uv run pip-audit --desc --skip-editable
```

### Markdown Linting

```powershell
# Lint documentation (as used in CI)
uv run pymarkdown --config pyproject.toml scan docs/ README.md
```

### Run All Quality Checks

```powershell
# Run all checks matching CI pipeline
uv run ruff format --check src/ scripts/ tests/
uv run ruff check src/ scripts/ tests/
uv run mypy src/ scripts/ tests/ --ignore-missing-imports
uv run bandit -r src/ scripts/ -c pyproject.toml
uv run pymarkdown --config pyproject.toml scan docs/ README.md
uv run pytest

# PowerShell (semicolon separator)
uv run ruff format --check src/ scripts/ tests/ ; uv run ruff check src/ scripts/ tests/ ; uv run mypy src/ scripts/ tests/ --ignore-missing-imports ; uv run bandit -r src/ scripts/ -c pyproject.toml ; uv run pytest
```

## Pre-commit Setup (Optional)

Install pre-commit hooks to automatically run checks before commits. If you manage dev dependencies with `uv`, use `uv sync --group dev` to install dev deps (including `pre-commit`) if listed in `pyproject.toml`. Otherwise install pre-commit directly:

```powershell
# Ensure pre-commit is installed in the project environment
uv run pip install pre-commit

# Install the git hook
uv run pre-commit install
```

If you prefer Poetry:

```powershell
poetry add --group dev pre-commit
poetry run pre-commit install
```

## CI/CD Pipeline

The GitHub Actions workflows run automatically on pushes and pull requests.

### Pipeline Jobs

1. **Python Tests** (`python-tests.yml`)
   - Runs on Python 3.11 and 3.12
   - Downloads NLTK data (punkt, stopwords, punkt_tab)
   - Runs all tests with coverage
   - Enforces minimum 60% coverage
   - Uploads coverage reports to Codecov

2. **Python Linting** (`python-lint.yml`)
   - Runs on Python 3.11 and 3.12
   - Checks: `uv run ruff check src/ scripts/ tests/`
   - Checks: `uv run ruff format --check src/ scripts/ tests/`

3. **Type Checking** (`python-typecheck.yml`)
   - Runs on Python 3.11 and 3.12
   - Checks: `uv run mypy src/ scripts/ tests/ --ignore-missing-imports`
   - Status: Allowed to fail (informational only)

4. **Security Audit** (`security-audit.yml`)
   - Runs on all pushes/PRs + weekly schedule
   - Checks: `uv run bandit -r src/ scripts/ -c pyproject.toml`
   - Checks: `uv run pip-audit --desc --skip-editable`
   - Status: Allowed to fail (informational only)

5. **Markdown Lint** (`markdown-lint.yml`)
   - Runs when markdown files change
   - Checks: `uv run pymarkdown --config pyproject.toml scan docs/ README.md`
   - Status: Allowed to fail (informational only)

6. **Docker Build** (`build-push-docker.yml`)
   - Runs on push to main branch
   - Builds and pushes Docker images

### Coverage Reports

Coverage reports are automatically uploaded to:
- **Codecov**: For PR comments and history tracking
- **GitHub Artifacts**: HTML reports available for 30 days

## Test Structure

```text
tests/
├── __init__.py
├── test_preprocessing.py  # Unit tests for text processing
├── test_utils.py          # Unit tests for utilities
└── test_api.py            # Integration tests for API endpoints
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - API integration tests
- `@pytest.mark.requires_model` - Tests needing ML model (skipped in CI)
- `@pytest.mark.slow` - Slow-running tests

## Writing New Tests

### Unit Test Example

```python
import pytest
from src.preprocessing import clean_text

@pytest.mark.unit
def test_clean_text():
    text = "Hello World!"
    result = clean_text(text)
    assert isinstance(result, str)
```

### API Test Example

```python
import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.mark.integration
def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
```

## Coverage Goals

- **Target**: 70%+ overall coverage
- **Focus**: Core logic in `src/ tests/`
- **Exclude**: ML model internals, notebooks

## Troubleshooting

### NLTK Data Missing

```powershell
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('punkt_tab', quiet=True)"
```

### Import Errors

```powershell
# Reinstall dependencies (uv)
uv sync --group dev

# Or with Poetry
poetry install --with dev
```

### Slow Tests

```powershell
# Skip slow tests
uv run pytest -m "not slow"

# Skip model-dependent tests
uv run pytest -m "not requires_model"
```
