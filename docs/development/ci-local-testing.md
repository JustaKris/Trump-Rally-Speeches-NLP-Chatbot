# CI/CD Local Testing Guide

This guide shows you how to run **exactly the same checks** that GitHub Actions CI/CD pipeline runs, ensuring your code will pass before pushing.

## Overview

Our CI/CD pipeline uses these separate workflows:

- **Python Tests** - Unit/integration tests with coverage
- **Python Linting** - Ruff linting and formatting
- **Type Checking** - Mypy static type analysis  
- **Security Audit** - Bandit and pip-audit scans
- **Markdown Lint** - Documentation quality checks

## Prerequisites

```powershell
# Install all dev dependencies
uv sync --group dev

# Verify uv is working
uv --version
```

## Quick Check: Run All CI Checks Locally

This single command runs **all** checks matching the CI pipeline:

```powershell
# Run all CI checks (PowerShell)
uv run ruff format --check src/ scripts/ tests/ ; `
uv run ruff check src/ scripts/ tests/ ; `
uv run mypy src/ scripts/ tests/ --ignore-missing-imports ; `
uv run bandit -r src/ scripts/ -c pyproject.toml ; `
uv run pymarkdown --config pyproject.toml scan docs/ README.md ; `
uv run pytest
```

**Note:** If any command fails, the rest won't run. Fix issues and re-run.

## Individual CI Checks

Run checks individually for faster iteration:

### 1. Python Tests (`python-tests.yml`)

**What CI runs:**

```yaml
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Run tests with coverage
pytest tests/ -v \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=html:reports/coverage/html \
  --cov-report=xml:reports/coverage/coverage.xml \
  --cov-fail-under=60 \
  --tb=short
```

**Run locally:**

```powershell
# NLTK data (only needed once)
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Run tests exactly as CI does
uv run pytest tests/ -v `
  --cov=src `
  --cov-report=term-missing `
  --cov-report=html:reports/coverage/html `
  --cov-report=xml:reports/coverage/coverage.xml `
  --cov-fail-under=60 `
  --tb=short

# View coverage report
start reports/coverage/html/index.html
```

**Minimum coverage:** 60% (enforced by `--cov-fail-under=60`)

### 2. Python Linting (`python-lint.yml`)

**What CI runs:**

```yaml
# Check linting
ruff check src/ scripts/ tests/

# Check formatting
ruff format --check src/ scripts/ tests/
```

**Run locally:**

```powershell
# Lint check (as CI does)
uv run ruff check src/ scripts/ tests/

# Format check (as CI does)
uv run ruff format --check src/ scripts/ tests/

# Auto-fix linting issues
uv run ruff check --fix src/ scripts/ tests/

# Auto-format code
uv run ruff format src/ scripts/ tests/
```

### 3. Type Checking (`python-typecheck.yml`)

**What CI runs:**

```yaml
mypy src/ scripts/ tests/ --ignore-missing-imports
```

**Run locally:**

```powershell
# Type check (as CI does)
uv run mypy src/ scripts/ tests/ --ignore-missing-imports

# Show error codes for debugging
uv run mypy src/ scripts/ tests/ --ignore-missing-imports --show-error-codes
```

**Status:** Allowed to fail in CI (informational only)

### 4. Security Audit (`security-audit.yml`)

**What CI runs:**

```yaml
# Bandit security scan
bandit -r src/ scripts/ -c pyproject.toml -f json -o bandit-report.json || true
bandit -r src/ scripts/ -c pyproject.toml

# Dependency vulnerabilities
pip-audit --desc --skip-editable || true
```

**Run locally:**

```powershell
# Bandit scan (as CI does)
uv run bandit -r src/ scripts/ -c pyproject.toml

# Generate JSON report
uv run bandit -r src/ scripts/ -c pyproject.toml -f json -o bandit-report.json

# Dependency scan
uv run pip-audit --desc --skip-editable
```

**Status:** Allowed to fail in CI (informational only)

### 5. Markdown Linting (`markdown-lint.yml`)

**What CI runs:**

```yaml
pymarkdown --config pyproject.toml scan docs/ README.md
```

**Run locally:**

```powershell
# Lint markdown (as CI does)
uv run pymarkdown --config pyproject.toml scan docs/ README.md

# Scan specific directory
uv run pymarkdown --config pyproject.toml scan docs/development/

# Scan single file
uv run pymarkdown --config pyproject.toml scan README.md
```

**Status:** Allowed to fail in CI (informational only)

## Configuration Files

All CI checks use configuration from `pyproject.toml`:

### Ruff Configuration

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "D"]
ignore = ["E501", "B008", "B904", "D203", "D213", "D205", "D415", "D102"]
```

### Pytest Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "--verbose",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html:reports/coverage/html",
    "--cov-report=xml:reports/coverage/coverage.xml",
    "--cov-fail-under=60",
]
```

### Mypy Configuration

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
```

### Bandit Configuration

```toml
[tool.bandit]
exclude_dirs = [".venv", "venv", "notebooks", "site"]
skips = ["B101", "B104", "B108", "B110", "B608"]
```

### Pymarkdown Configuration

```toml
[tool.pymarkdown]
plugins.md013.enabled = false  # Line length
plugins.md033.enabled = false  # Inline HTML
plugins.md036.enabled = false  # Emphasis headings
```

## CI Workflow Matrix

CI tests run on **Python 3.11 and 3.12**. Local testing on your Python version is usually sufficient, but you can test both with:

```powershell
# Check your Python version
python --version

# If needed, test with specific Python version
uv run --python 3.11 pytest
uv run --python 3.12 pytest
```

## Troubleshooting

### Tests pass locally but fail in CI

1. **Check Python version:** CI runs on 3.11 and 3.12
2. **Check NLTK data:** CI downloads fresh NLTK data
3. **Check dependencies:** Run `uv sync --group dev` to update

### Bandit warnings about comments

Bandit may show warnings about `# nosec` comments:

```text
WARNING Test in comment: Intentional is not a test name or id, ignoring
```

This is expected and harmless. The `# nosec: B104 - Intentional for...` comments are for documentation.

### Coverage fails in CI but passes locally

1. **Check coverage reports folder:** Ensure `reports/coverage/` exists
2. **Run with same options:** Use the exact pytest command from CI
3. **Check `.gitignore`:** Coverage files should be gitignored

## Pre-commit Hooks (Optional)

To run checks automatically before each commit:

```powershell
# Install pre-commit hooks
uv run pre-commit install

# Run on all files
uv run pre-commit run --all-files
```

## Summary: Passing CI Checklist

Before pushing, ensure these pass:

- [ ] `uv run ruff format --check src/ scripts/ tests/` ✅
- [ ] `uv run ruff check src/ scripts/ tests/` ✅
- [ ] `uv run pytest` (60%+ coverage) ✅
- [ ] `uv run mypy src/ scripts/ tests/ --ignore-missing-imports` ℹ️ (allowed to fail)
- [ ] `uv run bandit -r src/ scripts/ -c pyproject.toml` ℹ️ (allowed to fail)
- [ ] `uv run pymarkdown --config pyproject.toml scan docs/ README.md` ℹ️ (allowed to fail)

**Legend:**
- ✅ Must pass (blocks CI)
- ℹ️ Informational (won't block CI)

## CI Workflows Reference

| Workflow | File | Triggers | Python Versions |
|----------|------|----------|----------------|
| Python Tests | `python-tests.yml` | All pushes/PRs | 3.11, 3.12 |
| Python Linting | `python-lint.yml` | Python file changes | 3.11, 3.12 |
| Type Checking | `python-typecheck.yml` | Python file changes | 3.11, 3.12 |
| Security Audit | `security-audit.yml` | Pushes/PRs + Weekly | 3.12 |
| Markdown Lint | `markdown-lint.yml` | Markdown changes | 3.12 |

## Additional Resources

- [Testing Guide](../howto/testing.md) - Comprehensive testing documentation
- [Linting Guide](linting.md) - Detailed linting configuration
- [Security Guide](security.md) - Security best practices
- [Markdown Linting](markdown-linting.md) - Documentation style guide
