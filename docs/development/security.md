# Security Guide

Security practices and vulnerability scanning for maintaining secure code.

## Overview

Security best practices include:

- **Static security scanning** with Bandit
- **Dependency vulnerability scanning** with pip-audit  
- **Secret scanning** via GitHub
- **Secure coding practices**
- **CI/CD security gates**

## Bandit Security Scanning

### Running Locally

```powershell
# Scan source code
uv run bandit -r src/

# Generate JSON report for CI
uv run bandit -r src/ -f json -o bandit-report.json

# Show only high severity
uv run bandit -r src/ -ll
```

### Common Issues & Fixes

**Unsafe Model Downloads (B615):**

HuggingFace model downloads are safe when model names come from configuration:

```python
# Add nosec when model names are from validated config
tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec B615
model = AutoModelForSequenceClassification.from_pretrained(model_name)  # nosec B615
```

**Try-Except-Continue (B112):**

Intentional for robust error handling:

```python
# Add nosec for intentional error handling patterns
except Exception:  # nosec B112
    # Skip failed analyses - intentional for robustness
    continue
```

**Assert in Production (B101):**

Replace asserts with proper exceptions:

```python
# Bad
assert isinstance(df, pd.DataFrame)

# Good  
if not isinstance(df, pd.DataFrame):
    raise TypeError("Expected DataFrame")
```

**Hardcoded Secrets (B105-B107):**

Use environment variables:

```python
# Bad
password = "example_value"

# Good
import os
password = os.getenv("DB_PASSWORD")
```

## Dependency Scanning

```powershell
# Scan for vulnerabilities
uv run pip-audit

# Update dependencies
uv lock --upgrade
```

## Secret Scanning

GitHub provides built-in secret scanning for public repositories. For private repositories, enable it in Settings → Security & Analysis → Secret scanning.

If a secret is detected:

1. Remove the secret from code
2. Rewrite git history if needed
3. Force push with `--force-with-lease`
4. Rotate the exposed secret immediately

## CI/CD Integration

Security scans run automatically in GitHub Actions:

```yaml
- name: Run bandit security scan
  run: |
    uv run bandit -r src/ scripts/ -c pyproject.toml -f json -o bandit-report.json || true
    uv run bandit -r src/ scripts/ -c pyproject.toml

- name: Check for known vulnerabilities
  run: |
    uv run pip-audit --desc --skip-editable || true
```

## Related Documentation

- **[Linting Guide](linting.md)** - Code quality checks
- **GitHub Actions Workflows** - See `.github/workflows/security.yml` for automated security scanning
