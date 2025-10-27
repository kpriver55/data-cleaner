# CI/CD Documentation

This document describes the Continuous Integration and Continuous Deployment setup for the Data Cleaning Agent project.

## Overview

The project uses GitHub Actions for automated testing, linting, and quality checks on every push and pull request.

## Workflows

### Main CI Workflow (`.github/workflows/ci.yml`)

Runs on:

- Push to `main`, `develop`, `dspy-optimization-support` branches
- Pull requests to `main`, `develop`

#### Jobs

**1. Test** (`test`)

- **Matrix**: Python 3.9, 3.10, 3.11
- **Steps**:
  - Install dependencies
  - Run unit tests with pytest
  - Generate coverage report
  - Upload coverage to Codecov

**2. Lint** (`lint`)

- **Steps**:
  - Check code formatting with Black
  - Check import sorting with isort
  - Lint with flake8
  - Type check with mypy

**3. Security** (`security`)

- **Steps**:
  - Check dependencies with Safety
  - Security scan with Bandit

**4. Documentation** (`docs`)

- **Steps**:
  - Check for broken links in markdown
  - Validate example configurations

## Local Development Setup

### 1. Install Development Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks

Pre-commit hooks run checks before each commit:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### 3. Run Tests Locally

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=optimization --cov-report=html

# Run specific test file
pytest optimization/test_evaluators.py -v

# Run with specific marker
pytest -m "not slow"
```

### 4. Format and Lint Code

```bash
# Format with Black
black .

# Sort imports
isort .

# Lint with flake8
flake8 .

# Type check
mypy optimization/ --ignore-missing-imports
```

## Code Quality Standards

### Formatting

- **Tool**: Black
- **Line length**: 100 characters
- **Target**: Python 3.9+

### Import Sorting

- **Tool**: isort
- **Profile**: black (compatible with Black)
- **Line length**: 100 characters

### Linting

- **Tool**: flake8
- **Max line length**: 100
- **Ignored**: E203 (whitespace before ':'), W503 (line break before binary operator)
- **Max complexity**: 10

### Type Checking

- **Tool**: mypy
- **Mode**: Lenient (no strict optional, ignore missing imports)
- **Target**: Python 3.9+

### Security

- **Dependencies**: Safety (checks known vulnerabilities)
- **Code**: Bandit (static security analysis)
- **Level**: Low to low severity issues flagged

## Coverage Requirements

- **Target**: >80% coverage
- **Tool**: pytest-cov
- **Reports**: Terminal, HTML, XML (for Codecov)

### Viewing Coverage Reports

```bash
# Generate HTML report
pytest --cov=optimization --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Pre-commit Hooks

The `.pre-commit-config.yaml` configures hooks that run automatically:

1. **General checks**:
   - Trailing whitespace
   - End-of-file fixer
   - YAML/JSON validation
   - Large file check (max 1MB)
   - Merge conflict detection

2. **Code formatting**:
   - Black (auto-format)
   - isort (sort imports)

3. **Linting**:
   - flake8 (PEP 8 compliance)

4. **Security**:
   - Bandit (security issues)

5. **Documentation**:
   - Markdown linting

### Skipping Hooks

```bash
# Skip all hooks for one commit
git commit --no-verify

# Skip specific hook
SKIP=flake8 git commit -m "message"
```

## Continuous Integration

### GitHub Actions Configuration

All workflows are in `.github/workflows/`:

```
.github/
├── workflows/
│   └── ci.yml                           # Main CI workflow
└── markdown-link-check-config.json      # Link checker config
```

### Status Badges

Add to README.md:

```markdown
![CI](https://github.com/yourusername/data-cleaner/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/yourusername/data-cleaner/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/data-cleaner)
```

## Release Process

### Versioning

We use Semantic Versioning (SemVer):

```
MAJOR.MINOR.PATCH

- MAJOR: Breaking changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes, backwards compatible
```

Current version: `0.3.0` (in `pyproject.toml`)

### Creating a Release

1. **Update version** in `pyproject.toml`:

   ```toml
   [project]
   version = "0.4.0"
   ```

2. **Update CHANGELOG** (if exists):

   ```markdown
   ## [0.4.0] - 2025-01-XX
   ### Added
   - New feature X
   ### Fixed
   - Bug Y
   ```

3. **Commit and tag**:

   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Bump version to 0.4.0"
   git tag v0.4.0
   git push origin main --tags
   ```

4. **Create GitHub Release** (optional):
   - Go to GitHub Releases
   - Create release from tag
   - Add release notes

## Package Distribution

### Building the Package

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*
```

### Publishing to PyPI (when ready)

```bash
# Test PyPI
twine upload --repository testpypi dist/*

# Production PyPI
twine upload dist/*
```

## Troubleshooting

### CI Failures

**Tests failing locally but pass in CI:**

- Check Python version matches (use `python --version`)
- Ensure all dependencies installed: `pip install -r requirements.txt -r requirements-dev.txt`
- Clear pytest cache: `rm -rf .pytest_cache __pycache__`

**Linting failures:**

- Run `black .` and `isort .` to auto-fix formatting
- Run `flake8 .` to see specific issues
- Check `.flake8` config if rules seem inconsistent

**Type checking failures:**

- mypy is configured to be lenient (continues on error)
- Add `# type: ignore` comments for false positives
- Update type stubs: `pip install --upgrade types-PyYAML types-requests`

### Pre-commit Issues

**Hooks failing:**

```bash
# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean

# Reinstall
pre-commit uninstall
pre-commit install
```

**Slow hooks:**

- Mypy can be slow - consider commenting it out in `.pre-commit-config.yaml`
- Run specific hooks: `pre-commit run <hook-id>`

## Best Practices

### Before Committing

1. Run tests: `pytest`
2. Check formatting: `black --check .`
3. Check linting: `flake8 .`
4. Review changes: `git diff`

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)
- [ ] No linting errors
- [ ] Coverage not decreased

### Code Review

1. Check CI status (all green)
2. Review code changes
3. Test locally if needed
4. Approve or request changes

## Configuration Files

### `.github/workflows/ci.yml`

Main CI workflow configuration

### `.pre-commit-config.yaml`

Pre-commit hooks configuration

### `pyproject.toml`

Python package configuration and tool settings:

- `[tool.black]` - Black formatter
- `[tool.isort]` - Import sorting
- `[tool.mypy]` - Type checking
- `[tool.pytest.ini_options]` - Pytest configuration
- `[tool.coverage.*]` - Coverage settings

### `requirements-dev.txt`

Development dependencies (testing, linting, etc.)

## Future Enhancements

- [ ] Add integration tests
- [ ] Add performance benchmarks
- [ ] Add security scanning (Dependabot, Snyk)
- [ ] Add automatic documentation building
- [ ] Add Docker image building
- [ ] Add deployment workflows
- [ ] Add release automation

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Semantic Versioning](https://semver.org/)
