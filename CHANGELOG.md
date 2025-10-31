# Changelog

All notable changes to CoreRec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LICENSE file (MIT License)
- Proper packaging configuration (pyproject.toml, setup.py in root)
- Comprehensive requirements files (requirements.txt, requirements-dev.txt, requirements-test.txt)
- CI/CD workflows using GitHub Actions (test.yml, lint.yml, docs.yml)
- CONTRIBUTING.md with detailed contribution guidelines
- Pre-commit hooks configuration (.pre-commit-config.yaml)
- Improved .gitignore to exclude build artifacts
- CRITICAL_ISSUES_ANALYSIS.md - comprehensive codebase analysis
- QUICKSTART_FIXES.md - guide for fixing critical issues
- CHANGELOG.md (this file)
- CLI tool with tab completion (`corerec` command)

### Changed
- Moved setup.py from production/ to root directory
- Updated .gitignore to follow Python best practices
- Improved package structure for better import consistency

### Fixed
- Build artifacts (*.o, *.pyc, __pycache__) cleanup from repository
- Package installation issues (setup.py now in correct location)

### Security
- Added Bandit security scanning to CI/CD
- Added Safety dependency vulnerability scanning

## [0.5.1] - 2024-10-XX

### Added
- Multiple recommendation engines (DCN, DeepFM, GNNRec, MIND, NASRec, SASRec)
- Unionized Filter Engine (collaborative filtering algorithms)
- Content Filter Engine (content-based filtering algorithms)
- Comprehensive API with BaseRecommender base class
- Extensive examples directory

### Known Issues
- Two base classes exist (BaseRecommender and BaseCorerec) - consolidation needed
- Circular import risks in core_rec.py
- Incomplete type hints
- C++ extensions not properly configured in setup
- Missing comprehensive test coverage reports

## [0.5.0] and earlier

See git history for previous changes.

---

## Version Number Guide

- MAJOR version: Incompatible API changes
- MINOR version: Added functionality (backwards-compatible)
- PATCH version: Bug fixes (backwards-compatible)

[Unreleased]: https://github.com/vishesh9131/CoreRec/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/vishesh9131/CoreRec/releases/tag/v0.5.1
