.. _contributing:

************
Contributing
************

We welcome contributions to PyFVCOM2. This guide covers the development setup,
checks, and review process for code, documentation, and bug reports.

Development Setup
=================

1. **Fork the repository** on GitHub if you plan to open a pull request.

2. **Clone your fork or the upstream repository**::

    git clone https://github.com/pmlmodelling/pyfvcom2.git
    cd pyfvcom2

3. **Create a development environment**::

    conda env create -f environment.yml
    conda activate pyfvcom2

4. **Install the package with development dependencies**::

    pip install -e ".[dev]"

Code Standards
==============

**Style Guide:**

- Follow PEP 8 for Python code style
- Use type hints for all function parameters and return values
- Write docstrings in NumPy/SciPy format
- Keep line length under 88 characters (Black formatter default)

**Required checks:**

- Run tests: ``pytest tests/``
- Check for Python syntax and undefined-name errors:
  ``flake8 pyfvcom2 --jobs=1 --count --select=E9,F63,F7,F82 --show-source --statistics``

**Optional local checks:**

- Run broader linting:
  ``flake8 pyfvcom2 --jobs=1 --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics``

**Testing:**

- Write unit tests for all new functions
- Aim for >90% code coverage
- Test edge cases and error conditions
- Use pytest fixtures for common test data

Submitting Changes
==================

1. **Create a feature branch**::

    git checkout -b feature/your-feature-name

2. **Make your changes** following the code standards

3. **Write or update tests** for your changes

4. **Update documentation** if needed

5. **Run the required checks**::

    pytest tests/
    flake8 pyfvcom2 --jobs=1 --count --select=E9,F63,F7,F82 --show-source --statistics

6. **Commit your changes**::

    git add .
    git commit -m "Add descriptive commit message"

7. **Push to your fork**::

    git push origin feature/your-feature-name

8. **Create a Pull Request** on GitHub

Pull Request Guidelines
=======================

**Before submitting:**

- Ensure all tests pass
- Write a clear PR description explaining the changes

**PR Review Process:**

- All PRs must be reviewed by at least one maintainer
- Automated checks must pass
- Documentation must be updated for API changes
- Breaking changes require discussion and approval

Types of Contributions
======================

**Code Contributions:**

- New features and functionality
- Bug fixes and performance improvements
- Code refactoring and cleanup
- Test coverage improvements

**Documentation:**

- API documentation improvements
- Tutorial and example development
- User guide enhancements
- Translation efforts

**Other Contributions:**

- Bug reports with reproducible examples
- Feature requests with use cases
- Performance benchmarking
- Answering questions in issues and pull requests

Reporting Issues
================

**Bug Reports:**

Include the following information:

- PyFVCOM2 version
- Python version and environment
- Minimal code example reproducing the issue
- Full error traceback
- Expected vs. actual behavior

**Feature Requests:**

- Clear description of the proposed feature
- Use cases and benefits
- Possible implementation approaches
- Willingness to contribute code

Communication
=============

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code review and technical discussion

Recognition
===========

Contributors are recognized in:

- GitHub contributor statistics
- Pull request and issue history
- Release notes when relevant

Thank you for contributing to PyFVCOM2!
