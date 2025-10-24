.. _contributing:

************
Contributing
************

We welcome contributions to PyFVCOM2! This guide will help you get started with contributing code, documentation, or bug reports.

Development Setup
=================

1. **Fork the repository** on GitHub

2. **Clone your fork**::

    git clone https://github.com/yourusername/pyfvcom2.git
    cd pyfvcom2

3. **Create a development environment**::

    conda env create -f environment.yml
    conda activate pyfvcom2

4. **Install in development mode**::

    pip install -e .

5. **Install development dependencies**::

    pip install -e ".[dev]"

Code Standards
==============

**Style Guide:**

- Follow PEP 8 for Python code style
- Use type hints for all function parameters and return values
- Write docstrings in NumPy/SciPy format
- Keep line length under 88 characters (Black formatter default)

**Code Quality:**

- Run tests: ``pytest tests/``
- Check formatting: ``black --check pyfvcom2/``
- Check imports: ``isort --check-only pyfvcom2/``
- Type checking: ``mypy pyfvcom2/``
- Linting: ``flake8 pyfvcom2/``

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

5. **Run the test suite**::

    pytest tests/

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
- Update CHANGELOG.rst with your changes
- Add yourself to AUTHORS.rst (if not already there)
- Write a clear PR description explaining the changes

**PR Review Process:**

- All PRs must be reviewed by at least one maintainer
- Automated checks (CI/CD) must pass
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
- Community support and discussions

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
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and technical discussion
- **Email**: Contact maintainers for sensitive issues

Recognition
===========

Contributors are recognized in:

- AUTHORS.rst file
- Release notes and changelog
- GitHub contributor statistics
- Conference presentations (with permission)

Thank you for contributing to PyFVCOM2!