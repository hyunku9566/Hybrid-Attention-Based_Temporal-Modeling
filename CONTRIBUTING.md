# Contributing to Baseline ADL Recognition

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [Issues](../../issues) section
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, PyTorch version)

### Suggesting Enhancements

We welcome suggestions for improvements:

1. Open an issue with the `enhancement` label
2. Describe the enhancement and its benefits
3. Provide examples or mockups if applicable

### Pull Requests

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards:
   - Write clear, documented code
   - Add docstrings to functions and classes
   - Follow PEP 8 style guide
   - Add type hints where applicable

3. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   black .
   flake8 .
   ```

4. **Commit your changes**:
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```
   
   Use conventional commit messages:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Docs:` for documentation changes
   - `Refactor:` for code refactoring
   - `Test:` for test additions/changes

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**:
   - Provide a clear description of changes
   - Reference related issues
   - Ensure all CI checks pass

## 📋 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception is raised
    """
    pass
```

### Testing

- Write unit tests for new features
- Maintain test coverage above 80%
- Place tests in `tests/` directory
- Use `pytest` for testing

Example test:

```python
def test_baseline_model():
    """Test baseline model forward pass"""
    model = BaselineModel(input_dim=114, hidden_dim=128, n_classes=5)
    X = torch.randn(2, 100, 114)
    output = model(X)
    assert output.shape == (2, 5)
```

## 🔍 Code Review Process

1. All pull requests must be reviewed by at least one maintainer
2. Address reviewer comments promptly
3. Keep pull requests focused and reasonably sized
4. Update documentation as needed

## 🌳 Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes

## 📝 Commit Message Guidelines

Format: `<type>: <description>`

**Types:**
- `Add:` New feature
- `Fix:` Bug fix
- `Docs:` Documentation
- `Refactor:` Code refactoring
- `Test:` Tests
- `Chore:` Maintenance

**Examples:**
```bash
Add: Implement multi-head attention option
Fix: Correct attention weight normalization
Docs: Update README with installation steps
Refactor: Simplify data loading pipeline
Test: Add unit tests for TCN block
```

## 🧪 Running Tests

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=models --cov=train --cov=evaluate tests/

# Check code style
black --check .
flake8 .
```

## 📚 Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/baseline-adl-recognition.git
   cd baseline-adl-recognition
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

4. **Install pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

- **Data augmentation**: New augmentation techniques for sensor data
- **Model architectures**: Alternative attention mechanisms or TCN variants
- **Optimization**: Training speed improvements
- **Documentation**: Tutorials, examples, API docs
- **Testing**: Increase test coverage
- **Benchmarking**: Comparisons with other methods
- **Visualization**: New visualization tools

## 📧 Contact

For questions or discussions:

- Open an issue with the `question` label
- Email: your.email@example.com

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

Thank you for contributing to make this project better!

---

**Happy Contributing! 🎉**
