# Contributing to sklearn-under-the-hood

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🎯 Project Goals

This project aims to:
1. Provide clear, educational implementations of ML algorithms
2. Help users understand how scikit-learn works internally
3. Offer comprehensive documentation and tutorials
4. Compare custom implementations with scikit-learn

## 🤝 How to Contribute

### Types of Contributions

1. **New Algorithms**
   - Implement additional ML algorithms (SVM, Naive Bayes, Neural Networks, etc.)
   - Ensure clear documentation and mathematical explanations

2. **Improvements**
   - Optimize existing implementations
   - Add features (e.g., regularization, cross-validation)
   - Improve code clarity and documentation

3. **Documentation**
   - Add tutorials and examples
   - Create Jupyter notebooks with visualizations
   - Improve mathematical explanations

4. **Testing**
   - Add more comprehensive tests
   - Test edge cases
   - Improve test coverage

5. **Bug Fixes**
   - Fix issues in existing code
   - Correct documentation errors

## 📋 Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add comprehensive docstrings (NumPy style)
- Keep functions focused and modular

Example:
```python
def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=None
        Controls the shuffling applied to the data before splitting.
        
    Returns
    -------
    X_train : ndarray
        Training data
    X_test : ndarray
        Test data
    y_train : ndarray
        Training labels
    y_test : ndarray
        Test labels
    """
    # Implementation here
```

### Documentation Standards

1. **Module-level docstrings**: Explain module purpose
2. **Class docstrings**: Describe class, parameters, attributes
3. **Method docstrings**: Detail parameters, returns, examples
4. **Inline comments**: Explain complex logic (sparingly)

### Testing Requirements

- All new code must have tests
- Tests should cover:
  - Basic functionality
  - Edge cases
  - Error handling
  - Comparison with scikit-learn (when applicable)

Example test:
```python
def test_linear_regression_fit():
    """Test basic fit functionality."""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    model = LinearRegression()
    model.fit(X, y)
    
    assert model.coef_[0] == pytest.approx(2.0, abs=0.01)
    assert model.intercept_ == pytest.approx(0.0, abs=0.01)
```

### Mathematical Documentation

For each algorithm, provide:

1. **Problem statement**: What problem does it solve?
2. **Mathematical formulation**: Equations and notation
3. **Algorithm steps**: Clear procedure
4. **Complexity analysis**: Time and space complexity
5. **Example**: Simple worked example

## 🔄 Contribution Process

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sklearn-under-the-hood.git
   cd sklearn-under-the-hood
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

4. **Run tests**
   ```bash
   pytest tests/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Describe your changes
   - Wait for review

## 📝 Pull Request Guidelines

### PR Title Format
- `Add: [Feature name]` - New features
- `Fix: [Issue description]` - Bug fixes
- `Docs: [Topic]` - Documentation updates
- `Test: [Area]` - Test additions/improvements
- `Refactor: [Component]` - Code refactoring

### PR Description Should Include
- What changes were made
- Why the changes were necessary
- How to test the changes
- Related issues (if any)

### Review Process
1. Automated tests must pass
2. Code review by maintainers
3. Address feedback if requested
4. Merge after approval

## 🐛 Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, package versions

Example:
```python
# Bug report
from sklearn_from_scratch.linear_models import LinearRegression

X = [[1], [2], [3]]
y = [1, 2, 3]

model = LinearRegression()
model.fit(X, y)  # Raises ValueError

# Expected: Model trains successfully
# Actual: ValueError: invalid dimensions
```

## 💡 Suggesting Features

Feature requests should include:

1. **Use case**: Why is this feature needed?
2. **Description**: What should it do?
3. **Examples**: How would it be used?
4. **Alternatives**: Have you considered other approaches?

## 📚 Algorithm Implementation Checklist

When implementing a new algorithm:

- [ ] Core implementation in appropriate module
- [ ] Comprehensive docstrings
- [ ] Unit tests (aim for >90% coverage)
- [ ] Comparison with scikit-learn
- [ ] Mathematical documentation
- [ ] Usage example
- [ ] Tutorial/notebook (optional but appreciated)

## 🎓 Resources for Contributors

### Understanding ML Algorithms
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pattern Recognition and Machine Learning](http://research.microsoft.com/~cmbishop/PRML/)
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)

### Python Best Practices
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)

### Testing
- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## ❓ Questions?

If you have questions:

1. Check existing issues
2. Read the documentation
3. Open a new issue with your question

## 🙏 Thank You!

Your contributions help make this project better for everyone learning ML!

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
