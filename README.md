# sklearn-under-the-hood

🔬 **Learn Machine Learning by Building It From Scratch**

The "sklearn under the hood" project explores how scikit-learn algorithms work by re-implementing them from scratch in Python. It analyzes the math, compares custom and official sklearn code, and offers insights so users can master ML theory and use scikit-learn more effectively for real-world projects.

## 🎯 Project Goals

- **Deep Understanding**: Learn how ML algorithms work internally
- **Mathematical Foundations**: Understand the math behind each algorithm
- **Practical Implementation**: Build working algorithms from scratch
- **Direct Comparisons**: Compare custom implementations with scikit-learn
- **Educational Resources**: Comprehensive tutorials and documentation

## 📚 Implemented Algorithms

### Supervised Learning

#### Linear Models
- **Linear Regression**: Ordinary least squares with normal equation
- **Logistic Regression**: Binary classification with gradient descent

#### Neighbors
- **K-Nearest Neighbors Classifier**: Distance-based classification
- **K-Nearest Neighbors Regressor**: Distance-based regression

#### Tree-Based Models
- **Decision Tree Classifier**: Recursive splitting with information gain
- **Random Forest Classifier**: Ensemble of decision trees with bagging

### Utilities

#### Metrics
- Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Regression: MSE, MAE, R² Score

#### Preprocessing
- **StandardScaler**: Standardization (mean=0, std=1)
- **MinMaxScaler**: Scale features to a given range
- **train_test_split**: Split data into train and test sets

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/HOYALIM/sklearn-under-the-hood.git
cd sklearn-under-the-hood

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from sklearn_from_scratch.linear_models import LinearRegression
from sklearn_from_scratch.preprocessing import train_test_split
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score}")
```

## 📖 Documentation

### Tutorials

- **[ML Algorithms Guide](tutorials/ML_ALGORITHMS_GUIDE.md)**: Comprehensive tutorial covering all algorithms with mathematical foundations and implementation details

### Examples

Run the comparison script to see custom implementations vs scikit-learn:

```bash
python examples/compare_with_sklearn.py
```

This will show side-by-side comparisons of:
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors
- Decision Trees

## 🧪 Testing

Run tests to verify implementations:

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_linear_models.py

# Run with verbose output
pytest -v tests/
```

## 📊 Algorithm Comparisons

Our implementations achieve comparable accuracy to scikit-learn:

| Algorithm | Custom Accuracy | Sklearn Accuracy | Speed Comparison |
|-----------|----------------|------------------|------------------|
| Linear Regression | R² ≈ 0.99 | R² ≈ 0.99 | Similar |
| Logistic Regression | ~85-90% | ~90-95% | Slower (pure Python) |
| KNN Classifier | ~85-90% | ~85-90% | Slower (no KD-tree) |
| Decision Tree | ~80-85% | ~80-85% | Slower (pure Python) |

*Note: Custom implementations prioritize clarity over performance*

## 🔍 Key Features

### Mathematical Foundations

Each algorithm includes:
- **Mathematical formulation**: Equations and notation
- **Cost functions**: How the algorithm optimizes
- **Gradient derivations**: Where applicable
- **Complexity analysis**: Time and space complexity

### Code Quality

- **Clean implementations**: Easy to read and understand
- **Comprehensive docstrings**: Every class and method documented
- **Type hints**: Clear parameter and return types
- **Tests**: Full test coverage for all implementations

### Educational Value

- **Step-by-step explanations**: Understand each component
- **Visual comparisons**: See custom vs sklearn results
- **Practical examples**: Real-world usage patterns
- **Best practices**: When to use each algorithm

## 🛠️ Project Structure

```
sklearn-under-the-hood/
├── sklearn_from_scratch/      # Main package
│   ├── linear_models/          # Linear & Logistic Regression
│   ├── neighbors/              # K-Nearest Neighbors
│   ├── tree/                   # Decision Trees
│   ├── ensemble/               # Random Forest
│   ├── metrics/                # Evaluation metrics
│   └── preprocessing/          # Data preprocessing
├── tests/                      # Unit tests
├── examples/                   # Example scripts
├── tutorials/                  # Educational materials
├── notebooks/                  # Jupyter notebooks
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## 🎓 Learning Path

1. **Start with Linear Regression**: Simplest algorithm, introduces key concepts
2. **Move to Logistic Regression**: Adds classification and gradient descent
3. **Try K-Nearest Neighbors**: Non-parametric, different paradigm
4. **Study Decision Trees**: Recursive algorithms and information theory
5. **Explore Random Forest**: Ensemble methods and variance reduction

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional algorithms (SVM, Naive Bayes, Neural Networks)
- Performance optimizations
- More comprehensive tutorials
- Jupyter notebooks with visualizations
- Additional metrics and preprocessing methods

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by scikit-learn's excellent API design
- Mathematical foundations from classic ML textbooks
- Community feedback and contributions

## 📧 Contact

For questions, suggestions, or discussions:
- Open an issue on GitHub
- Check existing tutorials and documentation

## 🔗 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Course - Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/)

---

**Happy Learning! 🎉**

*Understanding algorithms deeply makes you a better ML practitioner*
