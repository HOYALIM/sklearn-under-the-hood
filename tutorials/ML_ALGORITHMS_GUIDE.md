# Machine Learning Algorithms Tutorial

## Introduction

This tutorial provides a comprehensive guide to understanding machine learning algorithms by implementing them from scratch. Each algorithm includes mathematical foundations, implementation details, and practical examples.

## Table of Contents

1. [Linear Regression](#linear-regression)
2. [Logistic Regression](#logistic-regression)
3. [K-Nearest Neighbors](#k-nearest-neighbors)
4. [Decision Trees](#decision-trees)
5. [Random Forest](#random-forest)

---

## Linear Regression

### Mathematical Foundation

Linear regression models the relationship between input features and a continuous target variable using a linear function.

**Hypothesis Function:**
```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
```

**Cost Function (Mean Squared Error):**
```
J(θ) = (1/2m) * Σᵢ₌₁ᵐ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

**Normal Equation (Closed-form Solution):**
```
θ = (XᵀX)⁻¹Xᵀy
```

### Implementation Details

Our implementation uses the Normal Equation for optimal solution:

1. **Add intercept**: Augment X with a column of ones
2. **Compute XᵀX**: Matrix multiplication
3. **Solve**: Use least squares to find θ
4. **Extract coefficients**: Separate intercept from other coefficients

### Example Usage

```python
from sklearn_from_scratch.linear_models import LinearRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"R² Score: {model.score(X, y)}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

### Key Insights

- **Pros**: Fast, interpretable, works well with linear relationships
- **Cons**: Assumes linear relationship, sensitive to outliers
- **Use cases**: Price prediction, trend analysis, forecasting

---

## Logistic Regression

### Mathematical Foundation

Logistic regression is used for binary classification by modeling the probability of a sample belonging to a class.

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Hypothesis Function:**
```
h(x) = σ(θᵀx)
```

**Cost Function (Log Loss):**
```
J(θ) = -(1/m) * Σᵢ₌₁ᵐ [y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]
```

**Gradient:**
```
∂J/∂θⱼ = (1/m) * Σᵢ₌₁ᵐ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
```

### Implementation Details

Our implementation uses gradient descent:

1. **Initialize**: Start with θ = 0
2. **Forward pass**: Compute predictions using sigmoid
3. **Compute gradient**: Calculate ∂J/∂θ
4. **Update**: θ := θ - α * gradient
5. **Repeat**: Until convergence or max iterations

### Example Usage

```python
from sklearn_from_scratch.linear_models import LogisticRegression
import numpy as np

# Binary classification data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train model
model = LogisticRegression(learning_rate=0.1, max_iter=1000)
model.fit(X, y)

# Predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)
print(f"Accuracy: {model.score(X, y)}")
```

### Key Insights

- **Pros**: Probabilistic output, works well for binary classification
- **Cons**: Assumes linear decision boundary, requires feature scaling
- **Use cases**: Spam detection, medical diagnosis, credit scoring

---

## K-Nearest Neighbors

### Mathematical Foundation

KNN is a non-parametric algorithm that classifies samples based on the majority vote of their k nearest neighbors.

**Euclidean Distance:**
```
d(x, x') = √(Σᵢ₌₁ⁿ (xᵢ - x'ᵢ)²)
```

**Manhattan Distance:**
```
d(x, x') = Σᵢ₌₁ⁿ |xᵢ - x'ᵢ|
```

**Classification Rule:**
```
ŷ = mode(y₁, y₂, ..., yₖ)  where y₁, ..., yₖ are labels of k nearest neighbors
```

### Implementation Details

Our implementation:

1. **Store training data**: Keep all training samples
2. **Compute distances**: Calculate distance to all training samples
3. **Find k nearest**: Sort and select k closest neighbors
4. **Vote**: Return most common class (or average for regression)

### Example Usage

```python
from sklearn_from_scratch.neighbors import KNeighborsClassifier
import numpy as np

# Create data with two clusters
X = np.array([[1, 1], [2, 2], [10, 10], [11, 11]])
y = np.array([0, 0, 1, 1])

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Predict new samples
X_new = np.array([[1.5, 1.5], [10.5, 10.5]])
predictions = model.predict(X_new)
print(f"Predictions: {predictions}")
```

### Key Insights

- **Pros**: Simple, no training phase, works with non-linear boundaries
- **Cons**: Slow prediction, memory intensive, sensitive to scale
- **Use cases**: Recommendation systems, anomaly detection, pattern recognition

---

## Decision Trees

### Mathematical Foundation

Decision trees recursively split data based on features to maximize information gain.

**Entropy (Information Theory):**
```
H(S) = -Σᵢ₌₁ᶜ pᵢ * log₂(pᵢ)
```

**Gini Impurity:**
```
G(S) = 1 - Σᵢ₌₁ᶜ pᵢ²
```

**Information Gain:**
```
IG = H(parent) - Σ (|Sⱼ|/|S|) * H(Sⱼ)
```

### Implementation Details

Our recursive implementation:

1. **Base case**: Stop if max depth, pure node, or min samples
2. **Find best split**: Test all features and thresholds
3. **Split data**: Partition into left and right subsets
4. **Recurse**: Build left and right subtrees
5. **Predict**: Traverse tree until reaching a leaf

### Example Usage

```python
from sklearn_from_scratch.tree import DecisionTreeClassifier
import numpy as np

# Classification data
X = np.array([[1, 1], [2, 2], [10, 10], [11, 11]])
y = np.array([0, 0, 1, 1])

# Train model
model = DecisionTreeClassifier(max_depth=5, criterion='gini')
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print(f"Accuracy: {model.score(X, y)}")
```

### Key Insights

- **Pros**: Interpretable, handles non-linear relationships, no scaling needed
- **Cons**: Prone to overfitting, unstable (small changes → different tree)
- **Use cases**: Credit approval, medical diagnosis, customer segmentation

---

## Random Forest

### Mathematical Foundation

Random Forest combines multiple decision trees using bootstrap aggregating (bagging) and random feature selection.

**Bootstrap Sampling:**
- Create m datasets by sampling with replacement
- Each dataset has same size as original

**Random Feature Selection:**
- At each split, consider only a random subset of features
- Reduces correlation between trees

**Majority Voting:**
```
ŷ = mode(tree₁(x), tree₂(x), ..., treeₘ(x))
```

### Implementation Details

1. **Bootstrap**: Create random samples with replacement
2. **Build trees**: Train decision tree on each bootstrap sample
3. **Aggregate**: Combine predictions via majority vote (classification) or averaging (regression)

### Example Usage

```python
from sklearn_from_scratch.ensemble import RandomForestClassifier
import numpy as np

# Classification data
X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train model
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print(f"Accuracy: {model.score(X, y)}")
```

### Key Insights

- **Pros**: Reduces overfitting, robust, high accuracy
- **Cons**: Less interpretable, slower than single tree
- **Use cases**: Complex classification, feature importance, ensemble learning

---

## Best Practices

### Data Preprocessing

1. **Scaling**: Use StandardScaler or MinMaxScaler for distance-based algorithms
2. **Splitting**: Use train_test_split to evaluate generalization
3. **Missing values**: Handle before training

### Model Selection

- **Linear problems**: Linear/Logistic Regression
- **Non-linear, small data**: KNN or Decision Trees
- **Complex patterns**: Random Forest
- **Interpretability needed**: Linear models or Decision Trees

### Evaluation

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Regression**: R², MSE, MAE
- **Always use held-out test set**

---

## Mathematical Concepts

### Gradient Descent

Optimization algorithm for finding minimum of cost function:

```
θⱼ := θⱼ - α * ∂J/∂θⱼ
```

where α is learning rate.

### Bias-Variance Tradeoff

- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Goal**: Balance both for optimal generalization

### Regularization

Prevent overfitting by penalizing large coefficients:

- **L1 (Lasso)**: Σ|θᵢ| (feature selection)
- **L2 (Ridge)**: Σθᵢ² (coefficient shrinkage)

---

## Further Reading

1. "Pattern Recognition and Machine Learning" by Christopher Bishop
2. "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
3. Scikit-learn documentation: https://scikit-learn.org/
4. "Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani

---

## Exercises

1. Implement gradient descent for Linear Regression
2. Add regularization to Logistic Regression
3. Implement weighted KNN
4. Add pruning to Decision Trees
5. Implement feature importance for Random Forest
