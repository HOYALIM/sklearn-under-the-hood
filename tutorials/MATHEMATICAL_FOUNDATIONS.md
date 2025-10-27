# Mathematical Foundations

## Overview

This document provides detailed mathematical explanations for each algorithm implemented in this project. Understanding these foundations is crucial for mastering machine learning.

## Table of Contents

1. [Linear Regression](#linear-regression)
2. [Logistic Regression](#logistic-regression)
3. [K-Nearest Neighbors](#k-nearest-neighbors)
4. [Decision Trees](#decision-trees)
5. [Evaluation Metrics](#evaluation-metrics)

---

## Linear Regression

### Problem Statement

Given training data $(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})$, find the best linear function that maps inputs to outputs.

### Hypothesis Function

The hypothesis function for linear regression is:

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n = \theta^T x$$

Where:
- $\theta$ = parameter vector (weights)
- $x$ = feature vector
- $n$ = number of features

### Cost Function

Mean Squared Error (MSE):

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $m$ = number of training examples
- The factor $\frac{1}{2}$ simplifies derivative computation

### Normal Equation

The closed-form solution (no iteration needed):

$$\theta = (X^T X)^{-1} X^T y$$

Where:
- $X$ = design matrix (m × n)
- $y$ = target vector

**Derivation:**
1. Start with cost function: $J(\theta) = \frac{1}{2}(X\theta - y)^T(X\theta - y)$
2. Take gradient: $\nabla_\theta J = X^T X \theta - X^T y$
3. Set to zero: $X^T X \theta = X^T y$
4. Solve: $\theta = (X^T X)^{-1} X^T y$

### Evaluation: R² Score

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

Where:
- $SS_{res}$ = residual sum of squares
- $SS_{tot}$ = total sum of squares
- $\bar{y}$ = mean of true values

**Interpretation:**
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: Model performs as well as predicting the mean
- $R^2 < 0$: Model performs worse than predicting the mean

---

## Logistic Regression

### Problem Statement

Given training data with binary labels $(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})$ where $y \in \{0, 1\}$, learn a function that predicts the probability of class 1.

### Sigmoid Function

The logistic (sigmoid) function maps any real number to [0, 1]:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- $\sigma(0) = 0.5$
- $\lim_{z \to \infty} \sigma(z) = 1$
- $\lim_{z \to -\infty} \sigma(z) = 0$
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### Hypothesis Function

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

This gives the probability that $y = 1$ given $x$.

### Cost Function (Log Loss)

For binary classification:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$$

**Why this function?**
- Derived from maximum likelihood estimation
- Convex (has single global minimum)
- Penalizes wrong predictions heavily

### Gradient Descent

Update rule:

$$\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$$

Where the gradient is:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

**Algorithm:**
1. Initialize $\theta = 0$
2. Repeat until convergence:
   - Compute $h_\theta(x)$ for all training examples
   - Update all $\theta_j$ simultaneously
3. Return $\theta$

---

## K-Nearest Neighbors

### Algorithm Description

KNN is a non-parametric method:
1. Store all training data
2. For each test point, find K nearest neighbors
3. Classification: majority vote
4. Regression: average of neighbor values

### Distance Metrics

**Euclidean Distance:**

$$d(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}$$

**Manhattan Distance:**

$$d(x, x') = \sum_{i=1}^{n} |x_i - x'_i|$$

**Minkowski Distance (generalization):**

$$d(x, x') = \left(\sum_{i=1}^{n} |x_i - x'_i|^p\right)^{1/p}$$

Where $p=2$ gives Euclidean, $p=1$ gives Manhattan.

### Classification Rule

$$\hat{y} = \text{mode}(\{y^{(i_1)}, y^{(i_2)}, ..., y^{(i_K)}\})$$

Where $i_1, ..., i_K$ are indices of K nearest neighbors.

### Choosing K

- **Small K** (e.g., K=1):
  - More flexible, captures local patterns
  - High variance, sensitive to noise
  - Risk of overfitting

- **Large K** (e.g., K=20):
  - Smoother decision boundaries
  - High bias, may underfit
  - More robust to noise

**Rule of thumb:** $K = \sqrt{m}$ where m is training set size.

---

## Decision Trees

### Information Theory Background

**Entropy** measures impurity/disorder in a set:

$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Where:
- $c$ = number of classes
- $p_i$ = proportion of samples in class $i$

**Properties:**
- $H(S) = 0$ when all samples are same class (pure)
- $H(S)$ is maximum when classes are evenly distributed
- For binary: $H_{max} = 1$ bit

**Gini Impurity** (alternative to entropy):

$$G(S) = 1 - \sum_{i=1}^{c} p_i^2$$

**Properties:**
- $G(S) = 0$ when pure
- Computationally cheaper than entropy
- Maximum at uniform distribution

### Information Gain

Measures the reduction in entropy from a split:

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $A$ = attribute to split on
- $S_v$ = subset of $S$ where attribute $A$ has value $v$

**Algorithm:**
1. Calculate entropy of current node
2. For each feature and threshold:
   - Split data into left and right
   - Calculate weighted entropy of children
   - Compute information gain
3. Choose split with highest information gain
4. Recurse on child nodes

### Stopping Criteria

- **Max depth reached**: Prevent tree from growing too deep
- **Min samples to split**: Need enough data to split
- **Pure node**: All samples have same label
- **No improvement**: Information gain is zero

### Overfitting Prevention

- **Pruning**: Remove branches that don't help
- **Max depth**: Limit tree depth
- **Min samples**: Require minimum samples per split/leaf
- **Random Forest**: Use ensemble of trees

---

## Evaluation Metrics

### Classification Metrics

**Confusion Matrix:**

|              | Predicted Positive | Predicted Negative |
|--------------|-------------------|--------------------|
| **Actually Positive** | TP (True Positive)  | FN (False Negative) |
| **Actually Negative** | FP (False Positive) | TN (True Negative)  |

**Accuracy:**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision** (How many predicted positives are correct?):

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall/Sensitivity** (How many actual positives are found?):

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score** (Harmonic mean of precision and recall):

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

### Regression Metrics

**Mean Squared Error (MSE):**

$$MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE):**

$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2}$$

**Mean Absolute Error (MAE):**

$$MAE = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$$

**R² Score** (see Linear Regression section)

---

## Gradient Descent

### Basic Idea

Iteratively move in the direction of steepest descent to minimize cost function.

### Update Rule

$$\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$$

Where:
- $\alpha$ = learning rate (step size)
- $\frac{\partial J}{\partial \theta_j}$ = partial derivative (gradient)

### Learning Rate Selection

- **Too small**: Slow convergence
- **Too large**: May overshoot minimum, diverge
- **Adaptive**: Start large, decrease over time

### Variants

**Batch Gradient Descent:**
- Use all training data for each update
- Stable but slow for large datasets

**Stochastic Gradient Descent (SGD):**
- Use one sample for each update
- Faster but noisy

**Mini-batch Gradient Descent:**
- Use small batch (e.g., 32 samples)
- Balance between speed and stability

---

## Bias-Variance Tradeoff

### Decomposition of Expected Error

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Bias:**
- Error from wrong assumptions
- High bias → underfitting
- Examples: linear model for non-linear data

**Variance:**
- Error from sensitivity to training data
- High variance → overfitting
- Examples: very deep decision tree

**Irreducible Error:**
- Noise in data
- Cannot be reduced by any model

### Managing the Tradeoff

- **Reduce Bias**: More complex models, more features
- **Reduce Variance**: Regularization, ensemble methods, more data
- **Sweet Spot**: Balance both for optimal generalization

---

## References

1. **Pattern Recognition and Machine Learning** - Christopher Bishop
2. **The Elements of Statistical Learning** - Hastie, Tibshirani, Friedman
3. **Introduction to Statistical Learning** - James, Witten, Hastie, Tibshirani
4. **Machine Learning** - Tom Mitchell
5. **Deep Learning** - Goodfellow, Bengio, Courville

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $m$ | Number of training examples |
| $n$ | Number of features |
| $x^{(i)}$ | i-th training example (vector) |
| $x_j^{(i)}$ | j-th feature of i-th example |
| $y^{(i)}$ | i-th training label |
| $\theta$ | Parameter vector |
| $\alpha$ | Learning rate |
| $h_\theta(x)$ | Hypothesis function |
| $J(\theta)$ | Cost function |
| $\sigma(z)$ | Sigmoid function |

