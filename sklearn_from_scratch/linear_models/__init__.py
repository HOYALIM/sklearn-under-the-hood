"""
Linear Models Module

Implementations of linear regression and logistic regression from scratch.
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression implementation from scratch using ordinary least squares.
    
    Mathematical formulation:
    - Hypothesis: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
    - Cost function: J(θ) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    - Solution: θ = (XᵀX)⁻¹Xᵀy (Normal Equation)
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    """
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        
    def fit(self, X, y):
        """
        Fit linear model using the Normal Equation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        # Add intercept column if needed
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        # Normal equation: θ = (XᵀX)⁻¹Xᵀy
        theta = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]
        
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.coef_ = theta
            
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values
        """
        X = np.array(X)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """
        Return the coefficient of determination R² of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values for X
            
        Returns
        -------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LogisticRegression:
    """
    Logistic Regression implementation from scratch using gradient descent.
    
    Mathematical formulation:
    - Hypothesis: h(x) = σ(θᵀx) where σ(z) = 1/(1 + e⁻ᶻ)
    - Cost function: J(θ) = -(1/m) * Σ[y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]
    - Gradient: ∂J/∂θⱼ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the logistic regression problem.
    intercept_ : float
        Independent term in the logistic model.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """
        Fit logistic regression model using gradient descent.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1)
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        
        # Initialize parameters
        if self.fit_intercept:
            X = np.c_[np.ones(m), X]
            theta = np.zeros(n + 1)
        else:
            theta = np.zeros(n)
        
        # Gradient descent
        for _ in range(self.max_iter):
            h = self._sigmoid(X @ theta)
            gradient = (1 / m) * X.T @ (h - y)
            theta -= self.learning_rate * gradient
        
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.coef_ = theta
            
        return self
    
    def predict_proba(self, X):
        """
        Probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        prob : ndarray of shape (n_samples,)
            Returns the probability of the sample for class 1
        """
        X = np.array(X)
        z = X @ self.coef_ + self.intercept_
        return self._sigmoid(z)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels for X
            
        Returns
        -------
        score : float
            Mean accuracy
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
