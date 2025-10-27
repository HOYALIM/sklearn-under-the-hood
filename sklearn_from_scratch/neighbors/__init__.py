"""
K-Nearest Neighbors Module

Implementation of K-Nearest Neighbors algorithm from scratch.
"""

import numpy as np
from collections import Counter


class KNeighborsClassifier:
    """
    K-Nearest Neighbors classifier implementation from scratch.
    
    The algorithm works by:
    1. Computing distances between test sample and all training samples
    2. Finding K nearest neighbors
    3. Voting on the most common class among neighbors
    
    Mathematical formulation:
    - Euclidean distance: d(x, x') = √(Σ(xᵢ - x'ᵢ)²)
    - Manhattan distance: d(x, x') = Σ|xᵢ - x'ᵢ|
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    metric : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric to use.
        
    Attributes
    ----------
    X_train_ : ndarray of shape (n_samples, n_features)
        Training data.
    y_train_ : ndarray of shape (n_samples,)
        Training labels.
    """
    
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train_ = None
        self.y_train_ = None
        
    def _compute_distance(self, x1, x2):
        """Compute distance between two vectors."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X, y):
        """
        Fit the k-nearest neighbors classifier.
        
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
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        return self
    
    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for each data sample
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Compute distances to all training samples
            distances = self._compute_distance(self.X_train_, x)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train_[k_indices]
            
            # Vote for most common class
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
            
        return np.array(predictions)
    
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


class KNeighborsRegressor:
    """
    K-Nearest Neighbors regressor implementation from scratch.
    
    The algorithm predicts the target value by averaging the values
    of K nearest neighbors.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    metric : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric to use.
        
    Attributes
    ----------
    X_train_ : ndarray of shape (n_samples, n_features)
        Training data.
    y_train_ : ndarray of shape (n_samples,)
        Training target values.
    """
    
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train_ = None
        self.y_train_ = None
        
    def _compute_distance(self, x1, x2):
        """Compute distance between two vectors."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X, y):
        """
        Fit the k-nearest neighbors regressor.
        
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
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        return self
    
    def predict(self, X):
        """
        Predict the target for the provided data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values for each data sample
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Compute distances to all training samples
            distances = self._compute_distance(self.X_train_, x)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            
            # Get values of k nearest neighbors
            k_nearest_values = self.y_train_[k_indices]
            
            # Average the values
            predictions.append(np.mean(k_nearest_values))
            
        return np.array(predictions)
    
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
