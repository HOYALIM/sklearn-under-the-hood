"""
Ensemble Module

Implementation of ensemble learning methods from scratch.
"""

import numpy as np
from collections import Counter
from ..tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest classifier implementation from scratch.
    
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.
    max_depth : int, default=None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    max_features : int or None, default=None
        Number of features to consider when looking for the best split.
        If None, uses sqrt(n_features).
    random_state : int, default=None
        Controls randomness of the bootstrapping and feature sampling.
        
    Attributes
    ----------
    trees_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    """
    
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2,
                 max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        
    def _bootstrap_sample(self, X, y, rng):
        """Create a bootstrap sample of the dataset."""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """
        Build a forest of trees from the training set.
        
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
        
        rng = np.random.RandomState(self.random_state)
        self.trees_ = []
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y, rng)
            
            # Train decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
            
        return self
    
    def predict(self, X):
        """
        Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted classes
        """
        X = np.array(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees_])
        
        # Majority voting
        predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
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
