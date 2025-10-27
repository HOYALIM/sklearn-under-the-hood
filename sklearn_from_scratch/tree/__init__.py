"""
Decision Tree Module

Implementation of Decision Tree classifier from scratch.
"""

import numpy as np
from collections import Counter


class Node:
    """
    A node in the decision tree.
    
    Attributes
    ----------
    feature : int or None
        Feature index to split on (None for leaf nodes)
    threshold : float or None
        Threshold value for the split (None for leaf nodes)
    left : Node or None
        Left child node
    right : Node or None
        Right child node
    value : int or None
        Class value for leaf nodes (None for internal nodes)
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        """Check if node is a leaf."""
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree classifier implementation from scratch.
    
    The algorithm uses recursive binary splitting based on information gain
    from entropy or Gini impurity.
    
    Mathematical formulation:
    - Entropy: H(S) = -Σ(pᵢ * log₂(pᵢ))
    - Gini impurity: G(S) = 1 - Σ(pᵢ²)
    - Information Gain: IG = H(parent) - Σ(|Sᵢ|/|S|) * H(Sᵢ)
    
    Parameters
    ----------
    max_depth : int, default=None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
        
    Attributes
    ----------
    root_ : Node
        The root node of the fitted tree.
    n_classes_ : int
        Number of classes.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root_ = None
        self.n_classes_ = None
        
    def _entropy(self, y):
        """Calculate entropy of a set of labels."""
        proportions = np.bincount(y) / len(y)
        # Avoid log(0)
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))
    
    def _gini(self, y):
        """Calculate Gini impurity of a set of labels."""
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _impurity(self, y):
        """Calculate impurity based on criterion."""
        if self.criterion == 'entropy':
            return self._entropy(y)
        else:  # gini
            return self._gini(y)
    
    def _information_gain(self, X, y, feature, threshold):
        """Calculate information gain for a split."""
        # Parent impurity
        parent_impurity = self._impurity(y)
        
        # Split the data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Calculate weighted average of children impurities
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        impurity_left = self._impurity(y[left_mask])
        impurity_right = self._impurity(y[right_mask])
        
        child_impurity = (n_left / n) * impurity_left + (n_right / n) * impurity_right
        
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """Find the best split for a node."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth if self.max_depth else False) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set.
        
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
        self.n_classes_ = len(np.unique(y))
        self.root_ = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)
    
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
        return np.array([self._predict_sample(x, self.root_) for x in X])
    
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
