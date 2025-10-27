"""
Preprocessing Module

Implementation of common data preprocessing techniques from scratch.
"""

import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Mathematical formulation:
    z = (x - μ) / σ
    where μ is the mean and σ is the standard deviation.
    
    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    scale_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set.
    """
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : object
            Fitted scaler
        """
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be scaled
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Scaled data
        """
        X = np.array(X)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Scaled data
        """
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.
    
    Mathematical formulation:
    X_scaled = (X - X_min) / (X_max - X_min)
    
    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data.
    max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data.
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None
        
    def fit(self, X):
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : object
            Fitted scaler
        """
        X = np.array(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        """
        Scale features of X according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be scaled
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Scaled data
        """
        X = np.array(X)
        data_range = self.data_max_ - self.data_min_
        # Avoid division by zero
        data_range[data_range == 0] = 1.0
        
        X_scaled = (X - self.data_min_) / data_range
        
        # Scale to feature_range
        min_val, max_val = self.feature_range
        X_scaled = X_scaled * (max_val - min_val) + min_val
        
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Scaled data
        """
        return self.fit(X).transform(X)


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
    X = np.array(X)
    y = np.array(y)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
