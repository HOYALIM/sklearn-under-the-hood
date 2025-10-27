"""
Metrics Module

Implementation of common machine learning metrics from scratch.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Accuracy classification score.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
        
    Returns
    -------
    score : float
        Fraction of correctly classified samples.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """
    Mean squared error regression loss.
    
    MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
        
    Returns
    -------
    loss : float
        Mean squared error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Mean absolute error regression loss.
    
    MAE = (1/n) * Σ|yᵢ - ŷᵢ|
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
        
    Returns
    -------
    loss : float
        Mean absolute error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    R² (coefficient of determination) regression score.
    
    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(yᵢ - ŷᵢ)² and SS_tot = Σ(yᵢ - ȳ)²
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
        
    Returns
    -------
    score : float
        R² score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix to evaluate classification accuracy.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
        
    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return matrix


def precision_score(y_true, y_pred, pos_label=1):
    """
    Compute precision for binary classification.
    
    Precision = TP / (TP + FP)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int, default=1
        The positive class label.
        
    Returns
    -------
    precision : float
        Precision score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def recall_score(y_true, y_pred, pos_label=1):
    """
    Compute recall (sensitivity) for binary classification.
    
    Recall = TP / (TP + FN)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int, default=1
        The positive class label.
        
    Returns
    -------
    recall : float
        Recall score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def f1_score(y_true, y_pred, pos_label=1):
    """
    Compute F1 score (harmonic mean of precision and recall).
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int, default=1
        The positive class label.
        
    Returns
    -------
    f1 : float
        F1 score.
    """
    precision = precision_score(y_true, y_pred, pos_label)
    recall = recall_score(y_true, y_pred, pos_label)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)
