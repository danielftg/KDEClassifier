import math
import numpy as np
from typing import Any


class KDEClassifier:
    """
    Kernel Density Estimation (KDE) based classifier.

    This class implements a simple KDE-based classifier with a scikit-learn-like API
    (fit, predict_proba, predict). It also implements minimal sklearn-compatible helpers:
    get_params, set_params, _estimator_type, and score so it can be used with sklearn
    utilities (clone, GridSearchCV, etc.) without requiring scikit-learn as a runtime
    dependency.

    Parameters
    ----------
    mybandwidth : float, default=1.0
        Bandwidth parameter for the kernel. Stored on the instance as `bandwidth`.
        get_params exposes this name for sklearn compatibility.
    """

    _estimator_type = "classifier"

    def __init__(self, mybandwidth: float = 1.0):
        """
        Initialize the KDEClassifier.

        The constructor parameter name `mybandwidth` is preserved for compatibility.
        """
        self.bandwidth = mybandwidth
        self.trainsamples = None
        self.trainlabels = None
        self.trainlabels_set = None

    def fit(self, trainsamples, trainlabels):
        """
        Fit the KDE classifier.

        Parameters
        ----------
        trainsamples : array-like, shape (n_samples, n_features)
            Training data.
        trainlabels : array-like, shape (n_samples,)
            Training labels.

        Returns
        -------
        self : KDEClassifier
            Returns self for chaining.
        """
        X = np.asarray(trainsamples)
        y = np.asarray(trainlabels)
        
        if X.ndim == 1:
            # convert 1-D feature vectors to 2-D shape (n_samples, 1)
            X = X.reshape(-1, 1)
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("fit() received empty training data; X and y must have at least one sample")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of training samples and training labels must match")
        
        self.trainsamples = X
        self.trainlabels = y
        labels = sorted(set(self.trainlabels))
        self.trainlabels_set = list(labels)
        return self

    def set_bandwidth_default(self, trainsamples):
        """
        Compute and set a default bandwidth from data using a Silverman-like rule.

        Parameters
        ----------
        trainsamples : array-like, shape (n_samples, n_features)
            Training data used to compute a default bandwidth.
        """
        arr = np.asarray(trainsamples)
        std_list = arr.std(axis=0)
        bandwidth_list = 1.059 * std_list * math.pow(arr.shape[0], -0.2)
        bandwidth_default = np.mean(bandwidth_list)
        self.bandwidth = float(bandwidth_default)

    def predict_proba(self, testsamples):
        """
        Return probability estimates for the test data.

        Parameters
        ----------
        testsamples : array-like, shape (n_query, n_features)
            Query points to estimate class probabilities for.

        Returns
        -------
        predict_probs : ndarray, shape (n_query, n_classes)
            The class probabilities of the input samples. Classes are ordered by
            lexicographic order (the same order as self.trainlabels_set).
        """
        if self.trainsamples is None or self.trainlabels is None:
            raise ValueError("Model not fitted. Call fit(...) before predict_proba.")

        if self.bandwidth == 0:
            raise ValueError("bandwidth must be non-zero")

        Xq = np.asarray(testsamples)
        if Xq.ndim == 1:
            Xq = Xq.reshape(-1, self.trainsamples.shape[1])

        labels_sorted = self.trainlabels_set
        n_query = Xq.shape[0]
        n_classes = len(labels_sorted)
        
        if n_classes == 0:
            raise ValueError("No classes found in training labels")
        
        n_query = Xq.shape[0]
        if n_query == 0:
            # return consistent empty 2-D array: (0, n_classes)
            return np.empty((0, n_classes), dtype=float)


        probs = np.zeros((n_query, n_classes), dtype=float)

        for j, label in enumerate(labels_sorted):
            mask = (self.trainlabels == label)
            Xs = self.trainsamples[mask]
            if Xs.size == 0:
                probs[:, j] = 0.0
                continue

            diffs = Xq[:, None, :] - Xs[None, :, :]
            sqd = np.sum(diffs ** 2, axis=2)
            denom = math.sqrt(2 * math.pi) * self.bandwidth
            contributions = np.exp(-sqd / (2 * self.bandwidth ** 2)) / denom
            probability_expect = contributions.mean(axis=1)
            probs[:, j] = probability_expect

        row_sums = probs.sum(axis=1, keepdims=True)
        zero_sum_mask = (row_sums[:, 0] == 0)
        if np.any(~zero_sum_mask):
            probs[~zero_sum_mask] = probs[~zero_sum_mask] / row_sums[~zero_sum_mask]
        if np.any(zero_sum_mask):
            probs[zero_sum_mask] = 1.0 / float(n_classes)

        return probs

    def predict(self, testsamples):
        """
        Predict class labels for the provided data.

        Parameters
        ----------
        testsamples : array-like, shape (n_query, n_features)

        Returns
        -------
        predict_labels : ndarray, shape (n_query,)
            Predicted labels, ordered corresponding to testsamples.
        """
        predict_probs = self.predict_proba(testsamples)
        idx = np.argmax(predict_probs, axis=1)
        predict_labels = np.array([self.trainlabels_set[i] for i in idx])
        return predict_labels

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Mapping of parameter names to their values.
        """
        return {"mybandwidth": self.bandwidth}

    def set_params(self, **params: Any):
        """
        Set parameters for this estimator.

        Parameters
        ----------
        **params : dict
            Parameter names and values.

        Returns
        -------
        self : KDEClassifier
        """
        for key, value in params.items():
            if key == "mybandwidth" or key == "bandwidth":
                self.bandwidth = value
            else:
                setattr(self, key, value)
        return self

    def score(self, X, y):
        """
        Simple accuracy score.

        Parameters
        ----------
        X : array-like
            Input samples.
        y : array-like
            True labels.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) compared to y.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        return float(np.mean(y_pred == y))

    def __repr__(self):
        return f"{self.__class__.__name__}(bandwidth={self.bandwidth})"
