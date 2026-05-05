import numpy as np

"""
This class is used to create the decision tree regressor model
"""

class DecisionTreeRegressor:
    """
    An implementation of a Decision Tree Regressor built from scratch.
    
    A Decision Tree is a non-parametric supervised learning method used for regression.
    The goal is to create a model that predicts the value of a target variable by learning 
    simple decision rules inferred from the data features.
    
    Mathematical Intuition:
    1. **Splitting Criterion**: The tree uses Variance Reduction (Mean Squared Error) 
       to evaluate the quality of a split. For a given node with data D, it finds the feature 
       and threshold that minimizes the weighted sum of variances in the left and right child nodes:
       
       `MSE = (N_left * Var(y_left) + N_right * Var(y_right)) / N`
       
    2. **Recursive Partitioning**: The dataset is recursively split into subsets until 
       a stopping criterion is met (e.g., maximum depth or minimum samples required to split).
    3. **Leaf Prediction**: The prediction for any instance reaching a leaf node is the 
       mean value of the target variables in that leaf: `y_pred = mean(y_leaf)`.
       
    References to learn more:
    - "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman.
    - Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/tree.html#regression
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        """
        This method is used to initialize the decision tree regressor model
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        """
        This method is used to fit the decision tree regressor model
        """
        self.n_features = X.shape[1]
        self.features = np.arange(self.n_features)
        self.tree = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        """
        This method is used to build the decision tree regressor model
        """
        n_samples, n_features = X.shape
        
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split):
            return np.mean(y)

        feat_idxs = np.random.choice(self.features,
                                     self.max_features or n_features,
                                     replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            return np.mean(y)

        left_idxs = X[:, best_feature] <= best_thresh
        right_idxs = X[:, best_feature] > best_thresh

        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return (best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """
        This method is used to find the best split for the decision tree regressor model
        """
        best_mse = float("inf")
        split_idx, split_thresh = None, None

        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left = y[X[:, feat] <= t]
                right = y[X[:, feat] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                mse = (len(left) * np.var(left) + len(right) * np.var(right)) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    split_idx = feat
                    split_thresh = t

        return split_idx, split_thresh

    def predict(self, X):
        """
        This method is used to predict the values for the decision tree regressor model
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        """
        This method is used to traverse the decision tree regressor model
        """
        if not isinstance(node, tuple):
            return node

        feature, thresh, left, right = node

        if x[feature] <= thresh:
            return self._traverse_tree(x, left)
        return self._traverse_tree(x, right)