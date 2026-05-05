import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ml_fundamentals.chapter1.src.DecisionTreeRegression import DecisionTreeRegressor

"""
Random Forest Regression From Scratch
"""

# =========================
# Random Forest Regressor
# =========================
class RandomForestRegression:
    """
    An implementation of a Random Forest Regressor built from scratch.
    
    A Random Forest is a powerful ensemble learning method that builds multiple decision trees 
    during training and averages their predictions to output the final result. By leveraging 
    multiple uncorrelated trees, it significantly reduces the variance and risk of overfitting 
    compared to a single decision tree.
    
    Mathematical Intuition:
    1. **Bootstrapping**: For each tree, a random subset of the training data is sampled with replacement.
       If the original dataset has size N, each tree is trained on a bootstrap sample also of size N.
    2. **Random Subspaces**: When splitting a node in any given tree, only a random subset of features 
       is considered (often `sqrt(d)` where `d` is the total number of features). This decorrelates the trees.
    3. **Aggregation**: Once all `M` trees are built, the final prediction for a new instance `x` 
       is calculated as the unweighted average of the predictions from all the individual trees:
       
       `y_hat = (1 / M) * \sum_{m=1}^{M} f_m(x)`
       
       where `f_m(x)` is the prediction of the m-th decision tree.
       
    References to learn more:
    - "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman.
    - Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
    """

    def __init__(self, config):
        self.config = config
        self.trees = []
        self.X = None
        self.y = None

    def load_data(self):
        self.X = pd.read_csv(self.config['datasets'][0]['path'])
        self.y = self.X.pop(self.config['datasets'][0]['target'])
        return self.X, self.y

    def preprocess_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X.values,
            self.y.values,
            test_size=self.config['splits']['train_test_split']['test_size'],
            random_state=self.config['splits']['train_test_split']['random_state']
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def train(self):
        params = self.config.get('models', {}).get('Random Forest Regression', {}).get('parameters')
        n_trees = params.get('n_estimators', 10)
        max_depth = params.get('max_depth', 5)
        max_features = params.get('max_features', int(np.sqrt(self.X_train.shape[1])))

        self.trees = []

        for _ in range(n_trees):
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                max_features=max_features
            )

            X_s, y_s = self._bootstrap_sample(self.X_train, self.y_train)
            tree.fit(X_s, y_s)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

    def evaluate(self, y_test, y_pred):
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

    def save_model(self):
        joblib.dump(self.trees, self.config['saved_models']['dir'] + '/rf_model.pkl')

    def load_model(self):
        self.trees = joblib.load(self.config['saved_models']['dir'] + '/rf_model.pkl')

    def generate_plots(self, y_test, y_pred):
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Random Forest Predictions")
        plt.savefig(self.config['plots']['dir'] + '/rf_plot.png')
        plt.close()
        
    def generate_report(self, metrics):
        """
        This method is used to generate the report
        
        Args:
            metrics: dictionary of metrics
        """
        with open(self.config['reports']['csv']['dir'] + '/' + self.config['reports']['csv']['name'] + '_' + 'random_forest_regression' + '.csv', 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    import json, os

    config = json.load(open('ml_fundamentals/chapter1/configs/regression.json'))

    os.makedirs(config['saved_models']['dir'], exist_ok=True)
    os.makedirs(config['plots']['dir'], exist_ok=True)

    rf = RandomForestRegression(config)

    rf.load_data()
    rf.preprocess_data()
    rf.train()
    y_pred = rf.predict(rf.X_test)
    metrics = rf.evaluate(rf.y_test, y_pred)
    rf.save_model()
    rf.generate_plots(rf.y_test, y_pred)
    rf.generate_report(metrics)