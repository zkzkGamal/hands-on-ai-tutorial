import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class SupportVectorRegression:
    """
    An implementation of Support Vector Regression (SVR) from scratch, 
    utilizing a simplified Sequential Minimal Optimization (SMO) style training algorithm.
    
    SVR is a powerful algorithm that extends Support Vector Machines (SVM) to regression problems.
    Unlike standard linear regression that tries to minimize the error between predicted and actual values,
    SVR attempts to fit the error within a certain threshold (epsilon), while keeping the model as flat as possible.
    
    Mathematical Intuition:
    1. **Epsilon-Tube**: The model constructs an epsilon-insensitive tube around the predictions. 
       Errors within this tube are ignored, meaning no penalty is given for predictions that are "close enough."
    2. **Kernel Trick**: To capture non-linear relationships, the inputs can be mapped into high-dimensional 
       feature spaces using kernel functions (e.g., Linear, Polynomial, or RBF/Gaussian). This allows 
       the model to separate or fit data that isn't linearly separable in the original space.
    3. **SMO Optimization**: To find the optimal weights (alphas), we iteratively update individual 
       alpha coefficients (Lagrange multipliers) based on the prediction error. If the error falls outside 
       the epsilon boundary, the alphas are adjusted to reduce the error, bounded by a regularization parameter `C`.
       
    References to learn more:
    - "Pattern Recognition and Machine Learning" by Christopher M. Bishop.
    - Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/svm.html#svm-regression
    """

    def __init__(self, config):
        self.config = config
        self.X = None
        self.y = None

        self.alpha = None
        self.alpha_star = None
        self.b = 0

        self.X_train = None
        self.y_train = None

    def kernel(self, x1, x2):
        """
        Compute the kernel function between two data points.
        
        Args:
            x1: First data point.
            x2: Second data point.
            
        Returns:
            The result of the kernel function.
        """
        kernel_type = self.config['models']['Support Vector Regression']['parameters']['kernel']

        if kernel_type == 'linear':
            return np.dot(x1, x2)

        elif kernel_type == 'poly':
            degree = self.config['models']['Support Vector Regression']['parameters']['degree']
            return np.power(np.dot(x1, x2) + 1, degree)

        elif kernel_type == 'rbf':
            gamma = self.config['models']['Support Vector Regression']['parameters']['gamma']
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def load_data(self):
        """
        Load the dataset from the specified path in the configuration.
        """
        self.X = pd.read_csv(self.config['datasets'][0]['path'])
        self.y = self.X.pop(self.config['datasets'][0]['target'])
        return self.X, self.y

    def preprocess_data(self):
        """
        Preprocess the dataset by splitting it into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X.values,
            self.y.values,
            test_size=self.config['splits']['train_test_split']['test_size'],
            random_state=self.config['splits']['train_test_split']['random_state']
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        """
        Train the SVR model using a simplified SMO-style optimization algorithm.
        """
        params = self.config['models']['Support Vector Regression']['parameters']
        C = params['C']
        epsilon = params['epsilon']
        lr = params['learning_rate']
        epochs = params['epochs']

        n_samples = self.X_train.shape[0]

        self.alpha = np.zeros(n_samples)
        self.alpha_star = np.zeros(n_samples)

        # Precompute kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(self.X_train[i], self.X_train[j])

        for _ in range(epochs):
            for i in range(n_samples):
                prediction = self._predict_row(self.X_train[i])

                error = prediction - self.y_train[i]

                if error > epsilon:
                    self.alpha[i] -= lr
                    self.alpha_star[i] += lr
                elif error < -epsilon:
                    self.alpha[i] += lr
                    self.alpha_star[i] -= lr

                # Clip values
                self.alpha[i] = np.clip(self.alpha[i], 0, C)
                self.alpha_star[i] = np.clip(self.alpha_star[i], 0, C)

        # Compute bias
        self.b = np.mean([
            self.y_train[i] - self._predict_row(self.X_train[i])
            for i in range(n_samples)
        ])

    def _predict_row(self, x):
        """
        Predict the value for a single data point.
        
        Args:
            x: Data point to predict.
            
        Returns:
            Predicted value.
        """
        result = 0
        for i in range(len(self.X_train)):
            result += (self.alpha[i] - self.alpha_star[i]) * self.kernel(self.X_train[i], x)
        return result + self.b

    def predict(self, X):
        return np.array([self._predict_row(x) for x in X])

    def evaluate(self, y_test, y_pred):
        """
        Evaluate the model using mean squared error and R-squared.
        
        Args:
            y_test: True values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary of metrics.
        """
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

    def save_model(self):
        """
        Save the trained model to a file.
        """
        joblib.dump({
            'alpha': self.alpha,
            'alpha_star': self.alpha_star,
            'b': self.b,
            'X_train': self.X_train
        }, self.config['saved_models']['dir'] + '/svr_model.pkl')

    def load_model(self):
        """
        Load the trained model from a file.
        """
        data = joblib.load(self.config['saved_models']['dir'] + '/svr_model.pkl')
        self.alpha = data['alpha']
        self.alpha_star = data['alpha_star']
        self.b = data['b']
        self.X_train = data['X_train']

    def generate_plots(self, y_test, y_pred):
        """
        Generate plots to visualize the model performance.
        
        Args:
            y_test: True values.
            y_pred: Predicted values.
        """
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("SVR Predictions")
        plt.savefig(self.config['plots']['dir'] + '/svr_plot.png')
        plt.close()
    
    def generate_report(self, metrics):
        """
        This method is used to generate the report
        
        Args:
            metrics: dictionary of metrics
        """
        with open(self.config['reports']['csv']['dir'] + '/' + self.config['reports']['csv']['name'] + "_SVR" + '.csv', 'w') as f:
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

    svr = SupportVectorRegression(config)

    svr.load_data()
    svr.preprocess_data()
    svr.train()

    y_pred = svr.predict(svr.X_test)

    metrics = svr.evaluate(svr.y_test, y_pred)
    svr.generate_report(metrics)

    svr.save_model()
    svr.generate_plots(svr.y_test, y_pred)