import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import matplotlib.pyplot as plt

"""
Linear Regression Module

This module implements a custom linear regression model based on the normal equation.
It provides functionalities for data loading, preprocessing, model training,
prediction, evaluation, model persistence, and generating reports/plots.
"""

class LinearRegression:
    """
    An implementation of a Multiple Linear Regression model built from scratch 
    using the Normal Equation.
    
    Linear Regression assumes a linear relationship between the input features (X) 
    and the continuous target variable (y). 
    
    Mathematical Intuition:
    1. **Hypothesis**: The model predicts the target using a linear combination of the features:
       `y_pred = X * theta`
       where `X` is the feature matrix and `theta` represents the learned weights (coefficients).
       
    2. **Cost Function**: The goal is to minimize the Mean Squared Error (MSE) between 
       the predicted values and the actual target values.
       
    3. **Normal Equation (Closed-form Solution)**: Instead of using an iterative optimization 
       algorithm like Gradient Descent, this implementation finds the optimal weights directly 
       by taking the derivative of the cost function with respect to `theta` and setting it to zero:
       
       `theta = (X^T * X)^(-1) * X^T * y`
       
       This requires computing the inverse of the matrix `(X^T * X)`.
       
    References to learn more:
    - "Pattern Recognition and Machine Learning" by Christopher M. Bishop.
    - Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        This method is used to load the data from the CSV file
        
        Returns:
            X: data features
            y: data target
        """
        try:
            self.X = pd.read_csv(self.config['datasets'][0]['path'])
            self.y = self.X.pop(self.config['datasets'][0]['target'])
            return self.X, self.y
        except Exception as e:
            raise e
        
    def preprocess_data(self):
        """
        This method is used to preprocess the data
        
        Returns:
            X_train: training data features
            X_test: testing data features
            y_train: training data target
            y_test: testing data target
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                            test_size=self.config['splits']['train_test_split']['test_size'],
                                            random_state=self.config['splits']['train_test_split']['random_state'])
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def kfold_split(self):
        """
        This method is used to split the data into k folds
        
        Returns:
            X_train: training data features
            X_test: testing data features
            y_train: training data target
            y_test: testing data target
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                            test_size=self.config['splits']['kfold']['test_size'],
                                            random_state=self.config['splits']['kfold']['random_state'])
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self):
        """
        This method is used to train the linear regression model
        
        Args:
            X_train: training data features
            y_train: training data target
        """
        self.model = (np.linalg.inv(self.X_train.T.dot(self.X_train)).dot(self.X_train.T)).dot(self.y_train)
        return self.model
    
    def predict(self, X_test):
        """
        This method is used to predict the target values
        
        Args:
            X_test: testing data features
        
        Returns:
            y_pred: predicted target values
        """
        self.y_pred = X_test.dot(self.model)
        return self.y_pred

    def evaluate(self, y_test, y_pred):
        """
        This method is used to evaluate the model
        
        Args:
            y_test: testing data target
            y_pred: predicted target values
        
        Returns:
            metrics: dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(self.y_test, y_pred),
            'r2': r2_score(self.y_test, y_pred)
        }
        return metrics

    def save_model(self):
        """
        This method is used to save the model
        """
        joblib.dump(self.model, self.config['saved_models']['dir'] + '/' + self.config['saved_models']['names'][0])

    def load_model(self):
        """
        This method is used to load the model
        """
        self.model = joblib.load(self.config['saved_models']['dir'] + '/' + self.config['saved_models']['names'][0])

    def generate_report(self, metrics):
        """
        This method is used to generate the report
        
        Args:
            metrics: dictionary of metrics
        """
        with open(self.config['reports']['csv']['dir'] + '/' + self.config['reports']['csv']['name'] + '.csv', 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')

    def generate_plots(self):
        """
        This method is used to generate the plots
        """
        plt.plot(self.X_test, self.y_test, 'o')
        plt.plot(self.X_test, self.y_pred, 'r-')
        plt.savefig(self.config['plots']['dir'] + '/' + self.config['plots']['names'][0] + '.png')
        plt.close()
        


if __name__ == '__main__':
    # create the directory for the models and reports
    import json
    config = json.load(open('ml_fundamentals/chapter1/configs/regression.json'))
    import os
    os.makedirs(config['saved_models']['dir'], exist_ok=True)
    os.makedirs(config['reports']['csv']['dir'], exist_ok=True)
    os.makedirs(config['reports']['json']['dir'], exist_ok=True)
    os.makedirs(config['plots']['dir'], exist_ok=True)
    
    linear_regression = LinearRegression(config)
    linear_regression.load_data()
    linear_regression.preprocess_data()
    linear_regression.train()
    linear_regression.predict(linear_regression.X_test)
    linear_regression.evaluate(linear_regression.y_test, linear_regression.y_pred)
    linear_regression.save_model()
    linear_regression.generate_report(linear_regression.evaluate(linear_regression.y_test, linear_regression.y_pred))
    linear_regression.generate_plots()
