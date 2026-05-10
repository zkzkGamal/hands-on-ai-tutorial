
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LogisticRegression:
    """
    This class is used to create the logistic regression model
    
    Mathematical Intuition:
    1. **Sigmoid Function**: Maps any real-valued input to a probability between 0 and 1.
       σ(z) = 1 / (1 + e^(-z))
       
    2. **Cost Function (Binary Cross-Entropy)**: Measures the error between predicted probabilities and true labels.
       J(w, b) = -1/m * Σ[y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i)]
       
    3. **Gradient Descent**: Updates weights iteratively to minimize the cost.
       w = w - α * ∂J/∂w
       b = b - α * ∂J/∂b
       where ∂J/∂w = 1/m * X^T * (ŷ - y) and ∂J/∂b = 1/m * Σ(ŷ - y)
       
    References to learn more:
    - "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman.
    - Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    """
    
    def __init__(self, config):
        """
        This method is used to initialize the logistic regression model
        
        Args:
            config: configuration dictionary
            X: data features
            learning_rate: learning rate
            num_iters: number of iterations
        """
        self.config = config
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.weights = None
        self.bias = None
        self.n = None
        self.m = None
        self.y_pred = None
        self.max_iter = self.config['models']['Logistic Regression']['parameters']['max_iter']
        self.eta0 = 0.001

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
        self.m, self.n = self.X_train.shape
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
        

    def train(self, X, y):
        """
        This method is used to train the logistic regression model
        
        Args:
            X: training data features
            y: training data target
        """
        # Convert to numpy arrays to avoid pandas broadcasting errors
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        self.weights = np.zeros((self.n, 1))
        self.bias = 0

        for it in range(self.max_iter):
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
            cost = (
                -1
                / self.m
                * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
            )

            dw = 1 / self.m * np.dot(X.T, (y_predict - y))
            db = 1 / self.m * np.sum(y_predict - y)
            self.weights -= self.eta0 * dw
            self.bias -= self.eta0 * db
            if it % 100 == 0:
                print(f"Cost after iteration {it}: {cost}")

        return self.weights, self.bias

    def predict(self, X):
        """
        This method is used to predict the target values
        
        Args:
            X: testing data features
        
        Returns:
            y_pred: predicted target values
        """
        # Convert to numpy array to avoid pandas broadcasting errors
        X = np.array(X)
        self.y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
        self.y_predict_labels = (self.y_predict > 0.5).astype(int).flatten()

        return self.y_predict_labels

    def sigmoid(self, z):
        """
        This method is used to apply the sigmoid function
        
        Args:
            z: input to the sigmoid function
        
        Returns:
            sigmoid(z): output of the sigmoid function
        """
        return 1 / (1 + np.exp(-z))
    
    def save_model(self):
        """
        This method is used to save the model
        """
        model_params = {'weights': self.weights, 'bias': self.bias}
        joblib.dump(model_params, self.config['saved_models']['dir'] + '/' + self.config['saved_models']['names']['Logistic Regression'])

    def load_model(self):
        """
        This method is used to load the model
        """
        model_params = joblib.load(self.config['saved_models']['dir'] + '/' + self.config['saved_models']['names']['Logistic Regression'])
        self.weights = model_params['weights']
        self.bias = model_params['bias']

    def generate_report(self, metrics):
        """
        This method is used to generate the report.
        Writes both a CSV report and a JSON report.
        
        Args:
            metrics: dictionary of metrics
        """
        import json as _json

        # --- CSV report ---
        csv_path = (self.config['reports']['csv']['dir'] + '/'
                    + self.config['reports']['csv']['name'] + '_lr' + '.csv')
        with open(csv_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')

        # --- JSON report ---
        json_path = (self.config['reports']['json']['dir'] + '/'
                     + self.config['reports']['json']['name'] + '_lr' + '.json')
        with open(json_path, 'w') as f:
            _json.dump(metrics, f, indent=4)

    def generate_plots(self):
        """
        Generates two plots and saves them to the plots directory:
        1. Confusion matrix heatmap.
        2. Predicted probability distribution (histogram).
        """
        plots_dir = self.config['plots']['dir']
        y_test_arr = np.array(self.y_test).flatten()

        # --- Plot 1: Confusion Matrix ---
        cm = confusion_matrix(y_test_arr, self.y_predict_labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()
        plt.savefig(plots_dir + '/' + self.config['plots']['names']['confusion_matrix'] + '_lr' + '.png')
        plt.close()

        # --- Plot 2: Predicted Probability Distribution ---
        probs = self.y_predict.flatten()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(probs[y_test_arr == 0], bins=30, alpha=0.7, label='Class 0', color='steelblue')
        ax.hist(probs[y_test_arr == 1], bins=30, alpha=0.7, label='Class 1', color='tomato')
        ax.axvline(0.5, color='black', linestyle='--', label='Decision Boundary (0.5)')
        ax.set_title('Predicted Probability Distribution')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir + '/' + self.config['plots']['names']['roc_auc']+ '_lr' + '.png')
        plt.close()


if __name__ == "__main__":
    import json
    config = json.load(open('ml_fundamentals/chapter2/configs/classification.json'))
    import os
    os.makedirs(config['saved_models']['dir'], exist_ok=True)
    os.makedirs(config['reports']['csv']['dir'], exist_ok=True)
    os.makedirs(config['reports']['json']['dir'], exist_ok=True)
    os.makedirs(config['plots']['dir'], exist_ok=True)

    logreg = LogisticRegression(config)
    logreg.load_data()
    logreg.preprocess_data()
    logreg.train(logreg.X_train, logreg.y_train)
    logreg.predict(logreg.X_test)
    y_test_arr = np.array(logreg.y_test).flatten()
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    logreg.generate_report({"accuracy": accuracy_score(y_test_arr, logreg.y_predict_labels), 
                            "precision": precision_score(y_test_arr, logreg.y_predict_labels), 
                            "recall": recall_score(y_test_arr, logreg.y_predict_labels), 
                            "f1": f1_score(y_test_arr, logreg.y_predict_labels)})
    print(f"Accuracy: {accuracy_score(y_test_arr, logreg.y_predict_labels):.4f}")
    print(f"Precision: {precision_score(y_test_arr, logreg.y_predict_labels):.4f}")
    print(f"Recall: {recall_score(y_test_arr, logreg.y_predict_labels):.4f}")
    print(f"F1: {f1_score(y_test_arr, logreg.y_predict_labels):.4f}")
    logreg.save_model()
    logreg.generate_plots()