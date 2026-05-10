import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
from DecisionTreeClassifier import DecisionTreeClassifier

class RandomForestClassifier:
    """
    An implementation of a Random Forest Classifier built from scratch.
    
    A Random Forest is an ensemble learning method that operates by constructing 
    a multitude of decision trees at training time and outputting the class 
    that is the mode of the classes of the individual trees.
    
    Mathematical Intuition:
    1. **Bagging (Bootstrap Aggregating)**: Each tree is trained on a random sample 
       of the data with replacement (bootstrap sample). This reduces variance.
    2. **Feature Randomness**: When splitting a node, only a random subset of 
       features is considered. This decorrelates the trees.
    3. **Aggregation**: Final prediction is based on majority voting.
       
    References to learn more:
    - Breiman, L. (2001). Random Forests. Machine Learning.
    - Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/ensemble.html#forest
    """

    def __init__(self, config):
        """
        Initialize the Random Forest Classifier.
        """
        self.config = config
        self.model_config = self.config['models']['Random Forest Classifier']
        self.n_estimators = self.model_config['parameters']['n_estimators']
        self.max_depth = self.model_config['parameters']['max_depth']
        self.min_samples_split = 2 # Default if not in config
        self.max_features = None # Default if not in config
        
        self.trees = []
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        Load data from CSV.
        """
        try:
            dataset_info = self.config['datasets'][0]
            self.X = pd.read_csv(dataset_info['path'])
            self.y = self.X.pop(dataset_info['target'])
            return self.X, self.y
        except Exception as e:
            raise e

    def preprocess_data(self):
        """
        Split data into train and test sets.
        """
        split_config = self.config['splits']['train_test_split']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=split_config['test_size'],
            random_state=split_config['random_state']
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample.
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def train(self, X, y):
        """
        Train the random forest.
        """
        X = np.array(X)
        y = np.array(y)
        self.trees = []
        
        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            if (i + 1) % 10 == 0:
                print(f"Trained {i + 1}/{self.n_estimators} trees...")

    def predict(self, X):
        """
        Predict by majority vote.
        """
        X = np.array(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # tree_preds shape: (n_trees, n_samples)
        # Transpose to (n_samples, n_trees)
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        y_pred = [Counter(sample_preds).most_common(1)[0][0] for sample_preds in tree_preds]
        self.y_predict_labels = np.array(y_pred)
        return self.y_predict_labels

    def save_model(self):
        """
        Save the forest.
        """
        save_path = f"{self.config['saved_models']['dir']}/{self.config['saved_models']['names']['Random Forest Classifier']}"
        joblib.dump(self.trees, save_path)

    def load_model(self):
        """
        Load the forest.
        """
        load_path = f"{self.config['saved_models']['dir']}/{self.config['saved_models']['names']['Random Forest Classifier']}"
        self.trees = joblib.load(load_path)

    def generate_report(self, metrics):
        """
        Generate CSV and JSON reports.
        """
        import json as _json
        report_dir = self.config['reports']['csv']['dir']
        report_name = self.config['reports']['csv']['name'] + "_rf"
        
        # CSV
        with open(f"{report_dir}/{report_name}.csv", 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
        
        # JSON
        json_dir = self.config['reports']['json']['dir']
        json_name = self.config['reports']['json']['name'] + "_rf"
        with open(f"{json_dir}/{json_name}.json", 'w') as f:
            _json.dump(metrics, f, indent=4)

    def generate_plots(self):
        """
        Generate Confusion Matrix plot.
        """
        plots_dir = self.config['plots']['dir']
        y_test_arr = np.array(self.y_test).flatten()
        
        cm = confusion_matrix(y_test_arr, self.y_predict_labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
        ax.set_title('Random Forest - Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{self.config['plots']['names']['confusion_matrix']}_rf.png")
        plt.close()

if __name__ == "__main__":
    import json
    import os
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    config = json.load(open('ml_fundamentals/chapter2/configs/classification.json'))
    
    os.makedirs(config['saved_models']['dir'], exist_ok=True)
    os.makedirs(config['reports']['csv']['dir'], exist_ok=True)
    os.makedirs(config['reports']['json']['dir'], exist_ok=True)
    os.makedirs(config['plots']['dir'], exist_ok=True)

    rf = RandomForestClassifier(config)
    rf.load_data()
    rf.preprocess_data()
    rf.train(rf.X_train, rf.y_train)
    rf.predict(rf.X_test)
    
    y_test_arr = np.array(rf.y_test).flatten()
    metrics = {
        "accuracy": accuracy_score(y_test_arr, rf.y_predict_labels),
        "precision": precision_score(y_test_arr, rf.y_predict_labels, average='weighted'),
        "recall": recall_score(y_test_arr, rf.y_predict_labels, average='weighted'),
        "f1": f1_score(y_test_arr, rf.y_predict_labels, average='weighted')
    }
    
    rf.generate_report(metrics)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    rf.save_model()
    rf.generate_plots()
