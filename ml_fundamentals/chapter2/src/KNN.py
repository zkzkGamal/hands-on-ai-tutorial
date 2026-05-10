import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix ,accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter

class KNearestNeighbor:
    """
    An implementation of a K-Nearest Neighbor (KNN) Classifier built from scratch.
    
    KNN is a non-parametric, type of lazy learning algorithm. It doesn't learn a 
    discriminative function from the training data but "memorizes" the training dataset instead.
    
    Mathematical Intuition:
    1. **Distance Metric**: The algorithm typically uses Euclidean distance to find the 
       nearest neighbors. For two points p and q, it is sqrt(Σ(p_i - q_i)^2).
    2. **Vectorized Distance**: We use the formula (a-b)^2 = a^2 - 2ab + b^2 to compute 
       all-to-all distances efficiently.
    3. **Voting**: For a given test point, it finds the k nearest training points and 
       predicts the class that appears most frequently among them.
       
    References to learn more:
    - Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE transactions on information theory.
    - Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/neighbors.html
    """

    def __init__(self, config):
        """
        Initialize the KNN Classifier.
        """
        self.config = config
        self.model_config = self.config['models']['K-Nearest Neighbors Classifier']
        self.k = self.model_config['parameters']['n_neighbors']
        self.eps = 1e-8
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X = None
        self.y = None
        self.y_predict_labels = None

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
        # Store as numpy for performance
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, X, y):
        """
        KNN doesn't require explicit training, just stores the data.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def compute_distances(self, X_test):
        """
        Heavily vectorized distance calculation.
        Idea: (a-b)^2 = a^2 - 2ab + b^2
        """
        X_test_squared = np.sum(X_test ** 2, axis=1, keepdims=True)
        X_train_squared = np.sum(self.X_train ** 2, axis=1, keepdims=True)
        two_X_test_X_train = np.dot(X_test, self.X_train.T)

        return np.sqrt(
            self.eps + X_test_squared - 2 * two_X_test_X_train + X_train_squared.T
        )

    def predict(self, X):
        """
        Predict classes for X.
        """
        X = np.array(X)
        distances = self.compute_distances(X)
        num_test = X.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            # Find indices of k smallest distances
            y_indices = np.argsort(distances[i, :])
            k_closest_classes = self.y_train[y_indices[: self.k]].astype(int)
            # Majority vote
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))

        self.y_predict_labels = y_pred.astype(int)
        return self.y_predict_labels

    def save_model(self):
        """
        Save the model (stored data).
        """
        save_path = f"{self.config['saved_models']['dir']}/{self.config['saved_models']['names']['K-Nearest Neighbors Classifier']}"
        model_data = {'X_train': self.X_train, 'y_train': self.y_train, 'k': self.k}
        joblib.dump(model_data, save_path)

    def load_model(self):
        """
        Load the model.
        """
        load_path = f"{self.config['saved_models']['dir']}/{self.config['saved_models']['names']['K-Nearest Neighbors Classifier']}"
        model_data = joblib.load(load_path)
        self.X_train = model_data['X_train']
        self.y_train = model_data['y_train']
        self.k = model_data['k']

    def generate_report(self, metrics):
        """
        Generate CSV and JSON reports.
        """
        import json as _json
        report_dir = self.config['reports']['csv']['dir']
        report_name = self.config['reports']['csv']['name'] + "_knn"
        
        # CSV
        with open(f"{report_dir}/{report_name}.csv", 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
        
        # JSON
        json_dir = self.config['reports']['json']['dir']
        json_name = self.config['reports']['json']['name'] + "_knn"
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('KNN - Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{self.config['plots']['names']['confusion_matrix']}_knn.png")
        plt.close()
        
    def plot_k_vs_accuracy(self):
        k_values = range(1, 30)
        accuracies = []

        for k in k_values:
            self.k = k
            self.predict(self.X_test)
            y_test_arr = np.array(self.y_test).flatten()
            acc = accuracy_score(y_test_arr, self.y_predict_labels)
            accuracies.append(acc)
            print(f"K={k}, Accuracy={acc:.4f}")
        
        best_k = k_values[np.argmax(accuracies)]
        best_acc = max(accuracies)
        print(f"\nBest K: {best_k}, Best Accuracy: {best_acc:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, marker='o', linestyle='-')
        plt.title('KNN: Impact of K on Accuracy')
        plt.xlabel('Number of Neighbors (K)')
        plt.ylabel('Accuracy')
        plt.xticks(k_values)
        plt.grid(True)
        plt.annotate(f'Best: K={best_k}\nAcc={best_acc:.4f}',
                     xy=(best_k, best_acc),
                     xytext=(best_k + 2, best_acc + 0.02),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{self.config['plots']['dir']}/knn_k_vs_accuracy.png")
        plt.close()

if __name__ == "__main__":
    import json
    import os    
    config = json.load(open('ml_fundamentals/chapter2/configs/classification.json'))
    
    os.makedirs(config['saved_models']['dir'], exist_ok=True)
    os.makedirs(config['reports']['csv']['dir'], exist_ok=True)
    os.makedirs(config['reports']['json']['dir'], exist_ok=True)
    os.makedirs(config['plots']['dir'], exist_ok=True)

    knn = KNearestNeighbor(config)
    knn.load_data()
    knn.preprocess_data()
    knn.train(knn.X_train, knn.y_train)
    knn.predict(knn.X_test)
    
    y_test_arr = np.array(knn.y_test).flatten()
    metrics = {
        "accuracy": accuracy_score(y_test_arr, knn.y_predict_labels),
        "precision": precision_score(y_test_arr, knn.y_predict_labels, average='weighted'),
        "recall": recall_score(y_test_arr, knn.y_predict_labels, average='weighted'),
        "f1": f1_score(y_test_arr, knn.y_predict_labels, average='weighted')
    }
    
    knn.generate_report(metrics)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    knn.save_model()
    knn.generate_plots()
    knn.plot_k_vs_accuracy()
