<div align="center">

<h1>🏷️ Chapter 2: Classification</h1>

<p><strong>Categorize the world with data — learn how machines predict distinct classes.</strong></p>

<p><em>Part of the <a href="../../README.md">Hands-on AI Tutorial</a> · Machine Learning Fundamentals Track</em></p>

<br/>

[![Track](https://img.shields.io/badge/Track-ML%20Fundamentals-blueviolet?style=flat-square)]()
[![Chapter](https://img.shields.io/badge/Chapter-2%20of%205-blue?style=flat-square)]()
[![Difficulty](https://img.shields.io/badge/Level-Beginner%20Friendly-brightgreen?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()

<br/>

| 🧮 Algorithms | 📊 Datasets | 📏 Metrics | 📓 Notebooks |
|:-------------:|:-----------:|:----------:|:------------:|
| **4** | **3** | Accuracy · F1 | **1** |

<br/>

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

<br/>

**[🧩 What is Classification?](#-what-is-classification-for-beginners)** · **[🧮 Algorithms](#-algorithms-covered)** · **[📊 Datasets](#-datasets)** · **[🚀 How to Run](#-how-to-run)** · **[🗺️ Learning Path](#️-learning-path-for-this-chapter)**

<br/>

---

> *"Classification is the art of drawing boundaries in data."*

</div>

---

## 🧩 What Is Classification? (For Beginners)

Imagine you receive an email and need to decide if it's **Spam** or **Not Spam**. Or a doctor looking at medical test results to diagnose whether a patient has a **disease** or is **healthy**. Classification is the technique that learns from past examples to assign new inputs into distinct categories (called **classes**).

In simple terms:

> **Classification = Finding the best decision boundary to separate data into different categories.**

Examples of classification problems:
- 📧 Identifying spam emails
- ❤️ Diagnosing heart disease from patient data
- 🌸 Identifying plant species from measurements
- 💳 Detecting fraudulent credit card transactions

---

## 🎯 What You Will Learn

- ✅ What classification is and how it differs from regression
- ✅ The math behind 4 major classification algorithms
- ✅ How to handle categorical predictions and probabilities
- ✅ How to evaluate classifiers beyond just "accuracy"
- ✅ How to read a Confusion Matrix
- ✅ How to visualize decision boundaries and probability distributions

---

## 🧮 Algorithms Covered

### 1. 📈 Logistic Regression
Despite its name, it's a classification algorithm. It squashes a linear combination of inputs into a probability between 0 and 1 using the Sigmoid function.

- **Best for**: Binary classification and understanding the probability of a class.
- **Key concept**: Sigmoid function and Binary Cross-Entropy (Log Loss).
- **Math**: `σ(z) = 1 / (1 + e^(-z))`
- 📄 [Full Documentation](docs/LogisticRegression.md)

---

### 2. 📍 K-Nearest Neighbors (KNN)
A simple but powerful lazy learner. It memorizes the training data and predicts the class of a new point based on the majority vote of its 'K' closest neighbors.

- **Best for**: Intuitive baselines, non-linear boundaries.
- **Key concept**: Distance metrics (Euclidean) and majority voting.
- **Math**: `Distance = sqrt(Σ(p_i - q_i)^2)`
- 📄 [Full Documentation](docs/KNN.md)

---

### 3. 🌳 Decision Tree Classifier
Builds a tree of if/else decisions to split data into groups. It aims to create pure leaves where all samples belong to a single class.

- **Best for**: Explainable models, mixed feature types.
- **Key concept**: Information Gain and Entropy (or Gini Impurity) to choose the best splits.
- **Math**: `Information Gain = Entropy(Parent) - Σ (N_child / N_parent) * Entropy(Child)`
- 📄 [Full Documentation](docs/DecisionTreeClassifier.md)

---

### 4. 🌲 Random Forest Classifier
An ensemble of many decision trees. Each tree votes on the class, and the majority wins.

- **Best for**: Robust performance, avoiding overfitting that single decision trees suffer from.
- **Key concept**: Bagging (Bootstrap Aggregating) and feature randomness.
- **Math**: Majority vote across M independent decision trees.
- 📄 [Full Documentation](docs/RandomForestClassifier.md)

---

## 📊 Datasets

All datasets are stored in `data/` and referenced in the config file.

| Dataset | File | Target Column | Description |
|---|---|---|---|
| Diabetes | `diabetes.csv` | `Outcome` | Predict whether a patient has diabetes |
| Heart Disease | `heart.csv` | `target` | Predict the presence of heart disease |
| Iris | `Iris.csv` | `Species` | Classify iris plants into three species |

---

## 📏 Evaluation Metrics

After training, classification models are evaluated using:

| Metric | Formula / Meaning | What It Means |
|---|---|---|
| **Accuracy** | Correct Predictions / Total Predictions | Overall correctness. Can be misleading with imbalanced data! |
| **Precision** | True Positives / (True Positives + False Positives) | When it predicts "Yes", how often is it right? |
| **Recall** | True Positives / (True Positives + False Negatives) | Out of all actual "Yes"s, how many did it find? |
| **F1 Score** | `2 * (Precision * Recall) / (Precision + Recall)` | Harmonic mean of precision and recall. Best single metric for imbalanced classes. |
| **Confusion Matrix** | A table of True Positives, True Negatives, False Positives, False Negatives | Shows exactly where the model is making mistakes. |

### 🧠 Deep Dive: Precision vs. Recall

In many real-world scenarios, **Accuracy** is not enough, especially when dealing with imbalanced datasets. This is where **Precision** and **Recall** become critical. There is often a trade-off between the two, and the "best" metric depends entirely on the problem you are solving.

- **Optimize for Precision (Quality over Quantity)**:
  - **Goal**: Minimize False Positives. When the model says "Yes", you want to be absolutely sure it's correct.
  - **Best Case**: Spam Email Detection. A False Positive means a legitimate, important email goes to the spam folder (bad user experience). It's better to let a few spam emails into the inbox (False Negatives) than to block an important work email.

- **Optimize for Recall (Quantity over Quality)**:
  - **Goal**: Minimize False Negatives. You want to catch *every single* positive case, even if it means raising some false alarms.
  - **Best Case & Medical Data**: **Recall is crucial for medical data** (like our Diabetes and Heart Disease datasets). A False Negative means a sick patient is told they are healthy, missing life-saving early treatment. A False Positive just means a healthy patient undergoes a few more tests to be sure. In healthcare, it is always better to be overly cautious; therefore, we heavily optimize for **Recall** to ensure no sick patient slips through the cracks.

---

## 🗂️ File Structure

```text
chapter2/
│
├── README.md                          ← You are here
│
├── configs/
│   └── classification.json            ← Dataset paths, model hyperparameters, split settings
│
├── data/
│   ├── diabetes.csv                   ← Diabetes dataset
│   ├── heart.csv                      ← Heart Disease dataset
│   └── Iris.csv                       ← Iris dataset
│
├── docs/                              ← Deep-dive algorithm docs with math
│   ├── LogisticRegression.md          ← Math + usage guide for Logistic Regression
│   ├── KNN.md                         ← Math + usage guide for KNN
│   ├── DecisionTreeClassifier.md      ← Math + usage guide for Decision Tree
│   └── RandomForestClassifier.md      ← Math + usage guide for Random Forest
│
├── models/
│   ├── logreg.pkl                     ← Saved Logistic Regression model
│   ├── knn.pkl                        ← Saved KNN model
│   └── rf.pkl                         ← Saved Random Forest model
│
├── notebooks/
│   └── Classification.ipynb           ← Interactive exploration notebook
│
├── results/
│   ├── classification_report.csv      ← Evaluation results (CSV)
│   └── classification_report.json     ← Evaluation results (JSON)
│
├── plots/
│   ├── confusion_matrix.png           ← Confusion Matrix visualizations
│   ├── roc_auc.png                    ← Probability distribution / ROC plots
│   └── knn_k_vs_accuracy.png          ← KNN hyperparameter tuning plot
│
└── src/
    ├── LogisticRegression.py          ← Logistic Regression (from scratch)
    ├── KNN.py                         ← K-Nearest Neighbors (from scratch)
    ├── DecisionTreeClassifier.py      ← Decision Tree Classifier (from scratch)
    └── RandomForestClassifier.py      ← Random Forest Classifier (uses DecisionTree)
```

---

## ⚙️ Configuration File (`configs/classification.json`)

The config file controls everything — datasets, model parameters, file paths. Here's what it contains:

```json
{
  "datasets": [...],        // List of datasets with file paths and target column
  "models": {...},          // Each model's hyperparameters (e.g., K for KNN, max_depth for trees)
  "metrics": {...},         // Which metrics to compute
  "splits": {...},          // Train/test split ratio and k-fold settings
  "reports": {...},         // Where to save CSV/JSON reports
  "saved_models": {...},    // Where to save trained model files
  "plots": {...}            // Where to save visualizations
}
```

---

## 🚀 How to Run

### Option A: Interactive (Recommended for Learners)

Open the Jupyter notebook to explore interactively:

```bash
jupyter notebook ml_fundamentals/chapter2/notebooks/Classification.ipynb
```

---

### Option B: Run a Script Directly

Each model can be run as a standalone Python script from the **project root**:

```bash
# Logistic Regression
python3 ml_fundamentals/chapter2/src/LogisticRegression.py

# K-Nearest Neighbors
python3 ml_fundamentals/chapter2/src/KNN.py

# Random Forest Classifier
python3 ml_fundamentals/chapter2/src/RandomForestClassifier.py
```

> ⚠️ Run from the **root** of the repository, not inside the chapter folder.

---

## 📋 Prerequisites

Make sure you have these Python libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## 🗺️ Learning Path for This Chapter

```text
Step 1: Read this README (you're doing it now ✅)

Step 2: Open the Notebook
  └── ml_fundamentals/chapter2/notebooks/Classification.ipynb
      ├── Load the datasets
      ├── Explore distributions and class balance
      └── See models in action visually

Step 3: Understand the Code
  └── See how the math translates to Python code in src/
      ├── Start with LogisticRegression.py
      └── See how KNN uses distance matrices

Step 4: Experiment!
  ├── Change 'n_neighbors' for KNN in classification.json and see the plot change!
  ├── Switch to the heart.csv dataset
  └── Compare metrics across models
```

---

## 💡 Tips for Beginners

- **Watch out for imbalanced data.** If 99% of emails are not spam, a model predicting "Not Spam" always will be 99% accurate but completely useless. Use F1 Score!
- **Visualize your results.** A confusion matrix tells a much better story than a single accuracy number.
- **Scaling matters.** Algorithms like KNN rely on distance, so if one feature is measured in millions and another in decimals, the larger one will dominate. Standardize your data!

---

## ➡️ What's Next?

Once you're comfortable with classification, move on to:

**→ Chapter 3: Clustering**

You'll learn how to find patterns in data when you *don't* have any labels (unsupervised learning).

---

> 📖 **Chapter 2 is part of the [Hands-on AI Tutorial](../../README.md)** — an open-source project for learning AI from scratch.
