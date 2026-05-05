<div align="center">

<h1>📈 Chapter 1: Regression</h1>

<p><strong>Predict the future with data — learn how machines estimate continuous values.</strong></p>

<p><em>Part of the <a href="../../README.md">Hands-on AI Tutorial</a> · Machine Learning Fundamentals Track</em></p>

<br/>

[![Track](https://img.shields.io/badge/Track-ML%20Fundamentals-blueviolet?style=flat-square)]()
[![Chapter](https://img.shields.io/badge/Chapter-1%20of%205-blue?style=flat-square)]()
[![Difficulty](https://img.shields.io/badge/Level-Beginner%20Friendly-brightgreen?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()

<br/>

| 🧮 Algorithms | 📊 Datasets | 📏 Metrics | 📓 Notebooks |
|:-------------:|:-----------:|:----------:|:------------:|
| **4** | **3** | MSE · R² | **1** |

<br/>

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-189AB4?style=flat-square)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

<br/>

**[🧩 What is Regression?](#-what-is-regression-for-beginners)** · **[🧮 Algorithms](#-algorithms-covered)** · **[📊 Datasets](#-datasets)** · **[🚀 How to Run](#-how-to-run)** · **[🗺️ Learning Path](#️-learning-path-for-this-chapter)**

<br/>

---

> *"You don't understand a regression algorithm until you've implemented the Normal Equation by hand."*

</div>

---

## 🧩 What Is Regression? (For Beginners)

Imagine you want to predict the **price of a house**. You know its size, number of rooms, and location. Regression is the technique that finds the mathematical relationship between those inputs (called **features**) and the output you want to predict (called the **target**).

In simple terms:

> **Regression = Finding the best line (or curve) that maps your inputs to a continuous output.**

Examples of regression problems:
- 🏡 Predicting house prices from square footage
- ❤️ Predicting life expectancy from health indicators
- 📚 Predicting a student's grade from study hours
- 📢 Predicting sales from advertising spend

---

## 🎯 What You Will Learn

- ✅ What regression is and when to use it
- ✅ The math behind 4 major regression algorithms
- ✅ How to load and preprocess real datasets
- ✅ How to train, evaluate, and save models
- ✅ How to interpret MSE and R² scores
- ✅ How to visualize model performance

---

## 🧮 Algorithms Covered

### 1. 📐 Linear Regression
The foundation of regression. It fits a straight line through your data by finding the weights that minimize prediction error.

- **Best for**: Data with a linear relationship between features and target
- **Key concept**: Normal Equation — solves for the optimal weights analytically (no iteration needed)
- **Math**: `θ = (XᵀX)⁻¹ Xᵀy`
- 📄 [Full Documentation](docs/LinearRegression.md)

---

### 2. 🌳 Decision Tree Regression
Builds a tree of if/else decisions to split data into groups and predict the mean value in each group.

- **Best for**: Non-linear data with complex interactions between features
- **Key concept**: Splits are chosen to minimize variance (MSE) in child nodes
- **Math**: `MSE_split = (N_left × Var(y_left) + N_right × Var(y_right)) / N`
- 📄 [Full Documentation](docs/DecisionTreeRegression.md)

---

### 3. 🌲 Random Forest Regression
An ensemble of many decision trees. Each tree is trained on a random sample of data and features, and final predictions are averaged across all trees.

- **Best for**: Most practical problems — robust, accurate, and resistant to overfitting
- **Key concept**: Bagging (Bootstrap AGGregating) + feature randomness = decorrelated trees
- **Math**: `ŷ = (1/M) × Σ f_m(x)` (average of M trees)
- 📄 [Full Documentation](docs/RandomForestRegression.md)

---

### 4. ⚡ Support Vector Regression (SVR)
Finds a "tube" around the prediction line within which errors are ignored, then optimizes the line to fit the data outside the tube.

- **Best for**: High-dimensional data, problems where you want to control how sensitive the model is to outliers
- **Key concept**: ε-insensitive tube + Kernel Trick for non-linear data
- **Kernels supported**: Linear, Polynomial, RBF (Gaussian)
- 📄 [Full Documentation](docs/SupportVectorRegression.md)

---

## 📊 Datasets

All datasets are stored in `data/` and referenced in the config file.

| Dataset | File | Target Column | Description |
|---|---|---|---|
| Advertising | `advertising.csv` | `Sales` | Predict sales based on TV, Radio, Newspaper spend |
| Life Expectancy | `LifeExpectancyData.csv` | *(configurable)* | Predict life expectancy from health/economic indicators |
| Student Data | `student_data.csv` | *(configurable)* | Predict student performance |

---

## 📏 Evaluation Metrics

After training, models are evaluated using:

| Metric | Formula | What It Means |
|---|---|---|
| **MSE** (Mean Squared Error) | `(1/n) Σ(ŷ - y)²` | Average squared error. Lower = better. Penalizes large errors heavily. |
| **R² Score** | `1 - SS_res/SS_tot` | How much variance the model explains. 1.0 = perfect, 0 = predicts the mean. |

> 💡 **Tip for beginners:** Start by trying to get a positive R² score. Negative means your model is *worse* than just predicting the average!

---

## 🗂️ File Structure

```
chapter1/
│
├── README.md                          ← You are here
│
├── configs/
│   └── regression.json                ← Dataset paths, model hyperparameters, split settings
│
├── data/
│   ├── advertising.csv                ← Advertising spend vs. Sales dataset
│   ├── LifeExpectancyData.csv         ← WHO Life Expectancy dataset
│   └── student_data.csv               ← Student performance dataset
│
├── docs/
│   ├── LinearRegression.md            ← Math + usage guide for Linear Regression
│   ├── DecisionTreeRegression.md      ← Math + usage guide for Decision Tree
│   ├── RandomForestRegression.md      ← Math + usage guide for Random Forest
│   ├── SupportVectorRegression.md     ← Math + usage guide for SVR
│   └── RegressionNotebook.md          ← Guide to the Jupyter notebook
│
├── models/
│   ├── linear_regression.joblib       ← Saved Linear Regression model
│   ├── rf_model.pkl                   ← Saved Random Forest model
│   └── svr_model.pkl                  ← Saved SVR model
│
├── notebooks/
│   └── Regression.ipynb               ← Interactive exploration notebook
│
├── results/
│   ├── regression_report.csv          ← Linear Regression evaluation results
│   ├── regression_report_random_forest_regression.csv
│   ├── regression_report_SVR.csv
│   ├── feature_importances.png        ← Feature importance bar chart
│   ├── residual_analysis.png          ← Residuals visualization
│   ├── target_distribution.png        ← Target variable distribution
│   ├── rf_plot.png                    ← Random Forest predictions plot
│   └── svr_plot.png                   ← SVR predictions plot
│
└── src/
    ├── LinearRegression.py            ← Linear Regression (from scratch)
    ├── DecisionTreeRegression.py      ← Decision Tree (from scratch)
    ├── RandomForestRegression.py      ← Random Forest (from scratch, uses Decision Tree)
    └── SupportVectorRegression.py     ← SVR (from scratch using SMO-style optimization)
```

---

## ⚙️ Configuration File (`configs/regression.json`)

The config file controls everything — datasets, model parameters, file paths. Here's what it contains:

```json
{
  "datasets": [...],        // List of datasets with file paths and target column
  "models": {...},          // Each model's hyperparameters
  "metrics": {...},         // Which metrics to compute
  "splits": {...},          // Train/test split ratio and k-fold settings
  "reports": {...},         // Where to save CSV/JSON reports
  "saved_models": {...}     // Where to save trained model files
}
```

You can swap datasets or tune model parameters without touching the source code — just edit this file!

---

## 🚀 How to Run

### Option A: Interactive (Recommended for Learners)

Open the Jupyter notebook to explore interactively:

```bash
jupyter notebook ml_fundamentals/chapter1/notebooks/Regression.ipynb
```

📄 See [RegressionNotebook.md](docs/RegressionNotebook.md) for a full guide to the notebook.

---

### Option B: Run a Script Directly

Each model can be run as a standalone Python script from the **project root**:

```bash
# Linear Regression
python -m ml_fundamentals.chapter1.src.LinearRegression

# Random Forest Regression
python -m ml_fundamentals.chapter1.src.RandomForestRegression

# Support Vector Regression
python -m ml_fundamentals.chapter1.src.SupportVectorRegression
```

> ⚠️ Run from the **root** of the repository, not inside the chapter folder.

---

## 📋 Prerequisites

Make sure you have these Python libraries installed:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost joblib
```

| Library | Why You Need It |
|---|---|
| `pandas` | Loading and manipulating CSV data |
| `numpy` | Numerical computations (matrix math) |
| `matplotlib` | Plotting charts and visualizations |
| `scikit-learn` | Train/test splits, evaluation metrics |
| `xgboost` | XGBoost model (used in the notebook) |
| `joblib` | Saving and loading trained models |

---

## 🗺️ Learning Path for This Chapter

```
Step 1: Read this README (you're doing it now ✅)

Step 2: Open the Notebook
  └── ml_fundamentals/chapter1/notebooks/Regression.ipynb
      ├── Load the dataset
      ├── Explore with df.head(), df.describe()
      └── Visualize the target distribution

Step 3: Understand the Math
  └── Read the docs/ files for the algorithm you're curious about
      ├── docs/LinearRegression.md
      ├── docs/DecisionTreeRegression.md
      ├── docs/RandomForestRegression.md
      └── docs/SupportVectorRegression.md

Step 4: Read the Source Code
  └── See how the math translates to Python code in src/

Step 5: Experiment!
  ├── Change hyperparameters in configs/regression.json
  ├── Switch to a different dataset
  └── Compare MSE and R² between models
```

---

## 💡 Tips for Beginners

- **Don't panic about the math.** Read the docs, then look at the code — the code makes the math concrete.
- **Run the notebook first.** Seeing outputs before understanding code helps a lot.
- **MSE alone doesn't tell you everything.** Always check R² too.
- **Experiment freely.** Change parameters, re-run, compare results. That's how you learn.
- **Linear Regression is your best friend.** Always start with a simple baseline before using complex models.

---

## ➡️ What's Next?

Once you're comfortable with regression, move on to:

**→ Chapter 2: Classification** *(coming soon)*

You'll learn how to predict *categories* instead of continuous numbers — like predicting whether an email is spam or not.

---

> 📖 **Chapter 1 is part of the [Hands-on AI Tutorial](../../README.md)** — an open-source project for learning AI from scratch.
