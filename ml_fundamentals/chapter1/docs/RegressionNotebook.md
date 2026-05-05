# Regression Exploration Notebook

The `Regression.ipynb` notebook serves as an interactive playground to explore various regression algorithms and datasets. It demonstrates the complete machine learning workflow from data loading and preprocessing to model training and evaluation.

## Contents

1.  **Data Loading**: Uses a flexible configuration-based approach to load different datasets (e.g., Life Expectancy, Advertising).
2.  **Preprocessing**: Handles data cleaning, feature selection, and encoding of categorical variables.
3.  **Model Training**: Implements and compares multiple regression models:
    *   Linear Regression
    *   Random Forest Regressor
    *   Support Vector Regressor (SVR)
    *   XGBoost Regressor
4.  **Evaluation**: Calculates and displays key performance metrics like:
    *   Mean Squared Error (MSE)
    *   Mean Absolute Error (MAE)
    *   R-squared (R²) Score
5.  **Visualization**: Generates plots to visualize predictions vs. actual values and analyze residuals.

## How to Use

1.  Open the notebook in a Jupyter environment (JupyterLab, VS Code, etc.).
2.  Ensure you have the required dependencies installed (`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`).
3.  The notebook relies on `ml_fundamentals/chapter1/configs/regression.json` for data paths and model parameters.
4.  Run the cells sequentially to observe the data analysis and model performance.

---
**File Location:** `ml_fundamentals/chapter1/notebooks/Regression.ipynb`
