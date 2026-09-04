# Healthcare Analytics SQL Project

## Project Overview

This project demonstrates my **Machine Learning and Data Science skills through the analysis and predictive modeling of a Carbon Emission Dataset.**

The project includes a structured dataset containing individual lifestyle, consumption, transport, energy, and demographic metrics. I performed extensive exploratory data analysis (EDA), data preprocessing, feature engineering, and trained multiple regression models to predict carbon emission levels and derive actionable environmental insights.

---

## Database Structure

The dataset consists of the following key features:

**Personal & Demographic Information**

Stores personal and physical details, including:

* Body Type
* Sex
* Diet
* Social Activity

**Household & Lifestyle Factors**

Stores daily routine and home consumption parameters, including:

* How Often Shower
* Heating Energy Source
* Monthly Grocery Bill
* How Many New Clothes Monthly
* Cooking_With (e.g., Stove, Oven, Microwave, Airfryer)

**Transport & Travel**

Stores commuting and mobility habits, including:

* Transport (e.g., public, private, walk/bicycle)
* Vehicle Type (e.g., petrol, diesel)
* Vehicle Monthly Distance Km
* Frequency of Traveling by Air

**Waste & Energy Management**

Stores sustainability and energy usage patterns, including:

* Waste Bag Size
* Waste Bag Weekly Count
* How Long TV PC Daily Hour
* How Long Internet Daily Hour
* Energy efficiency
* Recycling (e.g., Paper, Plastic, Glass, Metal)

**Target Variable**

* CarbonEmission: 'The calculated monthly carbon footprint score'.

---

## Database Relationships

Although this dataset is flattened for predictive modeling, key functional relationships exist across feature groups:

* Personal & Household Habits → Energy & Waste Generation
* Mobility & Travel Patterns → Fuel & Transport Emissions
* Food & Clothing Consumption → Indirect Household Emissions
* Recycling & Energy Efficiency → Emission Mitigation Impact

---

## Machine Learning & Python Concepts Used

This project demonstrates the following Data Science and ML concepts:

* Data Cleaning & Imputation (SimpleImputer)
* Exploratory Data Analysis (EDA) & Visualization (Matplotlib, Seaborn)
* Categorical Data Encoding (OneHotEncoder, Literal Evaluation)
* Feature Scaling (StandardScaler)
* Data Pipeline Construction (ColumnTransformer, Pipeline)
* Train-Test Splitting (train_test_split)
* Regression Algorithm Implementation
* Hyperparameter Tuning (GridSearchCV)
* Model Evaluation Metrics (R² Score, RMSE, MAE)
* Warning Suppression & Code Cleanliness

---

## Analysis & Modeling Performed

Some of the key steps and models implemented in this project are:

**1. Target Emission Distribution Analysis**

Retrieved statistical summaries for the `CarbonEmission` target column (Mean, Std, Min, Max, Quantiles).

**2. Data Integrity & Missing Value Audit**

Identified missing values across columns (e.g., `Vehicle Type` missingness due to non-vehicle owners) and verified zero duplicate rows.

**3. Feature Categorization & Pipeline Setup**

Constructed preprocessing pipelines separating numerical and categorical pipelines using `ColumnTransformer`.

**4. Linear Baseline Modeling**

Trained a baseline `LinearRegression` model to establish primary prediction benchmarks.

**5. Regularization Experiments**

Applied `Ridge` and `Lasso` regression models to prevent overfitting and analyze feature penalization.

**6. Tree-Based Ensemble Modeling**

Trained a `RandomForestRegressor` to capture non-linear feature interactions across lifestyle parameters.

**7. Gradient Boosting Regressor Implementation**

Executed `GradientBoostingRegressor` to iteratively optimize prediction accuracy.

**8. Model Performance View**

Created an evaluation script to summarize metrics across models:

```python
# Sample Model Evaluation Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}
```
**Hyperparameter Optimization**

Utilized GridSearchCV to search optimal parameter combinations for tree ensemble models.

---

## Project Files
 
| File Name | Description |
| :--- | :--- |
| `Carbon Emission - LMS.csv` |Dataset containing 10,000 individual records across 20 lifestyle and carbon emission features. |
| `Carbon Emission.ipynb` |Jupyter Notebook containing EDA, preprocessing, model training, hyperparameter tuning, and evaluation.|

---

## How to Run the Project

**Step 1: Install Dependencies**

Ensure you have Python installed, then install the necessary libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

**Step 2: Run the Notebook**

Open **Carbon Emission.ipynb** in Jupyter Notebook, JupyterLab, or VS Code and execute all cells sequentially to run data processing, visualization, and model evaluations.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-Learn

---

## Key Learning Outcomes

Through this project, I practiced:

- Structuring end-to-end Machine Learning workflows in Python.
- Handling domain-specific missing values and categorical features.
- Building modular scikit-learn Pipelines for automated feature transformation.
- Comparing linear, regularized, and ensemble regression algorithms.
- Evaluating regression models using R², RMSE, and MAE metrics.
- Fine-tuning machine learning algorithms using Grid Search Cross-Validation.
- Interpreting lifestyle factors affecting individual carbon footprints.

---
