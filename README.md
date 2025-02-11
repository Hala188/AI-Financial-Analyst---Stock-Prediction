# 📊 AI Financial Analyst - Predicting Stock Market Prices 📈

## Overview

This project uses machine learning algorithms to predict **stock market closing prices** based on historical data. The project employs different models including **Decision Trees**, **Random Forests**, and **XGBoost** to analyze stock market trends and make accurate predictions.

## Libraries and Dependencies

To run the project, the following Python libraries are required:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

To install the required libraries, use the following command:

```bash
pip install --upgrade xgboost
pip install --upgrade scikit-learn
Steps
1. Import necessary libraries
python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
2. Load the Stock Market Dataset
Upload your stock market dataset in CSV format.

python
Copy
Edit
df = pd.read_csv('stock_data.csv')
df.head()
3. Feature Engineering & Preprocessing
Prepare and scale the data for model training.

python
Copy
Edit
features = ['Opening Price', 'Low Price', 'High Price', 'Volume', 'Market Sentiment', 'Day', 'Month', 'Year', 'Weekday']
target = 'Closing Price'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
4. Train-Test Split
Split the dataset into training and testing sets.

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
5. Train a Decision Tree Regressor
Train the Decision Tree model.

python
Copy
Edit
dtree_model = DecisionTreeRegressor(random_state=42)
dtree_model.fit(X_train, y_train)
6. Train a Random Forest Regressor
Train the Random Forest model.

python
Copy
Edit
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
7. Train an XGBoost Regressor
Train the XGBoost model.

python
Copy
Edit
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
8. Hyperparameter Tuning
Optimize the performance of Random Forest and XGBoost using GridSearchCV and RandomizedSearchCV.

python
Copy
Edit
# Hyperparameter tuning for Random Forest
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, y_train)

# Hyperparameter tuning for XGBoost
xgb_params = {'n_estimators': np.arange(100, 300, 50), 'learning_rate': [0.01, 0.1, 0.2]}
random_xgb = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='neg_mean_squared_error', n_iter=10, n_jobs=-1, random_state=42)
random_xgb.fit(X_train, y_train)
9. Build a Stacked Model
Combine the tuned models using a voting regressor.

python
Copy
Edit
stacked_model = VotingRegressor([('rf', best_rf), ('xgb', best_xgb)])
stacked_model.fit(X_train, y_train)
10. Model Evaluation
Evaluate all models using RMSE.

python
Copy
Edit
model_results = {
    'Decision Tree': mean_squared_error(y_test, y_pred_dtree, squared=False),
    'Random Forest': mean_squared_error(y_test, y_pred_rf, squared=False),
    'Tuned RF': mean_squared_error(y_test, best_rf.predict(X_test), squared=False),
    'XGBoost': mean_squared_error(y_test, y_pred_xgb, squared=False),
    'Tuned XGBoost': mean_squared_error(y_test, best_xgb.predict(X_test), squared=False),
    'Stacked Model': mean_squared_error(y_test, y_pred_stacked, squared=False)
}

results_df = pd.DataFrame(list(model_results.items()), columns=['Model', 'RMSE'])
print(results_df)
11. Final Model Comparison
Compare the performance of all models.

python
Copy
Edit
plt.figure(figsize=(10,5))
plt.bar(results_df['Model'], results_df['RMSE'])
plt.xlabel('Models')
plt.ylabel('RMSE Score')
plt.title('Final Model Performance Comparison')
plt.xticks(rotation=45)
plt.show()
Model Results
Best Model: The Stacked Model achieved the lowest RMSE, outperforming individual models by combining the strengths of Random Forest and XGBoost.
Real-World Applications
Trading Strategies: By predicting stock trends, investors can create more informed strategies.
Automated Trading: The model can be integrated into automated trading systems for real-time decisions.
Risk Management: Predicting market volatility helps investors take necessary precautions.
Long-Term Analysis: The model can incorporate macroeconomic factors to predict long-term trends.
Challenges
Non-linearity of Financial Markets: Market behaviors are highly influenced by external factors, which are difficult to model accurately.
Sudden Events: Unexpected events such as economic crises or natural disasters can disrupt predictions.
Data Updates: Continuous updates to the model and data are necessary to maintain accuracy.
License
