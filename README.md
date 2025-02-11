# XGBoost Hyperparameter Tuning and Model Comparison

## Project Overview
This project focuses on training and evaluating multiple machine learning models for regression. The primary goal is to tune the hyperparameters of an XGBoost model using RandomizedSearchCV and compare its performance with other models.

## Requirements
To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn matplotlib xgboost
```

## Dataset
- The dataset is split into training (`X_train`, `y_train`) and testing (`X_test`, `y_test`) sets.
- The target variable is predicted using various machine learning models.

## Models Used
- **Decision Tree**
- **Random Forest** (Baseline and Tuned)
- **XGBoost** (Baseline and Tuned)
- **Stacked Model** (Combination of multiple models)

## Hyperparameter Tuning for XGBoost
We use `RandomizedSearchCV` to tune `n_estimators` and `learning_rate`:

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from xgboost import XGBRegressor

xgb_params = {
    'n_estimators': np.arange(100, 300, 50),
    'learning_rate': [0.01, 0.1, 0.2]
}

random_xgb = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='neg_mean_squared_error', n_iter=10, n_jobs=-1, random_state=42)
random_xgb.fit(X_train, y_train)

best_xgb = random_xgb.best_estimator_
print(f'Best XGBoost Params: {random_xgb.best_params_}')
```

## Model Evaluation
The models are evaluated using **Root Mean Squared Error (RMSE)**:

```python
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

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

plt.figure(figsize=(10,5))
plt.bar(results_df['Model'], results_df['RMSE'])
plt.xlabel('Models')
plt.ylabel('RMSE Score')
plt.title('Final Model Performance Comparison')
plt.xticks(rotation=45)
plt.show()
```

## Conclusion
- The tuned XGBoost model is expected to outperform other models.
- The stacked model may further improve performance by leveraging multiple models.

## Future Work
- Explore additional hyperparameters.
- Implement more advanced ensemble techniques.
- Optimize feature selection for better results.

