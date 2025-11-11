import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import math
import re

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_train = train_df.iloc[:, -30:]
y_train = train_df.iloc[:, 3]
X_test = test_df.iloc[:, -30:]
y_test = test_df.iloc[:, 3]

function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos','tan', 'log', 'abs', 'neg','sqrt','inv']

model = SymbolicRegressor(
    function_set = function_set,
    population_size=10000,
    generations=200,
    stopping_criteria=0.000001,
    p_crossover=0.7,
    p_subtree_mutation=0.2,
    p_hoist_mutation=0.05,
    p_point_mutation=0.05,
    max_samples=1,
    verbose=1,
    parsimony_coefficient=0.001,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

r2_train = r2_score(y_train, train_pred)
r2_test = r2_score(y_test, test_pred)

pearson_train, _ = pearsonr(y_train, train_pred)
pearson_test, _ = pearsonr(y_test, test_pred)

mse_train = mean_squared_error(y_train, train_pred)
mse_test = mean_squared_error(y_test, test_pred)

mae_train = mean_absolute_error(y_train, train_pred)
mae_test = mean_absolute_error(y_test, test_pred)

rmse_train = math.sqrt(mse_train)
rmse_test = math.sqrt(mse_test)

expression_str = str(model._program)

results = {
    "Metric": ["train R2", "Test R2", "train Pearson", "Test Pearson",
            "train MSE", "Test MSE", "train MAE", "Test MAE",
            "train RMSE", "Test RMSE", "expression_str"],
    "Value": [r2_train, r2_test, pearson_train, pearson_test, 
           mse_train, mse_test, mae_train, mae_test,
           rmse_train, rmse_test, expression_str]
}

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)