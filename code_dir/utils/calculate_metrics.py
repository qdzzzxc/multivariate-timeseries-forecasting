import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    mean_squared_log_error,
    explained_variance_score,
    median_absolute_error,
    max_error
)

def calculate_metrics(y_true, y_pred):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MSLE': mean_squared_log_error(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred),
        'Median AE': median_absolute_error(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred),
    }
    
    return metrics