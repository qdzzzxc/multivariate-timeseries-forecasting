import numpy as np
import pandas as pd
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

def calculate_sklearn_metrics(df: pd.DataFrame,
                            target_column: str = 'target',
                           naive_forecast_col: str | None = None,
                           metrics: list[str] = ['MASE', 'MAPE', 'MSE', 'MAE', 'SQL']) -> dict[str, float]:
    forecast_cols = ['0.1', '0.5', '0.9']
    for col in forecast_cols:
        if col not in df.columns:
            raise ValueError(f"Столбец с прогнозом '{col}' не найден в датафрейме")
    
    if 'MASE' in metrics and naive_forecast_col is None:
        df['naive_forecast'] = df['0.5'].shift(1)
        df.loc[df.index[0], 'naive_forecast'] = df.loc[df.index[0], '0.5']
        naive_forecast_col = 'naive_forecast'
    
    df = df.dropna(subset=['0.5'] + ([naive_forecast_col] if naive_forecast_col else []))
    
    if len(df) == 0:
        raise ValueError("После удаления NaN значений датафрейм пуст")
    
    results = {}
    
    y_true = df[target_column].values
    y_pred = df['0.5'].values
    
    if 'MSE' in metrics:
        results['MSE'] = mean_squared_error(y_true, y_pred)
    
    if 'MAE' in metrics:
        results['MAE'] = mean_absolute_error(y_true, y_pred)
    
    if 'MAPE' in metrics:
        mask = y_true != 0
        if not np.any(mask):
            results['MAPE'] = np.nan
        else:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            results['MAPE'] = mape
    
    if 'MASE' in metrics:
        if naive_forecast_col not in df.columns:
            results['MASE'] = np.nan
        else:
            naive_errors = np.abs(df['0.5'].values[1:] - df[naive_forecast_col].values[1:])
            denominator = np.mean(naive_errors)
            
            if denominator == 0:
                results['MASE'] = np.nan
            else:
                numerator = mean_absolute_error(y_true, y_pred)
                results['MASE'] = numerator / denominator
    
    if 'SQL' in metrics:
        sql_losses = []
        for forecast_col in forecast_cols:
            y_pred = df[forecast_col].values
            
            try:
                quantile = float(forecast_col)
            except ValueError:
                continue
            
            errors = y_true - y_pred
            sql_loss = np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
            sql_losses.append(sql_loss)
        
        if sql_losses:
            results['SQL'] = sum(sql_losses) / len(sql_losses)
        else:
            results['SQL'] = np.nan
    
    return results

def get_quantile_from_median(
    median_predictions, target_quantile, method="scale", scale_factor=0.2
):
    if not 0 <= target_quantile <= 1:
        raise ValueError("Квантиль должен быть между 0 и 1")

    if target_quantile == 0.5:
        return median_predictions.copy()

    direction = -1 if target_quantile < 0.5 else 1
    deviation = abs(target_quantile - 0.5) * 2

    if method == "scale":
        adjustment = direction * np.abs(median_predictions) * scale_factor * deviation

        return median_predictions + adjustment

    elif method == "offset":
        offset = (
            direction * np.mean(np.abs(median_predictions)) * scale_factor * deviation
        )

        return median_predictions + offset

    elif method == "dynamic":
        base_adjustment = np.mean(np.abs(median_predictions)) * scale_factor * deviation
        dynamic_factor = 1 + np.log1p(
            np.abs(median_predictions) / np.mean(np.abs(median_predictions))
        )
        adjustment = direction * base_adjustment * dynamic_factor

        return median_predictions + adjustment

    else:
        raise ValueError("Метод должен быть 'scale', 'offset' или 'dynamic'")