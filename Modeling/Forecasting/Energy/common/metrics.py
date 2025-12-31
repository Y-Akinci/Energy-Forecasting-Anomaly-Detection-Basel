import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": _mape(y_true, y_pred),
    }


def train_test_metrics(y_train, y_pred_train, y_test, y_pred_test):
    train = regression_metrics(y_train, y_pred_train)
    test = regression_metrics(y_test, y_pred_test)

    out = {}
    for k, v in train.items():
        out[f"{k}_train"] = v
    for k, v in test.items():
        out[f"{k}_test"] = v
    return out
