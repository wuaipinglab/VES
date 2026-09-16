import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def logistic(x):
    return 1 / (1 + np.exp(-x))


def standardization(x):
    """Assumes input is numpy array or pandas series"""
    return (x - x.mean()) / x.std()



def evaluate_prediction(
    y_true,
    y_pred,
    top_k=None
):
    """
    Parameters
    ----------
    y_true : array-like 
        experimental data
    y_pred : array-like
        VES predictions
    top_k : int or None
        Top-K Precision
    normalize : bool
        MinMaxScaler (0, 1)

    Returns
    -------
    dict
        evaluation metrics
    """

    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    metrics = {}

    metrics["MSE"] = mean_squared_error(y_true, y_pred)
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["R2"] = r2_score(y_true, y_pred)

    metrics["Pearson_r"], metrics["Pearson_p"] = pearsonr(y_true, y_pred)
    metrics["Spearman_r"], metrics["Spearman_p"] = spearmanr(y_true, y_pred)

    # Top-K Precision
    if top_k is not None:
        true_top_idx = np.argsort(y_true)[-top_k:]
        pred_top_idx = np.argsort(y_pred)[-top_k:]
        metrics[f"Precision@{top_k}"] = (
            len(set(true_top_idx) & set(pred_top_idx)) / top_k
        )

    return metrics


def bootstrap_evaluation(
    df,
    y_true_col,
    y_pred_col,
    n_sample=500,
    n_repeat=20,
    top_k=50,
    random_state=16
):
    """

    Parameters
    ----------
    df : pd.DataFrame
        data file containing model predictions and experimental data
    y_true_col : str
        experimental data
    y_pred_col : str
        model predictions
    n_sample : int
    n_repeat : int
    top_k : int
        precision@K
    random_state : int

    Returns
    -------
    pd.DataFrame
        evaluation results of each sampling
    """

    rng = np.random.default_rng(random_state)
    records = []

    for i in range(n_repeat):
        df_sample = df.sample(
            n=n_sample,
            replace=False,
            random_state=int(rng.integers(1e9))
        )

        metrics = evaluate_prediction(
            y_true=df_sample[y_true_col],
            y_pred=df_sample[y_pred_col],
            top_k=top_k
        )

        metrics["iteration"] = i + 1
        records.append(metrics)

    return pd.DataFrame(records)


df = pd.read_csv("./results/summaries/flu_ves.csv")
df["max_escape_experiment"] = np.log(logistic(standardization(df["max_escape_experiment"])))

results = bootstrap_evaluation(
    df,
    y_true_col="max_escape_experiment",
    y_pred_col="evescape",
    n_sample=500,
    n_repeat=20,
    top_k=50
)

results.to_csv("./sampled_evaluation/bootstrap_evaluation_metrics_VES.csv", index=False)


print("\n===== Summary =====")
print(results[["MSE"]].describe())
