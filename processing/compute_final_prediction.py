import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.impute import SimpleImputer


temperatures = {"fitness": 1}
flu_thresh = 0.054
flu = pd.read_csv("./results/prediction_w_exp/ves.csv")
flu_ablist = [col for col in flu.columns.values if "mutfracsurvive" in col]
flu_all_ab = [col for col in flu.columns.values if "mutfracsurvive" in col]


def logistic(x):
    return 1 / (1 + np.exp(-x))


def standardization(x):
    """Assumes input is numpy array or pandas series"""
    return (x - x.mean()) / x.std()


def make_predictors(summary_init, thresh, ablist, scores=True):

    summary = summary_init.copy()
    summary = summary.drop(
        columns=[col for col in summary.columns if "wcn_fill_" in col])
    summary = summary.drop(
        columns=[col for col in summary.columns if "wcn_sc" in col])
    summary = summary.drop(
        columns=[col for col in summary.columns if "diff" in col])
    summary["wcn_fill_r"] = -summary.wcn_fill
    summary = summary.drop(columns="wcn_fill")

    if scores:
        #Calculate max escape for each mutant
        summary["max_escape_experiment"] = summary[ablist].max(axis=1)
        #Calculate if escape>threshold for each mutant
        summary[
            "is_escape_experiment"] = summary["max_escape_experiment"] > thresh

    impute_cols = ["i", "model_prediction", "wcn_fill_r", "charge_ew-hydro"]

    df_imp = summary[impute_cols].copy()
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    df_imp = pd.DataFrame(imp.fit_transform(df_imp),
                          columns=df_imp.columns,
                          index=df_imp.index)
    df_imp = pd.concat([df_imp, summary[["wt", "mut"]]], axis=1)


    summary["ves_prediction"] = 0
    summary["ves_prediction"] += np.log(
        logistic(
            standardization(df_imp["model_prediction"]) * 1 /
            temperatures["fitness"]))
    summary = summary.drop(
        columns=[col for col in summary.columns if col == "wcn_fill"])

    summary = summary.rename(
        columns={
            "wcn_fill_r": "accessibility_wcn",
            "charge_ew-hydro": "dissimilarity_charge_hydro"
        })

    summary = summary.round(decimals=7)

    return (summary)

flu = make_predictors(flu, flu_thresh, flu_ablist)
flu = flu.drop(columns=flu_all_ab)

flu.to_csv("./results/summaries/flu_ves.csv",index=False)
def make_site(summary_init):

    summary = summary_init.copy()
    summary = summary.groupby(['i', 'wt']).mean(numeric_only=True).reset_index()

    return (summary)

flu_site = make_site(flu)
flu_site.to_csv('./results/summaries/flu_ves.csv',index=False)
