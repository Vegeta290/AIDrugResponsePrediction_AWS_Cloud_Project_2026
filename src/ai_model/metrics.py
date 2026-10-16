"""Regression and sensitivity-classification metrics for drug response prediction."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if len(y_true) > 2 and np.std(y_pred) > 0:
        out["pearson"] = float(pearsonr(y_true, y_pred)[0])
        out["spearman"] = float(spearmanr(y_true, y_pred)[0])
    return out


def classification_metrics(y_true, y_pred, drug_ids, thresholds):
    """A pair is 'sensitive' when ln(IC50) is below the drug's training-set median."""
    thr = np.array([thresholds.get(d, np.nan) for d in drug_ids], dtype=float)
    ok = ~np.isnan(thr)
    y_true, y_pred, thr = np.asarray(y_true)[ok], np.asarray(y_pred)[ok], thr[ok]
    true_sens = (y_true < thr).astype(int)
    pred_sens = (y_pred < thr).astype(int)
    out = {
        "accuracy": float(accuracy_score(true_sens, pred_sens)),
        "precision": float(precision_score(true_sens, pred_sens, zero_division=0)),
        "recall": float(recall_score(true_sens, pred_sens, zero_division=0)),
        "f1": float(f1_score(true_sens, pred_sens, zero_division=0)),
    }
    if len(set(true_sens)) == 2:
        # Lower predicted IC50 relative to the drug threshold = more sensitive.
        out["roc_auc"] = float(roc_auc_score(true_sens, thr - y_pred))
    return out


def per_drug_pearson(y_true, y_pred, drug_ids, min_samples=10):
    y_true, y_pred, drug_ids = np.asarray(y_true), np.asarray(y_pred), np.asarray(drug_ids)
    scores = {}
    for d in np.unique(drug_ids):
        sel = drug_ids == d
        if sel.sum() >= min_samples and np.std(y_pred[sel]) > 0 and np.std(y_true[sel]) > 0:
            scores[str(d)] = float(pearsonr(y_true[sel], y_pred[sel])[0])
    return scores


def full_report(y_true, y_pred, drug_ids, thresholds):
    report = regression_metrics(y_true, y_pred)
    report.update(classification_metrics(y_true, y_pred, drug_ids, thresholds))
    per_drug = per_drug_pearson(y_true, y_pred, drug_ids)
    if per_drug:
        report["mean_per_drug_pearson"] = float(np.mean(list(per_drug.values())))
    report["n"] = int(len(y_true))
    return report, per_drug
