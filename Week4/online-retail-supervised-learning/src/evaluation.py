import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def evaluate_classifier(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics dict from predictions.

    Returns dict with keys: accuracy, precision, recall, f1, (roc_auc if y_proba given).
    Macro averaging is used for precision/recall/f1 so minority class performance
    is not masked by a dominant majority class.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
    }

    pos_count = int((y_true_arr == 1).sum())
    neg_count = int((y_true_arr == 0).sum())
    metrics["support_positive"] = pos_count
    metrics["support_negative"] = neg_count
    total = pos_count + neg_count
    metrics["imbalance_ratio_neg_to_pos"] = (neg_count / pos_count) if pos_count > 0 else float("nan")

    if y_proba is not None:
        proba = np.asarray(y_proba)
        if proba.ndim == 2:
            if proba.shape[1] >= 2:
                proba_use = proba[:, 1]
            else:
                proba_use = proba[:, 0]
        else:
            proba_use = proba
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, proba_use))
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics["roc_auc"] = float("nan")

    logger.info(
        f"Evaluation — acc={metrics['accuracy']:.3f}  "
        f"prec={metrics['precision']:.3f}  "
        f"rec={metrics['recall']:.3f}  "
        f"f1={metrics['f1']:.3f}"
        + (f"  roc_auc={metrics['roc_auc']:.3f}" if "roc_auc" in metrics else "")
    )
    return metrics


def build_confusion_matrix_df(y_true, y_pred) -> pd.DataFrame:
    """
    Build a 2x2 (or nxn) confusion matrix as a labelled DataFrame.

    Rows = actual (True), Columns = predicted (Pred).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    labels = sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])).tolist())
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    row_index = pd.MultiIndex.from_tuples(
        [("Actual", int(lbl)) for lbl in labels],
        names=["Condition", "Label"],
    )
    col_index = pd.MultiIndex.from_tuples(
        [("Predicted", int(lbl)) for lbl in labels],
        names=["Prediction", "Label"],
    )
    return pd.DataFrame(cm, index=row_index, columns=col_index)


def run_cross_validation(
    X: pd.DataFrame,
    y,
    pipeline,
    cv: int = 5,
    random_state: int = RANDOM_SEED,
    scoring: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run stratified K-fold CV on the given pipeline, returning a per-fold metrics
    DataFrame plus an 'overall' summary row.

    Parameters
    ----------
    X : pd.DataFrame (n_samples, n_features) — raw, untransformed feature frame
        (the pipeline contains the preprocessor step).
    y : array-like target.
    pipeline : sklearn Pipeline (preprocessor + classifier).
    cv : int — number of stratified folds.
    random_state : int — seed for StratifiedKFold shuffle.
    scoring : optional list of sklearn scorer names; defaults to a
        classification set.

    Returns
    -------
    pd.DataFrame with a row per fold + 'overall' summary row.
    """
    logger.info(f"Running {cv}-fold stratified cross-validation")
    np.random.seed(random_state)

    if scoring is None:
        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    y_arr = np.asarray(y).astype(int)

    results = cross_validate(
        pipeline,
        X,
        y_arr,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        error_score="raise",
    )

    fold_df = pd.DataFrame()
    for key in scoring:
        res_key = f"test_{key}"
        short = key.replace("_macro", "")
        fold_df[short] = results[res_key]

    fold_df.insert(0, "fold", np.arange(1, len(fold_df) + 1))

    summary = {"fold": "overall"}
    for col in [c for c in fold_df.columns if c != "fold"]:
        vals = fold_df[col].values
        summary[f"{col}_mean"] = float(np.mean(vals))
        summary[f"{col}_std"] = float(np.std(vals, ddof=0))

    summary_df = pd.DataFrame([summary])
    fold_df = pd.concat([fold_df, summary_df], ignore_index=True)

    overall_row = fold_df.loc[fold_df["fold"] == "overall"].iloc[0]
    logger.info(
        f"  CV overall — accuracy={overall_row.get('accuracy_mean', float('nan')):.3f}  "
        f"f1={overall_row.get('f1_mean', float('nan')):.3f}  "
        f"roc_auc={overall_row.get('roc_auc_mean', float('nan')):.3f}"
    )
    return fold_df
