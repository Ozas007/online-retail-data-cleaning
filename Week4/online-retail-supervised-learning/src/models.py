import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def create_logistic_regression(random_state: int = RANDOM_SEED) -> LogisticRegression:
    """
    Return an unfitted LogisticRegression with sensible default hyperparameters.

    Uses class_weight='balanced' to compensate for typically-imbalanced churn /
    repeat-purchase tasks, increases max_iter to ensure convergence on
    standardized features, and uses l2 regularization via liblinear solver
    (good for small-to-medium datasets like the UCI retail customer matrix).
    """
    logger.info("Creating LogisticRegression (balanced class weights, l2 penalty)")
    np.random.seed(random_state)
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
        max_iter=2000,
    )


def create_decision_tree(random_state: int = RANDOM_SEED) -> DecisionTreeClassifier:
    """
    Return an unfitted DecisionTreeClassifier.

    Depth + min_samples restrictions avoid the tree memorizing noise on small
    customer-level datasets. class_weight='balanced' counters class imbalance.
    """
    logger.info("Creating DecisionTreeClassifier (max_depth=8, balanced weights)")
    np.random.seed(random_state)
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
    )


def create_random_forest(
    random_state: int = RANDOM_SEED,
    n_estimators: int = 200,
) -> RandomForestClassifier:
    """
    Return an unfitted RandomForestClassifier.

    Default n_estimators=200 (as per the project spec). Restricted depth +
    balanced class weights for robustness on imbalanced prediction tasks.
    """
    logger.info(f"Creating RandomForestClassifier (n_estimators={n_estimators}, balanced weights)")
    np.random.seed(random_state)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
