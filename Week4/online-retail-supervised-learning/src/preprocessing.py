import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def get_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer:
      - numeric features -> StandardScaler
      - categorical features -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Both transformers are "passthrough-safe" for empty feature lists.
    """
    logger.info(
        f"Building ColumnTransformer — numeric={numeric_features}  |  "
        f"categorical={categorical_features}"
    )
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ))
    if not transformers:
        raise ValueError("No numeric or categorical features supplied to preprocessor")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def build_full_pipeline(
    classifier_model,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Assemble [ColumnTransformer -> classifier_model] into a single Pipeline.

    Parameters
    ----------
    classifier_model : sklearn estimator (unfitted)
        E.g. LogisticRegression / DecisionTreeClassifier / RandomForestClassifier from models.py
    preprocessor : ColumnTransformer
        Output of get_preprocessor().

    Returns
    -------
    sklearn.Pipeline unfitted.
    """
    logger.info(
        f"Building sklearn Pipeline: preprocessor -> "
        f"{classifier_model.__class__.__name__}"
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier_model),
    ])
    return pipeline
