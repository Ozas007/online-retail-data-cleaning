import logging
import random
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    TABLES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    REPORT_DOCX,
    REPORT_DIR,
    LOG_FORMAT,
    INPUT_SHAPE,
    NUM_CLASSES,
    INPUT_DIM,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    DROPOUT_RATE,
    DENSE1_UNITS,
    DENSE2_UNITS,
    MODEL_SAVE_PATH,
)
from src.data_loading import load_mnist
from src.data_validation import validate_data_shapes, check_class_balance
from src.preprocessing import preprocess_images, preprocess_labels_categorical
from src.model import build_mlp_model, get_model_summary_string
from src.training import train_model
from src.evaluation import evaluate_trained_model
from src.visualization import run_all_visualizations
from src.report_generation import build_report

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "pipeline.log", mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )


def _ensure_directories() -> None:
    for d in (FIGURES_DIR, TABLES_DIR, MODELS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directories exist under {PROJECT_ROOT}")


def _set_global_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Could not set TF seed: {exc!r}")
    logger.info(f"Set global seeds (random/numpy/tf) = {seed}")


def _history_to_df(history) -> pd.DataFrame:
    if hasattr(history, "history"):
        return pd.DataFrame(history.history)
    return pd.DataFrame(history)


def _save_tables(
    *,
    class_balance_train: pd.DataFrame,
    class_balance_test: pd.DataFrame,
    history_df: pd.DataFrame,
    confusion_matrix_df: pd.DataFrame,
    eval_metrics: Dict[str, Any],
) -> None:
    class_balance_train.to_csv(TABLES_DIR / "01_class_balance_train.csv", index=False)
    class_balance_test.to_csv(TABLES_DIR / "02_class_balance_test.csv", index=False)
    history_df.to_csv(TABLES_DIR / "03_training_history.csv", index_label="epoch")

    per_class_df = pd.DataFrame({
        "label": list(range(len(eval_metrics["per_class_f1"]))),
        "f1": eval_metrics["per_class_f1"],
    })
    per_class_df.to_csv(TABLES_DIR / "04_per_class_f1.csv", index=False)

    flat_metrics = {k: v for k, v in eval_metrics.items() if k != "confusion_matrix_df" and k != "per_class_f1"}
    flat_metrics["overfitting_gap"] = None
    flat_metrics["best_epoch"] = None
    pd.DataFrame([flat_metrics]).to_csv(TABLES_DIR / "05_evaluation_metrics.csv", index=False)

    confusion_matrix_df.to_csv(TABLES_DIR / "06_confusion_matrix.csv")

    logger.info(f"Saved CSV tables to {TABLES_DIR}")


def main() -> Path:
    _setup_logging()
    _ensure_directories()
    _set_global_seeds(RANDOM_SEED)

    logger.info("=" * 70)
    logger.info("WEEK 5 — DEEP LEARNING MLP ON MNIST (TensorFlow/Keras only)")
    logger.info("=" * 70)
    logger.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
    logger.info(f"Random seed = {RANDOM_SEED}")
    logger.info(f"Hyperparams: epochs={EPOCHS}, batch={BATCH_SIZE}, "
                f"val_split={VALIDATION_SPLIT}, lr={LEARNING_RATE}, "
                f"patience={EARLY_STOPPING_PATIENCE}, dropout={DROPOUT_RATE}")

    logger.info("STEP 1: Load MNIST dataset")
    (X_train_raw, y_train), (X_test_raw, y_test) = load_mnist()
    n_train, n_test = len(X_train_raw), len(X_test_raw)

    logger.info("STEP 2: Validate shapes and label ranges")
    validate_data_shapes(X_train_raw, y_train, X_test_raw, y_test, input_shape=INPUT_SHAPE)
    class_balance_train = check_class_balance(y_train, num_classes=NUM_CLASSES)
    class_balance_test = check_class_balance(y_test, num_classes=NUM_CLASSES)

    logger.info("STEP 3: Preprocess images (flatten + normalize) and labels (one-hot)")
    X_train = preprocess_images(X_train_raw)
    X_test = preprocess_images(X_test_raw)
    y_train_cat = preprocess_labels_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_cat = preprocess_labels_categorical(y_test, num_classes=NUM_CLASSES)
    logger.info(f"X_train shape after preprocess = {X_train.shape}  X_test = {X_test.shape}")

    logger.info("STEP 4: Build MLP model")
    model = build_mlp_model(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        dense1_units=DENSE1_UNITS,
        dense2_units=DENSE2_UNITS,
        learning_rate=LEARNING_RATE,
    )
    model_summary = get_model_summary_string(model)
    logger.info("Model summary:\n" + model_summary)
    total_params = int(model.count_params())

    logger.info("STEP 5: Train model (with EarlyStopping + optional checkpoint)")
    history = train_model(
        model=model,
        X_train=X_train,
        y_train_cat=y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        patience=EARLY_STOPPING_PATIENCE,
        seed=RANDOM_SEED,
        checkpoint_path=None,
        verbose=1,
    )
    history_df = _history_to_df(history)
    final_train_acc = float(history_df["accuracy"].iloc[-1])
    final_val_acc = float(history_df["val_accuracy"].iloc[-1])
    overfitting_gap = final_train_acc - final_val_acc
    best_epoch = 1 + int(np.argmin(history_df["val_loss"].values))
    logger.info(f"Best epoch (min val_loss) = {best_epoch} | overfitting gap (train_acc - val_acc) = {overfitting_gap:+.4f}")

    logger.info("STEP 6: Evaluate on held-out test set")
    eval_metrics = evaluate_trained_model(model, X_test, y_test, y_test_cat, num_classes=NUM_CLASSES)
    confusion_matrix_df: pd.DataFrame = eval_metrics["confusion_matrix_df"]
    logger.info(
        f"Test results: loss={eval_metrics['test_loss']:.4f}, acc={eval_metrics['test_accuracy']:.4f}, "
        f"P_w={eval_metrics['precision_weighted']:.4f}, R_w={eval_metrics['recall_weighted']:.4f}, "
        f"F1_w={eval_metrics['f1_weighted']:.4f}"
    )
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    logger.info("STEP 7: Save trained Keras model")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_SAVE_PATH))
    logger.info(f"Model saved at: {MODEL_SAVE_PATH}  (size = {MODEL_SAVE_PATH.stat().st_size / 1024:.1f} KB)")

    logger.info("STEP 8: Generate all visualization PNGs (architecture diagram + curves + CM + samples)")
    paths_figs = run_all_visualizations(
        history=history,
        confusion_matrix_df=confusion_matrix_df,
        model=model,
        X_test_raw=X_test_raw,
        y_true=y_test,
        y_pred=y_pred,
    )

    logger.info("STEP 9: Save CSV tables (class balance, history, CM, per-class F1, eval metrics)")
    _save_tables(
        class_balance_train=class_balance_train,
        class_balance_test=class_balance_test,
        history_df=history_df,
        confusion_matrix_df=confusion_matrix_df,
        eval_metrics=eval_metrics,
    )

    hyperparams = {
        "INPUT_SHAPE": INPUT_SHAPE,
        "INPUT_DIM": INPUT_DIM,
        "NUM_CLASSES": NUM_CLASSES,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "VALIDATION_SPLIT": VALIDATION_SPLIT,
        "LEARNING_RATE": LEARNING_RATE,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "DROPOUT_RATE": DROPOUT_RATE,
        "DENSE1_UNITS": DENSE1_UNITS,
        "DENSE2_UNITS": DENSE2_UNITS,
        "RANDOM_SEED": RANDOM_SEED,
        "OPTIMIZER": "Adam",
        "LOSS": "CategoricalCrossentropy",
        "TOTAL_PARAMS": total_params,
    }

    logger.info("STEP 10: Build 26-section DOCX report")
    report_path = build_report(
        eval_metrics=eval_metrics,
        history_df=history_df,
        confusion_matrix_df=confusion_matrix_df,
        class_balance_train_df=class_balance_train,
        model_summary=model_summary,
        paths_figs=paths_figs,
        hyperparams=hyperparams,
        n_train=n_train,
        n_test=n_test,
        overfitting_gap=overfitting_gap,
        final_train_acc=final_train_acc,
        final_val_acc=final_val_acc,
        best_epoch=best_epoch,
        output_path=REPORT_DOCX,
    )

    logger.info("=" * 70)
    logger.info("WEEK 5 PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Train samples:                {n_train:,}")
    logger.info(f"Test samples:                 {n_test:,}")
    logger.info(f"Total MLP parameters:         {total_params:,}")
    logger.info(f"Epochs run / best epoch:      {len(history_df)} / {best_epoch}")
    logger.info(f"Final train accuracy:         {final_train_acc:.4f}")
    logger.info(f"Final val accuracy:           {final_val_acc:.4f}")
    logger.info(f"Overfitting gap (t-v):        {overfitting_gap:+.4f}")
    logger.info(f"Test accuracy:                {eval_metrics['test_accuracy']:.4f}")
    logger.info(f"Test loss (CCE):              {eval_metrics['test_loss']:.4f}")
    logger.info(f"Weighted F1:                  {eval_metrics['f1_weighted']:.4f}")
    logger.info(f"Saved Keras model:            {MODEL_SAVE_PATH}")
    logger.info(f"Figures directory:            {FIGURES_DIR}  ({len(list(FIGURES_DIR.glob('*.png')))} files)")
    logger.info(f"Tables directory:             {TABLES_DIR}  ({len(list(TABLES_DIR.glob('*.csv')))} files)")
    logger.info(f"Report saved at:              {report_path}")
    report_size_kb = report_path.stat().st_size / 1024 if report_path.exists() else 0
    logger.info(f"Report file size:             {report_size_kb:.1f} KB")
    return report_path


if __name__ == "__main__":
    main()
