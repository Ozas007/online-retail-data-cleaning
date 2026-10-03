from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"
MODELS_DIR: Path = OUTPUTS_DIR / "models"

REPORT_DIR: Path = PROJECT_ROOT / "report"
REPORT_DOCX: Path = REPORT_DIR / "Week_5_Deep_Learning_Report.docx"

MODEL_SAVE_PATH: Path = MODELS_DIR / "mnist_mlp.keras"

RANDOM_SEED: int = 42
FIGURE_DPI: int = 200

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

INPUT_SHAPE: tuple = (28, 28)
NUM_CLASSES: int = 10
INPUT_DIM: int = 28 * 28

EPOCHS: int = 15
BATCH_SIZE: int = 128
VALIDATION_SPLIT: float = 0.1
LEARNING_RATE: float = 0.001
EARLY_STOPPING_PATIENCE: int = 3
DROPOUT_RATE: float = 0.2

DENSE1_UNITS: int = 128
DENSE2_UNITS: int = 64
