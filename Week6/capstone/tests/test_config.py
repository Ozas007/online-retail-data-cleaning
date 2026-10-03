from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    EXPECTED_COLUMNS,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    RANDOM_SEED,
)


def test_project_root_exists():
    assert isinstance(PROJECT_ROOT, Path)
    assert (PROJECT_ROOT / "src").is_dir()


def test_raw_data_path_is_resolved():
    assert RAW_DATA_PATH is not None
    assert isinstance(RANDOM_SEED, int)
    assert RANDOM_SEED == 42
    assert len(EXPECTED_COLUMNS) >= 8
