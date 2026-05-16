"""NASA C-MAPSS turbofan engine degradation dataset loader.

The C-MAPSS benchmark contains four subsets (FD001-FD004) of simulated turbofan
engine run-to-failure trajectories. Each row is one operational cycle of one
engine with 21 sensor measurements + 3 operating settings.

Reference:
    Saxena, A., Goebel, K., Simon, D., Eklund, N.
    "Damage propagation modeling for aircraft engine run-to-failure simulation."
    IEEE Int. Conf. on Prognostics and Health Management, 2008.

Dataset URL:
    https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

The expected raw layout under `data/raw/CMAPSSData/` is:
    train_FD00X.txt
    test_FD00X.txt
    RUL_FD00X.txt        (true Remaining Useful Life for test trajectories)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.config import settings

SubsetName = Literal["FD001", "FD002", "FD003", "FD004"]

OPERATIONAL_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_FEATURE_COLUMNS = OPERATIONAL_SETTINGS + SENSOR_COLUMNS
COLUMN_NAMES = ["unit", "cycle", *ALL_FEATURE_COLUMNS]

# Sensors that are constant in FD001 (no information) — commonly dropped.
# Source: every C-MAPSS preprocessing reference uses essentially this list.
CONSTANT_SENSORS_FD001 = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


@dataclass
class CMAPSSDataset:
    """In-memory container for one C-MAPSS subset."""

    subset: SubsetName
    train: pd.DataFrame
    test: pd.DataFrame
    rul_test: pd.Series
    feature_columns: list[str]

    def __repr__(self) -> str:
        n_train_units = self.train["unit"].nunique()
        n_test_units = self.test["unit"].nunique()
        return (
            f"CMAPSSDataset(subset={self.subset!r}, "
            f"train_units={n_train_units}, test_units={n_test_units}, "
            f"features={len(self.feature_columns)})"
        )


def _read_cmapss_file(path: Path) -> pd.DataFrame:
    """Read a single C-MAPSS .txt file with whitespace separator."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )
    return df


def add_remaining_useful_life(
    df: pd.DataFrame,
    test_rul: pd.Series | None = None,
) -> pd.DataFrame:
    """Append a `RUL` column to a C-MAPSS DataFrame.

    For training data, RUL = max_cycle_for_this_unit - current_cycle.
    For test data, you must pass `test_rul` (one value per unit) — RUL at the
    last observed cycle is taken from there and decremented backwards.
    """
    df = df.copy()
    max_cycle_per_unit = df.groupby("unit")["cycle"].transform("max")

    if test_rul is None:
        df["RUL"] = max_cycle_per_unit - df["cycle"]
    else:
        # Test set: ground-truth RUL refers to the last cycle of each unit.
        rul_map = test_rul.to_dict()
        last_rul = df["unit"].map(rul_map)
        df["RUL"] = (max_cycle_per_unit - df["cycle"]) + last_rul

    return df


def select_feature_columns(
    df: pd.DataFrame,
    drop_constant_sensors: bool = True,
    subset: SubsetName = "FD001",
) -> list[str]:
    """Return the feature columns to use for modeling."""
    cols = list(ALL_FEATURE_COLUMNS)
    if drop_constant_sensors and subset == "FD001":
        cols = [c for c in cols if c not in CONSTANT_SENSORS_FD001]
    return cols


def load_cmapss(
    subset: SubsetName = "FD001",
    raw_dir: Path | None = None,
    drop_constant_sensors: bool = True,
) -> CMAPSSDataset:
    """Load one C-MAPSS subset.

    Parameters
    ----------
    subset: which of FD001..FD004 to load.
    raw_dir: location of CMAPSSData files. Defaults to settings.data_raw_dir / "CMAPSSData".
    drop_constant_sensors: drop sensors with no variance (only meaningful for FD001).
    """
    raw_dir = raw_dir or settings.data_raw_dir / "CMAPSSData"
    raw_dir = Path(raw_dir)

    train_path = raw_dir / f"train_{subset}.txt"
    test_path = raw_dir / f"test_{subset}.txt"
    rul_path = raw_dir / f"RUL_{subset}.txt"

    for p in (train_path, test_path, rul_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing C-MAPSS file: {p}. Run `make download-data` to fetch the dataset."
            )

    train = _read_cmapss_file(train_path)
    test = _read_cmapss_file(test_path)
    rul_test_values = np.loadtxt(rul_path)
    # The RUL file lists one value per unit, in unit order (1..N).
    rul_test = pd.Series(rul_test_values, index=range(1, len(rul_test_values) + 1), name="RUL")

    train = add_remaining_useful_life(train)
    test = add_remaining_useful_life(test, test_rul=rul_test)

    features = select_feature_columns(train, drop_constant_sensors, subset)

    return CMAPSSDataset(
        subset=subset,
        train=train,
        test=test,
        rul_test=rul_test,
        feature_columns=features,
    )


def make_sliding_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows of multivariate sensor data, per unit.

    Returns
    -------
    X: array of shape (n_windows, window_size, n_features)
    unit_ids: array of shape (n_windows,) — the unit each window came from.
    """
    windows: list[np.ndarray] = []
    unit_ids: list[int] = []

    for unit_id, group in df.groupby("unit"):
        values = group[feature_columns].to_numpy(dtype=np.float32)
        n_rows = len(values)
        if n_rows < window_size:
            continue
        for start in range(0, n_rows - window_size + 1, stride):
            windows.append(values[start : start + window_size])
            unit_ids.append(int(unit_id))

    if not windows:
        return np.empty((0, window_size, len(feature_columns)), dtype=np.float32), np.empty(
            0, dtype=int
        )

    return np.stack(windows), np.array(unit_ids)
