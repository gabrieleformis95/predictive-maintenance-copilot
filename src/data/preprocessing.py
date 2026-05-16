"""Preprocessing utilities: standardization and per-unit scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> StandardScaler:
    """Fit a StandardScaler on the training portion only."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns].to_numpy(dtype=np.float64))
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Return a copy of df with feature_columns scaled."""
    df = df.copy()
    df[feature_columns] = scaler.transform(df[feature_columns].to_numpy(dtype=np.float64))
    return df


def clip_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Clip RUL to a maximum value (common C-MAPSS preprocessing step).

    The intuition is that engines aren't really 'degrading' when far from
    failure, so capping RUL prevents the model from learning meaningless
    early-life regression targets.
    """
    df = df.copy()
    if "RUL" in df.columns:
        df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df
