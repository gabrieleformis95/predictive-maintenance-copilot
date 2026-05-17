"""Tests for src/data/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import apply_scaler, clip_rul, fit_scaler


def _make_df(n: int = 50) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(0)
    cols = ["f1", "f2", "f3"]
    data = rng.normal(loc=[10.0, -5.0, 100.0], scale=[2.0, 1.0, 20.0], size=(n, 3))
    return pd.DataFrame(data, columns=cols), cols


def test_fit_scaler_returns_fitted_scaler():
    """GIVEN a training DataFrame with 3 feature columns.
    WHEN fit_scaler is called.
    THEN a StandardScaler fitted on those columns is returned.
    """
    df, cols = _make_df()
    scaler = fit_scaler(df, cols)
    assert hasattr(scaler, "mean_")
    assert scaler.mean_.shape == (3,)


def test_apply_scaler_zero_means_training_data():
    """GIVEN a fitted scaler and the same training DataFrame.
    WHEN apply_scaler is called.
    THEN the mean of each feature column is approximately zero.
    """
    df, cols = _make_df()
    scaler = fit_scaler(df, cols)
    scaled = apply_scaler(df, cols, scaler)
    assert (scaled[cols].mean().abs() < 1e-10).all()


def test_apply_scaler_does_not_mutate_input():
    """GIVEN a DataFrame and a fitted scaler.
    WHEN apply_scaler is called.
    THEN the original DataFrame is unchanged.
    """
    df, cols = _make_df()
    scaler = fit_scaler(df, cols)
    original = df[cols].copy()
    apply_scaler(df, cols, scaler)
    pd.testing.assert_frame_equal(df[cols], original)


def test_clip_rul_clips_above_max():
    """GIVEN a DataFrame with RUL values [0, 50, 125, 150, 200] and max_rul=125.
    WHEN clip_rul is called.
    THEN values above 125 are capped and values below are unchanged.
    """
    df = pd.DataFrame({"RUL": [0, 50, 125, 150, 200]})
    out = clip_rul(df, max_rul=125)
    assert out["RUL"].tolist() == [0, 50, 125, 125, 125]


def test_clip_rul_no_rul_column_is_noop():
    """GIVEN a DataFrame with no RUL column.
    WHEN clip_rul is called.
    THEN the DataFrame is returned unchanged.
    """
    df = pd.DataFrame({"sensor_1": [1.0, 2.0]})
    out = clip_rul(df)
    pd.testing.assert_frame_equal(out, df)


def test_clip_rul_does_not_mutate_input():
    """GIVEN a DataFrame with RUL values [200, 300].
    WHEN clip_rul is called.
    THEN the original DataFrame is not modified.
    """
    df = pd.DataFrame({"RUL": [200, 300]})
    clip_rul(df)
    assert df["RUL"].tolist() == [200, 300]
