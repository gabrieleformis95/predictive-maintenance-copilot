"""Smoke tests for data loaders that don't require the real dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import (
    ALL_FEATURE_COLUMNS,
    add_remaining_useful_life,
    make_sliding_windows,
    select_feature_columns,
)


def _toy_cmapss(n_units: int = 3, cycles_per_unit: int = 80, n_features: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            row = {"unit": unit, "cycle": cycle}
            for col in ALL_FEATURE_COLUMNS:
                row[col] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


def test_add_rul_for_train_set_is_max_minus_cycle():
    """GIVEN a toy C-MAPSS DataFrame with 3 units of 80 cycles each.
    WHEN add_remaining_useful_life is called without test_rul.
    THEN the last cycle of each unit has RUL=0 and the first cycle has RUL=max_cycle-1.
    """
    df = _toy_cmapss()
    out = add_remaining_useful_life(df)
    last = out[out["unit"] == 1].sort_values("cycle").iloc[-1]
    assert last["RUL"] == 0
    first = out[out["unit"] == 1].sort_values("cycle").iloc[0]
    assert first["RUL"] == out[out["unit"] == 1]["cycle"].max() - 1


def test_select_feature_columns_drops_constants_for_fd001():
    """GIVEN a toy C-MAPSS DataFrame and drop_constant_sensors=True for FD001.
    WHEN select_feature_columns is called.
    THEN sensor_1 (constant in FD001) is excluded and sensor_2 is retained.
    """
    feats = select_feature_columns(_toy_cmapss(), drop_constant_sensors=True, subset="FD001")
    assert "sensor_1" not in feats
    assert "sensor_2" in feats


def test_make_sliding_windows_shapes():
    """GIVEN a DataFrame with 2 units of 100 cycles and window_size=30.
    WHEN make_sliding_windows is called.
    THEN the output arrays have shape (n_windows, 30, n_features) and (n_windows,).
    """
    df = _toy_cmapss(n_units=2, cycles_per_unit=100)
    feats = select_feature_columns(df, drop_constant_sensors=False)
    x, units = make_sliding_windows(df, feats, window_size=30, stride=1)
    expected = 2 * (100 - 30 + 1)
    assert x.shape == (expected, 30, len(feats))
    assert units.shape == (expected,)
    assert set(units.tolist()) == {1, 2}


def test_make_sliding_windows_skips_short_units():
    """GIVEN a DataFrame with 1 unit of only 10 cycles and window_size=30.
    WHEN make_sliding_windows is called.
    THEN both output arrays are empty (unit is too short to produce any window).
    """
    df = _toy_cmapss(n_units=1, cycles_per_unit=10)
    feats = select_feature_columns(df, drop_constant_sensors=False)
    x, units = make_sliding_windows(df, feats, window_size=30)
    assert x.shape[0] == 0
    assert units.shape[0] == 0
