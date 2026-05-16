"""Smoke tests for data loaders that don't require the real dataset."""

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
    df = _toy_cmapss()
    out = add_remaining_useful_life(df)
    # For unit 1, last cycle's RUL must be 0.
    last = out[(out["unit"] == 1)].sort_values("cycle").iloc[-1]
    assert last["RUL"] == 0
    # First cycle's RUL = max_cycle - 1.
    first = out[(out["unit"] == 1)].sort_values("cycle").iloc[0]
    assert first["RUL"] == first["cycle"] * 0 + (out[out["unit"] == 1]["cycle"].max() - 1)


def test_select_feature_columns_drops_constants_for_fd001():
    feats = select_feature_columns(_toy_cmapss(), drop_constant_sensors=True, subset="FD001")
    assert "sensor_1" not in feats
    assert "sensor_2" in feats


def test_make_sliding_windows_shapes():
    df = _toy_cmapss(n_units=2, cycles_per_unit=100)
    feats = select_feature_columns(df, drop_constant_sensors=False)
    x, units = make_sliding_windows(df, feats, window_size=30, stride=1)
    expected_windows_per_unit = 100 - 30 + 1
    assert x.shape == (2 * expected_windows_per_unit, 30, len(feats))
    assert units.shape == (2 * expected_windows_per_unit,)
    assert set(units.tolist()) == {1, 2}


def test_make_sliding_windows_skips_short_units():
    df = _toy_cmapss(n_units=1, cycles_per_unit=10)
    feats = select_feature_columns(df, drop_constant_sensors=False)
    x, units = make_sliding_windows(df, feats, window_size=30)
    assert x.shape[0] == 0
    assert units.shape[0] == 0
