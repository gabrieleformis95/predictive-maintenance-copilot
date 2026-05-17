"""Tests for src/data/loaders.py using synthetic C-MAPSS-format files."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loaders import (
    COLUMN_NAMES,
    CMAPSSDataset,
    _read_cmapss_file,
    add_remaining_useful_life,
    load_cmapss,
)


def test_read_cmapss_file_shape(cmapss_dir):
    """GIVEN a valid C-MAPSS train file with 3 units x 50 cycles.
    WHEN _read_cmapss_file is called.
    THEN it returns a DataFrame with 26 columns and 150 rows.
    """
    df = _read_cmapss_file(cmapss_dir / "train_FD001.txt")
    assert df.shape == (150, len(COLUMN_NAMES))


def test_read_cmapss_file_column_names(cmapss_dir):
    """GIVEN a valid C-MAPSS train file.
    WHEN _read_cmapss_file is called.
    THEN the returned DataFrame has the standard C-MAPSS column names.
    """
    df = _read_cmapss_file(cmapss_dir / "train_FD001.txt")
    assert list(df.columns) == COLUMN_NAMES


def test_load_cmapss_returns_dataset(cmapss_dir):
    """GIVEN valid C-MAPSS files for FD001.
    WHEN load_cmapss is called.
    THEN it returns a CMAPSSDataset with RUL columns on both train and test.
    """
    ds = load_cmapss("FD001", raw_dir=cmapss_dir)
    assert isinstance(ds, CMAPSSDataset)
    assert ds.subset == "FD001"
    assert "RUL" in ds.train.columns
    assert "RUL" in ds.test.columns


def test_load_cmapss_train_last_rul_is_zero(cmapss_dir):
    """GIVEN a loaded training set.
    WHEN checking the last cycle RUL for each unit.
    THEN RUL equals 0 (engine reached end of life).
    """
    ds = load_cmapss("FD001", raw_dir=cmapss_dir)
    for _, group in ds.train.groupby("unit"):
        last_rul = group.sort_values("cycle").iloc[-1]["RUL"]
        assert last_rul == 0


def test_load_cmapss_drop_constant_sensors(cmapss_dir):
    """GIVEN FD001 dataset loaded with drop_constant_sensors=True.
    WHEN inspecting feature_columns.
    THEN sensor_1 is excluded and sensor_2 is retained.
    """
    ds = load_cmapss("FD001", raw_dir=cmapss_dir, drop_constant_sensors=True)
    assert "sensor_1" not in ds.feature_columns
    assert "sensor_2" in ds.feature_columns


def test_load_cmapss_missing_file_raises(tmp_path):
    """GIVEN a directory with no C-MAPSS files.
    WHEN load_cmapss is called.
    THEN FileNotFoundError is raised.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_cmapss("FD001", raw_dir=empty)


def test_cmapss_dataset_repr(cmapss_dir):
    """GIVEN a loaded CMAPSSDataset.
    WHEN repr() is called.
    THEN the string includes the subset name and unit counts.
    """
    ds = load_cmapss("FD001", raw_dir=cmapss_dir)
    r = repr(ds)
    assert "FD001" in r
    assert "train_units" in r


def test_add_remaining_useful_life_test_set():
    """GIVEN a test DataFrame with 2 units and a ground-truth RUL series.
    WHEN add_remaining_useful_life is called with test_rul.
    THEN the last cycle of each unit gets exactly the ground-truth RUL value.
    """
    df = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2]})
    test_rul = pd.Series({1: 10.0, 2: 20.0})
    out = add_remaining_useful_life(df, test_rul=test_rul)
    assert out[out["unit"] == 1].sort_values("cycle").iloc[-1]["RUL"] == pytest.approx(10.0)
    assert out[out["unit"] == 2].sort_values("cycle").iloc[-1]["RUL"] == pytest.approx(20.0)
