import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from app.ml_engine import (
    prepare_features, build_targets, purged_time_series_split,
    get_historical_predictions, predict_latest_score,
    MODEL_PATH
)
from app.indicators import add_indicators

@pytest.fixture
def feature_data(mock_market_data):
    return add_indicators(mock_market_data)

def test_prepare_features(feature_data):
    processed = prepare_features(feature_data)
    assert 'RSI_14_Z' in processed.columns
    # Check if NaN rows are dropped (LOOKBACK_WINDOW is 252, so should be smaller)
    # mock_market_data has 300 rows.
    # 300 - 252 + 1 = 49 rows approx?
    # Actually rolling(252) produces 251 NaNs. So index 251 is the first valid value.
    # So we expect roughly 300 - 251 = 49 rows.
    assert len(processed) < len(feature_data)
    assert not processed.isnull().any().any()

def test_build_targets(feature_data):
    # Need to run preprocess first to get clean data, but build_targets works on market data + Next_Ret
    # But build_targets computes rolling stats on Close pct_change with window 252.
    # So it will also drop rows.

    # Let's use a longer mock data for this test to ensure we have enough data
    dates = pd.date_range(start="2015-01-01", periods=1000, freq="D")
    df = pd.DataFrame({
        "Close": np.random.rand(1000) * 100 + np.linspace(0, 100, 1000)
    }, index=dates)

    targets = build_targets(df)
    assert 'Target_Class' in targets.columns
    assert targets['Target_Class'].between(0, 4).all()

def test_purged_time_series_split():
    n_samples = 100
    train_window = 20
    test_window = 5
    purge = 2

    splits = list(purged_time_series_split(n_samples, train_window, test_window, purge))
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert len(train_idx) == train_window
        assert len(test_idx) == test_window
        assert max(train_idx) < min(test_idx) - (purge - 1) # simple check

@patch("app.ml_engine.fetch_data")
@patch("app.ml_engine.joblib.dump")
def test_train_model_runs(mock_dump, mock_fetch, mock_market_data):
    # Mock data return
    # Need enough data for rolling windows (warmup + features + targets + cv splits)
    dates = pd.date_range(start="2015-01-01", periods=2000, freq="D")
    df = pd.DataFrame({
        "Open": np.random.rand(2000) * 100,
        "High": np.random.rand(2000) * 100,
        "Low": np.random.rand(2000) * 100,
        "Close": np.random.rand(2000) * 100,
        "Volume": np.random.randint(100, 1000, 2000)
    }, index=dates)
    mock_fetch.return_value = df

    from app.ml_engine import train_model
    model = train_model()
    assert model is not None
    mock_dump.assert_called()

@patch("app.ml_engine.fetch_data")
@patch("app.ml_engine.joblib.dump")
def test_train_model_small_data(mock_dump, mock_fetch, mock_market_data):
    # Mock data return with very few rows
    dates = pd.date_range(start="2015-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Open": np.random.rand(10) * 100,
        "High": np.random.rand(10) * 100,
        "Low": np.random.rand(10) * 100,
        "Close": np.random.rand(10) * 100,
        "Volume": np.random.randint(100, 1000, 10)
    }, index=dates)
    mock_fetch.return_value = df

    from app.ml_engine import train_model
    model = train_model()
    # It should return None because data is insufficient
    assert model is None
    mock_dump.assert_not_called()

def test_get_historical_predictions(feature_data):
    # Need more data
    dates = pd.date_range(start="2015-01-01", periods=600, freq="D")
    df = pd.DataFrame({
        "Open": np.random.rand(600) * 100,
        "High": np.random.rand(600) * 100,
        "Low": np.random.rand(600) * 100,
        "Close": np.random.rand(600) * 100,
        "Volume": np.random.randint(100, 1000, 600)
    }, index=dates)
    df = add_indicators(df)

    preds = get_historical_predictions(df)
    assert len(preds) == len(df)
    # Values should be mostly 0 where data was insufficient, and non-zero later
    # But get_historical_predictions returns a Series aligned to input index.
