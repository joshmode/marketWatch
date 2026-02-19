import pytest
import pandas as pd
import numpy as np
from app.backtest import run_backtest

@pytest.fixture
def backtest_data(mock_market_data):
    # Add regime columns
    df = mock_market_data.copy()
    df['P_Expansion'] = 0.5
    df['P_Slowdown'] = 0.3
    df['P_Stress'] = 0.2
    # Add dummy ML score if needed, handled inside function
    # But function calls get_historical_predictions from ml_engine
    # We should mock that to avoid complex dependency
    return df

from unittest.mock import patch

@patch("app.backtest.get_historical_predictions")
def test_run_backtest(mock_get_preds, backtest_data):
    # Mock ML predictions as a series of 0s
    mock_get_preds.return_value = pd.Series(0, index=backtest_data.index)

    result = run_backtest(backtest_data)

    assert 'Returns' in result.columns
    assert 'Base_Signal' in result.columns
    assert 'Signal' in result.columns
    assert 'Strategy_Returns' in result.columns
    assert 'Cumulative_Strategy' in result.columns
    assert 'Drawdown' in result.columns
