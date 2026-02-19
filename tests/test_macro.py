import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from app.macro import load_macro_data, enrich_macro_data

@pytest.fixture
def mock_fetch_series():
    with patch("app.macro._fetch_series") as mock:
        yield mock

@patch("app.macro.FRED_API_KEY", "fake_key")
def test_load_macro_data(mock_fetch_series):
    # Mock return values for each series
    mock_series = pd.Series(np.random.rand(100), index=pd.date_range("2020-01-01", periods=100))
    mock_fetch_series.return_value = mock_series

    df = load_macro_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "growth" in df.columns # Check if keys from SERIES are present

@patch("app.macro.FRED_API_KEY", "fake_key")
def test_enrich_macro_data(mock_fetch_series, mock_market_data):
    # Mock macro data return
    mock_series = pd.Series(np.random.rand(300), index=pd.date_range("2020-01-01", periods=300))
    mock_fetch_series.return_value = mock_series

    # Ensure market data index aligns somewhat
    enriched = enrich_macro_data(mock_market_data)

    expected_cols = [
        'yield_curve', 'curve_inversion_depth', 'inflation_yoy',
        'macro_score', 'recession_probability'
    ]
    for col in expected_cols:
        assert col in enriched.columns
