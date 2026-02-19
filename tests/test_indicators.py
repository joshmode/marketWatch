import pandas as pd
import numpy as np
from app.indicators import (
    calculate_sma, calculate_rsi, calculate_atr, calculate_volatility_z_score,
    calculate_macd, calculate_bollinger_bands, calculate_momentum,
    calculate_momentum_drift, calculate_market_stress, calculate_fear_greed_proxy,
    add_indicators
)

def test_calculate_sma(mock_market_data):
    sma = calculate_sma(mock_market_data, window=10)
    assert len(sma) == len(mock_market_data)
    assert sma.iloc[0:9].isnull().all()
    assert not sma.iloc[9:].isnull().any()

def test_calculate_rsi(mock_market_data):
    rsi = calculate_rsi(mock_market_data, window=14)
    assert len(rsi) == len(mock_market_data)
    # RSI should be between 0 and 100
    assert rsi.min() >= 0
    assert rsi.max() <= 100

def test_calculate_atr(mock_market_data):
    atr = calculate_atr(mock_market_data, window=14)
    assert len(atr) == len(mock_market_data)
    # First value is not NaN because high-low diff is valid
    assert not np.isnan(atr.iloc[0])

def test_calculate_volatility_z_score(mock_market_data):
    mock_market_data['ATR_14'] = calculate_atr(mock_market_data, window=14)
    z_score = calculate_volatility_z_score(mock_market_data, atr_col='ATR_14', window=50)
    assert len(z_score) == len(mock_market_data)

def test_calculate_macd(mock_market_data):
    macd, signal, hist = calculate_macd(mock_market_data)
    assert len(macd) == len(mock_market_data)
    assert len(signal) == len(mock_market_data)
    assert len(hist) == len(mock_market_data)

def test_add_indicators(mock_market_data):
    df = add_indicators(mock_market_data)
    expected_cols = [
        'SMA_50', 'SMA_200', 'RSI_14', 'ATR_14', 'ATR_Z',
        'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower',
        'Momentum_10', 'Momentum_Drift', 'Market_Stress', 'Fear_Greed'
    ]
    for col in expected_cols:
        assert col in df.columns
