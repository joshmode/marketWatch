import pytest
import pandas as pd
from app.indicators import calculate_sma, calculate_rsi
from app.models import MarketData

def test_imports():
    assert calculate_sma is not None
    assert calculate_rsi is not None
    assert MarketData is not None

def test_indicators():
    df = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
    sma = calculate_sma(df, window=2)
    assert sma.iloc[-1] == 4.5
