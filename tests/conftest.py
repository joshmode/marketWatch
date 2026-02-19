import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def mock_market_data():
    dates = pd.date_range(start="2020-01-01", periods=300, freq="D")
    df = pd.DataFrame({
        "Open": np.random.rand(300) * 100,
        "High": np.random.rand(300) * 100,
        "Low": np.random.rand(300) * 100,
        "Close": np.random.rand(300) * 100,
        "Volume": np.random.randint(100, 1000, 300)
    }, index=dates)
    return df

@pytest.fixture
def mock_macro_data():
    dates = pd.date_range(start="2020-01-01", periods=300, freq="D")
    df = pd.DataFrame({
        "growth": np.random.rand(300),
        "inflation": np.random.rand(300),
        "yield_10y": np.random.rand(300),
        "yield_2y": np.random.rand(300),
        "credit": np.random.rand(300),
        "unemployment": np.random.rand(300),
        "rates": np.random.rand(300),
        "dollar": np.random.rand(300),
        "core_inflation": np.random.rand(300),
    }, index=dates)
    return df
