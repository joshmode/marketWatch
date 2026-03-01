import asyncio
from unittest.mock import AsyncMock
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
from app.macro import load_macro_data, enrich_macro_data
from app.ml_engine import train_model, MODEL_PATH

def test_load_macro_data_missing_key():
    # Simulate missing API key
    with patch("app.macro.FRED_API_KEY", None):
        import asyncio
        df = asyncio.run(load_macro_data())
        assert isinstance(df, pd.DataFrame)
        assert df.empty

def test_enrich_macro_data_missing_macro(mock_market_data):
    # Simulate load_macro_data returning empty DF
    with patch("app.macro.load_macro_data", new_callable=AsyncMock, return_value=pd.DataFrame()):
        enriched = enrich_macro_data(mock_market_data)

        # Check if macro columns are present (should be filled with 0s)
        expected_cols = [
            'macro_score', 'recession_probability',
            'yield_curve', 'inflation_yoy'
        ]
        for col in expected_cols:
            assert col in enriched.columns
            # Assert values are 0 (or close to 0)
            # inflation_yoy is 0.0
            # macro_score is 0.0 if all Z-scores are 0
            # yield_curve is 0
            # recession_probability is sigmoid(0) = 0.5?
            # Let's check logic:
            # recession_proxy = -0.4*0 + ... = 0
            # prob = 1 / (1 + exp(0)) = 0.5
            if col == 'recession_probability':
                assert (enriched[col] == 0.0).all()
            else:
                assert (enriched[col] == 0).all()

def test_train_model_with_provided_data():
    # Generate sufficient data for training
    dates = pd.date_range(start="2015-01-01", periods=1000, freq="D")
    df = pd.DataFrame({
        "Open": np.random.rand(1000) * 100,
        "High": np.random.rand(1000) * 100,
        "Low": np.random.rand(1000) * 100,
        "Close": np.random.rand(1000) * 100,
        "Volume": np.random.randint(100, 1000, 1000)
    }, index=dates)

    # Needs indicators
    from app.indicators import add_indicators
    df = add_indicators(df)

    with patch("app.ml_engine.joblib.dump") as mock_dump:
        model = train_model(market_data=df)
        assert model is not None
        mock_dump.assert_called()
