from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
from app.main import app
import traceback

client = TestClient(app)

@patch("app.main.fetch_data")
@patch("app.main.add_indicators")
@patch("app.main.enrich_macro_data")
@patch("app.main.compute_bayesian_regime")
@patch("app.main.build_overlay_signal")
def test_overlay_endpoint(mock_build_overlay, mock_bayes, mock_enrich, mock_indicators, mock_fetch):
    mock_df = pd.DataFrame({"Close": [100, 101], "Open": [99, 100]}, index=pd.date_range("2021-01-01", periods=2))
    mock_fetch.return_value = mock_df
    mock_indicators.return_value = mock_df
    mock_enrich.return_value = mock_df

    mock_bayes_df = pd.DataFrame({"P_Expansion": [0.5, 0.5]}, index=mock_df.index)
    mock_bayes.return_value = mock_bayes_df

    mock_build_overlay.return_value = {"signal": "buy"}

    response = client.get("/api/overlay")
    print('STATUS', response.status_code)
    if response.status_code == 500:
        print('BODY', response.json())

test_overlay_endpoint()
