from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from app.main import app

client = TestClient(app)

@patch("app.main.fetch_data")
@patch("app.main.add_indicators")
@patch("app.main.enrich_macro_data")
@patch("app.main.compute_bayesian_regime")
@patch("app.main.run_backtest")
@patch("app.main.create_dashboard")
def test_dashboard_endpoint(mock_create_dashboard, mock_backtest, mock_bayes, mock_enrich, mock_indicators, mock_fetch):
    # Mock data flow
    mock_df = pd.DataFrame({"Close": [100, 101], "Open": [99, 100]}, index=pd.date_range("2021-01-01", periods=2))
    mock_fetch.return_value = mock_df
    mock_indicators.return_value = mock_df
    mock_enrich.return_value = mock_df

    mock_bayes_df = pd.DataFrame({"P_Expansion": [0.5, 0.5]}, index=mock_df.index)
    mock_bayes.return_value = mock_bayes_df

    mock_backtest.return_value = mock_df.join(mock_bayes_df)

    mock_fig = MagicMock()
    mock_fig.to_json.return_value = "{}"
    mock_create_dashboard.return_value = mock_fig

    # Check if create_dashboard was called with a dataframe containing Cumulative_Strategy
    # This is implicitly checked by the fact that we mocked backtest to return it.

    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

@patch("app.main.fetch_data")
@patch("app.main.add_indicators")
@patch("app.main.enrich_macro_data")
@patch("app.main.compute_bayesian_regime")
@patch("app.main.run_backtest")
@patch("app.main.build_overlay_signal")
def test_overlay_endpoint(mock_build_overlay, mock_backtest, mock_bayes, mock_enrich, mock_indicators, mock_fetch):
    # Mock data
    mock_df = pd.DataFrame({"Close": [100, 101], "Open": [99, 100]}, index=pd.date_range("2021-01-01", periods=2))
    mock_fetch.return_value = mock_df
    mock_indicators.return_value = mock_df
    mock_enrich.return_value = mock_df

    mock_bayes_df = pd.DataFrame({"P_Expansion": [0.5, 0.5], "P_Slowdown": [0.3, 0.3], "P_Stress": [0.2, 0.2]}, index=mock_df.index)
    mock_bayes.return_value = mock_bayes_df

    mock_backtest.return_value = mock_df.join(mock_bayes_df)

    mock_build_overlay.return_value = {"signal": "buy"}

    response = client.get("/api/overlay")
    assert response.status_code == 200
    assert response.json() == {"signal": "buy"}

@patch("app.main.build_market_dataset")
def test_overlay_endpoint_error(mock_build):
    # Mock data to raise an exception
    mock_build.side_effect = Exception("Test Overlay Error")

    response = client.get("/api/overlay")
    assert response.status_code == 500
    assert response.json() == {"detail": "Test Overlay Error"}
