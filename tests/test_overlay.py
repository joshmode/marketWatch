import pandas as pd
import pytest
from app.overlay import build_overlay_signal

def test_build_overlay_signal_empty_data():
    """Test that build_overlay_signal returns an empty dict when given an empty DataFrame."""
    empty_df = pd.DataFrame()
    result = build_overlay_signal(empty_df)
    assert result == {}

def test_build_overlay_signal_valid_data():
    """Test that build_overlay_signal processes valid data correctly."""
    data = {
        "P_Expansion": [0.6],
        "P_Slowdown": [0.3],
        "P_Stress": [0.1],
        "macro_score": [0.75],
        "recession_probability": [0.15],
        "Fear_Greed": [60.0],
        "Market_Stress": [0.2],
        "macro_liquidity_z": [0.5],
        "liquidity_stress_index": [0.3]
    }
    df = pd.DataFrame(data, index=pd.date_range("2023-01-01", periods=1))

    result = build_overlay_signal(df)

    assert "timestamp" in result
    assert result["timestamp"] == "2023-01-01 00:00:00"

    assert "regime_probabilities" in result
    assert result["regime_probabilities"]["expansion"] == 0.6
    assert result["regime_probabilities"]["slowdown"] == 0.3
    assert result["regime_probabilities"]["stress"] == 0.1

    assert result["macro_score"] == 0.75
    assert result["recession_probability"] == 0.15

    assert "risk_diagnostics" in result
    assert result["risk_diagnostics"]["fear_greed"] == 60.0
    assert result["risk_diagnostics"]["market_stress"] == 0.2
    assert result["risk_diagnostics"]["liquidity_stress_z"] == 0.5
    assert result["risk_diagnostics"]["liquidity_stress_index"] == 0.3

    assert "recommended_risk_level" in result
    assert "equity_beta_overlay" in result
    assert result["confidence"] == 0.6
