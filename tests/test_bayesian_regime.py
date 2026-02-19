import pytest
import pandas as pd
import numpy as np
from app.bayesian_regime import compute_bayesian_regime

def test_compute_bayesian_regime(mock_market_data):
    # Add required columns
    mock_market_data['macro_score'] = 0.5
    mock_market_data['credit_stress_z'] = -0.5
    mock_market_data['curve_slope'] = 1.0
    mock_market_data['macro_liquidity_z'] = 0.5
    mock_market_data['dollar_regime_z'] = 0.0

    regime_df = compute_bayesian_regime(mock_market_data)

    assert 'P_Expansion' in regime_df.columns
    assert 'P_Slowdown' in regime_df.columns
    assert 'P_Stress' in regime_df.columns
    assert 'Regime' in regime_df.columns

    # Probabilities should sum to 1 (approx)
    probs = regime_df[['P_Expansion', 'P_Slowdown', 'P_Stress']].sum(axis=1)
    assert np.allclose(probs, 1.0)
