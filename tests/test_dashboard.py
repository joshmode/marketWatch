import pytest
import pandas as pd
import numpy as np
from app.dashboard import create_dashboard

def test_create_dashboard_with_benchmark():
    # Create sample data
    dates = pd.date_range("2021-01-01", periods=10)
    df = pd.DataFrame({
        "Close": np.linspace(100, 200, 10),
        "Cumulative_Strategy": np.linspace(10000, 20000, 10),
        "P_Expansion": np.random.rand(10),
        "P_Slowdown": np.random.rand(10),
        "P_Stress": np.random.rand(10)
    }, index=dates)

    fig = create_dashboard(df)

    # Check traces
    trace_names = [trace.name for trace in fig.data if trace.name]
    assert "Strategy Equity" in trace_names
    assert "Benchmark (Buy & Hold)" in trace_names
    # "S&P 500" trace is only added if strategy is missing
    assert "S&P 500" not in trace_names

def test_create_dashboard_with_live_data():
    dates = pd.date_range("2021-01-01", periods=10)
    df = pd.DataFrame({
        "Close": np.linspace(100, 200, 10)
    }, index=dates)

    live_data = [
        {"symbol": "SPY", "price": 400.0, "change": 2.0, "pct_change": 0.5},
        {"symbol": "QQQ", "price": 300.0, "change": -1.0, "pct_change": -0.3}
    ]

    fig = create_dashboard(df, live_data=live_data)

    # Check for indicators
    # Indicators don't always have a 'name' attribute in the same way, but we can check type
    indicators = [trace for trace in fig.data if trace.type == "indicator"]
    assert len(indicators) == 2
    assert indicators[0].title.text == "SPY"
    assert indicators[1].title.text == "QQQ"

def test_create_dashboard_with_fear_greed():
    dates = pd.date_range("2021-01-01", periods=10)
    df = pd.DataFrame({
        "Close": np.linspace(100, 200, 10),
        "Fear_Greed": np.linspace(-50, 50, 10)
    }, index=dates)

    fig = create_dashboard(df)

    trace_names = [trace.name for trace in fig.data if trace.name]
    assert "Fear & Greed Proxy" in trace_names
