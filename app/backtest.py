import pandas as pd
import numpy as np

from app.ml_engine import get_historical_predictions


def run_backtest(market_data: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    # work on a copy so upstream data is untouched
    backtest = market_data.copy()

    if "P_Expansion" not in backtest.columns:
        raise ValueError("Bayesian regime probabilities missing.")

    backtest["Returns"] = backtest["Close"].pct_change()

    base_signal = (
        backtest["P_Expansion"] * 1.0 +
        backtest["P_Slowdown"] * 0.5 +
        backtest["P_Stress"] * 0.0
    )
    backtest["Base_Signal"] = base_signal

    try:
        ml_score = get_historical_predictions(backtest)
    except Exception:
        ml_score = 0.0

    backtest["ML_Score"] = ml_score

    combined_signal = backtest["Base_Signal"] + (backtest["ML_Score"] * 0.2)
    backtest["Signal"] = combined_signal.clip(0.0, 1.2)

    # apply signal with one-period lag
    strategy_returns = backtest["Signal"].shift(1) * backtest["Returns"]
    backtest["Strategy_Returns"] = strategy_returns.fillna(0)

    equity_curve = (1 + backtest["Strategy_Returns"]).cumprod() * initial_capital
    backtest["Cumulative_Strategy"] = equity_curve

    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak
    backtest["Drawdown"] = drawdown

    # dynamic risk reduction based on drawdown
    de_risk_factor = np.clip(1 + drawdown * 2, 0.5, 1.0)
    backtest["Signal"] = backtest["Signal"] * de_risk_factor

    return backtest
