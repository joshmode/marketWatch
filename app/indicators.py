import pandas as pd
import numpy as np


def calculate_sma(market_data: pd.DataFrame, window: int = 200, column: str = "Close") -> pd.Series:
    return market_data[column].rolling(window=window).mean()


def calculate_rsi(market_data: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.Series:
    price_change = market_data[column].diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(market_data: pd.DataFrame, window: int = 14) -> pd.Series:
    high = market_data["High"]
    low = market_data["Low"]
    close = market_data["Close"]

    true_range = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    return true_range.ewm(alpha=1 / window, adjust=False).mean()


def calculate_volatility_z_score(market_data: pd.DataFrame, atr_column: str = "ATR_14", window: int = 50) -> pd.Series:
    if atr_column not in market_data.columns:
        raise ValueError(f"ATR column '{atr_column}' missing.")

    rolling = market_data[atr_column].rolling(window=window)
    return (market_data[atr_column] - rolling.mean()) / rolling.std()


def calculate_macd(
    market_data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = "Close"
):
    price = market_data[column]

    fast_ema = price.ewm(span=fast_period, adjust=False).mean()
    slow_ema = price.ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    market_data: pd.DataFrame,
    window: int = 20,
    num_std: int = 2,
    column: str = "Close"
):
    rolling = market_data[column].rolling(window=window)
    middle = rolling.mean()
    std = rolling.std()

    upper = middle + std * num_std
    lower = middle - std * num_std

    return upper, lower


def calculate_momentum(market_data: pd.DataFrame, window: int = 10, column: str = "Close") -> pd.Series:
    return market_data[column].pct_change(periods=window) * 100


def calculate_momentum_drift(market_data: pd.DataFrame, momentum_column: str = "Momentum_10", window: int = 5) -> pd.Series:
    if momentum_column not in market_data.columns:
        return pd.Series(index=market_data.index, dtype=float)

    return market_data[momentum_column].diff(periods=window)


def calculate_market_stress(market_data: pd.DataFrame, atr_z_column: str = "ATR_Z", rsi_column: str = "RSI_14") -> pd.Series:
    if atr_z_column not in market_data.columns or rsi_column not in market_data.columns:
        return pd.Series(index=market_data.index, dtype=float)

    rsi_component = (market_data[rsi_column] - 50).abs() / 25
    volatility_component = market_data[atr_z_column].clip(lower=0)

    return volatility_component + rsi_component


def calculate_fear_greed_proxy(market_data: pd.DataFrame, rsi_column="RSI_14", atr_z_column="ATR_Z") -> pd.Series:
    if rsi_column not in market_data.columns or atr_z_column not in market_data.columns:
        return pd.Series(index=market_data.index, dtype=float)

    rsi_scaled = (market_data[rsi_column] - 50) / 50
    raw_score = rsi_scaled * 100 - (market_data[atr_z_column] * 20)

    return 100 * (1 / (1 + np.exp(-raw_score / 40)))


def add_indicators(market_data: pd.DataFrame) -> pd.DataFrame:
    enriched = market_data.copy()

    enriched["SMA_50"] = calculate_sma(enriched, window=50)
    enriched["SMA_200"] = calculate_sma(enriched, window=200)

    enriched["RSI_14"] = calculate_rsi(enriched)
    enriched["ATR_14"] = calculate_atr(enriched)
    enriched["ATR_Z"] = calculate_volatility_z_score(enriched)

    macd, signal, hist = calculate_macd(enriched)
    enriched["MACD"] = macd
    enriched["MACD_Signal"] = signal
    enriched["MACD_Hist"] = hist

    upper, lower = calculate_bollinger_bands(enriched)
    enriched["BB_Upper"] = upper
    enriched["BB_Lower"] = lower

    enriched["Momentum_10"] = calculate_momentum(enriched)
    enriched["Momentum_Drift"] = calculate_momentum_drift(enriched)

    enriched["Market_Stress"] = calculate_market_stress(enriched)
    enriched["Fear_Greed"] = calculate_fear_greed_proxy(enriched)

    return enriched
