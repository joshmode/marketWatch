import os
import logging
import asyncio
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "growth": "GDPC1",
    "ind_prod": "INDPRO",
    "payrolls": "PAYEMS",
    "inflation": "CPIAUCSL",
    "core_inflation": "CPILFESL",
    "rates": "FEDFUNDS",
    "unemployment": "UNRATE",
    "yield_10y": "DGS10",
    "yield_2y": "DGS2",
    "credit": "BAMLH0A0HYM2",
    "dollar": "DTWEXBGS",
}

_cache: Dict[str, Any] = {}
_cache_expiry: Dict[str, datetime] = {}


def _series_or_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(index=df.index, dtype=float)


async def fetch_series(series_id: str, client: httpx.AsyncClient) -> pd.Series:
    now = datetime.now()

    if series_id in _cache and now < _cache_expiry.get(series_id, datetime.min):
        return _cache[series_id]

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    try:
        response = await client.get(FRED_URL, params=params, timeout=10.0)
        response.raise_for_status()

        observations = response.json().get("observations", [])
        df = pd.DataFrame(observations)

        if df.empty:
            series = pd.Series(dtype=float)
        else:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            # Memory explicit instruction: All time-series data must be explicitly sorted by index (`df.sort_index()`) immediately after loading.
            series = df.set_index("date")["value"].sort_index()

        _cache[series_id] = series
        _cache_expiry[series_id] = now + timedelta(hours=24)

        return series

    except Exception as e:
        logger.error(f"Failed to fetch {series_id}: {e}")
        return pd.Series(dtype=float)


async def _load_macro_data_async() -> pd.DataFrame:
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set. Running in neutral macro mode.")
        return pd.DataFrame()

    async with httpx.AsyncClient() as client:
        names = list(SERIES.keys())
        tasks = [fetch_series(code, client) for code in SERIES.values()]
        results = await asyncio.gather(*tasks)

        data = dict(zip(names, results))

    macro = pd.DataFrame(data)
    macro = macro.ffill()
    macro.dropna(how="all", inplace=True)

    return macro


def load_macro_data() -> pd.DataFrame:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If an event loop is already running (e.g., in a testing context or specific async runner),
        # we cannot use asyncio.run(). Instead, we run the async task in a separate thread.
        import threading

        result = None
        exception = None

        def _run_in_thread():
            nonlocal result, exception
            try:
                result = asyncio.run(_load_macro_data_async())
            except Exception as e:
                exception = e

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        thread.join()

        if exception:
            raise exception
        return result
    else:
        return asyncio.run(_load_macro_data_async())


def get_macro_summary() -> Dict[str, Any]:
    if not FRED_API_KEY:
        return {k: {"value": "N/A", "date": "No API Key"} for k in SERIES.keys()}

    macro = load_macro_data()
    if macro.empty:
        return {k: {"value": "N/A", "date": "No Data"} for k in SERIES.keys()}

    summary = {}

    for name in SERIES.keys():
        series = macro.get(name, pd.Series(dtype=float)).dropna()

        if series.empty:
            summary[name] = {"value": "N/A", "date": "-"}
        else:
            summary[name] = {
                "value": round(series.iloc[-1], 2),
                "date": series.index[-1].strftime("%Y-%m-%d"),
            }

    return summary


def enrich_macro_data(market_data: pd.DataFrame) -> pd.DataFrame:
    macro = load_macro_data()

    if macro.empty:
        aligned = pd.DataFrame(index=market_data.index)
    else:
        aligned = macro.reindex(market_data.index, method="ffill")

    yield_10y = _series_or_nan(aligned, "yield_10y")
    yield_2y = _series_or_nan(aligned, "yield_2y")
    inflation = _series_or_nan(aligned, "inflation")
    core_inflation = _series_or_nan(aligned, "core_inflation")
    rates = _series_or_nan(aligned, "rates")
    credit = _series_or_nan(aligned, "credit")
    unemployment = _series_or_nan(aligned, "unemployment")
    growth = _series_or_nan(aligned, "growth")
    dollar = _series_or_nan(aligned, "dollar")

    aligned["yield_curve"] = yield_10y - yield_2y
    aligned["curve_inversion_depth"] = aligned["yield_curve"].clip(upper=0).abs()

    inverted = (aligned["yield_curve"] < 0).astype(float)
    aligned["inversion_persistence"] = inverted.rolling(60).sum()

    aligned["inflation_yoy"] = inflation.pct_change(252) * 100
    aligned["core_inflation_yoy"] = core_inflation.pct_change(252) * 100
    aligned["inflation_persistence"] = aligned["inflation_yoy"] - aligned["core_inflation_yoy"]

    rate_acceleration = yield_10y.diff().diff().rolling(20).std()
    credit_change = credit.diff().rolling(20).mean()
    aligned["macro_shock_raw"] = rate_acceleration + credit_change

    aligned["real_rates"] = rates - aligned["inflation_yoy"]

    unemployment_gap = unemployment - unemployment.rolling(252).mean()
    aligned["policy_tightness"] = rates - (aligned["inflation_yoy"] - 0.5 * unemployment_gap)

    z_inputs = {
        "growth": growth,
        "inflation": inflation,
        "yield_curve": aligned["yield_curve"],
        "credit": credit,
        "unemployment": unemployment,
        "dollar": dollar,
        "policy_tightness": aligned["policy_tightness"],
        "macro_shock_raw": aligned["macro_shock_raw"],
    }

    for name, series in z_inputs.items():
        mean = series.rolling(252).mean()
        std = series.rolling(252).std()
        aligned[f"{name}_z"] = (series - mean) / std

    aligned["macro_score"] = (
        +0.30 * aligned.get("growth_z")
        -0.25 * aligned.get("inflation_z")
        +0.20 * aligned.get("yield_curve_z")
        -0.15 * aligned.get("credit_z")
        -0.10 * aligned.get("unemployment_z")
    )

    recession_signal = (
        -0.4 * aligned.get("yield_curve_z")
        +0.3 * aligned.get("credit_z")
        +0.3 * aligned.get("unemployment_z")
        +0.2 * aligned.get("policy_tightness_z")
    )

    aligned["recession_probability"] = 1 / (1 + np.exp(-recession_signal))

    aligned["credit_stress_z"] = aligned.get("credit_z")
    aligned["curve_slope"] = aligned.get("yield_curve_z")
    aligned["macro_liquidity_z"] = (
        aligned.get("credit_z") + aligned.get("policy_tightness_z")
    ) / 2

    aligned["dollar_regime_z"] = aligned.get("dollar_z")

    aligned["liquidity_stress_index"] = 100 * (1 / (1 + np.exp(-aligned["macro_liquidity_z"])))

    result = market_data.join(aligned, how="left")
    result = result.ffill()
    result = result.fillna(0)

    return result
