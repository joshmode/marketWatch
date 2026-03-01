import logging
import pickle
import pandas as pd
import requests
import time
import random
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"


def ensure_cache_directory():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_file_path(ticker: str, period: str) -> Path:
    safe_ticker = ticker.replace("^", "").replace(".", "_")
    return CACHE_DIR / f"{safe_ticker}_{period}.pkl"


def load_cached_data(ticker: str, period: str) -> Optional[pd.DataFrame]:
    ensure_cache_directory()
    path = cache_file_path(ticker, period)

    if not path.exists():
        return None

    try:
        modified_time = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - modified_time > timedelta(hours=12):
            return None

        with open(path, "rb") as f:
            logger.info(f"Loaded cached data for {ticker}")
            return pickle.load(f)

    except Exception as e:
        logger.warning(f"Failed to load cache for {ticker}: {e}")
        return None


def write_cache(ticker: str, period: str, data: pd.DataFrame):
    ensure_cache_directory()
    path = cache_file_path(ticker, period)

    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Cached data for {ticker}")
    except Exception as e:
        logger.warning(f"Failed to write cache for {ticker}: {e}")


def yahoo_range_from_period(period: str) -> str:
    return "10y" if period == "max" else period


def request_headers() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }


def fetch_data(ticker: str = "^GSPC", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    cached = load_cached_data(ticker, period)
    if cached is not None:
        return cached

    params = {
        "range": yahoo_range_from_period(period),
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true",
    }

    t = urllib.parse.quote(ticker)
    endpoints = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{t}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{t}",
    ]

    last_error = None

    for url in endpoints:
        try:
            time.sleep(random.uniform(1, 3))

            logger.info(f"Fetching data for {ticker}")
            response = requests.get(url, params=params, headers=request_headers(), timeout=15)

            if response.status_code == 429:
                continue

            response.raise_for_status()
            payload = response.json()

            result = payload.get("chart", {}).get("result", [])
            if not result:
                raise ValueError("No data returned")

            chart = result[0]
            timestamps = chart.get("timestamp", [])
            quote = chart.get("indicators", {}).get("quote", [{}])[0]

            if not timestamps:
                raise ValueError("Empty timestamp list")

            df = pd.DataFrame(
                {
                    "Open": quote.get("open", []),
                    "High": quote.get("high", []),
                    "Low": quote.get("low", []),
                    "Close": quote.get("close", []),
                    "Volume": quote.get("volume", []),
                },
                index=pd.to_datetime(timestamps, unit="s"),
            )

            df.index.name = "Date"
            df = df.sort_index().dropna()

            if df.empty:
                raise ValueError("Processed dataframe empty")

            write_cache(ticker, period, df)
            return df

        except Exception as e:
            logger.warning(f"Fetch failed from {url}: {e}")
            last_error = e

    raise RuntimeError(f"Data fetch failed for {ticker}: {last_error}")


def fetch_live_ticker(ticker: str) -> Dict[str, Any]:
    params = {
        "range": "1d",
        "interval": "1m",
        "includePrePost": "true",
    }

    t = urllib.parse.quote(ticker)
    endpoints = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{t}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{t}",
    ]

    for url in endpoints:
        try:
            time.sleep(random.uniform(0.5, 1.5))

            response = requests.get(url, params=params, headers=request_headers(), timeout=10)
            if response.status_code == 429:
                continue

            response.raise_for_status()
            payload = response.json()

            result = payload.get("chart", {}).get("result", [])
            if not result:
                continue

            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice")
            prev_close = meta.get("chartPreviousClose")

            if price is None:
                quote = result[0].get("indicators", {}).get("quote", [{}])[0]
                closes = quote.get("close", [])
                valid_closes = [c for c in closes if c is not None]
                if valid_closes:
                    price = valid_closes[-1]

            if price is None or prev_close is None:
                continue

            change = price - prev_close
            pct_change = (change / prev_close) * 100



            return {
                "symbol": ticker,
                "price": round(price, 2),
                "change": round(change, 2),
                "pct_change": round(pct_change, 2),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception:
            continue

    return {
        "symbol": ticker,
        "price": 0.0,
        "change": 0.0,
        "pct_change": 0.0,
        "timestamp": datetime.now().isoformat(),
    }
