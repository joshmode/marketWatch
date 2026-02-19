import os
import joblib
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score

from app.data import fetch_data
from app.indicators import add_indicators

MODEL_PATH = "ml_model.joblib"

LOOKBACK = 252
TEST_WINDOW = 63
PURGE_GAP = 5

BASE_FEATURES = [
    "RSI_14",
    "ATR_14",
    "MACD",
    "Momentum_10",
    "SMA_50",
    "SMA_200",
    "Dist_SMA_50",
    "Dist_SMA_200",
]

logger = logging.getLogger(__name__)


def prepare_features(market_data: pd.DataFrame) -> pd.DataFrame:
    data = market_data.copy()

    if "Dist_SMA_50" not in data.columns:
        data["Dist_SMA_50"] = (data["Close"] - data["SMA_50"]) / data["SMA_50"]

    if "Dist_SMA_200" not in data.columns:
        data["Dist_SMA_200"] = (data["Close"] - data["SMA_200"]) / data["SMA_200"]

    for col in BASE_FEATURES:
        if col in data.columns:
            rolling = data[col].rolling(window=LOOKBACK)
            data[f"{col}_Z"] = (data[col] - rolling.mean()) / rolling.std()

    return data.dropna()


def build_targets(feature_data: pd.DataFrame) -> pd.DataFrame:
    data = feature_data.copy()

    data["Next_Return"] = data["Close"].shift(-1) / data["Close"] - 1


    rolling_mean = data["Close"].pct_change(fill_method=None).rolling(LOOKBACK).mean()
    rolling_std = data["Close"].pct_change(fill_method=None).rolling(LOOKBACK).std()

    data["Target_Z"] = (data["Next_Return"] - rolling_mean) / rolling_std

    bins = [
        (data["Target_Z"] < -0.84),
        (data["Target_Z"] >= -0.84) & (data["Target_Z"] < -0.25),
        (data["Target_Z"] >= -0.25) & (data["Target_Z"] < 0.25),
        (data["Target_Z"] >= 0.25) & (data["Target_Z"] < 0.84),
        (data["Target_Z"] >= 0.84),
    ]

    data["Target_Class"] = np.select(bins, [0, 1, 2, 3, 4], default=2)

    return data.dropna()


def purged_time_series_split(n_samples: int):
    start = LOOKBACK + PURGE_GAP

    for i in range(start, n_samples - TEST_WINDOW, TEST_WINDOW):
        train_start = i - LOOKBACK - PURGE_GAP
        train_end = i - PURGE_GAP

        test_start = i
        test_end = i + TEST_WINDOW

        if train_start >= 0:
            yield list(range(train_start, train_end)), list(range(test_start, test_end))


def get_model():
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        metric="multi_logloss",
        verbosity=-1,
        boosting_type="gbdt",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
    )


def train_model(ticker: str = "^GSPC", market_data: pd.DataFrame = None):
    logger.info(f"training model for {ticker}")

    if market_data is None:
        raw = fetch_data(ticker, period="5y")
        market_data = add_indicators(raw)

    features = prepare_features(market_data)
    dataset = build_targets(features)

    if dataset.empty:
        logger.warning("no data available for training")
        return None

    feature_cols = [c for c in dataset.columns if c.endswith("_Z") and "Target" not in c]
    X = dataset[feature_cols]
    y = dataset["Target_Class"]

    model = get_model()

    scores = []
    splits = list(purged_time_series_split(len(X)))

    for train_idx, test_idx in splits:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], preds))

    if scores:
        logger.info(f"mean cv accuracy: {np.mean(scores):.4f}")

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    return model


def get_historical_predictions(market_data: pd.DataFrame) -> pd.Series:
    try:
        features = prepare_features(market_data)
        dataset = build_targets(features)
    except Exception:
        return pd.Series(0, index=market_data.index)

    if dataset.empty:
        return pd.Series(0, index=market_data.index)

    feature_cols = [c for c in dataset.columns if c.endswith("_Z") and "Target" not in c]
    X = dataset[feature_cols]
    y = dataset["Target_Class"]

    predictions = pd.Series(index=X.index, dtype=float)
    model = get_model()

    for train_idx, test_idx in purged_time_series_split(len(X)):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])

        expected_value = np.sum(probs * np.arange(5), axis=1)
        score = (expected_value - 2.0) / 2.0

        predictions.iloc[test_idx] = score

    return predictions.reindex(market_data.index).fillna(0)


def predict_latest_score(market_data: pd.DataFrame) -> float:
    if not os.path.exists(MODEL_PATH):
        train_model(market_data=market_data)

    if not os.path.exists(MODEL_PATH):
        return 0.0

    model = joblib.load(MODEL_PATH)
    features = prepare_features(market_data)

    if features.empty:
        return 0.0

    feature_cols = [c for c in features.columns if c.endswith("_Z") and "Target" not in c]
    latest = features.iloc[[-1]][feature_cols]

    probs = model.predict_proba(latest)[0]
    expected_value = np.sum(probs * np.arange(5))

    return float((expected_value - 2.0) / 2.0)


if __name__ == "__main__":
    train_model()
