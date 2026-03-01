from sqlalchemy.orm import Session
from app.models import MarketData
import pandas as pd
import numpy as np


def save_market_data(db: Session, market_data: pd.DataFrame, ticker: str):
    # replace existing records for this ticker
    db.query(MarketData).filter(MarketData.ticker == ticker).delete()

    data_to_store = market_data.copy()

    if data_to_store.index.tz is not None:
        data_to_store.index = data_to_store.index.tz_localize(None)

    expected_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_200", "RSI_14", "ATR_14", "ATR_Z", "Regime"
    ]

    for col in expected_columns:
        if col not in data_to_store.columns:
            data_to_store[col] = None

    data_to_store = data_to_store.replace({np.nan: None})

    records = []
    for row in data_to_store.itertuples():
        records.append(
            MarketData(
                ticker=ticker,
                date=row.Index,
                open=row.Open,
                high=row.High,
                low=row.Low,
                close=row.Close,
                volume=row.Volume,
                sma_200=row.SMA_200,
                rsi_14=row.RSI_14,
                atr_14=row.ATR_14,
                atr_z=row.ATR_Z,
                regime=row.Regime,
            )
        )

    db.add_all(records)
    db.commit()


def get_market_data(db: Session, ticker: str):
    return (
        db.query(MarketData)
        .filter(MarketData.ticker == ticker)
        .order_by(MarketData.date)
        .all()
    )
