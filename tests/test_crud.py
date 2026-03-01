import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models import MarketData
from app.crud import save_market_data, get_market_data

@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_save_data(db):
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [12.0, 13.0],
            "Low": [9.0, 10.0],
            "Close": [11.0, 12.0],
            "Volume": [100, 200],
            "SMA_200": [10.5, 11.5],
            "RSI_14": [50.0, 60.0],
            "ATR_14": [1.5, 1.6],
            "ATR_Z": [0.5, 0.6],
            "Regime": ["Bull", "Bull"]
        },
        index=pd.date_range("2023-01-01", periods=2)
    )
    save_market_data(db, df, "AAPL")
    res = get_market_data(db, "AAPL")

    assert len(res) == 2
    assert res[0].ticker == "AAPL"
    assert res[0].open == 10.0
    assert res[1].regime == "Bull"

def test_save_replace(db):
    df1 = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=pd.date_range("2023-01-01", periods=1)
    )
    save_market_data(db, df1, "TSLA")
    res1 = get_market_data(db, "TSLA")
    assert len(res1) == 1
    assert res1[0].open == 1.0

    df2 = pd.DataFrame(
        {"Open": [5.0], "High": [6.0], "Low": [4.0], "Close": [5.5], "Volume": [20]},
        index=pd.date_range("2023-01-01", periods=1)
    )
    save_market_data(db, df2, "TSLA")
    res2 = get_market_data(db, "TSLA")
    assert len(res2) == 1
    assert res2[0].open == 5.0

def test_save_tz(db):
    df = pd.DataFrame(
        {"Open": [10.0], "High": [12.0], "Low": [9.0], "Close": [11.0], "Volume": [100]},
        index=pd.date_range("2023-01-01", periods=1, tz="UTC")
    )
    save_market_data(db, df, "MSFT")
    res = get_market_data(db, "MSFT")

    assert len(res) == 1
    assert res[0].date.tzinfo is None

def test_save_missing(db):
    df = pd.DataFrame(
        {"Close": [100.0, np.nan], "Volume": [1000, 2000]},
        index=pd.date_range("2023-01-01", periods=2)
    )
    save_market_data(db, df, "GOOG")
    res = get_market_data(db, "GOOG")

    assert len(res) == 2
    assert res[0].open is None
    assert res[0].close == 100.0
    assert res[1].close is None
