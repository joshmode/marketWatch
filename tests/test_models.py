import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import MarketData
from app.database import Base

@pytest.fixture(scope="module")
def engine():
    return create_engine("sqlite:///:memory:")

@pytest.fixture(scope="module")
def tables(engine):
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(engine, tables):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

def test_market_data_instantiation():
    dt = datetime(2023, 1, 1, 12, 0)
    md = MarketData(
        ticker="AAPL",
        date=dt,
        open=150.0,
        high=155.0,
        low=149.0,
        close=154.0,
        volume=1000000,
        sma_200=145.0,
        rsi_14=60.0,
        atr_14=5.0,
        atr_z=1.2,
        regime="Bull"
    )

    assert md.ticker == "AAPL"
    assert md.date == dt
    assert md.open == 150.0
    assert md.high == 155.0
    assert md.low == 149.0
    assert md.close == 154.0
    assert md.volume == 1000000
    assert md.sma_200 == 145.0
    assert md.rsi_14 == 60.0
    assert md.atr_14 == 5.0
    assert md.atr_z == 1.2
    assert md.regime == "Bull"

def test_market_data_persistence(db_session):
    dt = datetime(2023, 1, 1, 12, 0)
    md = MarketData(
        ticker="MSFT",
        date=dt,
        open=250.0,
        high=255.0,
        low=249.0,
        close=254.0,
        volume=2000000,
        sma_200=240.0,
        rsi_14=55.0,
        atr_14=4.0,
        atr_z=0.8,
        regime="Bear"
    )

    db_session.add(md)
    db_session.commit()

    retrieved = db_session.query(MarketData).filter_by(ticker="MSFT").first()

    assert retrieved is not None
    assert retrieved.id is not None
    assert retrieved.ticker == "MSFT"
    assert retrieved.date == dt
    assert retrieved.open == 250.0
    assert retrieved.high == 255.0
    assert retrieved.low == 249.0
    assert retrieved.close == 254.0
    assert retrieved.volume == 2000000
    assert retrieved.sma_200 == 240.0
    assert retrieved.rsi_14 == 55.0
    assert retrieved.atr_14 == 4.0
    assert retrieved.atr_z == 0.8
    assert retrieved.regime == "Bear"

def test_market_data_nullable_fields():
    dt = datetime(2023, 1, 2, 12, 0)
    md = MarketData(
        ticker="GOOGL",
        date=dt,
        open=100.0,
        high=105.0
    )
    assert md.ticker == "GOOGL"
    assert md.date == dt
    assert md.open == 100.0
    assert md.high == 105.0
    assert md.low is None
    assert md.close is None

def test_market_data_persistence_nullable(db_session):
    dt = datetime(2023, 1, 2, 12, 0)
    md = MarketData(
        ticker="AMZN",
        date=dt,
        open=100.0,
        high=105.0
    )

    db_session.add(md)
    db_session.commit()

    retrieved = db_session.query(MarketData).filter_by(ticker="AMZN").first()

    assert retrieved is not None
    assert retrieved.id is not None
    assert retrieved.ticker == "AMZN"
    assert retrieved.open == 100.0
    assert retrieved.low is None
    assert retrieved.close is None
