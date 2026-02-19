from sqlalchemy import Column, Integer, String, Float, DateTime
from app.database import Base


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)

    ticker = Column(String, index=True)
    date = Column(DateTime, index=True)

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

    sma_200 = Column(Float)
    rsi_14 = Column(Float)
    atr_14 = Column(Float)
    atr_z = Column(Float)

    regime = Column(String)
