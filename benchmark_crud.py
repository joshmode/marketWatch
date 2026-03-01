import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, MarketData
from app.crud import save_market_data

# Create an in-memory SQLite database
engine = create_engine('sqlite:///:memory:')
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# Create some dummy data
dates = pd.date_range(start='2020-01-01', periods=10000, freq='D')
df = pd.DataFrame({
    'Open': np.random.rand(10000),
    'High': np.random.rand(10000),
    'Low': np.random.rand(10000),
    'Close': np.random.rand(10000),
    'Volume': np.random.randint(1000, 100000, 10000),
    'SMA_200': np.random.rand(10000),
    'RSI_14': np.random.rand(10000),
    'ATR_14': np.random.rand(10000),
    'ATR_Z': np.random.rand(10000),
    'Regime': np.random.randint(-1, 2, 10000)
}, index=dates)

def run_benchmark():
    db = SessionLocal()
    start_time = time.time()
    save_market_data(db, df, "TEST")
    end_time = time.time()
    db.close()
    return end_time - start_time

if __name__ == "__main__":
    t = run_benchmark()
    print(f"Time taken: {t:.4f} seconds")
