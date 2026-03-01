import sys
import time
from datetime import datetime, timedelta

# Create dummy modules
class DummyModule:
    pass

class DummySeries:
    def __init__(self, data=None, index=None, dtype=None):
        if data is None:
            data = []
        self.empty = len(data) == 0
        self.iloc = data
        self.index = [DummyDate(d) for d in index] if index else []
    def dropna(self): return self

class DummyDataFrame:
    def __init__(self, data=None):
        self.empty = False if data else True
    def ffill(self): return self
    def dropna(self, **kwargs): return self
    def get(self, name, default):
        return DummySeries([1.23], ["2023-01-01"])

class DummyDate:
    def __init__(self, d): self.d = d
    def strftime(self, fmt): return self.d

pd = DummyModule()
pd.DataFrame = DummyDataFrame
pd.Series = DummySeries

np = DummyModule()
requests = DummyModule()

sys.modules['pandas'] = pd
sys.modules['numpy'] = np
sys.modules['requests'] = requests

# Now we can import app.macro
import app.macro as macro

# Mock missing things in app.macro
macro.FRED_API_KEY = "mock_key"
def mock_load_macro_data():
    # Simulate a slow pandas operation
    time.sleep(0.05)
    return pd.DataFrame({"growth": DummySeries([1.23], ["2023-01-01"])})

macro.load_macro_data = mock_load_macro_data

def run_benchmark():
    start = time.perf_counter()
    for _ in range(100):
        macro.get_macro_summary()
    end = time.perf_counter()
    return (end - start) * 1000 / 100

print(f"Patched Time: {run_benchmark():.2f} ms per call")
