import time
import asyncio
from unittest.mock import patch, MagicMock
import app.macro
from app.macro import load_macro_data

async def mock_get(*args, **kwargs):
    await asyncio.sleep(0.5) # Simulate network delay
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"observations": [{"date": "2023-01-01", "value": "1.0"}]}
    return mock_response

@patch('httpx.AsyncClient.get', side_effect=mock_get)
@patch('app.macro.FRED_API_KEY', 'fake_key')
def run_benchmark(mock_get_func):
    app.macro._cache.clear()
    start = time.time()
    df = load_macro_data()
    end = time.time()
    print(f"Time taken to load macro data (async w/ mocked 0.5s network delay): {end - start:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
