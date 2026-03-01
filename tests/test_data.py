import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from app.data import fetch_data

MOCK_YAHOO_JSON = {
    "chart": {
        "result": [
            {
                "meta": {
                    "currency": "USD",
                    "symbol": "^GSPC",
                },
                "timestamp": [1672531200, 1672617600, 1672704000],
                "indicators": {
                    "quote": [
                        {
                            "open": [3800.0, 3810.0, 3820.0],
                            "high": [3850.0, 3860.0, 3870.0],
                            "low": [3790.0, 3800.0, 3810.0],
                            "close": [3825.0, 3835.0, 3845.0],
                            "volume": [1000000, 1100000, 1200000]
                        }
                    ]
                }
            }
        ],
        "error": None
    }
}

@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.json.return_value = MOCK_YAHOO_JSON
    mock.status_code = 200
    return mock

def test_fetch_data_success(mock_response):
    with patch("requests.get", return_value=mock_response) as mock_get, \
         patch("app.data.write_cache") as mock_write_cache, \
         patch("app.data.load_cached_data", return_value=None):

        df = fetch_data("^GSPC", "1mo")

        assert not df.empty
        assert len(df) == 3
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

        args, kwargs = mock_get.call_args
        assert "finance.yahoo.com" in args[0]
        assert "^GSPC" in args[0]

        mock_write_cache.assert_called_once()

def test_fetch_data_uses_cache():
    cached_df = pd.DataFrame(
        {"Open": [100], "Close": [102]},
        index=pd.to_datetime(["2024-01-01"])
    )

    with patch("app.data.load_cached_data", return_value=cached_df) as mock_load_cache, \
         patch("requests.get") as mock_get:

        df = fetch_data("^GSPC", "2y")
        assert not df.empty
        mock_load_cache.assert_called_once()
        mock_get.assert_not_called()

def test_fetch_data_failure():
    with patch("requests.get", side_effect=Exception("Network error")), \
         patch("app.data.load_cached_data", return_value=None):

        with pytest.raises(RuntimeError, match="Data fetch failed"):
            fetch_data("^GSPC", "2y")

def test_fetch_data_empty_response():
    mock_empty = MagicMock()
    mock_empty.json.return_value = {"chart": {"result": None, "error": {"code": "Not Found"}}}
    mock_empty.status_code = 404

    with patch("requests.get", return_value=mock_empty), \
         patch("app.data.load_cached_data", return_value=None):

        with pytest.raises(RuntimeError, match="Data fetch failed"):
            fetch_data("^GSPC", "2y")
