import pytest
from unittest.mock import MagicMock
from app.crud import get_market_data, save_market_data
from app.models import MarketData
import pandas as pd
import numpy as np

def test_get_market_data():
    # Arrange
    mock_db = MagicMock()
    mock_query = mock_db.query.return_value
    mock_filter = mock_query.filter.return_value
    mock_order_by = mock_filter.order_by.return_value

    # Set up expected return
    expected_result = [MarketData(ticker="^GSPC", close=100.0)]
    mock_order_by.all.return_value = expected_result

    # Act
    result = get_market_data(mock_db, "^GSPC")

    # Assert
    mock_db.query.assert_called_once_with(MarketData)

    # Verify filter was called (checking expression equality with mocks can be tricky,
    # but we can verify the method was called)
    mock_query.filter.assert_called_once()

    # Verify order_by was called
    mock_filter.order_by.assert_called_once()

    # Verify all was called
    mock_order_by.all.assert_called_once()

    # Verify the return value matches what we expect
    assert result == expected_result

def test_save_market_data():
    # Arrange
    mock_db = MagicMock()

    dates = pd.date_range("2024-01-01", periods=2)
    df = pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [105.0, 106.0],
        "Low": [95.0, 96.0],
        "Close": [102.0, 103.0],
        "Volume": [1000, 1100],
    }, index=dates)

    # Act
    save_market_data(mock_db, df, "^GSPC")

    # Assert
    # Verify existing records are deleted
    mock_db.query.assert_called_once_with(MarketData)
    mock_db.query.return_value.filter.assert_called_once()
    mock_db.query.return_value.filter.return_value.delete.assert_called_once()

    # Verify add_all is called with the right number of records
    mock_db.add_all.assert_called_once()
    added_records = mock_db.add_all.call_args[0][0]
    assert len(added_records) == 2

    assert added_records[0].ticker == "^GSPC"
    assert added_records[0].close == 102.0
    assert added_records[1].ticker == "^GSPC"
    assert added_records[1].close == 103.0

    # Verify commit
    mock_db.commit.assert_called_once()
