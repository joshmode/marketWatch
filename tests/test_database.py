import pytest
from unittest.mock import MagicMock, patch
from app.database import get_db

def test_get_db_yields_session():
    mock_session = MagicMock()
    with patch('app.database.SessionLocal', return_value=mock_session):
        gen = get_db()
        session = next(gen)

        assert session == mock_session
        mock_session.close.assert_not_called()

        with pytest.raises(StopIteration):
            next(gen)

        mock_session.close.assert_called_once()

def test_get_db_closes_session_on_exception():
    mock_session = MagicMock()
    with patch('app.database.SessionLocal', return_value=mock_session):
        gen = get_db()
        session = next(gen)

        assert session == mock_session
        mock_session.close.assert_not_called()

        with pytest.raises(ValueError):
            try:
                raise ValueError("Test Error")
            finally:
                gen.close() # Close generator to trigger finally block, simulating exception in caller

        mock_session.close.assert_called_once()
