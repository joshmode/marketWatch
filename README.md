# PELOSI: Macro-Driven Market Regime Intelligence Engine

PELOSI is a sophisticated market intelligence engine designed to detect market regimes (Expansion, Slowdown, Stress) using a fusion of macroeconomic data, technical structure, and liquidity signals. It is built for research and signal generation, not automated execution.

## Core Features

### 1. Market Data Layer
*   **Direct API Ingestion:** Fetches historical and live market data (OHLCV) directly from Yahoo Finance without third-party wrappers, ensuring robustness in restricted environments.
*   **Live Tickers:** Real-time monitoring of S&P 500 (`^GSPC`), NASDAQ (`^IXIC`), and VIX (`^VIX`).

### 2. Macro Data Integration (FRED)
*   **Economic Indicators:** Ingests live data from the Federal Reserve Economic Data (FRED) API, including:
    *   **Growth:** Real GDP, Industrial Production, Payrolls.
    *   **Inflation:** CPI, Core CPI, Inflation Persistence.
    *   **Policy:** Fed Funds Rate, Yield Curve (10Y-2Y), Real Rates.
    *   **Liquidity:** Credit Spreads, Financial Conditions.
*   **Macro Score:** Computes a composite Z-score of macro conditions to drive regime probability.

### 3. Regime Detection Engine
*   **Bayesian Switching Model:** A recursive Bayesian filter that estimates the probability of three market regimes:
    *   **Expansion:** High growth, stable inflation, ample liquidity.
    *   **Slowdown:** Decelerating growth, tightening liquidity.
    *   **Stress:** Recession risk, credit widening, volatility spikes.
*   **Recession Probability:** Logistic regression model estimating real-time recession risk based on yield curve, credit, and labor data.

### 4. Technical & Risk Overlay
*   **Market Structure:** SMA trends, Momentum drift, RSI positioning.
*   **Fear & Greed Proxy:** Custom sentiment index derived from RSI and Volatility Z-scores.
*   **Liquidity Stress:** Measures systemic strain via credit spreads and rate volatility.

### 5. Machine Learning Layer
*   **Contextual Alpha:** LightGBM model trained on normalized macro-technical features to predict forward returns (research component).
*   **Purged Cross-Validation:** Uses purged walk-forward validation to prevent data leakage.

## Architecture

The system is modular and API-first:

*   `app/data.py`: Robust data fetching (Yahoo Finance).
*   `app/macro.py`: FRED integration and macro feature engineering.
*   `app/bayesian_regime.py`: Probabilistic regime modeling.
*   `app/indicators.py`: Technical analysis library.
*   `app/dashboard.py`: Plotly-based interactive visualization.
*   `app/main.py`: FastAPI entry point serving the dashboard and JSON APIs.

## Installation & Setup

### Prerequisites
*   Python 3.10+
*   FRED API Key (Get one free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html))

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/pelosi-engine.git
    cd pelosi-engine
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Set your FRED API key as an environment variable:
    ```bash
    export FRED_API_KEY="your_api_key_here"
    ```

4.  **Run the Application:**
    ```bash
    uvicorn app.main:app --reload
    ```

5.  **Access Dashboard:**
    Open your browser to `http://127.0.0.1:8000`.

## Usage

*   **Dashboard:** View the interactive dashboard with live tickers, macro summary, and regime charts.
*   **API Endpoint:** Get the raw overlay signal JSON:
    *   `GET /api/overlay?ticker=^GSPC&period=2y`

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice. Trading involves risk.
