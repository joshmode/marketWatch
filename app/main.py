import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.data import fetch_data, fetch_live_ticker
from app.indicators import add_indicators
from app.macro import enrich_macro_data, get_macro_summary
from app.bayesian_regime import compute_bayesian_regime
from app.backtest import run_backtest
from app.dashboard import create_dashboard
from app.overlay import build_overlay_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Macro Signal Engine")

templates = Jinja2Templates(directory="app/templates")


def build_market_dataset(ticker: str, period: str):
    price_data = fetch_data(ticker, period)
    price_data = add_indicators(price_data)
    price_data = enrich_macro_data(price_data)

    regime = compute_bayesian_regime(price_data)
    price_data = price_data.join(regime)

    return run_backtest(price_data)


@app.get("/", response_class=HTMLResponse)
def dashboard_view(request: Request, ticker: str = "^GSPC", period: str = "2y"):
    try:
        dataset = build_market_dataset(ticker, period)

        live_symbols = ["^GSPC", "^IXIC", "^VIX"]
        live_data = []

        for symbol in live_symbols:
            info = fetch_live_ticker(symbol)

            if symbol == "^GSPC":
                info["symbol"] = "S&P 500"
            elif symbol == "^IXIC":
                info["symbol"] = "NASDAQ"
            elif symbol == "^VIX":
                info["symbol"] = "VIX"

            live_data.append(info)

        macro_summary = get_macro_summary()

        figure = create_dashboard(dataset, live_data, macro_summary)
        chart_json = figure.to_json()

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "macro_summary": macro_summary,
                "chart_json": chart_json,
            },
        )

    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return HTMLResponse(f"<h1>Error</h1><pre>{str(e)}</pre>", status_code=500)


@app.get("/api/overlay")
def macro_overlay(ticker: str = "^GSPC", period: str = "2y"):
    try:
        dataset = build_market_dataset(ticker, period)
        overlay = build_overlay_signal(dataset)
        return JSONResponse(overlay)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
