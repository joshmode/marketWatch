import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

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


def build_market_dataset(ticker: str, period: str):
    price_data = fetch_data(ticker, period)
    price_data = add_indicators(price_data)
    price_data = enrich_macro_data(price_data)

    regime = compute_bayesian_regime(price_data)
    price_data = price_data.join(regime)

    return run_backtest(price_data)


@app.get("/", response_class=HTMLResponse)
async def dashboard_view(ticker: str = "^GSPC", period: str = "2y"):
    try:
        # Run synchronous blocking tasks in a separate thread
        dataset = await asyncio.to_thread(build_market_dataset, ticker, period)
        macro_summary = await asyncio.to_thread(get_macro_summary)

        live_symbols = ["^GSPC", "^IXIC", "^VIX"]

        # Concurrently fetch live tickers
        live_infos = await asyncio.gather(*(fetch_live_ticker(sym) for sym in live_symbols))

        live_data = []
        for symbol, info in zip(live_symbols, live_infos):
            if symbol == "^GSPC":
                info["symbol"] = "S&P 500"
            elif symbol == "^IXIC":
                info["symbol"] = "NASDAQ"
            elif symbol == "^VIX":
                info["symbol"] = "VIX"

            live_data.append(info)

        figure = await asyncio.to_thread(create_dashboard, dataset, live_data, macro_summary)
        chart_json = figure.to_json()

        html = f"""
        <html>
        <head>
            <title>marketWatch by @joshmode</title>
            <style>
                body {{ font-family: sans-serif; background-color: #111; color: #eee; margin: 0; padding: 20px; }}
                .container {{ max_width: 1600px; margin: 0 auto; }}
                .macro-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em; }}
                .macro-table th, .macro-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #333; }}
                .macro-table th {{ color: #888; }}
                h2 {{ color: #00ffcc; }}
            </style>
        </head>
        <body>
        <div class="container">
            <h2>marketWatch by @joshmode</h2>

            <div style="display: grid; grid-template-columns: 1fr 3fr; gap: 20px;">
                <div>
                    <h3>Macro State</h3>
                    <table class="macro-table">
                        <tr><th>Indicator</th><th>Value</th><th>Date</th></tr>
                        {''.join([f"<tr><td>{k}</td><td>{v['value']}</td><td>{v['date']}</td></tr>" for k, v in macro_summary.items()])}
                    </table>
                </div>
                <div id="chart"></div>
            </div>
        </div>

        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
        var chart = {chart_json};
        Plotly.newPlot('chart', chart.data, chart.layout);
        </script>
        </body>
        </html>
        """

        return HTMLResponse(html)

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
