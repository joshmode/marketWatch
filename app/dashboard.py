import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional


def create_dashboard(
    market_data: pd.DataFrame,
    live_data: Optional[List[Dict[str, Any]]] = None,
    macro_summary: Optional[Dict[str, Any]] = None
):
    specs = [
        [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        [{"colspan": 3}, None, None],
        [{"colspan": 3}, None, None],
        [{"colspan": 3}, None, None],
        [{"colspan": 3}, None, None],
    ]

    figure = make_subplots(
        rows=5,
        cols=3,
        specs=specs,
        row_heights=[0.15, 0.35, 0.2, 0.15, 0.15],
        vertical_spacing=0.03,
        subplot_titles=(
            "", "", "",
            "Market + Macro Signal Overlay",
            "Macro Regime Probabilities",
            "Recession Risk & Liquidity Stress",
            "Fear & Greed / Market Stress",
        ),
    )

    # live indicators
    if live_data:
        for i, ticker in enumerate(live_data):
            if i >= 3:
                break

            color = "green" if ticker["change"] >= 0 else "red"

            figure.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=ticker["price"],
                    delta={
                        "reference": ticker["price"] - ticker["change"],
                        "relative": True,
                        "valueformat": ".2%",
                        "position": "bottom",
                    },
                    title={"text": ticker["symbol"]},
                    number={"font": {"color": color}},
                ),
                row=1,
                col=i + 1,
            )
    else:
        figure.add_trace(
            go.Indicator(mode="number", value=0, title={"text": "Live Data Unavailable"}),
            row=1,
            col=1,
        )

    # IMPORTANT: convert everything to lists before plotting
    x_vals = market_data.index.tolist()

    if "Cumulative_Strategy" in market_data.columns:
        figure.add_trace(
            go.Scatter(
                x=x_vals,
                y=market_data["Cumulative_Strategy"].tolist(),
                name="Strategy Equity",
                line=dict(color="lime"),
            ),
            row=2,
            col=1,
        )

        initial_capital = market_data["Cumulative_Strategy"].iloc[0]
        benchmark = (market_data["Close"] / market_data["Close"].iloc[0]) * initial_capital

        figure.add_trace(
            go.Scatter(
                x=x_vals,
                y=benchmark.tolist(),
                name="Benchmark (Buy & Hold)",
                line=dict(color="white", width=1),
                opacity=0.5,
            ),
            row=2,
            col=1,
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=x_vals,
                y=market_data["Close"].tolist(),
                name="Price",
                line=dict(color="white", width=1),
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

    if "P_Expansion" in market_data.columns:
        figure.add_trace(go.Scatter(x=x_vals, y=market_data["P_Expansion"].tolist(),
                                   name="Expansion Prob", line=dict(color="green"), stackgroup="one"), row=3, col=1)

        figure.add_trace(go.Scatter(x=x_vals, y=market_data["P_Slowdown"].tolist(),
                                   name="Slowdown Prob", line=dict(color="orange"), stackgroup="one"), row=3, col=1)

        figure.add_trace(go.Scatter(x=x_vals, y=market_data["P_Stress"].tolist(),
                                   name="Stress Prob", line=dict(color="red"), stackgroup="one"), row=3, col=1)

    if "recession_probability" in market_data.columns:
        figure.add_trace(go.Scatter(x=x_vals, y=market_data["recession_probability"].tolist(),
                                   name="Recession Prob", line=dict(color="red", dash="dot")), row=4, col=1)

    if "liquidity_stress_index" in market_data.columns:
        figure.add_trace(go.Scatter(x=x_vals, y=market_data["liquidity_stress_index"].tolist(),
                                   name="Liquidity Stress Index", line=dict(color="cyan")), row=4, col=1)
    elif "macro_liquidity_z" in market_data.columns:
        figure.add_trace(go.Scatter(x=x_vals, y=market_data["macro_liquidity_z"].tolist(),
                                   name="Liquidity Stress (Z)", line=dict(color="cyan")), row=4, col=1)

    if "Fear_Greed" in market_data.columns:
        figure.add_trace(go.Scatter(x=x_vals, y=market_data["Fear_Greed"].tolist(),
                                   name="Fear & Greed Proxy", line=dict(color="yellow")), row=5, col=1)

        figure.add_trace(go.Scatter(x=x_vals, y=[80] * len(x_vals),
                                   line=dict(color="green", dash="dot", width=1),
                                   showlegend=False, hoverinfo="skip"), row=5, col=1)

        figure.add_trace(go.Scatter(x=x_vals, y=[20] * len(x_vals),
                                   line=dict(color="red", dash="dot", width=1),
                                   showlegend=False, hoverinfo="skip"), row=5, col=1)

    if "Market_Stress" in market_data.columns:
        figure.add_trace(go.Scatter(x=x_vals, y=market_data["Market_Stress"].tolist(),
                                   name="Market Stress", line=dict(color="magenta")), row=5, col=1)

    figure.update_layout(template="plotly_dark", height=1400,
                         title_text="Follow me on GitHub @joshmode")

    return figure
