import pandas as pd


def build_overlay_signal(market_data: pd.DataFrame) -> dict:
    if market_data.empty:
        return {}

    latest = market_data.iloc[-1]

    expansion_prob = latest.P_Expansion
    slowdown_prob = latest.P_Slowdown
    stress_prob = latest.P_Stress

    # blended risk posture derived from regime probabilities
    recommended_risk = (
        expansion_prob * 1.0 +
        slowdown_prob * 0.6 +
        stress_prob * 0.2
    )

    equity_beta = (
        expansion_prob * 1.1 +
        slowdown_prob * 0.7 +
        stress_prob * 0.3
    )

    recession_prob = latest.get("recession_probability", 0.0)
    fear_greed = latest.get("Fear_Greed", 0.0)
    market_stress = latest.get("Market_Stress", 0.0)
    liquidity_z = latest.get("macro_liquidity_z", 0.0)
    liquidity_index = latest.get("liquidity_stress_index", 0.0)

    confidence = max(expansion_prob, slowdown_prob, stress_prob)

    return {
        "timestamp": str(latest.name),

        "regime_probabilities": {
            "expansion": float(expansion_prob),
            "slowdown": float(slowdown_prob),
            "stress": float(stress_prob),
        },

        "macro_score": float(latest.macro_score),
        "recession_probability": float(recession_prob),

        "risk_diagnostics": {
            "fear_greed": float(fear_greed),
            "market_stress": float(market_stress),
            "liquidity_stress_z": float(liquidity_z),
            "liquidity_stress_index": float(liquidity_index),
        },

        "recommended_risk_level": float(recommended_risk),
        "equity_beta_overlay": float(equity_beta),

        "confidence": float(confidence),
    }
