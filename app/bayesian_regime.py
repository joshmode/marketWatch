import numpy as np
import pandas as pd


def _normal_pdf(x, mean, std):
    std = max(std, 1e-6)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def compute_bayesian_regime(market_data: pd.DataFrame) -> pd.DataFrame:
    required_fields = [
        "macro_score",
        "credit_stress_z",
        "curve_slope",
        "macro_liquidity_z",
        "dollar_regime_z"
    ]

    for field in required_fields:
        if field not in market_data.columns:
            market_data[field] = 0.0

    # each regime defined by how it "expects" each macro input to behave
    regime_templates = {
        "Expansion": {
            "macro_score": (1.0, 0.8),
            "credit_stress_z": (-0.5, 1.0),
            "curve_slope": (1.0, 1.0),
            "macro_liquidity_z": (-0.5, 1.0),
            "dollar_regime_z": (-0.2, 1.0),
        },
        "Slowdown": {
            "macro_score": (0.0, 0.7),
            "credit_stress_z": (0.5, 1.0),
            "curve_slope": (0.0, 1.0),
            "macro_liquidity_z": (0.5, 1.0),
            "dollar_regime_z": (0.5, 1.0),
        },
        "Stress": {
            "macro_score": (-1.5, 0.8),
            "credit_stress_z": (2.0, 1.2),
            "curve_slope": (-1.0, 1.5),
            "macro_liquidity_z": (1.5, 1.2),
            "dollar_regime_z": (1.5, 1.2),
        },
    }

    regime_names = list(regime_templates.keys())
    prior = np.full(len(regime_names), 1 / len(regime_names))

    probability_path = []

    for _, row in market_data.iterrows():
        likelihoods = []

        for regime in regime_names:
            regime_definition = regime_templates[regime]

            likelihood = 1.0
            for factor, (mean, std) in regime_definition.items():
                likelihood *= _normal_pdf(row[factor], mean, std)

            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)

        if likelihoods.sum() == 0:
            likelihoods[:] = 1 / len(likelihoods)

        posterior = prior * likelihoods
        posterior /= posterior.sum()

        probability_path.append(posterior)

        # smooth transition so regimes don't flip unrealistically fast
        prior = posterior * 0.9 + (1 / len(prior)) * 0.1

    regime_df = pd.DataFrame(
        probability_path,
        columns=[f"P_{name}" for name in regime_names],
        index=market_data.index
    )

    regime_df["Regime"] = regime_df.idxmax(axis=1).str.replace("P_", "")

    return regime_df
