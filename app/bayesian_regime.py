import numpy as np
import pandas as pd


def _normal_pdf(x, m, s):
    s = max(s, 1e-6)
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def compute_bayesian_regime(df):
    req = [
        "macro_score",
        "credit_stress_z",
        "curve_slope",
        "macro_liquidity_z",
        "dollar_regime_z"
    ]

    for c in req:
        if c not in df.columns:
            df[c] = 0.0

    tmpl = {
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

    names = list(tmpl.keys())
    n = len(df)
    n_reg = len(names)

    lik = np.ones((n, n_reg))

    for j, name in enumerate(names):
        for f, (m, s) in tmpl[name].items():
            vals = df[f].to_numpy()
            lik[:, j] *= _normal_pdf(vals, m, s)

    prior = np.full(n_reg, 1.0 / n_reg)
    probs = []

    for i in range(n):
        r_lik = lik[i]

        if r_lik.sum() == 0:
            r_lik[:] = 1.0 / n_reg

        post = prior * r_lik
        post /= post.sum()

        probs.append(post)
        prior = post * 0.9 + (1.0 / n_reg) * 0.1

    res = pd.DataFrame(
        probs,
        columns=[f"P_{name}" for name in names],
        index=df.index
    )

    res["Regime"] = res.idxmax(axis=1).str.replace("P_", "")

    return res
