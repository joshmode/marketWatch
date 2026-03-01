import time
import numpy as np
import pandas as pd
from app.bayesian_regime import compute_bayesian_regime

np.random.seed(42)
n_rows = 10000
market_data = pd.DataFrame({
    "macro_score": np.random.randn(n_rows),
    "credit_stress_z": np.random.randn(n_rows),
    "curve_slope": np.random.randn(n_rows),
    "macro_liquidity_z": np.random.randn(n_rows),
    "dollar_regime_z": np.random.randn(n_rows)
})

start_time = time.time()
res = compute_bayesian_regime(market_data.copy())
end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")
