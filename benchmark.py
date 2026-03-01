import time
import numpy as np
import pandas as pd
from app.bayesian_regime import compute_bayesian_regime

# Generate some fake data
np.random.seed(42)
n_rows = 10000
data = {
    "macro_score": np.random.randn(n_rows),
    "credit_stress_z": np.random.randn(n_rows),
    "curve_slope": np.random.randn(n_rows),
    "macro_liquidity_z": np.random.randn(n_rows),
    "dollar_regime_z": np.random.randn(n_rows)
}
df = pd.DataFrame(data)

start_time = time.time()
result = compute_bayesian_regime(df)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")
print(result.head())
