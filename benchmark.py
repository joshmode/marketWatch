import time
import pandas as pd
from app.macro import load_macro_data, enrich_macro_data

def run_benchmark():
    # Warmup
    print("Warming up...")
    df = pd.DataFrame(index=pd.date_range("2020-01-01", "2023-01-01"))
    load_macro_data()

    print("Benchmarking load_macro_data...")
    start = time.perf_counter()
    for _ in range(100):
        load_macro_data()
    end = time.perf_counter()
    print(f"load_macro_data: {(end - start) / 100:.6f} seconds per call")

    print("Benchmarking enrich_macro_data...")
    start = time.perf_counter()
    for _ in range(10):
        enrich_macro_data(df)
    end = time.perf_counter()
    print(f"enrich_macro_data: {(end - start) / 10:.6f} seconds per call")

if __name__ == "__main__":
    run_benchmark()
