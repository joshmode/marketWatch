import time
import requests

try:
    print("Testing network connection...")
    response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/^GSPC?range=1d&interval=1d", timeout=5)
    print("Success! Status code:", response.status_code)
except Exception as e:
    print("Failed to reach Yahoo Finance:", e)
