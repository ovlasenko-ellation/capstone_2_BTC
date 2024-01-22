import requests
import json

#response = requests.post(url, json=data)
#result = response.json()
#print(json.dumps(result, indent=1))

import requests

if __name__ == "__main__":
    url = f"http://localhost:8080/predict"

    input = {
        "date": "2023-12-27",
        "open": 42518.468750,
        "high": 43683.160156,
        "low": 42167.582031,
        "close": 43442.855469,
        "adj_close": 43442.855469,
        "volume": 25260941032,
        "price_change": 922.453125,
        "rsi": 52.953577,
        "is_bull": "true",
        "overbought": "false",
        "oversold": "false"
    }

    response = requests.post(url, json=input)
    print(response)
    print(response.json())