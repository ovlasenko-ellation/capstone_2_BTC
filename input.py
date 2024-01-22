import json

import requests

if __name__ == "__main__":
    #url = 'https://g72eqje1bk.execute-api.us-west-2.amazonaws.com/STAGE/predict'

    url = f"http://localhost:8080/predict"    #DOCKET URL
    #url = "http://127.0.0.1:9696/predict"    #local host

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