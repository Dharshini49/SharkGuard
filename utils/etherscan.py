# utils/etherscan.py
"""
Simple wrapper to fetch Ethereum transactions from the Etherscan API.

Requires an Etherscan API key.
Usage:
    from utils.etherscan import fetch_transactions
    txs = fetch_transactions("0x1234...", "YOUR_API_KEY")
"""

import requests

ETHERSCAN_BASE = "https://api.etherscan.io/api"


def fetch_transactions(address, api_key, startblock=0, endblock=99999999, sort="asc"):
    """
    Return a list of transactions (dicts) for a given address.
    If API key is missing or API returns an error, returns [].
    """
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": startblock,
        "endblock": endblock,
        "sort": sort,
        "apikey": api_key,
    }
    try:
        r = requests.get(ETHERSCAN_BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "1":
            return []  # no txs or error
        return data.get("result", [])
    except Exception as e:
        print("⚠️  Etherscan request failed:", e)
        return []
