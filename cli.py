# cli.py
"""
Command-line tool for SharkGuard
--------------------------------
Usage examples (run in PowerShell / CMD):
    python cli.py --wallet 0x1234567890abcdef... --etherscan_key YOUR_API_KEY
    python cli.py --wallet 0x1234567890abcdef...   # (no etherscan fetch, empty txs)

Requires:
 - sharkguard/core.py
 - utils/etherscan.py
 - A trained model saved in models/isolation_model.joblib
"""

import argparse
import json
from sharkguard.core import SharkGuardModel, txs_to_dataframe, extract_wallet_features
from utils.etherscan import fetch_transactions


def main():
    parser = argparse.ArgumentParser(description="SharkGuard CLI Wallet Analyzer")
    parser.add_argument("--wallet", required=True, help="Wallet address (0x...)")
    parser.add_argument("--etherscan_key", required=False, help="Etherscan API Key")
    parser.add_argument("--model", default="models/isolation_model.joblib",
                        help="Path to trained model (default: models/isolation_model.joblib)")
    args = parser.parse_args()

    wallet = args.wallet
    api_key = args.etherscan_key

    # Fetch transactions (if API key provided)
    txs = []
    if api_key:
        try:
            txs = fetch_transactions(wallet, api_key)
        except Exception as e:
            print("⚠️  Etherscan fetch failed:", e)
            txs = []

    # Convert to DataFrame + extract features
    df = txs_to_dataframe(txs)
    feat = extract_wallet_features(df, wallet)

    # Load trained model
    sg = SharkGuardModel()
    try:
        sg.load(args.model)
    except Exception as e:
        print("❌ Could not load model:", e)
        print("Make sure you have trained and saved one at", args.model)
        return

    # Predict
    res = sg.predict_score(feat)

    # Output as JSON
    out = {
        "wallet": wallet,
        "features": feat,
        "result": res
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
