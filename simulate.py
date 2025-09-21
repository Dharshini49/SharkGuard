# data/simulate.py
"""
Simulate synthetic wallet transactions and extract features for training SharkGuard.

Usage:
    python data/simulate.py

This will generate:
    data/simulated_features.csv
which you can then use to train a model with sharkguard.core.train_and_persist_model().
"""

import random
from datetime import datetime, timedelta
import os

# Import feature extraction tools from core
from sharkguard.core import txs_to_dataframe, extract_wallet_features, features_to_df


def simulate_wallet(num_txs=50, start_days_ago=180, wallet_addr="0xwallet"):
    """
    Generate a list of fake transactions for a wallet.
    Each transaction has fields similar to Etherscan API output.
    """
    txs = []
    now = datetime.utcnow()
    for i in range(num_txs):
        ts = now - timedelta(seconds=random.randint(0, start_days_ago * 24 * 3600))
        other = "0x" + "".join(random.choices("0123456789abcdef", k=40))
        value = random.random() * (0.1 if random.random() > 0.95 else 1.0)
        tx = {
            "hash": "0x" + "".join(random.choices("0123456789abcdef", k=64)),
            "from": wallet_addr if random.random() < 0.5 else other,
            "to": other if random.random() < 0.5 else wallet_addr,
            "value": int(value * 1e18),  # in wei
            "gas": random.randint(21000, 200000),
            "gasPrice": random.randint(20, 200) * (10**9),  # in wei
            "timeStamp": int(ts.timestamp())
        }
        txs.append(tx)
    return txs


def main():
    # Create synthetic dataset of multiple wallets
    wallets = []
    for i in range(200):
        w = "0x" + "".join(random.choices("0123456789abcdef", k=40))
        if i < 20:
            # make some wallets look more suspicious
            txs = simulate_wallet(num_txs=random.randint(100, 200), wallet_addr=w)
        else:
            txs = simulate_wallet(num_txs=random.randint(5, 80), wallet_addr=w)
        wallets.append({"wallet": w, "txs": txs})

    # Extract features
    feats = []
    for entry in wallets:
        df = txs_to_dataframe(entry["txs"])
        feat = extract_wallet_features(df, entry["wallet"])
        feats.append(feat)

    df_feats = features_to_df(feats)

    # Save to CSV
    out_path = os.path.join("data", "simulated_features.csv")
    df_feats.to_csv(out_path, index=False)
    print(f"✅ Saved {out_path}")


if __name__ == "__main__":
    main()
