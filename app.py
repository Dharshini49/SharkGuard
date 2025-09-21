# app.py
"""
SharkGuard Streamlit app
- Place this file in the project root (same level as 'sharkguard/' and 'utils/' folders).
- Ensure you have a Python package 'sharkguard' (create an empty sharkguard/__init__.py if needed).
- Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import traceback

# Paths
MODEL_PATH = Path("models/isolation_model.joblib")
SIM_FEATURES = Path("data/simulated_features.csv")
MODELS_DIR = MODEL_PATH.parent
DATA_DIR = SIM_FEATURES.parent

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Attempt to import required project modules; show an informative error if they are missing.
try:
    from sharkguard.core import SharkGuardModel, txs_to_dataframe, extract_wallet_features, train_and_persist_model
except Exception as e:
    st.set_page_config(page_title="SharkGuard - Missing modules", layout="centered")
    st.title("🦈 SharkGuard — Missing project files")
    st.error(
        "Could not import `sharkguard.core`. Make sure:\n"
        "  • `sharkguard/core.py` exists, AND\n"
        "  • `sharkguard/__init__.py` exists (an empty file is fine) so Python treats it as a package.\n\n"
        "Error details:\n" + repr(e)
    )
    st.stop()

# Try to import Etherscan util. If missing we provide a simple stub (app will continue but fetch won't work).
try:
    from utils.etherscan import fetch_transactions
except Exception:
    def fetch_transactions(address, api_key, *args, **kwargs):
        # fallback: return empty list so UI still runs
        return []

# Cache model loading so Streamlit doesn't reload it every rerun
@st.cache_resource
def load_model(path: str):
    sg = SharkGuardModel()
    sg.load(path)
    return sg

# UI
st.set_page_config(page_title="SharkGuard", layout="centered")
st.title("🦈 SharkGuard — Web3 Fake Account Detector")

st.sidebar.header("Settings")
etherscan_key = st.sidebar.text_input("Etherscan API Key (optional)", type="password")
wallet = st.sidebar.text_input("Wallet address (0x...)")
use_sample = st.sidebar.checkbox("Use simulated sample features (no on-chain fetch)", value=False)

# Model controls
st.sidebar.markdown("---")
st.sidebar.subheader("Model")
model_status = "Not loaded"
sg = None
if MODEL_PATH.exists():
    try:
        sg = load_model(str(MODEL_PATH))
        model_status = f"Loaded from {MODEL_PATH}"
    except Exception as e:
        model_status = f"Failed to load ({e})"
st.sidebar.write(model_status)

if st.sidebar.button("🔁 Reload model"):
    if MODEL_PATH.exists():
        try:
            sg = load_model(str(MODEL_PATH))
            st.sidebar.success("Model reloaded.")
        except Exception as e:
            st.sidebar.error("Reload failed: " + str(e))
    else:
        st.sidebar.warning("No model file found. Train or upload one.")

# Upload model (.joblib)
uploaded = st.sidebar.file_uploader("Upload model (.joblib)", type=["joblib"])
if uploaded is not None:
    try:
        data = uploaded.read()
        MODEL_PATH.write_bytes(data)
        sg = load_model(str(MODEL_PATH))
        st.sidebar.success("Uploaded and loaded model.")
    except Exception as e:
        st.sidebar.error("Failed to save/load uploaded model: " + str(e))

# Train from simulated features (if available)
st.sidebar.markdown("---")
if st.sidebar.button("⚙️ Train model from simulated data"):
    if SIM_FEATURES.exists():
        try:
            df = pd.read_csv(SIM_FEATURES)
            train_and_persist_model(df, path=str(MODEL_PATH))
            sg = load_model(str(MODEL_PATH))
            st.sidebar.success("Model trained and saved to " + str(MODEL_PATH))
        except Exception as e:
            st.sidebar.error("Training failed: " + str(e))
            st.sidebar.text(traceback.format_exc())
    else:
        st.sidebar.warning("Simulated features file not found. Run `python data/simulate.py` first.")

st.sidebar.markdown("---")
st.sidebar.markdown("Need help? Make sure `sharkguard/core.py` and `utils/etherscan.py` are present.")

# Main area
st.write("Use the sidebar to load/train/upload a model. Then enter a wallet and click Analyze.")
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Analyze wallet")
    analyze = st.button("🔍 Analyze")

with col2:
    if SIM_FEATURES.exists():
        st.caption(f"Sim features: {SIM_FEATURES.name}")
    else:
        st.caption("No simulated features found")

# When Analyze is pressed
if analyze:
    if use_sample:
        # show a random row from simulated features CSV
        if SIM_FEATURES.exists():
            df_feats = pd.read_csv(SIM_FEATURES)
            st.success("Loaded simulated features — showing a random sample")
            st.dataframe(df_feats.sample(1).T)
        else:
            st.error("Simulated features file not present. Run: python data/simulate.py")
    else:
        if not wallet:
            st.error("Please enter a wallet address (0x...) or enable 'Use simulated sample features'.")
        else:
            # Fetch transactions (may be empty)
            txs = []
            if etherscan_key:
                try:
                    with st.spinner("Fetching transactions from Etherscan..."):
                        txs = fetch_transactions(wallet, etherscan_key)
                except Exception as e:
                    st.error("Etherscan fetch failed: " + str(e))
                    txs = []
            else:
                st.info("No Etherscan key provided — will not fetch on-chain transactions (empty).")

            # Convert to dataframe and extract features
            try:
                df = txs_to_dataframe(txs)
            except Exception as e:
                st.error("Failed to convert transactions to DataFrame: " + str(e))
                df = pd.DataFrame()

            feat = extract_wallet_features(df, wallet)
            st.subheader("Extracted features")
            st.json(feat)

            # Model inference
            if sg is None:
                st.warning("No model loaded. Train or upload a model to get a suspicion score.")
            else:
                try:
                    res = sg.predict_score(feat)
                    st.metric("Suspicion Score (0 = normal, 1 = suspicious)", f"{res['score']:.3f}")
                    st.write("Label:", res["label"])
                    st.write("Raw model score:", res["raw"])
                except Exception as e:
                    st.error("Model prediction failed: " + str(e))
                    st.text(traceback.format_exc())

            # Heuristic explanations
            st.subheader("Heuristic signals")
            expl = []
            if feat.get("tx_count", 0) < 3:
                expl.append("Very few transactions — new or dormant account")
            if feat.get("tx_freq_per_day", 0) > 50:
                expl.append("Extremely high transaction frequency — bot-like behavior")
            if feat.get("repeated_ratio", 0) > 0.6:
                expl.append("High repeated_ratio — interacting with the same counterparty often")
            if feat.get("hour_entropy", 10) < 1.0:
                expl.append("Low hour entropy — very regular timing")
            if not expl:
                expl.append("No strong heuristic flags detected ✅")
            for e in expl:
                st.write("- ", e)

            # Show raw txs
            st.subheader("Raw transactions (first 20)")
            if not df.empty:
                st.dataframe(df.head(20))
            else:
                st.write("No transactions to show (empty).")
