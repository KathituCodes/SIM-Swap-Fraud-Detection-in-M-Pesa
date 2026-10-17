import inspect
import json
import re
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

"""
M-Pesa SIM Swap Fraud Detection Dashboard
Design : Inspired by the My Safaricom App visual identity
Author : Urbanus Kathitu (Kat.Codes)
GitHub : github.com/KathituCodes/SIM-Swap-Fraud-Detection-M-Pesa-ML
"""

# PAGE CONFIG: must be the very first Streamlit call
st.set_page_config(
    page_title="M-Pesa Fraud Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
# Model and config files live in ./models next to app.py
# (falls back to the app folder if there is no models/ folder)
MODEL_DIR = BASE_DIR / "models" if (BASE_DIR / "models").is_dir() else BASE_DIR

# SAFARICOM BRAND COLORS
SAF_GREEN = "#30B54A"
SAF_DARK_GREEN = "#006633"
SAF_RED = "#E2001A"
SAF_LIGHT = "#F0FAF2"
SAF_GREY = "#F5F5F5"
SAF_DARK_TEXT = "#1A1A1A"
SAF_MID_TEXT = "#555555"
WHITE = "#FFFFFF"


# ─────────────────────────────────────────────────────────────
# VERSION-SAFE WIDTH HELPER
# Newer Streamlit uses width="stretch"; older versions use
# use_container_width=True. This picks whichever the installed
# version supports so the app does not break on upgrades.
# ─────────────────────────────────────────────────────────────
def stretch(fn):
    params = inspect.signature(fn).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


st.markdown(f"""
<style>
[data-testid="stAppViewContainer"], .main {{ background-color: #F8FAF8; }}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {SAF_DARK_GREEN} 0%, #004d26 100%);
}}
[data-testid="stMetric"], [data-testid="metric-container"] {{
    background-color: {WHITE};
    border: 1px solid #E0E0E0;
    border-left: 5px solid {SAF_GREEN};
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div {{
    color: {SAF_MID_TEXT} !important;
}}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {{
    color: {SAF_DARK_TEXT} !important;
}}
.fraud-alert {{
    background: linear-gradient(135deg, #FFF0F0, #FFE5E5);
    border: 2px solid {SAF_RED};
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 12px 0;
    box-shadow: 0 4px 12px rgba(226,0,26,0.15);
}}
.legit-alert {{
    background: linear-gradient(135deg, #F0FAF2, #E8F5E9);
    border: 2px solid {SAF_GREEN};
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 12px 0;
    box-shadow: 0 4px 12px rgba(48,181,74,0.15);
}}
.info-box {{
    background-color: {SAF_LIGHT};
    border-left: 5px solid {SAF_GREEN};
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: {SAF_DARK_GREEN};
    line-height: 1.6;
}}
.warning-box {{
    background-color: #FFF8E1;
    border-left: 5px solid #FFA000;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: #5D4037;
    line-height: 1.6;
}}
.section-header {{
    background: linear-gradient(90deg, {SAF_DARK_GREEN}, {SAF_GREEN});
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 16px;
    letter-spacing: 0.3px;
}}
.stat-card {{
    background: white;
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #E8F5E9;
    border-top: 4px solid {SAF_GREEN};
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    height: 100%;
}}
/* Main-page widget labels: dark text on the light background */
label[data-testid="stWidgetLabel"] p {{
    font-weight: 600 !important;
    color: {SAF_DARK_TEXT} !important;
    font-size: 14px !important;
}}
/* Sidebar widget labels: white text on the dark green sidebar only */
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [role="radiogroup"] p {{
    color: #FFFFFF !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(90deg, {SAF_DARK_GREEN}, {SAF_GREEN});
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 32px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 12px rgba(48,181,74,0.3);
}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS AND CONFIG
# Paths are resolved relative to this file so the app works no
# matter which folder Streamlit is launched from.
# JSON formats are preferred because they survive library upgrades;
# .pkl files are kept as a fallback.
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "high_value_threshold": 3800.0,
    "wipeout_threshold": 100.0,
    "decision_threshold": 0.5,
    "feature_ranges": {},
    "metrics": None,
    "dataset": {"rows": 120000, "fraud": 3510, "fraud_rate": 2.93, "imbalance": 33.2},
}


@st.cache_resource
def load_models():
    from xgboost import XGBClassifier

    xgb_json = MODEL_DIR / "xgboost_fraud_model.json"
    if xgb_json.exists():
        xgb_model = XGBClassifier()
        xgb_model.load_model(str(xgb_json))
    else:
        xgb_model = joblib.load(MODEL_DIR / "xgboost_fraud_model.pkl")

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    woe_model = joblib.load(MODEL_DIR / "woe_scorecard_model.pkl")

    bins_json = MODEL_DIR / "woe_bins.json"
    if bins_json.exists():
        with open(bins_json) as f:
            woe_bins = json.load(f)
    else:
        with open(MODEL_DIR / "woe_bins.pkl", "rb") as f:
            raw = pickle.load(f)
        woe_bins = {k: v[["bin", "woe"]].to_dict("records") for k, v in raw.items()}

    config = dict(DEFAULT_CONFIG)
    cfg_path = MODEL_DIR / "app_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            config.update(json.load(f))

    return xgb_model, scaler, woe_model, woe_bins, config


try:
    xgb_model, scaler, woe_model, woe_bins, CONFIG = load_models()
except Exception as e:
    st.error(
        "Model files could not be loaded. Check that the model files sit in the "
        f"models/ folder next to app.py and that requirements.txt versions match training.\n\n"
        f"Details: {type(e).__name__}: {e}"
    )
    st.stop()

HIGH_VALUE_T = float(CONFIG["high_value_threshold"])
WIPEOUT_T = float(CONFIG["wipeout_threshold"])
DECISION_T = float(CONFIG["decision_threshold"])

XGB_FEATURES = list(xgb_model.get_booster().feature_names or [])
SCALE_COLS = list(scaler.feature_names_in_)
WOE_FEATURES = list(woe_model.feature_names_in_)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# Each input has a persistent value (e.g. "amount") and a widget
# key (e.g. "w_amount"). Widgets read and write only their own key,
# so the Quick Test buttons set both. The persistent copy survives
# page switches, when Streamlit discards state of unrendered widgets.
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "amount": 500.0, "sender_before": 5000.0, "sender_after": 4500.0,
    "receiver_before": 1000.0, "receiver_after": 1500.0, "hour": 14,
    "txn_type": "peer", "device": "smartphone", "region": "Nairobi", "day": "Tue",
    "woe_amount": 500.0, "woe_s_before": 5000.0, "woe_s_after": 4500.0,
}

FRAUD_EXAMPLE = {
    "amount": 4950.0, "sender_before": 5000.0, "sender_after": 50.0,
    "receiver_before": 1000.0, "receiver_after": 5950.0, "hour": 2,
    "txn_type": "peer", "device": "smartphone", "region": "Nairobi", "day": "Tue",
    "woe_amount": 4950.0, "woe_s_before": 5000.0, "woe_s_after": 50.0,
}

LEGIT_EXAMPLE = {
    "amount": 300.0, "sender_before": 25000.0, "sender_after": 24700.0,
    "receiver_before": 3000.0, "receiver_after": 3300.0, "hour": 14,
    "txn_type": "paybill", "device": "smartphone", "region": "Nairobi", "day": "Wed",
    "woe_amount": 300.0, "woe_s_before": 25000.0, "woe_s_after": 24700.0,
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("xgb_result", None)
st.session_state.setdefault("woe_result", None)


def _apply_values(values):
    for k, v in values.items():
        st.session_state[k] = v
        st.session_state["w_" + k] = v
    st.session_state.xgb_result = None
    st.session_state.woe_result = None


def load_fraud():
    """Fraud pattern: KES 5,000 balance, KES 4,950 sent, KES 50 left (ratio ~0.01)."""
    _apply_values(FRAUD_EXAMPLE)


def load_legit():
    """Legitimate pattern: KES 25,000 balance, KES 300 sent (ratio ~0.99)."""
    _apply_values(LEGIT_EXAMPLE)


def _sync(name, result_key):
    st.session_state[name] = st.session_state["w_" + name]
    st.session_state[result_key] = None


def bound(name, result_key="xgb_result"):
    """Keyword args that bind a widget to its persistent value."""
    wkey = "w_" + name
    if wkey not in st.session_state:
        st.session_state[wkey] = st.session_state[name]
    return {"key": wkey, "on_change": _sync, "args": (name, result_key)}


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (must mirror the notebook exactly)
# ─────────────────────────────────────────────────────────────
def engineer_features(amount, sender_bal_before, sender_bal_after,
                      receiver_bal_before, receiver_bal_after,
                      hour, transaction_type, device_type,
                      region, day_of_week):

    balance_depletion_rate = amount / (sender_bal_before + 1)
    is_high_value = int(amount > HIGH_VALUE_T)
    is_balance_wipeout = int(sender_bal_after < WIPEOUT_T)
    sender_balance_ratio = sender_bal_after / (sender_bal_before + 1)

    # One-hot columns were bool in training (pd.get_dummies default)
    features = {
        'amount': amount,
        'sender_balance_before': sender_bal_before,
        'sender_balance_after': sender_bal_after,
        'receiver_balance_before': receiver_bal_before,
        'receiver_balance_after': receiver_bal_after,
        'hour': hour,
        'transaction_type_peer': transaction_type == "peer",
        'transaction_type_till': transaction_type == "till",
        'device_type_smartphone': device_type == "smartphone",
        'region_Kisumu': region == "Kisumu",
        'region_Mombasa': region == "Mombasa",
        'region_Nairobi': region == "Nairobi",
        'region_Nakuru': region == "Nakuru",
        'day_of_week_Mon': day_of_week == "Mon",
        'day_of_week_Sat': day_of_week == "Sat",
        'day_of_week_Sun': day_of_week == "Sun",
        'day_of_week_Thu': day_of_week == "Thu",
        'day_of_week_Tue': day_of_week == "Tue",
        'day_of_week_Wed': day_of_week == "Wed",
        'balance_depletion_rate': balance_depletion_rate,
        'is_high_value': is_high_value,
        'is_balance_wipeout': is_balance_wipeout,
        'sender_balance_ratio': sender_balance_ratio,
    }
    X = pd.DataFrame([features])
    if XGB_FEATURES:
        missing = [c for c in XGB_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"App is not producing model features: {missing}")
        X = X[XGB_FEATURES]

    return X, {
        'balance_depletion_rate': balance_depletion_rate,
        'is_high_value': is_high_value,
        'is_balance_wipeout': is_balance_wipeout,
        'sender_balance_ratio': sender_balance_ratio,
    }


# ─────────────────────────────────────────────────────────────
# WoE TRANSFORMATION (replicates sc.woebin_ply without scorecardpy)
# scorecardpy bins are left-inclusive, right-exclusive, e.g.
# "[-inf,4000.0)", "[4000.0,9000.0)", "[9000.0,inf)".
# float() parses "inf" and "-inf" natively, so the edges work.
# ─────────────────────────────────────────────────────────────
_BIN_RE = re.compile(r"^\s*[\[\(]\s*([^,]+?)\s*,\s*([^\]\)]+?)\s*[\]\)]\s*$")


def parse_bins(records):
    parsed = []
    for r in records:
        label = str(r["bin"])
        m = _BIN_RE.match(label)
        if not m:
            continue  # 'missing' or categorical bins
        parsed.append((float(m.group(1)), float(m.group(2)), float(r["woe"])))
    return sorted(parsed)


PARSED_BINS = {k: parse_bins(v) for k, v in woe_bins.items()}


def get_woe(value, feature):
    bins = PARSED_BINS.get(feature, [])
    for lower, upper, woe in bins:
        if lower <= value < upper:
            return woe
    raise ValueError(f"Value {value} for '{feature}' did not fall in any WoE bin")


def prepare_woe_input(raw_values):
    """raw_values: dict of raw feature name -> value. Output columns follow
    woe_model.feature_names_in_ exactly."""
    row = {}
    for col in WOE_FEATURES:
        base = col[:-4] if col.endswith("_woe") else col
        row[col] = get_woe(raw_values[base], base)
    return pd.DataFrame([row])[WOE_FEATURES]


def input_warnings(amount, sender_before, sender_after, receiver_before=None, receiver_after=None):
    msgs = []
    if amount > sender_before:
        msgs.append("Amount is larger than the sender's starting balance.")
    # Only enforce balance arithmetic if the training data itself follows it
    if CONFIG.get("sender_balance_consistent", True) and abs((sender_before - amount) - sender_after) > 1:
        msgs.append(f"Sender balance after should be about KES {sender_before - amount:,.0f} "
                    f"(before minus amount), but KES {sender_after:,.0f} was entered.")
    if (receiver_before is not None and CONFIG.get("receiver_balance_consistent", True)
            and abs((receiver_before + amount) - receiver_after) > 1):
        msgs.append(f"Receiver balance after should be about KES {receiver_before + amount:,.0f} "
                    f"(before plus amount), but KES {receiver_after:,.0f} was entered.")
    ranges = CONFIG.get("feature_ranges") or {}
    for name, val in [("amount", amount), ("sender_balance_before", sender_before),
                      ("sender_balance_after", sender_after)]:
        if name in ranges:
            lo, hi = ranges[name]
            if val < lo or val > hi:
                msgs.append(f"{name} = {val:,.0f} is outside the training range "
                            f"({lo:,.0f} to {hi:,.0f}); the score is less reliable.")
    return msgs


# ─────────────────────────────────────────────────────────────
# METRICS (read from app_config.json written by the notebook)
# ─────────────────────────────────────────────────────────────
FALLBACK_METRICS = {
    'Model': ['Logistic Regression', 'WoE Scorecard', 'LightGBM', 'XGBoost'],
    'Type': ['Baseline', 'Regulatory', 'High Performance', 'Primary'],
    'Precision': [0.5219, 0.2280, 0.9603, 1.0000],
    'Recall': [0.6624, 0.6852, 0.6538, 0.6553],
    'F1-Score': [0.5838, 0.3421, 0.7780, 0.7917],
    'AUC-ROC': [0.8343, 0.8250, 0.8267, 0.8303],
    'False Positives': [426, 1629, 19, 0],
    'Fraud Caught': [465, 481, 459, 460],
}
comp_df = pd.DataFrame(CONFIG.get("metrics") or FALLBACK_METRICS)
test_size = int(CONFIG.get("test_size", 24000))


def metric(model, col):
    row = comp_df[comp_df['Model'] == model]
    return float(row[col].iloc[0]) if len(row) else float('nan')


XGB_P, XGB_R, XGB_F1 = metric('XGBoost', 'Precision'), metric('XGBoost', 'Recall'), metric('XGBoost', 'F1-Score')
XGB_FP = int(metric('XGBoost', 'False Positives'))
WOE_P, WOE_R = metric('WoE Scorecard', 'Precision'), metric('WoE Scorecard', 'Recall')
WOE_CAUGHT = int(metric('WoE Scorecard', 'Fraud Caught'))
AUC_MIN, AUC_MAX = comp_df['AUC-ROC'].min(), comp_df['AUC-ROC'].max()
DS = CONFIG["dataset"]


# SIDEBAR
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:40px;'>🔐</div>
        <div style='color:white; font-size:18px; font-weight:800;
                    letter-spacing:0.5px; margin-top:8px;'>
            M-Pesa Fraud Detector
        </div>
        <div style='color:#A8D5B5; font-size:11px; margin-top:4px;'>
            by Kat.Codes
        </div>
    </div>
    <hr style='border-color:#004d26; margin:12px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color:#A8D5B5; font-size:12px; font-weight:600; margin-bottom:8px;'>NAVIGATE</p>",
                unsafe_allow_html=True)

    page = st.radio(
        label="Navigate",
        options=["🏠 Home", "🔍 Check a Transaction", "📋 Scorecard Audit", "📊 Model Results"],
        label_visibility="collapsed",
        key="page",
    )

    st.markdown("<hr style='border-color:#004d26; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A8D5B5; font-size:12px; font-weight:600; margin-bottom:8px;'>QUICK TEST</p>",
                unsafe_allow_html=True)

    st.button("⚠️ Load Fraud Example", on_click=load_fraud, **stretch(st.button))
    st.button("✅ Load Safe Example", on_click=load_legit, **stretch(st.button))

    st.markdown("<hr style='border-color:#004d26; margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='color:#A8D5B5; font-size:11px; line-height:1.9;'>
        <p style='color:white; font-size:12px; font-weight:600; margin-bottom:6px;'>DATASET</p>
        📱 {DS['rows']:,} synthetic M-Pesa transactions<br>
        ⚠️ {DS['fraud']:,} fraud cases ({DS['fraud_rate']:.2f}%)<br>
        ⚖️ {DS['imbalance']:.0f}:1 class imbalance<br><br>
        <p style='color:white; font-size:12px; font-weight:600; margin-bottom:6px;'>TEST-SET PERFORMANCE</p>
        🎯 XGBoost Precision: {XGB_P:.2f}<br>
        🔎 WoE Scorecard Recall: {WOE_R:.2f}<br>
        📈 AUC range: {AUC_MIN:.2f} to {AUC_MAX:.2f}<br><br>
        <p style='color:white; font-size:12px; font-weight:600; margin-bottom:6px;'>CONTEXT</p>
        💸 KSh 810M lost in 2024<br>
        📊 344% increase YoY (CBK)
    </div>
    """, unsafe_allow_html=True)


# PAGE: HOME
if page == "🏠 Home":

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {SAF_DARK_GREEN} 0%, {SAF_GREEN} 100%);
                padding: 36px 32px; border-radius: 16px; margin-bottom: 28px;
                box-shadow: 0 8px 24px rgba(0,102,51,0.25);'>
        <div style='display:flex; align-items:center; gap:16px;'>
            <div style='font-size:48px;'>🔐</div>
            <div>
                <h1 style='color:white; margin:0; font-size:28px; font-weight:800;'>
                    M-Pesa SIM Swap Fraud Detection
                </h1>
                <p style='color:#C8EDD0; margin:8px 0 0 0; font-size:15px; line-height:1.5;'>
                    Fraud scoring powered by XGBoost and a WoE Scorecard,
                    trained on synthetic M-Pesa transaction data.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions Analysed", f"{DS['rows']:,}")
    c2.metric("Fraud Cases", f"{DS['fraud']:,}", delta=f"{DS['fraud_rate']:.2f}% of total", delta_color="inverse")
    c3.metric("XGBoost Precision", f"{XGB_P:.2f}", delta=f"{XGB_FP} false positives on test set")
    c4.metric("WoE Recall", f"{WOE_R:.2f}", delta="Highest fraud catch rate")
    c5.metric("AUC-ROC", f"{AUC_MIN:.2f} to {AUC_MAX:.2f}", delta="All four models")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">What does this tool do?</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="stat-card">
            <div style='font-size:32px; margin-bottom:12px;'>🔍</div>
            <b style='color:{SAF_DARK_GREEN}; font-size:15px;'>Check a Transaction</b>
            <p style='color:{SAF_MID_TEXT}; font-size:13px; margin-top:8px; line-height:1.6;'>
                Enter transaction details and get an instant fraud risk score
                from the XGBoost model, which produced {XGB_FP} false positives
                on {test_size:,} held-out test transactions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="stat-card">
            <div style='font-size:32px; margin-bottom:12px;'>📋</div>
            <b style='color:{SAF_DARK_GREEN}; font-size:15px;'>Scorecard Audit</b>
            <p style='color:{SAF_MID_TEXT}; font-size:13px; margin-top:8px; line-height:1.6;'>
                Run the WoE Logistic Regression Scorecard for a fully explainable
                fraud decision. Every flag can be justified feature by feature
                to a regulator or auditor.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="stat-card">
            <div style='font-size:32px; margin-bottom:12px;'>📊</div>
            <b style='color:{SAF_DARK_GREEN}; font-size:15px;'>Model Results</b>
            <p style='color:{SAF_MID_TEXT}; font-size:13px; margin-top:8px; line-height:1.6;'>
                Compare all four models side by side. See how XGBoost, LightGBM,
                Logistic Regression, and the WoE Scorecard perform against
                each other on the same dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">How the Two-Layer System Works</div>', unsafe_allow_html=True)

    lay1, lay2 = st.columns(2)
    with lay1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #FFF0F0, #FFE5E5);
                    border: 1.5px solid {SAF_RED}; border-radius: 12px;
                    padding: 20px; height:100%;'>
            <div style='font-size:28px;'>⚡</div>
            <b style='color:{SAF_RED}; font-size:16px;'>Layer 1: Real-Time Blocking</b>
            <p style='color:{SAF_DARK_TEXT}; font-size:13px; margin-top:10px; line-height:1.7;'>
                <b>Model:</b> XGBoost<br>
                <b>Precision:</b> {XGB_P:.2f} ({XGB_FP} false positives on the test set)<br>
                <b>Use case:</b> Automated transaction blocking<br><br>
                High-precision flags suited to automatic blocking. Precision on
                synthetic test data will not fully carry over to live traffic.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with lay2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {SAF_LIGHT}, #E8F5E9);
                    border: 1.5px solid {SAF_GREEN}; border-radius: 12px;
                    padding: 20px; height:100%;'>
            <div style='font-size:28px;'>📋</div>
            <b style='color:{SAF_DARK_GREEN}; font-size:16px;'>Layer 2: Regulatory Audit</b>
            <p style='color:{SAF_DARK_TEXT}; font-size:13px; margin-top:10px; line-height:1.7;'>
                <b>Model:</b> WoE Logistic Regression Scorecard<br>
                <b>Recall:</b> {WOE_R:.2f}, catches the most fraud<br>
                <b>Use case:</b> Compliance review and audit trails<br><br>
                When a regulator asks why a transaction was flagged,
                the scorecard explains every contributing feature.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <b>Why this matters:</b> The CBK's 2024 Financial Sector Stability Report documented
        KSh 810 million in mobile banking losses, a 344% increase in a single year, driven
        primarily by SIM swap fraud. Existing defenses are reactive. This system scores
        fraud risk at the moment of the transaction.
    </div>
    """, unsafe_allow_html=True)


# PAGE: CHECK A TRANSACTION
elif page == "🔍 Check a Transaction":

    st.markdown(f"""
    <h2 style='color:{SAF_DARK_GREEN}; margin-bottom:4px;'>🔍 Check a Transaction</h2>
    <p style='color:{SAF_MID_TEXT}; margin-bottom:20px;'>
        Enter the transaction details below and click <b>Check Transaction</b>
        to get an instant fraud risk score from the XGBoost model.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <b>XGBoost Layer:</b> Precision {XGB_P:.2f} with {XGB_FP} false positives on
        {test_size:,} held-out test transactions. Use the <b>Quick Test</b>
        buttons in the sidebar to try a pre-built example.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">💳 Transaction Details</div>',
                    unsafe_allow_html=True)

        amount = st.number_input("Transaction Amount (KES)", min_value=1.0, max_value=500000.0,
                                 step=100.0, **bound("amount"))
        sender_before = st.number_input("Sender Balance Before (KES)", min_value=0.0, max_value=1000000.0,
                                        step=500.0, **bound("sender_before"))
        sender_after = st.number_input("Sender Balance After (KES)", min_value=0.0, max_value=1000000.0,
                                       step=500.0, **bound("sender_after"))
        receiver_before = st.number_input("Receiver Balance Before (KES)", min_value=0.0, max_value=1000000.0,
                                          step=500.0, **bound("receiver_before"))
        receiver_after = st.number_input("Receiver Balance After (KES)", min_value=0.0, max_value=1000000.0,
                                         step=500.0, **bound("receiver_after"))
        hour = st.slider("Hour of Transaction (0 = Midnight, 23 = 11pm)", 0, 23, **bound("hour"))

        st.markdown('<div class="section-header">📱 Transaction Context</div>',
                    unsafe_allow_html=True)

        txn_type = st.selectbox("Transaction Type", ["peer", "till", "paybill"],
                                help="peer = person to person | till = business till | paybill = bill payment",
                                **bound("txn_type"))
        device = st.selectbox("Device Type", ["smartphone", "feature"],
                              help="smartphone = Android or iOS | feature = basic USSD phone",
                              **bound("device"))
        region = st.selectbox("Region", ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"],
                              **bound("region"))
        day = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                           **bound("day"))

        if st.button("🔍 Check Transaction", type="primary", **stretch(st.button)):
            X_input, eng = engineer_features(
                amount, sender_before, sender_after,
                receiver_before, receiver_after,
                hour, txn_type, device, region, day
            )
            X_scaled = X_input.copy()
            X_scaled[SCALE_COLS] = scaler.transform(X_input[SCALE_COLS])
            prob = float(xgb_model.predict_proba(X_scaled)[0][1])
            st.session_state.xgb_result = {
                "prob": prob,
                "prediction": int(prob >= DECISION_T),
                "eng": eng,
                "warnings": input_warnings(amount, sender_before, sender_after,
                                           receiver_before, receiver_after),
            }

    with col_result:
        st.markdown('<div class="section-header">🎯 Detection Result</div>',
                    unsafe_allow_html=True)

        res = st.session_state.xgb_result
        if res:
            prob, prediction, eng = res["prob"], res["prediction"], res["eng"]
            pct = round(prob * 100, 1)

            for w in res["warnings"]:
                st.warning(w)

            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <div style='font-size:48px;'>⚠️</div>
                    <h2 style='color:{SAF_RED}; margin:8px 0;'>FRAUDULENT</h2>
                    <h3 style='color:{SAF_RED}; margin:4px 0; font-size:22px;'>
                        Model Risk Score: {pct}%
                    </h3>
                    <p style='color:#555; margin:12px 0 0 0; font-size:13px;'>
                        This transaction matches SIM swap fraud patterns.<br>
                        <b>Recommended action: Block and flag for investigation.</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                    <div style='font-size:48px;'>✅</div>
                    <h2 style='color:{SAF_DARK_GREEN}; margin:8px 0;'>LEGITIMATE</h2>
                    <h3 style='color:{SAF_DARK_GREEN}; margin:4px 0; font-size:22px;'>
                        Model Risk Score: {pct}%
                    </h3>
                    <p style='color:#555; margin:12px 0 0 0; font-size:13px;'>
                        No SIM swap fraud pattern detected.<br>
                        <b>Recommended action: Allow transaction to proceed.</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.caption("The risk score is the model's raw output. Class weighting during training "
                       "inflates it, so treat it as a ranking score, not a calibrated probability.")

            fig, ax = plt.subplots(figsize=(7, 1.5))
            bar_color = SAF_RED if prediction == 1 else SAF_GREEN
            ax.barh(['Risk Level'], [pct], color=bar_color, alpha=0.85, height=0.5)
            ax.barh(['Risk Level'], [100 - pct], left=[pct], color='#E0E0E0', alpha=0.5, height=0.5)
            ax.axvline(x=DECISION_T * 100, color='orange', linestyle='--', linewidth=1.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Model Risk Score (%)')
            if pct >= 10:
                ax.text(pct / 2, 0, f'{pct}%', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=12)
            else:
                ax.text(pct + 2, 0, f'{pct}%', ha='left', va='center',
                        color=SAF_DARK_TEXT, fontweight='bold', fontsize=12)
            ax.set_title('Fraud Risk Score', fontsize=11, color=SAF_DARK_GREEN, fontweight='bold')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">🧬 SIM Swap Behavioral Signals</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <p style='color:#555; font-size:13px; margin-bottom:12px;'>
                These four engineered features capture the account drainage
                pattern of a SIM swap attack.
            </p>
            """, unsafe_allow_html=True)

            s1, s2 = st.columns(2)
            dr = eng['balance_depletion_rate']
            br = eng['sender_balance_ratio']
            s1.metric("Balance Depletion Rate", f"{dr:.4f}",
                      delta="⚠️ High, account drain" if dr > 0.9 else "✅ Normal",
                      delta_color="inverse" if dr > 0.9 else "normal")
            s1.metric("Sender Balance Ratio", f"{br:.4f}",
                      delta="⚠️ Low, account drained" if br < 0.1 else "✅ Normal",
                      delta_color="inverse" if br < 0.1 else "normal")
            s2.metric("High Value Transaction",
                      "YES ⚠️" if eng['is_high_value'] else "NO ✅",
                      delta=f"Above KES {HIGH_VALUE_T:,.0f}" if eng['is_high_value'] else "Normal",
                      delta_color="inverse" if eng['is_high_value'] else "normal")
            s2.metric("Balance Wipeout",
                      "YES ⚠️" if eng['is_balance_wipeout'] else "NO ✅",
                      delta=f"Below KES {WIPEOUT_T:,.0f} left" if eng['is_balance_wipeout'] else "Normal",
                      delta_color="inverse" if eng['is_balance_wipeout'] else "normal")
        else:
            st.markdown(f"""
            <div style='text-align:center; padding:60px 20px;
                        background:{SAF_LIGHT}; border-radius:12px;
                        border: 2px dashed {SAF_GREEN};'>
                <div style='font-size:48px; margin-bottom:16px;'>🔍</div>
                <h3 style='color:{SAF_DARK_GREEN};'>Ready to check a transaction</h3>
                <p style='color:{SAF_MID_TEXT}; font-size:14px;'>
                    Fill in the transaction details on the left<br>
                    and click <b>Check Transaction</b>.<br><br>
                    Or use the <b>Quick Test</b> buttons in the sidebar<br>
                    to load a sample fraud or legitimate transaction.
                </p>
            </div>
            """, unsafe_allow_html=True)


# PAGE: SCORECARD AUDIT
elif page == "📋 Scorecard Audit":

    st.markdown(f"""
    <h2 style='color:{SAF_DARK_GREEN}; margin-bottom:4px;'>📋 Scorecard Audit</h2>
    <p style='color:{SAF_MID_TEXT}; margin-bottom:20px;'>
        The WoE Logistic Regression Scorecard explains every fraud decision
        feature by feature. Use this for regulatory reporting and audit trails.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <b>WoE Scorecard Layer:</b> Unlike XGBoost, every decision this model makes
        can be explained to a regulator. Higher Recall ({WOE_R:.2f}) means it catches more
        fraud than XGBoost, at the cost of more false alarms (Precision: {WOE_P:.2f}).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_woe_form, col_woe_result = st.columns([1, 1], gap="large")

    with col_woe_form:
        st.markdown('<div class="section-header">💳 Transaction Inputs</div>',
                    unsafe_allow_html=True)

        woe_amount = st.number_input("Amount (KES)", min_value=1.0, max_value=500000.0,
                                     step=100.0, **bound("woe_amount", "woe_result"))
        woe_s_before = st.number_input("Sender Balance Before (KES)", min_value=0.0, max_value=1000000.0,
                                       step=500.0, **bound("woe_s_before", "woe_result"))
        woe_s_after = st.number_input("Sender Balance After (KES)", min_value=0.0, max_value=1000000.0,
                                      step=500.0, **bound("woe_s_after", "woe_result"))

        if st.button("📋 Run Scorecard Audit", type="primary", **stretch(st.button)):
            raw = {
                "amount": woe_amount,
                "sender_balance_before": woe_s_before,
                "sender_balance_after": woe_s_after,
                "sender_balance_ratio": woe_s_after / (woe_s_before + 1),
                "balance_depletion_rate": woe_amount / (woe_s_before + 1),
                "is_high_value": int(woe_amount > HIGH_VALUE_T),
                "is_balance_wipeout": int(woe_s_after < WIPEOUT_T),
            }
            try:
                woe_input = prepare_woe_input(raw)
                woe_prob = float(woe_model.predict_proba(woe_input)[0][1])
                st.session_state.woe_result = {
                    "input": woe_input.to_dict("records")[0],
                    "prob": woe_prob,
                    "pred": int(woe_prob >= 0.5),
                    "warnings": input_warnings(woe_amount, woe_s_before, woe_s_after),
                }
            except (KeyError, ValueError) as e:
                st.session_state.woe_result = {"error": str(e)}

        st.markdown("""
        <div class="warning-box" style='margin-top:16px;'>
            <b>How WoE scoring works:</b><br>
            Each feature value is mapped to a WoE score based on how common
            that value is in fraud vs legitimate transactions.<br><br>
            <b>Positive WoE</b> = this value appears more in fraud<br>
            <b>Negative WoE</b> = this value appears more in legitimate
        </div>
        """, unsafe_allow_html=True)

    with col_woe_result:
        st.markdown('<div class="section-header">🎯 Scorecard Result</div>',
                    unsafe_allow_html=True)

        res = st.session_state.woe_result
        if res and "error" in res:
            st.error(f"Scorecard could not score this input: {res['error']}")
        elif res:
            woe_pct = round(res["prob"] * 100, 1)
            for w in res["warnings"]:
                st.warning(w)

            if res["pred"] == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <div style='font-size:48px;'>⚠️</div>
                    <h2 style='color:{SAF_RED}; margin:8px 0;'>FRAUD FLAG</h2>
                    <h3 style='color:{SAF_RED}; margin:4px 0; font-size:20px;'>
                        Scorecard Risk Score: {woe_pct}%
                    </h3>
                    <p style='color:#555; margin:12px 0 0 0; font-size:13px;'>
                        Scorecard flags this transaction for regulatory review.<br>
                        <b>Recommended: Queue for credit officer audit.</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                    <div style='font-size:48px;'>✅</div>
                    <h2 style='color:{SAF_DARK_GREEN}; margin:8px 0;'>CLEAR</h2>
                    <h3 style='color:{SAF_DARK_GREEN}; margin:4px 0; font-size:20px;'>
                        Scorecard Risk Score: {woe_pct}%
                    </h3>
                    <p style='color:#555; margin:12px 0 0 0; font-size:13px;'>
                        Scorecard does not flag this transaction.<br>
                        <b>No regulatory action required.</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📊 Feature Score Breakdown</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <p style='color:#555; font-size:13px; margin-bottom:12px;'>
                WoE is the bin's fraud evidence. Contribution = coefficient x WoE,
                which is what actually moves the score.
            </p>
            """, unsafe_allow_html=True)

            coefs = dict(zip(WOE_FEATURES, woe_model.coef_[0]))
            woe_display = pd.DataFrame([
                {
                    "Feature": f.replace("_woe", "").replace("_", " ").title(),
                    "WoE Score": round(v, 4),
                    "Contribution": round(coefs[f] * v, 4),
                    "Signal": ("🔴 Fraud signal" if coefs[f] * v > 0
                               else "🟢 Legitimate signal" if coefs[f] * v < 0 else "⚪ Neutral"),
                }
                for f, v in res["input"].items()
            ])
            st.dataframe(woe_display, hide_index=True, **stretch(st.dataframe))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚖️ Model Coefficients</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <p style='color:#555; font-size:13px; margin-bottom:12px;'>
                A longer red bar means that feature pushes more strongly toward fraud.
            </p>
            """, unsafe_allow_html=True)

            coef_df = pd.DataFrame({
                'Feature': [f.replace("_woe", "").replace("_", " ").title() for f in WOE_FEATURES],
                'Coefficient': woe_model.coef_[0],
            }).sort_values('Coefficient')
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            colors = [SAF_RED if c > 0 else SAF_GREEN for c in coef_df['Coefficient']]
            ax2.barh(coef_df['Feature'], coef_df['Coefficient'],
                     color=colors, alpha=0.85, edgecolor='white', height=0.5)
            ax2.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
            ax2.set_xlabel('Coefficient Value')
            ax2.set_title('Red = pushes toward fraud | Green = toward legitimate',
                          fontsize=10, color=SAF_MID_TEXT)
            fig2.patch.set_facecolor('white')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        else:
            st.markdown(f"""
            <div style='text-align:center; padding:60px 20px;
                        background:{SAF_LIGHT}; border-radius:12px;
                        border: 2px dashed {SAF_GREEN};'>
                <div style='font-size:48px; margin-bottom:16px;'>📋</div>
                <h3 style='color:{SAF_DARK_GREEN};'>Ready for scorecard audit</h3>
                <p style='color:{SAF_MID_TEXT}; font-size:14px;'>
                    Enter the transaction amount and balance details<br>
                    on the left and click <b>Run Scorecard Audit</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)


# PAGE: MODEL RESULTS
elif page == "📊 Model Results":

    st.markdown(f"""
    <h2 style='color:{SAF_DARK_GREEN}; margin-bottom:4px;'>📊 Model Performance Results</h2>
    <p style='color:{SAF_MID_TEXT}; margin-bottom:20px;'>
        All four models were trained and tested on the same synthetic dataset
        using an 80/20 stratified split ({test_size:,} test transactions).
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        All four models land in a narrow AUC-ROC band ({AUC_MIN:.2f} to {AUC_MAX:.2f}).
        The difference is mainly in <b>where</b> each model sets its risk tolerance.
        No single model meets every business target, which is why XGBoost and the
        WoE Scorecard are deployed together.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Full Comparison Scorecard</div>',
                unsafe_allow_html=True)

    def highlight_best(s):
        if s.name in ['Precision', 'F1-Score', 'AUC-ROC', 'Fraud Caught', 'Recall']:
            return ['background-color:#E8F5E9; color:#006633; font-weight:bold;' if v == s.max() else '' for v in s]
        if s.name == 'False Positives':
            return ['background-color:#E8F5E9; color:#006633; font-weight:bold;' if v == s.min() else '' for v in s]
        return ['' for _ in s]

    st.dataframe(
        comp_df.style.apply(highlight_best,
                            subset=['Precision', 'Recall', 'F1-Score', 'AUC-ROC',
                                    'False Positives', 'Fraud Caught'])
                     .format({'Precision': '{:.4f}', 'Recall': '{:.4f}',
                              'F1-Score': '{:.4f}', 'AUC-ROC': '{:.4f}'}),
        hide_index=True, **stretch(st.dataframe)
    )
    st.markdown("<p style='color:#888; font-size:12px;'>Green = best value in that column</p>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_ch1, col_ch2 = st.columns(2)
    short = {'Logistic Regression': 'LR Baseline'}
    models_s = [short.get(m, m) for m in comp_df['Model']]
    x = np.arange(len(models_s))
    w = 0.35
    clrs = ['#888888', SAF_DARK_GREEN, '#FFA000', SAF_RED][:len(models_s)]

    with col_ch1:
        st.markdown('<div class="section-header">Precision vs Recall</div>',
                    unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.bar(x - w/2, comp_df['Precision'], w, color=clrs, alpha=0.9, edgecolor='white',
                label='Precision (solid)')
        ax3.bar(x + w/2, comp_df['Recall'], w, color=clrs, alpha=0.45, edgecolor='white',
                label='Recall (transparent)')
        ax3.axhline(y=0.80, color='black', linestyle='--', linewidth=0.8, alpha=0.4,
                    label='Target (0.80)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models_s, rotation=10, fontsize=9)
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1.15)
        ax3.legend(fontsize=9)
        ax3.set_title('Precision vs Recall by Model', fontsize=11,
                      color=SAF_DARK_GREEN, fontweight='bold')
        fig3.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_ch2:
        st.markdown('<div class="section-header">False Positives vs Fraud Caught</div>',
                    unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.bar(x - w/2, comp_df['False Positives'], w,
                label='False Positives (innocent customers flagged)',
                color=SAF_RED, alpha=0.75, edgecolor='white')
        ax4.bar(x + w/2, comp_df['Fraud Caught'], w,
                label='Fraud Caught', color=SAF_GREEN, alpha=0.85, edgecolor='white')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models_s, rotation=10, fontsize=9)
        ax4.set_ylabel('Number of Transactions')
        ax4.legend(fontsize=9)
        ax4.set_title(f'False Positives vs Fraud Caught\n(out of {test_size:,} test transactions)',
                      fontsize=10, color=SAF_DARK_GREEN, fontweight='bold')
        fig4.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🚀 Deployment Recommendation</div>',
                unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #FFF0F0, #FFE5E5);
                    border:1.5px solid {SAF_RED}; border-radius:12px; padding:20px;'>
            <b style='color:{SAF_RED}; font-size:15px;'>⚡ Layer 1: XGBoost</b>
            <p style='color:{SAF_DARK_TEXT}; font-size:13px; margin-top:10px; line-height:1.7;'>
                Precision {XGB_P:.2f} | F1 {XGB_F1:.2f} | False Positives: <b>{XGB_FP}</b><br><br>
                Block transactions automatically on high-confidence flags.
                Re-validate precision on live data before relying on it.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {SAF_LIGHT}, #E8F5E9);
                    border:1.5px solid {SAF_GREEN}; border-radius:12px; padding:20px;'>
            <b style='color:{SAF_DARK_GREEN}; font-size:15px;'>📋 Layer 2: WoE Scorecard</b>
            <p style='color:{SAF_DARK_TEXT}; font-size:13px; margin-top:10px; line-height:1.7;'>
                Recall {WOE_R:.2f} | Fraud Caught: <b>{WOE_CAUGHT}</b><br><br>
                Explain every flagged transaction to a regulator with
                a full feature-by-feature audit trail.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
        <b>PSI Monitoring:</b> All WoE features were STABLE at deployment
        (training vs test). In production, recompute PSI monthly on incoming transactions.
        Alert at PSI above 0.10. Retrain the model if PSI exceeds 0.25.
    </div>
    """, unsafe_allow_html=True)
