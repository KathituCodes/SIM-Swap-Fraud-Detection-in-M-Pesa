# SIM Swap Fraud Detection in Kenya's M-Pesa Ecosystem

A supervised machine learning system that flags SIM swap style account drains in M-Pesa transactions and explains every decision for regulatory audit.

**[Live App](https://sim-swap-fraud-detection-in-m-pesa-pzuecggagee65k9kdsyzs4.streamlit.app/)** | **[Documentation Report]([reports/SIM_Swap_Fraud_Detection_Documentation_Report.docx](https://drive.google.com/file/d/18rPBH-QXOq9LO_g_YNWePTxK2QCi00MJ/view?usp=drive_link)** | **[Dataset](https://www.kaggle.com/datasets/calebboen/mpesa-transactions-fraud)**

---

## The Problem

SIM swap fraud lets criminals take over a victim's phone number, reset their M-Pesa PIN and drain their wallet and linked credit lines (Fuliza, M-Shwari) within minutes. Kenyan mobile banking fraud losses rose 344% to KSh 810 million in 2024, and INTERPOL reports a 327% rise in SIM swap fraud in Kenya. Current defences are reactive. This project scores each transaction as it happens.

## Approach

| Phase | What was done |
|---|---|
| 1. Exploration & preparation | 120,000 synthetic transactions, 33:1 class imbalance. Engineered four SIM swap proxy features (balance wipeout, depletion rate, balance ratio, high value) |
| 2. Model selection | Non-linear boundaries and multicollinearity pointed to gradient boosted trees |
| 3. Training | Logistic Regression baseline, XGBoost (primary), LightGBM. Tuned with RandomizedSearchCV and class weighting |
| 4. Interpretability | WoE logistic regression scorecard for regulator-friendly explanations, plus PSI drift monitoring |
| 5. Deployment | Two-layer Streamlit app on Streamlit Community Cloud |

## Results (24,000-transaction test set)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **XGBoost** (real-time blocking) | 0.99 | 0.66 | 0.79 | 0.82 |
| LightGBM | 0.96 | 0.65 | 0.78 | 0.83 |
| **WoE Scorecard** (audit layer) | 0.40 | 0.67 | 0.50 | 0.83 |
| Logistic Regression | 0.52 | 0.66 | 0.58 | 0.83 |

XGBoost raised only 4 false positives in 24,000 transactions. Recall is capped at about 0.66 because the remaining fraud looks identical to legitimate traffic without true SIM swap signals (swap timestamps, device changes). See the documentation for the full analysis and limitations.

## The App

- **Check a Transaction:** instant XGBoost fraud risk score
- **Scorecard Audit:** feature-by-feature explanation of each decision
- **Model Results:** side-by-side comparison of all four models

## Repository Structure

```
├── app.py                  Streamlit app
├── requirements.txt        Pinned dependencies
├── .streamlit/config.toml  Theme settings
├── models/                 Trained models, WoE bins and app config
├── notebooks/              Full Colab notebook (EDA to deployment)
└── reports/                Documentation report and data profile
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.13.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, scorecardpy, Matplotlib, Streamlit

---

**Author:** Urbanus Kathitu ([Kat.Codes](https://github.com/KathituCodes))
