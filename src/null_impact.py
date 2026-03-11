import numpy as np
import pandas as pd
import joblib
import sqlite3
from pathlib import Path
from src.features import engineer_features, build_feature_matrix
from src.missingness import (add_missingness_indicators, MNAR_FEATURES,
                              MAR_FEATURES, NULLABLE_FEATURES)

DB_PATH = Path("data/creditdb.sqlite")

def build_baseline_customer():
    return {
        "monthly_income": 5200.0,
        "revolving_utilization": 0.30,
        "age": 42,
        "debt_ratio": 0.28,
        "open_credit_lines": 7,
        "total_late_payments": 0,
        "real_estate_loans": 1,
        "dependents": 1,
        "loan_amount": 12000.0,
        "loan_purpose": "personal",
        "months_employed": 36,
        "owns_property": 1.0,
        "owns_vehicle": 1.0
    }

def customer_to_features(customer):
    model = joblib.load("models/credit_model.pkl")
    df = pd.DataFrame([customer])
    df = add_missingness_indicators(df)
    df = engineer_features(df)
    X = build_feature_matrix(df)
    return model, X

def run_null_impact_analysis():
    baseline = build_baseline_customer()
    model, X_base = customer_to_features(baseline)
    base_prob = model.predict_proba(X_base)[0][1]
    print(f"Baseline probability all filled: {base_prob:.4f}\n")

    results = []
    for feat in NULLABLE_FEATURES:
        if feat not in baseline:
            continue
        nulled = baseline.copy()
        nulled[feat] = None
        _, X_null = customer_to_features(nulled)
        null_prob = model.predict_proba(X_null)[0][1]
        delta = null_prob - base_prob
        direction = (
            "INCREASES risk" if delta > 0.01
            else "DECREASES risk" if delta < -0.01
            else "Negligible"
        )
        cls = "MNAR" if feat in MNAR_FEATURES else "MAR"
        results.append({
            "feature": feat,
            "classification": cls,
            "baseline": round(base_prob, 4),
            "null_prob": round(null_prob, 4),
            "delta": round(delta, 4),
            "direction": direction
        })

    df_r = pd.DataFrame(results).sort_values("delta", ascending=False, key=abs)
    print("=== NULL IMPACT ANALYSIS ===")
    print(df_r.to_string(index=False))

    with sqlite3.connect(DB_PATH) as conn:
        df_r.to_sql("null_impact_report", conn, if_exists="replace", index=False)
    print(f"\nSaved to null_impact_report in {DB_PATH}")

if __name__ == "__main__":
    run_null_impact_analysis()