import pandas as pd
import numpy as np

# I'll use MNAR since missingness IS an important signal
# REMEMBER DIPSHIT - don't pass nan to xgboost, allways, ALLWAYS add an indicator

MNAR_FEATURES = ["monthly_income", "debt_ratio", "months_employed"]

# MAR for missingness that is related with other variables

MAR_FEATURES = {
"revolving_utilization": "median",
"total_late_payments": 0,
"dependents": 0,
"real_estate_loans": 0,
"owns_property": 0,
"owns_vehicle": 0,
"open_credit_lines": "median",
}

# MAR, i'll just pass nan in this case its fine

MCAR_FEATURES = {"age": "median"}

NULLABLE_FEATURES = MNAR_FEATURES + list(MAR_FEATURES.keys())

def add_missingness_indicators(df):
    df = df.copy()

    # MNAR -
    for feat in MNAR_FEATURES:
        if feat in df.columns:
            df[f"{feat}_missing"] = df[feat].isna().astype(int)

    # MAR -
    for feat, fill in MAR_FEATURES.items():
        if feat in df.columns:
            if fill == "median":
                df[feat] = df[feat].fillna(df[feat].median())
            else:
                df[feat] = df[feat].fillna(fill)

    # MCAR -
    for feat, fill in MAR_FEATURES.items():
        if feat in df.columns:
            if fill == "median":
                df[feat] = df[feat].fillna(df[feat].median())
            else:
                df[feat] = df[feat].fillna(fill)

    return df

def missingness_report(df):
    records = []
    all_feats = (
            MNAR_FEATURES
            + list(MAR_FEATURES.keys())
            + list(MCAR_FEATURES.keys())
    )
    for feat in all_feats:
        if feat not in df.columns: continue
        pct = df[feat].isna().mean() * 100
        if feat in MNAR_FEATURES:
            cls, strat = "MNAR", "Indicator + NaN (XGBoost Native)"
        elif feat in MAR_FEATURES:
            cls = "MAR"
            strat = f"Indicator + fill({MAR_FEATURES[feat]})"
        else:
            cls, strat = "MCAR", f"fill({MCAR_FEATURES[feat]})"

        records.append({"feature": feat, "missing_pct": round(pct, 2),
                        "classification": cls, "strategy": strat})

        report = pd.DataFrame(records).sort_values(
            "missing_pct", ascending=False)
        print("\n=== Missingness Report ===")
        print(report.to_string(index=False))
        return report