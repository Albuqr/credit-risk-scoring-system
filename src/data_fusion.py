## Map give me some credit columns to a common schema

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sqlite3

from src.synthetic_generator import generate_synthetic_slice, validate_synthetic

DB_PATH = Path("data/creditdb.sqlite")
GMC_PATH = Path("/opt/credit-risk-data/gmc/cs-training.csv")
LC_PATH = Path("/opt/credit-risk-data/lc/loan.csv")

def get_conn():
    return sqlite3.connect(DB_PATH)

LC_COLS = [
    "loan_amnt", "purpose", "annual_inc", "emp_length",
    "home_ownership", "revol_util", "open_acc", "dti", "loan_status",
]

UNIFIED_COLS = [
    "defaulted", "monthly_income", "revolving_utilization", "age",
    "debt_ratio", "open_credit_lines", "total_late_payments",
    "real_estate_loans", "dependents", "loan_amount", "loan_purpose",
    "months_employed", "owns_property", "owns_vehicle", "source",
]

def load_gmc() -> pd.DataFrame:
    df = pd.read_csv(GMC_PATH, index_col=0)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.rename(columns={"seriousdlqin2yrs": "defaulted"}, inplace=True)
    df = df[df["age"] >= 18].copy()

    df["monthly_income"] = df["monthlyincome"]
    df["revolving_utilization"] = df[
    "revolvingutilizationofunsecuredlines"].clip(0, 1)
    df["debt_ratio"] = df["debtratio"].clip(0, 10)
    df["open_credit_lines"] = df["numberofopencreditlinesandloans"]
    df["real_estate_loans"] = df["numberrealestateloansorlines"]
    df["dependents"] = df["numberofdependents"]
    late_30 = df.get("numberoftime30-59dayspastduenotworse", pd.Series(np.nan, index=df.index))
    late_60 = df.get("numberoftime60-89dayspastduenotworse", pd.Series(np.nan, index=df.index))
    late_90 = df.get("numberoftimes90dayslate", pd.Series(np.nan, index=df.index))
    df["total_late_payments"] = late_30.fillna(0) + late_60.fillna(0) + late_90.fillna(0)

# Give me some credit has no loan_amount, purpose, employment, vehicle so -
    df["loan_amount"] = np.nan
    df["loan_purpose"] = np.nan
    df["months_employed"] = np.nan
    df["owns_property"] = np.nan
    df["owns_vehicle"] = np.nan
    df["source"] = "gmc"
    cleaned = df[UNIFIED_COLS].copy()
    with get_conn() as conn:
        cleaned.to_sql("gmc_raw", conn, if_exists="replace", index=False)
    print(" Written gmc_raw to creditdb.sqlite")
    return cleaned

# Loan and clean Lending Club

def load_lc() -> pd.DataFrame:
    print("Loading Lending Club (this may take 30-60 seconds)...")
    df = pd.read_csv(LC_PATH, usecols=LC_COLS, low_memory=False)

# i'll only keep unambiguous outcomes

    df = df[df["loan_status"].isin(
        ["Fully Paid", "Charged Off", "Default"]
    )].copy()

    df["defaulted"] = df["loan_status"].isin(
        ["Charged Off", "Default"]).astype(int)

    df["monthly_income"] = df["annual_inc"] / 12
    df["revolving_utilization"] = pd.to_numeric(
        df["revol_util"], errors="coerce") / 100
    df["revolving_utilization"] = df["revolving_utilization"].clip(0, 1)
    df["debt_ratio"] = pd.to_numeric(df["dti"], errors="coerce") / 100
    df["open_credit_lines"] = df["open_acc"]
    df["loan_amount"] = df["loan_amnt"]
    df["loan_purpose"] = df["purpose"]

# rules for employment length
# "10+ years" -> 120, " < 1 year" -> 6, "5 years" -> 60
    def parse_emp(x):
        if pd.isna(x): return np.nan
        x = str(x).lower()
        if "10+" in x: return 120
        if "< 1" in x: return 6
        digits = "".join(c for c in x if c.isdigit())
        return int(digits) * 12 if digits else np.nan

    df["months_employed"] = df["emp_length"].apply(parse_emp)
    df["owns_property"] = df["home_ownership"].isin(
        ["MORTGAGE", "OWN"]).astype(float)
    df.loc[df["home_ownership"].isna(), "owns_property"] = np.nan

# Lending club has no age, late payment data, real estate loans, dependent or vehicle

    df["age"] = np.nan
    df["total_late_payments"] = np.nan
    df["real_estate_loans"] = np.nan
    df["dependents"] = np.nan
    df["owns_vehicle"] = np.nan
    df["source"] = "lc"

    cleaned = df[UNIFIED_COLS].copy()
    with get_conn() as conn:
        cleaned.to_sql("lc_raw", conn, if_exists="replace", index=False)
        print(" Written lc_raw to creditdb.sqlite")
        return cleaned

def synthetic_fill(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = df.copy()
    real_mask = df["source"].isin(["gmc", "lc"])

# Add age to Lending club - beta distribution fit to give me some credit ages

    gmc_ages = df.loc[df["source"] == "gmc", "age"].dropna()
    a, b, loc, scale = stats.beta.fit(
        np.clip(gmc_ages / 100, 0.01, 0.99), floc=0, fscale=1)
    lc_mask = df["source"] == "lc"
    n_lc = lc_mask.sum()
    df.loc[lc_mask, "age"] = np.clip(
        rng.beta(a, b, n_lc) * 100, 18, 90).astype(int)

# Total late payments for Lending Club

    lc_def = (df["source"] == "lc") & (df["defaulted"] == 1)
    lc_nodef = (df["source"] == "lc") & (df["defaulted"] == 0)
    df.loc[lc_def, "total_late_payments"] = rng.negative_binomial(
        2, 0.4, lc_def.sum())
    df.loc[lc_nodef, "total_late_payments"] = rng.negative_binomial(
        1, 0.8, lc_nodef.sum())

# Real estate loand + dependents for Lending Club

    for col in ["real_estate_loans", "dependents"]:
        gmc_vals = df.loc[df["source"] == "gmc", col].dropna()
        lc_missing = (df["source"] == "lc") & df[col].isna()
        df.loc[lc_missing, col] = rng.choice(
        gmc_vals.values, lc_missing.sum())

# Owns vehicle - log function of income (thhis should use real data only for reasons)

    income = df["monthly_income"].fillna(3000)
    p_vehicle = (1 / (1 + np.exp(-(income - 3000) / 2000))).clip(0.1, 0.9)
    veh_miss = df["owns_vehicle"].isna() & real_mask
    df.loc[veh_miss, "owns_vehicle"] = (
            rng.random(veh_miss.sum()) < p_vehicle[veh_miss]).astype(float)

# Loan ammount for Give me some credit - sampled from Lending Club

    lc_loans = df.loc[df["source"] == "lc", "loan_amount"].dropna()
    gmc_loan = (df["source"] == "gmc") & df["loan_amount"].isna()
    df.loc[gmc_loan, "loan_amount"] = rng.choice(
        lc_loans.values, gmc_loan.sum())

# Loan purpose for Give me some credit - sampled from Lending Club

    lc_purp = df.loc[df["source"] == "lc", "loan_purpose"].dropna()
    purp_counts = lc_purp.value_counts(normalize=True)
    gmc_purp = (df["source"] == "gmc") & df["loan_purpose"].isna()
    df.loc[gmc_purp, "loan_purpose"] = rng.choice(
        purp_counts.index, gmc_purp.sum(), p=purp_counts.values)
    return df

def balance_classes(df, target_ratio=0.20):
    rng = np.random.default_rng(42)
    defaulters = df[df["defaulted"] == 1]
    non_defaulters = df[df["defaulted"] == 0]
    n_needed = int(
        (target_ratio * len(non_defaulters)) / (1 - target_ratio) - len(defaulters)
    )
    if n_needed <= 0:
        return df

    print(f"Generating {n_needed} synthetic defaulter rows...")
    base = defaulters.sample(n=n_needed, replace=True, random_state=42).copy()

    # Exclude binary columns from noise — they must stay 0/1
    binary_cols = ["owns_property", "owns_vehicle"]
    numeric_cols = [c for c in base.select_dtypes(include=[np.number]).columns
                    if c not in ["defaulted"] + binary_cols and base[c].std() > 0]

    for col in numeric_cols:
        noise = rng.normal(0, base[col].std() * 0.05, len(base))
        base[col] = base[col] + noise

    # Restore binary columns to clean 0/1 after sampling
    for col in binary_cols:
        if col in base.columns:
            base[col] = base[col].round().clip(0, 1)

    base["source"] = "synthetic_balance"
    combined = pd.concat([df, base], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f" New default rate: {combined['defaulted'].mean():.1%}")

    return combined

def compare_distributions(combined):
    numeric_features = ["monthly_income", "revolving_utilization", "age",
    "debt_ratio", "loan_amount", "total_late_payments"]
    print("\n=== Distribution Comparison by Source ===")

    for feat in numeric_features:
        if feat not in combined.columns: continue
        print(f"\n{feat}:")

        for source in ["gmc", "lc", "synthetic", "synthetic_balance"]:
            subset = combined.loc[combined["source"] == source, feat].dropna()
            if len(subset) == 0: continue
            miss = combined.loc[combined["source"] == source, feat].isna().mean()
            print(f" {source:20s} mean={subset.mean():10.2f} std={subset.std():9.2f} missing={miss:.1%}")

def run_fusion():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Step 1/6 Loading Give Me Some Credit...")

    gmc = load_gmc()
    print(f" GMC: {len(gmc):,} rows, {gmc['defaulted'].mean():.1%} default")

    print("Step 2/6 Loading Lending Club...")
    lc = load_lc()
    print(f" LC: {len(lc):,} rows, {lc['defaulted'].mean():.1%} default")

    print("Step 3/6 Generating synthetic Brazilian-profile slice...")
    synthetic = generate_synthetic_slice(n_rows=50_000, random_seed=42)
    print(f" Synthetic: {len(synthetic):,} rows, "
          f"{synthetic['defaulted'].mean():.1%} default")

    validate_synthetic(synthetic)
    with get_conn() as conn:
        synthetic.to_sql("synthetic_slice", conn,
                         if_exists="replace", index=False)

    print(" Written synthetic_slice to creditdb.sqlite")


    print("Step 4/6 Combining all three sources...")
    combined = pd.concat([gmc, lc, synthetic], ignore_index=True)
    print(f" Combined: {len(combined):,} rows")

    print("Step 5/6 Synthetic gap-filling for real data...")
    combined = synthetic_fill(combined)

    print("Step 6/6 Balancing classes and writing to SQLite...")
    combined = balance_classes(combined, target_ratio=0.20)
    compare_distributions(combined)

    with get_conn() as conn:
        combined.to_sql("unified_credit_data", conn,
                        if_exists="replace", index=False)
    print(f"\nWritten unified_credit_data to {DB_PATH}")

    print(f"Final shape: {combined.shape}")
    for source in combined["source"].unique():
        s = combined[combined["source"] == source]
        print(f" {source:20s}: {len(s):>8,} rows "
          f"{s['defaulted'].mean():.1%} default")

if __name__ == "__main__":
    run_fusion()
