import pandas as pd
import numpy as np

UNIFIED_COLS = [
"defaulted", "monthly_income", "revolving_utilization", "age",
"debt_ratio", "open_credit_lines", "total_late_payments",
"real_estate_loans", "dependents", "loan_amount", "loan_purpose",
"months_employed", "owns_property", "owns_vehicle", "source",
]
PURPOSE_DIST = {
"debt_consolidation": 0.28, "personal": 0.22,
"small_business": 0.15, "home_improvement": 0.10,
"medical": 0.08, "car": 0.07, "educational": 0.05, "other": 0.05,
}
PURPOSE_LOAN_MULTIPLIER = {
"mortgage": 3.5, "car": 1.2, "small_business": 2.0, "educational": 0.8,
}


# i'll be using a MNAR biased approach to deal with missingness
# my college professor brought it to my attention since i'm dealing with finance here
# not all null's are equal, if a value is missing it should be taken into account, it may be important that it's not there

def apply_missing(arr, rate, rng, mnar_bias_mask=None):
    arr = arr.astype(float)
    if mnar_bias_mask is not None:
        p = np.full(len(arr), rate * 0.6)
        p[mnar_bias_mask] = rate * 1.8  # 2x more likely
        p = np.clip(p, 0, 0.95)
        missing = rng.random(len(arr)) < p
    else:
        missing = rng.random(len(arr)) < rate
    arr[missing] = np.nan
    return arr

def apply_missing_str(arr, rate, rng):
    missing = rng.random(len(arr)) < rate
    arr = arr.astype(object)
    arr[missing] = None
    return arr


def generate_synthetic_slice(n_rows=50_000, default_rate=0.18, random_seed=42):
    rng = np.random.default_rng(random_seed)
    n_default = int(n_rows * default_rate)
    n_non_default = n_rows - n_default

# This will be my target variable

    defaulted = np.concatenate([
        np.ones(n_default, dtype=int),
        np.zeros(n_non_default, dtype=int)])
    rng.shuffle(defaulted)
    is_def = defaulted == 1
    is_nondef = ~is_def

# Scaling and stuff

    age = np.clip(
        rng.beta(3, 5, n_rows) * (72 - 18) + 18, 18, 90).astype(int)

    monthly_income = np.empty(n_rows)
    monthly_income[is_nondef] = rng.lognormal(8.5, 0.6, is_nondef.sum())
    monthly_income[is_def] = rng.lognormal(7.8, 0.8, is_def.sum())

    dependents = np.empty(n_rows, dtype=float)
    dependents[is_nondef] = rng.poisson(1.2, is_nondef.sum())
    dependents[is_def] = rng.poisson(1.8, is_def.sum())
    dependents = np.clip(dependents, 0, 8)

# Small correlation -  more dependents -> slightly higher income

    monthly_income = monthly_income * (1 + dependents * 0.05)

    revolving_utilization = np.empty(n_rows)
    revolving_utilization[is_nondef] = rng.beta(2, 5, is_nondef.sum())
    revolving_utilization[is_def] = rng.beta(5, 2, is_def.sum())
    revolving_utilization = np.clip(revolving_utilization, 0, 1)

    debt_ratio = np.empty(n_rows)
    debt_ratio[is_nondef] = np.clip(
        rng.lognormal(-1.5, 0.5, is_nondef.sum()), 0, 2)
    debt_ratio[is_def] = np.clip(
        rng.lognormal(-0.8, 0.7, is_def.sum()), 0, 3)

    open_credit_lines = np.empty(n_rows, dtype=int)
    open_credit_lines[is_nondef] = rng.poisson(5, is_nondef.sum())
    open_credit_lines[is_def] = rng.poisson(8, is_def.sum())

    total_late_payments = np.zeros(n_rows, dtype=float)
    nd_idx = np.where(is_nondef)[0]
    nd_late = rng.random(is_nondef.sum()) > 0.85
    total_late_payments[nd_idx[nd_late]] = rng.negative_binomial(
        1, 0.7, nd_late.sum())
    d_idx = np.where(is_def)[0]
    d_late = rng.random(is_def.sum()) > 0.40
    total_late_payments[d_idx[d_late]] = rng.negative_binomial(
        3, 0.4, d_late.sum())

    real_estate_loans = np.empty(n_rows, dtype=int)
    real_estate_loans[is_nondef] = rng.poisson(0.4, is_nondef.sum())
    real_estate_loans[is_def] = rng.poisson(0.3, is_def.sum())
    real_estate_loans = np.clip(real_estate_loans, 0, 5)

    loan_amount = np.empty(n_rows)
    loan_amount[is_nondef] = rng.lognormal(9.2, 0.7, is_nondef.sum())
    loan_amount[is_def] = rng.lognormal(9.5, 0.9, is_def.sum())

    purposes = list(PURPOSE_DIST.keys())
    probs = list(PURPOSE_DIST.values())
    loan_purpose = rng.choice(purposes, n_rows, p=probs)
    for p, mult in PURPOSE_LOAN_MULTIPLIER.items():
        loan_amount[loan_purpose == p] *= mult

    months_employed = np.empty(n_rows)
    months_employed[is_nondef] = rng.exponential(48, is_nondef.sum())
    months_employed[is_def] = rng.exponential(18, is_def.sum())
    months_employed = np.clip(months_employed, 0, 480).astype(float)

# Constraint - someone can't work longer than their age allows ( can't be 24 with 10 years of experience )

    months_employed = np.minimum(
        months_employed, (age - 16) * 12).astype(int)

    owns_property = np.empty(n_rows, dtype=float)
    owns_property[is_nondef] = rng.binomial(1, 0.42, is_nondef.sum())
    owns_property[is_def] = rng.binomial(1, 0.28, is_def.sum())
    debt_ratio[owns_property == 1] += 0.10  # mortgage component
    owns_vehicle = np.empty(n_rows, dtype=float)
    owns_vehicle[is_nondef] = rng.binomial(1, 0.48, is_nondef.sum())
    owns_vehicle[is_def] = rng.binomial(1, 0.35, is_def.sum())

# Small correlation: income -> loan_amount
    for _ in range(3):
        too_high = loan_amount > monthly_income * 36
        if too_high.sum() == 0: break
        loan_amount[too_high] = rng.lognormal(
            np.log(monthly_income[too_high] * 18), 0.4)

    # Correlation check for low-risk defaulters
    suspicious = (is_def & (total_late_payments == 0)
                  & (revolving_utilization < 0.3))
    flip_mask = suspicious & (rng.random(n_rows) < 0.7)
    defaulted[flip_mask] = 0

    # Boost utilization when late more that twice
    boost = (total_late_payments > 2) & (revolving_utilization < 0.4)
    revolving_utilization[boost] = np.clip(
        revolving_utilization[boost]
        + rng.uniform(0.15, 0.35, boost.sum()), 0, 1)

    # Dealing with mnar biased toward high-risk profiles
    monthly_income = apply_missing(monthly_income, 0.12, rng,
        mnar_bias_mask=is_def)

    debt_ratio = apply_missing(debt_ratio, 0.15, rng,
        mnar_bias_mask=(debt_ratio > np.nanmedian(debt_ratio)))

    months_employed_f = apply_missing(months_employed.astype(float),
        0.22, rng, mnar_bias_mask=(months_employed < 12))

    # MAR for dealing with standard random missingness
    total_late_f = apply_missing(total_late_payments, 0.10, rng)

    dependents_f = apply_missing(dependents, 0.20, rng,
        mnar_bias_mask=(dependents == 0))

    revolving_f = apply_missing(revolving_utilization, 0.08, rng)
    owns_prop_f = apply_missing(owns_property, 0.18, rng)
    owns_veh_f = apply_missing(owns_vehicle, 0.15, rng)
    loan_purpose = apply_missing_str(loan_purpose, 0.05, rng)

    # Build DataFrame
    df = pd.DataFrame({
        "defaulted": defaulted, "monthly_income": monthly_income,
        "revolving_utilization": revolving_f, "age": age,
        "debt_ratio": debt_ratio, "open_credit_lines": open_credit_lines,
        "total_late_payments": total_late_f,
        "real_estate_loans": real_estate_loans,
        "dependents": dependents_f, "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "months_employed": months_employed_f,
        "owns_property": owns_prop_f, "owns_vehicle": owns_veh_f,
        "source": "synthetic",
    })
    return df[UNIFIED_COLS]

# Validation checks :D

def validate_synthetic(df):
    checks = {
        "age range [18,90]":
            ((df["age"] >= 18) & (df["age"] <= 90)).mean() > 0.999,
        "revolving_utilization in [0,1]":
            df["revolving_utilization"].dropna().between(0, 1).mean() > 0.999,
        "no negative loan amounts":
            (df["loan_amount"] >= 0).all(),
        "default rate in [0.15, 0.22]":
            0.15 <= df["defaulted"].mean() <= 0.22,
        "months_employed age constraint":
            (df["months_employed"].dropna() <=
            (df.loc[df["months_employed"].notna(), "age"] - 16) * 12
            ).mean() > 0.98,
        "loan_to_income < 10":
            (df.loc[df["monthly_income"].notna(), "loan_amount"] /
             (df.loc[df["monthly_income"].notna(), "monthly_income"] * 12 + 1)
            ).lt(10).mean() > 0.995,
        "source is synthetic":
            (df["source"] == "synthetic").all(),
    }

    print("\n=== Synthetic Data Validation ===")

    for name, passed in checks.items():
        print(f" [{'PASS' if passed else 'WARN'}] {name}")

if __name__ == "__main__":
    df = generate_synthetic_slice(n_rows=50_000, random_seed=42)
    print(f"Shape: {df.shape}, Default rate: {df['defaulted'].mean():.1%}")

    validate_synthetic(df)
