import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / 'models'

##this is the contact between the preprocessing and the actual model
##changing this list requires re-training the model

FEATURE_NAMES = [
#  Original features - Give me some credit
"revolving_utilization", "age", "log_income", "debt_ratio",
"log_debt_to_income", "open_credit_lines", "total_late_payments",
"has_late_payment", "real_estate_loans", "dependents",

#  New loan features - Lenders club
"loan_amount", "log_loan_amount", "loan_purpose_risk",
"loan_to_income", "payment_to_income",

#  New personal asset features
"owns_property", "owns_vehicle", "asset_score",

#  New employment features
"employment_stability", "is_stable_employed",

#  Interaction terms - how groups of features affect the outcome
"age_x_log_loan", "income_x_property",
"utilization_x_late", "debt_ratio_x_loan_to_income",

#  Missingness indicators for MNAR
"monthly_income_missing", "debt_ratio_missing",
"months_employed_missing",

#  Missingness indicators for MAR
"revolving_utilization_missing", "total_late_payments_missing",
"dependents_missing", "owns_property_missing", "owns_vehicle_missing",
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['monthlyincome'] = df['monthlyincome'].fillna(df['monthlyincome'].median())
    df['numberofdependents'] = df['numberofdependents'].fillna(0)
    df = df[df['age'] >= 18].copy()
    df['revolvingutilizationofunsecuredlines'] = (
        df['revolvingutilizationofunsecuredlines'].clip(upper=1.0)
    )
    return df

## Engineering and Matrix Builder

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

## Turn raw debt-to-income in Dollars

    df['debt_to_income'] = df['debtratio'] * df['monthlyincome']

## Log Transforms: log(x+1) handles zero safely, compresses right skew

    df['log_income'] = np.log1p(df['monthlyincome'])
    df['log_debt_to_income'] = np.log1p(df['debt_to_income'])

## Aggregate late payment history across all severity buckets

    df['total_late_payments'] = (
        df['numberoftime30_59dayspastduenotworse'] +
        df['numberoftime60_89dayspastduenotworse'] +
        df['numberoftimes90dayslate']
     )

    # Loan features
    df["log_loan_amount"] = np.log1p(df["loan_amount"].fillna(0))

    PURPOSE_RISK = {
        "mortgage": 0, "car": 1, "home_improvement": 1,
        "educational": 2, "major_purchase": 2,
        "medical": 3, "personal": 3, "vacation": 3,
        "moving": 4, "small_business": 4,
        "debt_consolidation": 5, "other": 5, "renewable_energy": 3,
    }

    df["loan_purpose_risk"] = df["loan_purpose"].map(PURPOSE_RISK).fillna(3)

    df["loan_to_income"] = np.where(
        df["monthly_income"].notna() & (df["monthly_income"] > 0),
        df["loan_amount"] / (df["monthly_income"] * 12 + 1), np.nan)

    monthly_payment = df["loan_amount"] * 0.02

    df["payment_to_income"] = np.where(
        df["monthly_income"].notna() & (df["monthly_income"] > 0),
        monthly_payment / (df["monthly_income"] + 1), np.nan)

    # Asset features
    df["owns_property"] = df["owns_property"].fillna(0)

    df["owns_vehicle"] = df["owns_vehicle"].fillna(0)
    df["asset_score"] = df["owns_property"] * 2 + df["owns_vehicle"]


    # Employment features
    df["employment_stability"] = np.log1p(df["months_employed"])
    df["is_stable_employed"] = np.where(
        df["months_employed"].notna(),
        (df["months_employed"] >= 24).astype(int), np.nan)


    # Interaction terms
    df["age_x_log_loan"] = df["age"] * df["log_loan_amount"]
    df["income_x_property"] = df["log_income"] * df["owns_property"]

    df["utilization_x_late"] = (
            df["revolving_utilization"] * df["total_late_payments"].fillna(0))

    df["debt_ratio_x_loan_to_income"] = (
            df["debt_ratio"].fillna(0) * df["loan_to_income"].fillna(0))

    # Question : Has this person EVER been late? Strong signal

    df['has_late_payment'] = (df['total_late_payments'] > 0).astype(int)
    return df

# This will return the columns in the exact order defined in FEATURE_NAMES
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for feat in FEATURE_NAMES:
        if feat in df.columns:
            result[feat] = df[feat].values
        else:
            result[feat] = np.full(len(df), np.nan)
        return pd.DataFrame(result)

# THIS IS IMPORTANT DICKHEAD : fit only training data, never test or validation

def fit_and_save_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    # fill null stuff wit 0 for scaler fitting
    X_fill = (
    X_train.fillna(0) if isinstance(X_train, pd.DataFrame)
    else np.nan_to_num(X_train, nan=0.0)
)
    scaler.fit(X_fill)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    return scaler

def load_scaler() -> StandardScaler:
    return joblib.load(MODELS_DIR / 'scaler.pkl')