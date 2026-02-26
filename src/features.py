import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

MODELS_DIR = Path('models')

##this is the contact between the preprocessing and the actual model
##changing this list requires re-training the model

FEATURE_NAMES = [
'revolving_utilization', 'age', 'log_income',
'debt_ratio', 'log_debt_to_income',
'open_credit_lines', 'total_late_payments',
'has_late_payment', 'real_estate_loans', 'dependents',
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

# Question : Has this person EVER been late? Strong binary signal

    df['has_late_payment'] = (df['total_late_payments'] > 0).astype(int)
    return df

# This will return the columns in the exact order defined in FEATURE_NAMES
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'revolving_utilization': df['revolvingutilizationofunsecuredlines'],
        'age': df['age'],
        'log_income': df['log_income'],
        'debt_ratio': df['debtratio'],
        'log_debt_to_income': df['log_debt_to_income'],
        'open_credit_lines': df['numberofopencreditlinesandloans'],
        'total_late_payments': df['total_late_payments'],
        'has_late_payment': df['has_late_payment'],
        'real_estate_loans': df['numberrealestateloansorlines'],
        'dependents': df['numberofdependents'],
    })

# THIS IS IMPORTANT DICKHEAD : fit only training data, never test or validation

def fit_and_save_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    return scaler

def load_scaler() -> StandardScaler:
    return joblib.load(MODELS_DIR / 'scaler.pkl')