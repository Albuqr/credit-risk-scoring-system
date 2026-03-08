import pandas as pd
from sklearn.model_selection import train_test_split
from src.database import query
from src.features import (
clean_raw_data, engineer_features,
build_feature_matrix, fit_and_save_scaler
)
import sqlite3
from pathlib import Path
from src.missingness import add_missingness_indicators, missingness_report

DB_PATH = Path(__file__).parent.parent / "data" / "creditdb.sqlite"

def run_pipeline(test_size=0.2, random_state=42):
    print("1/6 Loading unified dataset from SQLite...")
    with sqlite3.connect(DB_PATH) as conn:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        if "unified_credit_data" not in tables:
            raise RuntimeError("Run: python -m src.data_fusion first.")
        df = pd.read_sql("SELECT * FROM unified_credit_data", conn)
    print(f"    Loaded {len(df):,} rows")

    print("2/6 Missingness analysis...")
    missingness_report(df)

    print("3/6 Cleaning and adding missingness indicators...")
    df = clean_raw_data(df)
    df = add_missingness_indicators(df)

    print("4/6 Engineering features...")
    df = engineer_features(df)

    print("5/6 Building feature matrix...")
    X = build_feature_matrix(df)
    y = df["defaulted"].values

    print("6/6 Splitting and scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = fit_and_save_scaler(X_train)
    X_train_scaled = scaler.transform(X_train.fillna(0))
    X_test_scaled = scaler.transform(X_test.fillna(0))

    print(f"    Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"    Default rate: train={y_train.mean():.3f}  test={y_test.mean():.3f}")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test