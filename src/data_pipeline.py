import pandas as pd
from sklearn.model_selection import train_test_split
from src.database import query
from src.features import (
clean_raw_data, engineer_features,
build_feature_matrix, fit_and_save_scaler
)

def run_pipeline(test_size: float = 0.2, random_state: int = 42):
    print('1/5 Loading data from database...')
    df= query('SELECT * FROM raw_credit_data')

    print('2/5 Cleaning data...')
    df = clean_raw_data(df)

    print('3/5 Engineering features...')
    df = engineer_features(df)

    print('4/5 Building feature matrix...')
    X = build_feature_matrix(df)
    y = df['defaulted'].values

# Stratify = y ensures the ratio of 93.7% non defaulters remain the same between the train test splits
    print('5/5 Splitting and scaling...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler =  fit_and_save_scaler(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')
    print(f'Default rate - train {y_train.mean():.3f} test {y_test.mean():.3f}')

    return X_train, X_test, y_train, y_test