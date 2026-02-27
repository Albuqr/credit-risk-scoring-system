import mlflow, mlflow.sklearn
import json, joblib, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
roc_auc_score, precision_score, recall_score, f1_score,
classification_report
)
from xgboost import XGBClassifier
from src.data_pipeline import run_pipeline

MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

def evaluate(model, X_test, y_test, threshold=0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        'auc': round(roc_auc_score(y_test, proba), 4),
        'precision': round(precision_score(y_test, preds, zero_division=0), 4),
        'recall': round(recall_score(y_test, preds, zero_division=0), 4),
        'f1': round(f1_score(y_test, preds, zero_division=0), 4)
    }

# Training all three models

def train_all():
    X_train, X_test, y_train, y_test = run_pipeline()

    ## Scale pos_weight tells xgboost the amount of times more neg

    spw = round((y_train == 0).sum() / (y_train == 1).sum(), 2)
    print(f'scale_pos_weight = {spw}')

    mlflow.set_experiment('credit-risk')

    with mlflow.start_run(run_name='logistic_regression'):
        params = {'C':1.0, 'class_weight':'balanced', 'max_iter':1000}
        lr = LogisticRegression(**params, random_state=42)
        lr.fit(X_train, y_train)
        m = evaluate(lr, X_test, y_test)
        mlflow.log_params(params); mlflow.log_metrics(m)
        mlflow.sklearn.log_model(lr, 'model')
        print(f'LR AUC: {m['auc']}')


    with mlflow.start_run(run_name='random_forest'):
        params = {'n_estimators':200, 'max_depth':10,
                  'class_weight':'balanced', 'n_jobs':-1}
        rf = RandomForestClassifier(**params, random_state=42)
        rf.fit(X_train, y_train)
        m = evaluate(rf, X_test, y_test)
        mlflow.log_metrics(m); mlflow.log_metrics(m)
        mlflow.sklearn.log_model(rf, 'model')
        print(f'RF AUC: {m["auc"]}')

    with mlflow.start_run(run_name='xgboost') as run:
        params = {'n_estimators':300, 'max_depth':5,'learning_rate':0.05,
                  'subsample':0.8, 'colsample_bytree':0.8, 'scale_pos_weight':spw,
                  'eval_metric':'auc'}
        xgb = XGBClassifier(**params, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        m = evaluate(xgb, X_test, y_test)
        mlflow.log_metrics(m); mlflow.log_metrics(m)
        mlflow.sklearn.log_model(xgb, 'model')
        print(f'XGB AUC: {m["auc"]}')

        print(classification_report(y_test,(xgb.predict_proba(X_test)[:, 1] >= 0.5).astype(int),
              target_names=['No Default', 'Default']))

        joblib.dump(xgb, MODELS_DIR / 'credit_model.pkl')
        json.dump({**m, 'run_id': run.info.run_id},
                  open(MODELS_DIR / 'metrics.json', 'w'), indent=2)

        return xgb

if __name__ == '__main__':
    train_all()

