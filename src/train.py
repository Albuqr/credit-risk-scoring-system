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

    ## Cap scale_pos_weight at 3 — the raw ratio (~14) causes severe over-prediction of defaults
    raw_spw = (y_train == 0).sum() / (y_train == 1).sum()
    spw = round(min(raw_spw, 3.0), 2)
    print(f'scale_pos_weight (raw={raw_spw:.2f}, capped={spw})')

    mlflow.set_experiment('credit-risk')

    with mlflow.start_run(run_name='logistic_regression'):
        params = {'C':1.0, 'class_weight':'balanced', 'max_iter':1000}
        lr = LogisticRegression(**params, random_state=42)
        lr.fit(X_train, y_train)
        m = evaluate(lr, X_test, y_test)
        train_auc = round(roc_auc_score(y_train, lr.predict_proba(X_train)[:, 1]), 4)
        mlflow.log_params(params); mlflow.log_metrics({**m, 'train_auc': train_auc})
        mlflow.sklearn.log_model(lr, 'model')
        print(f'LR  train_auc={train_auc}  test_auc={m["auc"]}')


    with mlflow.start_run(run_name='random_forest'):
        params = {'n_estimators':200, 'max_depth':10,
                  'class_weight':'balanced', 'n_jobs':-1}
        rf = RandomForestClassifier(**params, random_state=42)
        rf.fit(X_train, y_train)
        m = evaluate(rf, X_test, y_test)
        train_auc = round(roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1]), 4)
        mlflow.log_params(params); mlflow.log_metrics({**m, 'train_auc': train_auc})
        mlflow.sklearn.log_model(rf, 'model')
        print(f'RF  train_auc={train_auc}  test_auc={m["auc"]}')

    with mlflow.start_run(run_name='xgboost_v2') as run:
        params = {
            'n_estimators': 300,
            'max_depth': 4,           # reduced from 5 — less overfitting
            'learning_rate': 0.05,
            'subsample': 0.7,         # reduced from 0.8
            'colsample_bytree': 0.7,  # reduced from 0.8
            'scale_pos_weight': spw,  # capped at 3 instead of ~14
            'min_child_weight': 5,    # added — prevents splits on few samples
            'reg_alpha': 0.1,         # added — L1 regularization
            'reg_lambda': 2.0,        # added — L2 regularization
            'eval_metric': 'auc'
        }
        xgb = XGBClassifier(**params, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)

        train_auc = round(roc_auc_score(y_train, xgb.predict_proba(X_train)[:, 1]), 4)
        m = evaluate(xgb, X_test, y_test)
        mlflow.log_params(params); mlflow.log_metrics({**m, 'train_auc': train_auc})
        mlflow.sklearn.log_model(xgb, 'model')
        print(f'XGB train_auc={train_auc}  test_auc={m["auc"]}  gap={round(train_auc - m["auc"], 4)}')

        print(classification_report(y_test,(xgb.predict_proba(X_test)[:, 1] >= 0.5).astype(int),
              target_names=['No Default', 'Default']))

        # Probability distribution on test set
        proba_test = xgb.predict_proba(X_test)[:, 1]
        print('\n--- Predicted probability distribution (test set) ---')
        for pct in [10, 25, 50, 75, 90, 95, 99]:
            print(f'  p{pct}: {np.percentile(proba_test, pct):.4f}')

        # Risk category distribution with current thresholds
        low    = (proba_test < 0.10).mean()
        medium = ((proba_test >= 0.10) & (proba_test < 0.30)).mean()
        high   = (proba_test >= 0.30).mean()
        print(f'\n--- Risk category distribution (thresholds: LOW<0.10, MEDIUM<0.30, HIGH>=0.30) ---')
        print(f'  LOW:    {low:.1%}')
        print(f'  MEDIUM: {medium:.1%}')
        print(f'  HIGH:   {high:.1%}')

        joblib.dump(xgb, MODELS_DIR / 'credit_model.pkl')
        json.dump({**m, 'train_auc': train_auc, 'run_id': run.info.run_id},
                  open(MODELS_DIR / 'metrics.json', 'w'), indent=2)

        return xgb

if __name__ == '__main__':
    train_all()
