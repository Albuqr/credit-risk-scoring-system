import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from api.schemas import CustomerInput, PredictionOutput, RiskFactor, RiskCategory
from src.features import FEATURE_NAMES, engineer_features, build_feature_matrix
from src.missingness import add_missingness_indicators

MODELS_DIR = Path(__file__).parent.parent / "models"

class CreditRiskPredictor:

    def __init__(self):
        # Loads model scaler and SHAP explainer when the API starts

        # All prediction requests share a single instance bc loading on every request adds latency
        self.model = joblib.load(MODELS_DIR / 'credit_model.pkl')
        self.scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        self.explainer = joblib.load(MODELS_DIR / 'shap_explainer.pkl')
        self.version = '3.0.0'
        print(f'Model loaded. Features: {self.model.n_features_in_}')

    def _confidence(self, c: CustomerInput) -> str:
        # HIGH confidence = all MNAR fields provided
        # Confidence drops as MNAR fields are missing
        mnar_missing = sum([
            c.monthly_income is None,
            c.debt_ratio is None,
            c.months_employed is None,
        ])
        if mnar_missing == 0: return 'HIGH'
        if mnar_missing == 1: return 'MEDIUM'
        return 'LOW'

    def _build_features(self, c: CustomerInput) -> pd.DataFrame:
        # Build single-row DataFrame — None becomes NaN automatically
        raw = pd.DataFrame([{
            'monthly_income':        c.monthly_income,
            'revolving_utilization': c.revolving_utilization,
            'age':                   c.age,
            'debt_ratio':            c.debt_ratio,
            'open_credit_lines':     c.open_credit_lines,
            'total_late_payments':   c.total_late_payments,
            'real_estate_loans':     c.real_estate_loans,
            'dependents':            c.dependents,
            'loan_amount':           c.loan_amount,
            'loan_purpose':          c.loan_purpose,
            'months_employed':       c.months_employed,
            'owns_property':         c.owns_property,
            'owns_vehicle':          c.owns_vehicle,
        }])

        # Add missingness indicators and applies MAR/MCAR imputation
        # MNAR fields are left as NaN — XGBoost handles them so it ok

        raw = add_missingness_indicators(raw)
        raw = engineer_features(raw)
        return build_feature_matrix(raw)

    def predict(self, customer: CustomerInput) -> PredictionOutput:
        X_raw = self._build_features(customer)
        X_scaled = self.scaler.transform(X_raw.fillna(0))

        probability = float(self.model.predict_proba(X_raw)[0, 1])

        # High prob = Low score  |  Low prob = High score
        risk_score = int(max(300, min(850, 850 - probability * 550)))

        if probability < 0.10:   category = RiskCategory.LOW
        elif probability < 0.30: category = RiskCategory.MEDIUM
        else:                    category = RiskCategory.HIGH

        # SHAP explanation for this customer
        shap_vals = self.explainer.shap_values(X_raw)[0]
        ranked = sorted(zip(FEATURE_NAMES, shap_vals),
                        key=lambda x: abs(x[1]), reverse=True)

        top_factors = [
            RiskFactor(
                feature=feat,
                shap_value=round(float(val), 4),
                direction='increases_risk' if val > 0 else 'decreases_risk'
            )
            for feat, val in ranked[:5]
        ]

        return PredictionOutput(
            default_probability=round(probability, 4),
            risk_category=category,
            risk_score=risk_score,
            top_risk_factors=top_factors,
            prediction_confidence=self._confidence(customer),
            model_version=self.version
        )