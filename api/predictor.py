import numpy as np
import joblib
from pathlib import Path
from api.schemas import CustomerInput, PredictionOutput, RiskFactor, RiskCategory
from src.features import FEATURE_NAMES

MODELS_DIR = Path(__file__).parent.parent / "models"

class CreditRiskPredictor:

    def __init__(self):
        # Loads model, scaler and SHAP explainer when the API starts
        # All prediction requests share a single instance
        # Loading on every request adds latency
        self.model = joblib.load(MODELS_DIR / 'credit_model.pkl')
        self.scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        self.explainer = joblib.load(MODELS_DIR / 'shap_explainer.pkl')
        self.version = '1.0.0'
        print(f'Model loaded. Features: {self.model.n_features_in_}')

    def _build_features(self, c: CustomerInput) -> np.ndarray:
        # This recreates the same steps as src/features.py
        dti = c.debt_ratio * c.monthly_income
        return np.array([[
            c.revolving_utilization,
            c.age,
            np.log1p(c.monthly_income), c.debt_ratio,
            np.log1p(dti), c.open_credit_lines,
            c.total_credit_lines,

            1 if c.total_late_payments > 0 else 0,
            c.real_estate_loans, c.dependents,
        ]])

    def predict(self, customer: CustomerInput) -> PredictionOutput:
        raw = self._build_features(customer)
        scaled = self.scaler.transform(raw)

        probability = float(self.model.predict_proba(scaled)[0, 1])

        # Here i'll need to convert the probability to a 300-850 credit-score-style int
        # High prob = Low score | Low prob = High Score

        risk_score = int(max(300, min(850, 850 - probability * 550)))

        if probability < 0.10: category = RiskCategory.LOW
        elif probability < 0.30: category = RiskCategory.MEDIUM
        else: category = RiskCategory.HIGH

        # Shap exp for this specific customer

        shap_vals = self.explainer.shap_values(scaled)[0]
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
            default_probability = round(probability, 4),
            risk_category = category,
            risk_score = risk_score,
            top_risk_factors = top_factors,
            model_version = self.version
        )