from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class RiskCategory(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class CustomerInput(BaseModel):
    revolving_utilization: float = Field(..., ge=0, le=1,description="Credit card balance divided by credit limit. 0 to 1.")
    age: int = Field(..., ge=18, le=120)
    monthly_income: float = Field(..., ge=0)
    debt_ratio: float = Field(..., ge=0)
    open_credit_lines: int = Field(..., ge=0)
    total_late_payments: int = Field(..., ge=0)
    total_credit_lines: int = Field(..., ge=0)
    real_estate_loans: int = Field(..., ge=0)
    dependents: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {'example': {
                'revolving_utilization': 0.35, 'age': 42,
                'monthly_income': 6500.0, 'debt_ratio': 0.28,
                'open_credit_lines': 7, 'total_late_payments': 0,
                'real_estate_loans': 1, 'dependents': 2}}


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str # increases_risk or decreases_risk

class PredictionOutput(BaseModel):
    default_probability: float
    risk_category: RiskCategory
    risk_score: int # 300-850 scale, the higher the 'safer'
    top_risk_factors: List[RiskFactor]
    model_version: str

class BatchInput(BaseModel):
    customers: List[CustomerInput] = Field(..., min_length=1, max_length=100)

class BatchOutput(BaseModel):
    predictions: list[PredictionOutput]
    total_processed: int


