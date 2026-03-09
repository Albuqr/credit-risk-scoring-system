from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class RiskCategory(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class CustomerInput(BaseModel):
    revolving_utilization: float = Field(..., ge=0, le=1)
    age: int = Field(..., ge=18, le=120)
    monthly_income: Optional[float] = Field(None, ge=0)        # MNAR
    debt_ratio: Optional[float] = Field(None, ge=0)            # MNAR
    open_credit_lines: int = Field(..., ge=0)
    total_late_payments: Optional[int] = Field(None, ge=0)     # MAR
    real_estate_loans: Optional[int] = Field(None, ge=0)       # MAR
    dependents: Optional[int] = Field(None, ge=0)              # MAR
    loan_amount: float = Field(..., ge=0)
    loan_purpose: Optional[str] = None                         # MAR
    months_employed: Optional[int] = Field(None, ge=0)        # MNAR
    owns_property: Optional[float] = None                      # MAR
    owns_vehicle: Optional[float] = None                       # MAR

    class Config:
        json_schema_extra = {'example': {
            'revolving_utilization': 0.35, 'age': 42,
            'monthly_income': 6500.0, 'debt_ratio': 0.28,
            'open_credit_lines': 7, 'total_late_payments': 0,
            'real_estate_loans': 1, 'dependents': 2,
            'loan_amount': 15000.0, 'loan_purpose': 'personal',
            'months_employed': 36, 'owns_property': 1.0,
            'owns_vehicle': 0.0}}

class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str

class PredictionOutput(BaseModel):
    default_probability: float
    risk_category: RiskCategory
    risk_score: int
    top_risk_factors: List[RiskFactor]
    prediction_confidence: str  # HIGH / MEDIUM / LOW
    model_version: str

class BatchInput(BaseModel):
    customers: List[CustomerInput] = Field(..., min_length=1, max_length=100)

class BatchOutput(BaseModel):
    predictions: list[PredictionOutput]
    total_processed: int


