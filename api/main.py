from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from sklearn.metrics import precision_recall_fscore_support

from api.schemas import (
CustomerInput, PredictionOutput, BatchInput, BatchOutput
)
from api.predictor import CreditRiskPredictor
import json
from pathlib import Path

from src.features import FEATURE_NAMES

predictor: CreditRiskPredictor = None

# This runs once after the server starts, loading here = model ready when 1st request comes thru
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = CreditRiskPredictor()
    yield

app = FastAPI(
title = 'Credit Risk Scoring API',
description = 'XGBoost based credit risk model sith SHAP per-prediction explanations',
version = '1.0.0',
lifespan=lifespan
 )

@app.get('/health')
def health():
    return {'status': 'Healthy', 'version': '1.0.0'}

@app.get('/model-info')
def model_info():
    metrics = json.loads(Path('models/metrics.json').read_text())
    return {'model': 'XGBoost', 'metrics': metrics, 'features': FEATURE_NAMES}


## Score a single customer and return a full explanation
## Real time endpoint
@app.post('/predict', response_model=PredictionOutput)
def predict(customer: CustomerInput):
    try:
        return predictor.predict(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong at /predict: {str(e)}")


## Score a list of customers in one request
## A 'real' scoring pipeline would process thousands of applicants overnight, i'm trying to mimic that, seeing as this is a project
@app.post('/batch-predict', response_model = BatchOutput)
def batch_predict(batch: BatchInput):
    try:
        preds = [predictor.predict(c) for c in batch.customers]
        return BatchOutput(predictions=preds, total_processed=len(preds))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong at /batch-predict: {str(e)}")

# Run this shit in terinal - uvicorn api.main:app --reload, reload makes it restart when the code is changed, remove for prod

