import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from sklearn.metrics import precision_recall_fscore_support

from api.schemas import (
CustomerInput, PredictionOutput, BatchInput, BatchOutput
)
from api.predictor import CreditRiskPredictor
import json
from pathlib import Path

from src.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

predictor: CreditRiskPredictor = None
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = CreditRiskPredictor()
    yield

app = FastAPI(
title = 'Credit Risk Scoring API',
description = 'XGBoost based credit risk model with SHAP per-prediction explanations',
version = '1.0.0',
lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:8501'],
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type'],
)

@app.get('/health')
def health():
    return {'status': 'Healthy', 'version': '1.0.0'}

@app.get('/model-info')
def model_info():
    raw = json.loads(Path('models/metrics.json').read_text())
    metrics = {k: v for k, v in raw.items() if k != 'run_id'}
    return {'model': 'XGBoost', 'metrics': metrics, 'features': FEATURE_NAMES}


## Score a single customer and return a full explanation
## Real time endpoint
@app.post('/predict', response_model=PredictionOutput)
@limiter.limit('30/minute')
def predict(request: Request, customer: CustomerInput):
    try:
        return predictor.predict(customer)
    except Exception as e:
        logger.error('Prediction failed: %s', e, exc_info=True)
        raise HTTPException(status_code=500, detail='Prediction failed. Please try again later.')


## Score a list of customers in one request
## A 'real' scoring pipeline would process thousands of applicants overnight, i'm trying to mimic that, seeing as this is a project
@app.post('/batch-predict', response_model = BatchOutput)
@limiter.limit('5/minute')
def batch_predict(request: Request, batch: BatchInput):
    try:
        preds = [predictor.predict(c) for c in batch.customers]
        return BatchOutput(predictions=preds, total_processed=len(preds))
    except Exception as e:
        logger.error('Batch prediction failed: %s', e, exc_info=True)
        raise HTTPException(status_code=500, detail='Batch prediction failed. Please try again later.')

# Run this shit in terinal - uvicorn api.main:app --reload, reload makes it restart when the code is changed, remove for prod
