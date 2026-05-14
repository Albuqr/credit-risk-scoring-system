[BR Português](#sistema-de-scoring-de-crédito) | [US English](#credit-risk-scoring-system)

---

<a id="sistema-de-scoring-de-crédito"></a>

# Sistema de Scoring de Crédito

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square) ![Model](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square) ![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square) ![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square) ![Infra](https://img.shields.io/badge/Infra-Docker-2496ED?style=flat-square) ![DB](https://img.shields.io/badge/DB-SQLite-003B57?style=flat-square) ![Tracking](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=flat-square) ![Explainability](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Visão Geral

Sistema de scoring de crédito com foco em produção, treinado no dataset Give Me Some Credit (150.000 registros) com XGBoost como modelo principal. A API retorna probabilidade de inadimplência, score entre 300 e 850 e os 5 principais fatores de risco por predição via SHAP, viabilizando conformidade com o Artigo 20 da LGPD. O projeto foi desenvolvido como portfólio técnico direcionado a posições de ciência de dados em fintechs como Nubank, Itaú, Bradesco e Santander.

## Demo ao Vivo

[![API Docs](https://img.shields.io/badge/API-Docs-009688?style=flat-square)](https://api.albuqr.com/docs) [![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-FF4B4B?style=flat-square)](https://dashboard.albuqr.com)

## Arquitetura

```
Give Me Some Credit (CSV)
          |
          v
    load_data.py
          |
          v
  data/creditdb.sqlite
          |
          v
  src/data_pipeline.py
          |
          v
     src/features.py
          |
          v
  src/train.py (MLflow)
          |
          v
  models/credit_model.pkl
  models/scaler.pkl
  models/shap_explainer.pkl
          |
       +--+--+
       |     |
       v     v
  FastAPI  Streamlit
  :8000    :8501
       |     |
       v     v
api.albuqr.com  dashboard.albuqr.com
        (Traefik + SSL, Easypanel)
```

## Stack Tecnológica

| Camada | Tecnologia | Finalidade |
|---|---|---|
| Dados | SQLite + pandas | Pipeline SQL-first; sem leitura direta de CSV em produção |
| Modelo | XGBoost | Classificação binária de risco de crédito |
| Baselines | Logistic Regression, Random Forest | Comparação de performance no dashboard |
| Explicabilidade | SHAP TreeExplainer | Top 5 fatores de risco por predição |
| Rastreamento | MLflow | Registro de experimentos e métricas |
| API | FastAPI + Pydantic + Uvicorn | Endpoints /predict, /batch-predict, /health |
| Dashboard | Streamlit | Performance do modelo, predição interativa, distribuições |
| Containerização | Docker + docker-compose | Dois serviços isolados: API e dashboard |
| Deploy | Easypanel + Traefik | VPS Ubuntu, SSL automático, roteamento por domínio |

## Resultados do Modelo

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.869 |
| Precision | 0.423 |
| Recall | 0.450 |
| F1 Score | 0.436 |

Baselines avaliados: Logistic Regression e Random Forest. Comparação disponível no dashboard.

## Estrutura do Projeto

```
credit-risk-scoring-system/
├── api/
│   ├── main.py               # FastAPI: /predict, /batch-predict, /health
│   ├── predictor.py          # Carregamento do modelo e lógica de predição
│   └── schemas.py            # Schemas Pydantic de entrada e saída
├── src/
│   ├── data_pipeline.py      # Leitura do SQLite, split, scaling
│   ├── features.py           # Engenharia de features, FEATURE_NAMES
│   └── train.py              # Treinamento com MLflow, XGBoost + baselines
├── dashboard/
│   └── app.py                # Dashboard Streamlit (3 abas)
├── models/
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   ├── shap_explainer.pkl
│   └── metrics.json
├── data/
│   └── creditdb.sqlite
├── docs/
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   ├── class_imbalance.png
│   ├── correlations.png
│   └── distributions.png
├── notebooks/
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── load_data.py
├── requirements.txt
├── requirements-docker.txt
└── .gitignore
```

## Como Executar

**1. Clone o repositório**
```bash
git clone https://github.com/Albuqr/credit-risk-scoring-system.git
cd credit-risk-scoring-system
```

**2. Instale as dependências**
```bash
pip install -r requirements.txt
```

**3. Suba os containers**
```bash
docker-compose up
```

API disponível em `http://localhost:8000/docs`, dashboard em `http://localhost:8501`.

## API — Entrada e Saída

**POST /predict**

Requisição:
```json
{
  "revolving_utilization": 0.45,
  "age": 34,
  "monthly_income": 5000.0,
  "debt_ratio": 0.38,
  "open_credit_lines": 6,
  "total_late_payments": 1,
  "real_estate_loans": 1,
  "dependents": 2
}
```

Resposta:
```json
{
  "default_probability": 0.312,
  "risk_score": 621,
  "risk_category": "MEDIUM",
  "prediction_confidence": "HIGH",
  "top_risk_factors": [
    {"feature": "revolving_utilization", "direction": "increases risk", "value": 0.45},
    {"feature": "total_late_payments", "direction": "increases risk", "value": 1},
    {"feature": "debt_ratio", "direction": "increases risk", "value": 0.38},
    {"feature": "age", "direction": "decreases risk", "value": 34},
    {"feature": "monthly_income", "direction": "decreases risk", "value": 5000.0}
  ]
}
```

## Conformidade LGPD

Cada predição retorna os 5 fatores que mais influenciaram o resultado, calculados via SHAP TreeExplainer com direção de impacto. Isso atende ao Artigo 20 da LGPD, que exige explicação e possibilidade de revisão humana em decisões automatizadas com efeito significativo sobre o titular — tornando o modelo defensável juridicamente em contextos de crédito automatizado no Brasil.

## Limitações

- Os dados de treinamento são exclusivamente americanos (Give Me Some Credit, Kaggle); distribuições de renda, endividamento e comportamento de crédito diferem do mercado brasileiro
- O modelo não foi validado em dados de produção real; métricas refletem desempenho em conjunto de teste do mesmo dataset
- SQLite não suporta concorrência de escrita; inadequado para cargas multi-usuário em produção
- Os endpoints da API não possuem autenticação; não devem ser expostos publicamente sem camada de segurança adicional
- Os limiares LOW / MEDIUM / HIGH não foram calibrados contra taxas de inadimplência reais do mercado brasileiro

## Licença

MIT License © 2026 João Pedro Castro Albuquerque

---

Built by Albuquerque · [albuqr.com](https://albuqr.com) · [GitHub: Albuqr](https://github.com/Albuqr)

---

<a id="credit-risk-scoring-system"></a>

# Credit Risk Scoring System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square) ![Model](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square) ![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square) ![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square) ![Infra](https://img.shields.io/badge/Infra-Docker-2496ED?style=flat-square) ![DB](https://img.shields.io/badge/DB-SQLite-003B57?style=flat-square) ![Tracking](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=flat-square) ![Explainability](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Overview

A production-grade credit risk scoring system trained on the Give Me Some Credit dataset (150,000 records) using XGBoost as the primary model. The API returns default probability, a 300–850 risk score, and the top 5 SHAP-explained risk factors per prediction, supporting compliance with Article 20 of Brazil's LGPD. Built as a technical portfolio project targeting data science roles at Brazilian fintechs including Nubank, Itaú, Bradesco, and Santander.

## Live Demo

[![API Docs](https://img.shields.io/badge/API-Docs-009688?style=flat-square)](https://api.albuqr.com/docs) [![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-FF4B4B?style=flat-square)](https://dashboard.albuqr.com)

## Architecture

```
Give Me Some Credit (CSV)
          |
          v
    load_data.py
          |
          v
  data/creditdb.sqlite
          |
          v
  src/data_pipeline.py
          |
          v
     src/features.py
          |
          v
  src/train.py (MLflow)
          |
          v
  models/credit_model.pkl
  models/scaler.pkl
  models/shap_explainer.pkl
          |
       +--+--+
       |     |
       v     v
  FastAPI  Streamlit
  :8000    :8501
       |     |
       v     v
api.albuqr.com  dashboard.albuqr.com
        (Traefik + SSL, Easypanel)
```

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data | SQLite + pandas | SQL-first pipeline; no raw CSV reads in production |
| Model | XGBoost | Binary classification of credit default risk |
| Baselines | Logistic Regression, Random Forest | Performance comparison shown in dashboard |
| Explainability | SHAP TreeExplainer | Top 5 risk factors per prediction with direction |
| Experiment Tracking | MLflow | Run logging, metric tracking, model registry |
| API | FastAPI + Pydantic + Uvicorn | /predict, /batch-predict, /health endpoints |
| Dashboard | Streamlit | Model performance, live predictor, feature distributions |
| Containerisation | Docker + docker-compose | Two isolated services: API and dashboard |
| Deployment | Easypanel + Traefik | Ubuntu VPS, automatic SSL, domain-based routing |

## Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.869 |
| Precision | 0.423 |
| Recall | 0.450 |
| F1 Score | 0.436 |

Baselines evaluated: Logistic Regression and Random Forest. Full comparison available in the dashboard.

## Project Structure

```
credit-risk-scoring-system/
├── api/
│   ├── main.py               # FastAPI app: /predict, /batch-predict, /health
│   ├── predictor.py          # Model loading and prediction logic
│   └── schemas.py            # Pydantic input/output schemas
├── src/
│   ├── data_pipeline.py      # SQLite reads, train/test split, scaling
│   ├── features.py           # Feature engineering, FEATURE_NAMES list
│   └── train.py              # MLflow training: XGBoost + baselines
├── dashboard/
│   └── app.py                # Streamlit dashboard (3 tabs)
├── models/
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   ├── shap_explainer.pkl
│   └── metrics.json
├── data/
│   └── creditdb.sqlite
├── docs/
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   ├── class_imbalance.png
│   ├── correlations.png
│   └── distributions.png
├── notebooks/
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── load_data.py
├── requirements.txt
├── requirements-docker.txt
└── .gitignore
```

## Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/Albuqr/credit-risk-scoring-system.git
cd credit-risk-scoring-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the containers**
```bash
docker-compose up
```

The API will be available at `http://localhost:8000/docs` and the dashboard at `http://localhost:8501`.

## API Reference

**POST /predict**

Request body:
```json
{
  "revolving_utilization": 0.45,
  "age": 34,
  "monthly_income": 5000.0,
  "debt_ratio": 0.38,
  "open_credit_lines": 6,
  "total_late_payments": 1,
  "real_estate_loans": 1,
  "dependents": 2
}
```

Response:
```json
{
  "default_probability": 0.312,
  "risk_score": 621,
  "risk_category": "MEDIUM",
  "prediction_confidence": "HIGH",
  "top_risk_factors": [
    {"feature": "revolving_utilization", "direction": "increases risk", "value": 0.45},
    {"feature": "total_late_payments", "direction": "increases risk", "value": 1},
    {"feature": "debt_ratio", "direction": "increases risk", "value": 0.38},
    {"feature": "age", "direction": "decreases risk", "value": 34},
    {"feature": "monthly_income", "direction": "decreases risk", "value": 5000.0}
  ]
}
```

## LGPD Compliance

Every prediction returns the five features that most influenced the outcome, computed via SHAP TreeExplainer with explicit impact direction. This satisfies Article 20 of Brazil's Lei Geral de Proteção de Dados, which requires that automated decisions with significant effects on data subjects be explainable and subject to human review. The model output is designed to be legally defensible in automated credit decision contexts.

## Limitations

- Training data is exclusively US-based (Give Me Some Credit, Kaggle); income distributions, debt behaviour, and default patterns differ materially from the Brazilian market
- The model has not been validated against live production data; reported metrics reflect held-out test set performance from the same dataset
- SQLite does not support concurrent writes; it is not suitable for multi-user production workloads
- API endpoints have no authentication layer; they must not be exposed publicly without additional security controls
- LOW / MEDIUM / HIGH risk thresholds were not calibrated against real Brazilian default rates and should not be used for actual credit decisions without recalibration

## License

MIT License © 2026 João Pedro Castro Albuquerque

---

Built by Albuquerque · [albuqr.com](https://albuqr.com) · [GitHub: Albuqr](https://github.com/Albuqr)
