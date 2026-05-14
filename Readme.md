[BR Português](#sistema-de-scoring-de-crédito) | [US English](#credit-risk-scoring-system)

---

<a id="sistema-de-scoring-de-crédito"></a>

# Sistema de Scoring de Crédito

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square) ![Model](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square) ![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square) ![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square) ![Infra](https://img.shields.io/badge/Infra-Docker-2496ED?style=flat-square) ![DB](https://img.shields.io/badge/DB-SQLite-003B57?style=flat-square) ![Tracking](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=flat-square) ![Explainability](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Sobre

Pipeline de scoring de crédito do dado bruto ao deploy. Classifica solicitantes por probabilidade de inadimplência e devolve um score entre 300 e 850, junto dos 5 fatores que mais pesaram na decisão — calculados via SHAP para cada predição individualmente, não como média global.

O pipeline lê exclusivamente de tabelas SQLite. Nenhum CSV é lido em nenhuma etapa de treinamento ou serving. Cada run de treinamento é registrado no MLflow. A API tem rate limiting por IP, validação de entrada via Pydantic e documentação interativa gerada automaticamente.

[![API Docs](https://img.shields.io/badge/API-api.albuqr.com/docs-009688?style=flat-square)](https://api.albuqr.com/docs) [![Dashboard](https://img.shields.io/badge/Dashboard-dashboard.albuqr.com-FF4B4B?style=flat-square)](https://dashboard.albuqr.com)

## Endpoints

| Método | Rota | Limite | Descrição |
|---|---|---|---|
| GET | `/health` | — | Status do serviço |
| GET | `/model-info` | — | Métricas e features do modelo em produção |
| POST | `/predict` | 30/min por IP | Score de um solicitante com explicação SHAP |
| POST | `/batch-predict` | 5/min por IP | Score em lote |

## Performance

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.869 |
| Precision | 0.424 |
| Recall | 0.450 |
| F1 Score | 0.437 |

Treinado no Give Me Some Credit (Kaggle) — 150.000 registros, ~7% de inadimplência. Desbalanceamento tratado com `scale_pos_weight`. Comparado contra Logistic Regression e Random Forest no dashboard.

## Stack

| Camada | Tecnologia |
|---|---|
| Modelo | XGBoost + Logistic Regression + Random Forest |
| Explicabilidade | SHAP TreeExplainer |
| Rastreamento | MLflow |
| API | FastAPI + Pydantic + slowapi |
| Dashboard | Streamlit |
| Banco de Dados | SQLite |
| Containerização | Docker + docker-compose |
| Deploy | Easypanel + Traefik + VPS Ubuntu |

## Conformidade LGPD

Cada resposta da API inclui os 5 fatores individuais que influenciaram aquela decisão específica, com direção de impacto. Atende ao Artigo 20 da LGPD, que exige explicabilidade e possibilidade de revisão humana em decisões automatizadas com efeito significativo sobre o titular.

## Limitações

- Dados de treinamento americanos; comportamento de crédito e distribuições de renda diferem do mercado brasileiro
- Limiares LOW / MEDIUM / HIGH não calibrados contra taxas reais de inadimplência brasileiras
- SQLite sem suporte a escrita concorrente
- Sem autenticação nos endpoints

## Licença

MIT © 2026 João Pedro Castro Albuquerque

---

Built by Albuquerque · [albuqr.com](https://albuqr.com) · [GitHub: Albuqr](https://github.com/Albuqr)

---

<a id="credit-risk-scoring-system"></a>

# Credit Risk Scoring System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square) ![Model](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square) ![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square) ![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square) ![Infra](https://img.shields.io/badge/Infra-Docker-2496ED?style=flat-square) ![DB](https://img.shields.io/badge/DB-SQLite-003B57?style=flat-square) ![Tracking](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=flat-square) ![Explainability](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## About

Credit risk scoring pipeline from raw data to deployment. Classifies applicants by default probability and returns a 300–850 risk score alongside the 5 features that most influenced the decision — computed via SHAP per prediction, not as a global average.

The pipeline reads exclusively from SQLite tables. No CSV is read at any stage of training or serving. Every training run is logged in MLflow. The API has per-IP rate limiting, Pydantic input validation, and auto-generated interactive documentation.

[![API Docs](https://img.shields.io/badge/API-api.albuqr.com/docs-009688?style=flat-square)](https://api.albuqr.com/docs) [![Dashboard](https://img.shields.io/badge/Dashboard-dashboard.albuqr.com-FF4B4B?style=flat-square)](https://dashboard.albuqr.com)

## Endpoints

| Method | Route | Rate Limit | Description |
|---|---|---|---|
| GET | `/health` | — | Service status |
| GET | `/model-info` | — | Live model metrics and feature list |
| POST | `/predict` | 30/min per IP | Score a single applicant with SHAP explanation |
| POST | `/batch-predict` | 5/min per IP | Score a batch of applicants |

## Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.869 |
| Precision | 0.424 |
| Recall | 0.450 |
| F1 Score | 0.437 |

Trained on Give Me Some Credit (Kaggle) — 150,000 records, ~7% default rate. Class imbalance handled via `scale_pos_weight`. Benchmarked against Logistic Regression and Random Forest baselines, compared in the dashboard.

## Stack

| Layer | Technology |
|---|---|
| Model | XGBoost + Logistic Regression + Random Forest |
| Explainability | SHAP TreeExplainer |
| Experiment Tracking | MLflow |
| API | FastAPI + Pydantic + slowapi |
| Dashboard | Streamlit |
| Database | SQLite |
| Containerisation | Docker + docker-compose |
| Deployment | Easypanel + Traefik + Ubuntu VPS |

## LGPD Compliance

Every API response includes the 5 individual features that influenced that specific decision, with impact direction. This satisfies Article 20 of Brazil's LGPD, which requires that automated decisions with significant effects on data subjects be explainable and subject to human review.

## Limitations

- Training data is US-based; credit behaviour and income distributions differ from the Brazilian market
- LOW / MEDIUM / HIGH thresholds not calibrated against real Brazilian default rates
- SQLite has no concurrent write support
- No authentication on API endpoints

## License

MIT © 2026 João Pedro Castro Albuquerque

---

Built by Albuquerque · [albuqr.com](https://albuqr.com) · [GitHub: Albuqr](https://github.com/Albuqr)
