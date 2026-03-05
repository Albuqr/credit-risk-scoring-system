import pandas as pd
import streamlit as st
import joblib
import json
import numpy as np
import plotly.express as px
from pathlib import Path
import sys; sys.path.append('.')
BASE_DIR = Path(__file__).parent.parent
from api.predictor import CreditRiskPredictor
from api.schemas import CustomerInput

st.set_page_config(page_title='Credit Risk Dashboard', layout='wide', page_icon='🏦')
@st.cache_resource
def load_predictor(): return CreditRiskPredictor()

predictor = load_predictor()
metrics = json.loads(Path('models/metrics.json').read_text())

tab1,tab2,tab3 = st.tabs([
'📊 Model Performance', '🔍 Live Predictor', '📈 Distributions'
])

with tab1:
    st.header('Model Performance')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('AUC-ROC', f'{metrics["auc"]:.3f}')
    c2.metric('Precision', f'{metrics["precision"]:.3f}')
    c3.metric('Recall', f'{metrics["recall"]:.3f}')
    c4.metric('F1 Score', f'{metrics["f1"]:.3f}')

    comp = pd.DataFrame({'Model':['Logistic Reg.','Random Forest','XGBoost'],
                         'AUC':[0.820, 0.851, metrics['auc']]})

    fig = px.bar(comp, x='Model', y='AUC', color='AUC',
                 color_continuous_scale='blues', title='Model Comparison')
    fig.update_layout(yaxis_range=[0.7,1.0])
    st.plotly_chart(fig, use_container_width=True)
    st.image(str(BASE_DIR / 'docs' / 'shap_summary.png'), caption='SHAP Global Importance', use_container_width=True)

with tab2:
    st.header('Score a Customer')
    c1,c2 = st.columns(2)
    with c1:
        util = st.slider('Credit Utilization', 0.0, 1.0, 0.35)
        age = st.number_input('Age', 18, 100, 40)
        income = st.number_input('Monthly Income ($)', 0, 200000, 5000)
        debt = st.slider('Debt Ratio', 0.0, 2.0, 0.30)

    with c2:
        lines = st.number_input('Open Credit Lines', 0, 50, 7)
        total_lines = st.number_input('Total Credit Lines', 0, 50, 10)
        late = st.number_input('Total Late Payments', 0, 100, 0)
        re = st.number_input('Real Estate Loans', 0, 20, 1)
        deps = st.number_input('Dependents', 0, 20, 0)

    if st.button('Score', type='primary'):
        r = predictor.predict(CustomerInput(
            revolving_utilization=util, age=age, monthly_income=income,
            debt_ratio=debt, open_credit_lines=lines, total_credit_lines=total_lines,
            total_late_payments=late, real_estate_loans=re, dependents=deps))
        col = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
        rc = r.risk_category.value

        st.divider()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'### Category: :{col[rc]}[{rc}]')
            st.caption('Risk classification based on default probability. LOW < 10%. MEDIUM 10–30%. HIGH > 30%.')
        with m2:
            st.metric('Default Probability', f'{r.default_probability:.1%}')
            st.caption('Probability the customer will default within 2 years. Above 30% = HIGH risk.')
        with m3:
            st.metric('Risk Score (300-850)', r.risk_score)
            st.caption('Credit score from 300 (highest risk) to 850 (lowest risk). Below 580 = HIGH risk.')

        st.subheader('Top 5 Risk Factors (SHAP)')
        for col, f in zip(st.columns(5), r.top_risk_factors):
            icon = '🔴' if f.direction == 'increases_risk' else '🟢'
            with col:
                st.markdown(f'**{icon} {f.feature}**')
                st.write(f'{f.shap_value:+.4f}')
                st.caption('Pushes default probability UP' if f.direction == 'increases_risk' else 'Pushes default probability DOWN')

with tab3:
    st.header('Feature Distributions by Default Status')
    import sqlite3

    conn = sqlite3.connect('data/creditdb.sqlite')
    sample = pd.read_sql_query(
        'SELECT * FROM raw_credit_data ORDER BY RANDOM() LIMIT 5000', conn)
    conn.close()
    feat = st.selectbox('Select feature to inspect', sample.columns[1:])
    fig = px.histogram(sample, x=feat, color='defaulted',
                       barmode='overlay', opacity=0.7,
                       color_discrete_map={0: '#4f8ef7', 1: '#f05a5a'},
                       title=f'{feat} distribution by default status')
    st.plotly_chart(fig, use_container_width=True)
