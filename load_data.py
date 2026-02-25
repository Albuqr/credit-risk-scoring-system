from src.database import load_csv_to_db, query

load_csv_to_db('data/cs-training.csv')

print(query('SELECT COUNT(*) as rows FROM raw_credit_data'))

print(query('''
    SELECT defaulted,
           COUNT(*) as n,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
    FROM raw_credit_data GROUP BY defaulted
'''))