import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path('data/creditdb.sqlite')

def get_connection():
    return sqlite3.connect(DB_PATH)

def load_csv_to_db(csv_path: str) -> None:
    df = pd.read_csv(csv_path, index_col=0)

#Normalization of column names: lowercase, no hypens

    df.columns = [c.strip().lower().replace('-','_') for c in df.columns]
    df.rename(columns={'seriousdlqin2yrs': 'defaulted'}, inplace=True)

#db connection

    conn = get_connection()
    df.to_sql('raw_credit_data', conn, if_exists='replace', index=True , index_label='customer_id')
    conn.close()
    print(f'Loaded {len(df):,} rows.')

def query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    result = pd.read_sql_query(sql, conn)
    conn.close()
    return result
