import pandas as pd
import numpy as np
import os

DATA_PATH = "data/amazon.csv"

def load_or_generate_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        
        if 'discount_percentage' in df.columns:
            df['discount_percentage'] = (
                df['discount_percentage'].astype(str).str.replace('%', '')
                .apply(pd.to_numeric, errors='coerce').fillna(0)
            )

        for col in ['discounted_price', 'actual_price', 'rating_count']:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str).str.replace('₹', '').str.replace(',', '')
                    .apply(pd.to_numeric, errors='coerce').fillna(0)
                )

        if 'rating' in df.columns:
            df['rating'] = (
                df['rating'].astype(str).str.replace('|', '0', regex=False) 
                .apply(pd.to_numeric, errors='coerce').fillna(0)
            )
        df=df.drop_duplicates(subset=['product_id'], keep='first')
        return df.fillna("")
    
    print("Dataset not found. Generating synthetic data...")
    os.makedirs("data", exist_ok=True)
    return pd.DataFrame() 