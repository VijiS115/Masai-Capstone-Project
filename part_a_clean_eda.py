import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

def run_part_a():
    raw_df = pd.read_csv('data/raw_shipments.csv')
    hubs_df = pd.read_csv('data/hubs.csv')
    
    # 1. Clean
    clean_df, medians = clean_shipments(raw_df)
    
    # 2. Build features
    merged_df = build_features(clean_df, hubs_df)
    merged_df['is_delayed'] = (merged_df['delivery_hours'] > merged_df['promised_hours']).astype(int)
    
    # Save modeling table
    merged_df.to_csv('data/modelling_table.csv', index=False)
    print("Successfully created data/modelling_table.csv")

if __name__ == '__main__':
    run_part_a()
