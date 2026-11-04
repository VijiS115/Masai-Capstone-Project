import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES

def run_part_c():
    modelling_df = pd.read_csv('data/modelling_table.csv')
    X = modelling_df[FEATURES]
    y_reg = modelling_df['delivery_hours']
    y_strat = modelling_df['is_delayed']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_reg, test_size=0.3, random_state=42, stratify=y_strat)
    print("Part C Classification checks ready.")

if __name__ == '__main__':
    run_part_c()
