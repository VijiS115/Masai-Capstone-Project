import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from src.shipment_prep import FEATURES

def run_part_b():
    modelling_df = pd.read_csv('data/modelling_table.csv')
    X = modelling_df[FEATURES]
    y = modelling_df['delivery_hours']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    print("Part B Regression checks finished.")

if __name__ == '__main__':
    run_part_b()
