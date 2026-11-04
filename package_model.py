import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

def package():
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # 1. Read configurations from previous runs
    final_config = pd.read_csv('outputs/part_c_final_config.csv').iloc[0]
    selector_k = int(final_config['selector_k'])
    chosen_threshold = float(final_config['threshold'])

    # 2. Rebuild the exact Part C stratified splits
    modelling_df = pd.read_csv('data/modelling_table.csv')
    X = modelling_df[FEATURES]
    y_reg = modelling_df['delivery_hours']
    y_strat = modelling_df['is_delayed']

    # Train (70%) vs Temporary (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_reg, test_size=0.3, random_state=42, stratify=y_strat
    )

    # Validation (15% overall) vs Test (15% overall)
    y_temp_strat = modelling_df.loc[X_temp.index, 'is_delayed']
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_strat
    )

    y_train_clf = modelling_df.loc[X_train.index, 'is_delayed']
    y_test_clf = modelling_df.loc[X_test.index, 'is_delayed']

    # 3. Create and fit the final pipeline on the training subset
    final_pipeline = Pipeline([
        ('selector', SelectKBest(f_classif, k=selector_k)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    final_pipeline.fit(X_train, y_train_clf)

    # 4. Serialize artifacts
    joblib.dump(final_pipeline, 'models/delay_model.pkl')
    pd.DataFrame({'threshold': [chosen_threshold]}).to_csv('models/delay_threshold.csv', index=False)
    pd.DataFrame({'feature': FEATURES}).to_csv('models/feature_order.csv', index=False)

    # Extract clean shipment medians from shipments table dynamically
    raw_df = pd.read_csv('data/raw_shipments.csv')
    shipments_clean = raw_df.drop_duplicates().copy()
    shipments_clean['declared_value_inr'] = shipments_clean['declared_value'].str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
    shipments_clean.loc[shipments_clean['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan

    impute_df = pd.DataFrame({
        'column': ['distance_km', 'parcel_weight_kg', 'declared_value_inr'],
        'median': [
            float(shipments_clean['distance_km'].median()),
            float(shipments_clean['parcel_weight_kg'].median()),
            float(shipments_clean['declared_value_inr'].median())
        ]
    })
    impute_df.to_csv('models/impute_values.csv', index=False)
    print("Successfully packaged and serialized final model assets!")

if __name__ == '__main__':
    package()
