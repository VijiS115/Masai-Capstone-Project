import os
import joblib
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

def clean_shipments(df, medians):
    # 1. Drop exact duplicate rows
    df = df.drop_duplicates().copy()
    
    # 2. clean service_type
    df['service_type'] = df['service_type'].str.strip().str.title()
    
    # 3. clean payment_mode
    df['payment_mode'] = df['payment_mode'].fillna('Unknown')
    
    # 4. clean declared_value -> declared_value_inr
    df['declared_value_inr'] = df['declared_value'].str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df = df.drop('declared_value', axis=1)
    
    # 5. convert dispatch_time to datetime
    df['dispatch_time'] = pd.to_datetime(df['dispatch_time'])
    
    # 6. treat sentinel weights > 100 as missing
    df.loc[df['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan
    
    # 7. fill missing values with stored medians
    for col, med_val in medians.items():
        df[col] = df[col].fillna(med_val)
        
    return df

def build_features(df, hubs_df):
    # 1. left merge hubs
    merged = df.merge(
        hubs_df[['hub_id', 'hub_city', 'region', 'hub_tier', 'hub_daily_capacity']],
        on='hub_id',
        how='left',
        validate='many_to_one'
    )
    
    # 2. promised_hours
    merged['promised_hours'] = merged['service_type'].map(PROMISED_HOURS).astype(int)
    
    # 3. is_cod
    merged['is_cod'] = (merged['payment_mode'] == 'COD').astype(int)
    
    # 4. is_weekend
    merged['is_weekend'] = (merged['dispatch_time'].dt.dayofweek >= 5).astype(int)
    
    # 5. dispatch_hour
    merged['dispatch_hour'] = merged['dispatch_time'].dt.hour.astype(int)
    
    # 6. log_declared_value
    merged['log_declared_value'] = np.log(merged['declared_value_inr'])
    
    return merged

def main():
    # Load datasets
    new_df = pd.read_csv('data/new_shipments.csv')
    hubs_df = pd.read_csv('data/hubs.csv')
    segments_df = pd.read_csv('outputs/hub_segments.csv')
    
    # Load models and configurations
    pipeline = joblib.load('models/delay_model.pkl')
    threshold = float(pd.read_csv('models/delay_threshold.csv').iloc[0]['threshold'])
    feature_order = pd.read_csv('models/feature_order.csv')['feature'].tolist()
    
    impute_data = pd.read_csv('models/impute_values.csv')
    medians = dict(zip(impute_data['column'], impute_data['median']))
    
    # Prep and clean shipments
    cleaned = clean_shipments(new_df, medians)
    featurized = build_features(cleaned, hubs_df)
    
    # Extract features in correct sequence
    X = featurized[feature_order]
    
    # Calculate delay probability
    probs = pipeline.predict_proba(X)[:, 1]
    featurized['delay_probability'] = np.round(probs, 4)
    
    # Calculate flagged
    featurized['flagged'] = (featurized['delay_probability'] >= threshold).astype(int)
    
    # Join hub segments
    final_df = featurized.merge(segments_df[['hub_id', 'segment']], on='hub_id', how='left')
    
    # Assign actions based on operational playbooks
    def assign_action(row):
        if row['flagged'] == 1:
            if row['segment'] == 'Escalate':
                return 'Escalate to hub manager'
            else:
                return 'Proactive customer call'
        return 'No action'
        
    final_df['action'] = final_df.apply(assign_action, axis=1)
    
    # Select final target columns in original file order
    output_cols = ['shipment_id', 'hub_id', 'segment', 'delay_probability', 'flagged', 'action']
    output_df = final_df[output_cols]
    
    os.makedirs('outputs', exist_ok=True)
    output_df.to_csv('outputs/new_shipment_actions.csv', index=False)
    print("Successfully saved outputs/new_shipment_actions.csv!")

if __name__ == '__main__':
    main()
