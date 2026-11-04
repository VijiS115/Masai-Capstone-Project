import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(df, medians=None):
    """Cleans raw shipment dataframe."""
    df = df.drop_duplicates().copy()
    df['service_type'] = df['service_type'].str.strip().str.title()
    df['payment_mode'] = df['payment_mode'].fillna('Unknown')
    df['declared_value_inr'] = df['declared_value'].str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df = df.drop('declared_value', axis=1)
    df['dispatch_time'] = pd.to_datetime(df['dispatch_time'])
    df.loc[df['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan

    if medians is None:
        medians = {
            'distance_km': round(float(df['distance_km'].median()), 2),
            'parcel_weight_kg': round(float(df['parcel_weight_kg'].median()), 2),
            'declared_value_inr': round(float(df['declared_value_inr'].median()), 2)
        }

    for col, med_val in medians.items():
        df[col] = df[col].fillna(med_val)

    return df, medians

def build_features(df, hubs_df):
    """Enriches cleaned shipments with features."""
    merged = df.merge(
        hubs_df[['hub_id', 'hub_city', 'region', 'hub_tier', 'hub_daily_capacity']],
        on='hub_id',
        how='left',
        validate='many_to_one'
    )
    merged['promised_hours'] = merged['service_type'].map(PROMISED_HOURS).astype(int)
    merged['is_cod'] = (merged['payment_mode'] == 'COD').astype(int)
    merged['is_weekend'] = (merged['dispatch_time'].dt.dayofweek >= 5).astype(int)
    merged['dispatch_hour'] = merged['dispatch_time'].dt.hour.astype(int)
    merged['log_declared_value'] = np.log(merged['declared_value_inr'])
    return merged
