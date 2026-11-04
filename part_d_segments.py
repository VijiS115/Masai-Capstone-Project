import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_part_d():
    hub_profile = pd.read_csv('outputs/hub_profile.csv')
    features = ['mean_delivery_hours', 'delay_rate', 'mean_distance_km', 'mean_n_stops']
    X_hubs = hub_profile[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hubs)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile['cluster'] = kmeans.fit_predict(X_scaled)
    print("Part D Segmentation ready.")

if __name__ == '__main__':
    run_part_d()
