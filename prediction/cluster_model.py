import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Approximate centroids for Indian states/UTs used if Latitude/Longitude are not present
STATE_COORDINATES = {
    'Andhra Pradesh': (15.9129, 79.7400),
    'Arunachal Pradesh': (28.2180, 94.7278),
    'Assam': (26.2006, 92.9376),
    'Bihar': (25.0961, 85.3131),
    'Chhattisgarh': (21.2787, 81.8661),
    'Goa': (15.2993, 74.1240),
    'Gujarat': (23.0225, 72.5714),
    'Haryana': (29.0588, 76.0856),
    'Himachal Pradesh': (31.1048, 77.1734),
    'Jharkhand': (23.6102, 85.2799),
    'Karnataka': (15.3173, 75.7139),
    'Kerala': (10.8505, 76.2711),
    'Madhya Pradesh': (22.9734, 78.6569),
    'Maharashtra': (19.7515, 75.7139),
    'Manipur': (24.6637, 93.9063),
    'Meghalaya': (25.4670, 91.3662),
    'Mizoram': (23.1645, 92.9376),
    'Nagaland': (26.1584, 94.5624),
    'Odisha': (20.9517, 85.0985),
    'Punjab': (30.9090, 75.8544),
    'Rajasthan': (27.0238, 74.2179),
    'Sikkim': (27.5330, 88.5122),
    'Tamil Nadu': (11.1271, 78.6569),
    'Telangana': (18.1124, 79.0193),
    'Tripura': (23.9408, 91.9882),
    'Uttar Pradesh': (26.8467, 80.9462),
    'Uttarakhand': (30.0668, 79.0193),
    'West Bengal': (22.9868, 87.8550),
    'Andaman and Nicobar Islands': (11.7401, 92.6586),
    'Chandigarh': (30.7333, 76.7794),
    'Dadra and Nagar Haveli and Daman and Diu': (20.1809, 73.0169),
    'Delhi': (28.7041, 77.1025),
    'Jammu and Kashmir': (34.0837, 74.7973),
    'Ladakh': (34.2268, 77.4165),
    'Lakshadweep': (10.5667, 72.6417),
    'Puducherry': (11.9416, 79.8083),
}

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'ml' / 'accidents.csv'
MODEL_PATH = BASE_DIR / 'ml' / 'risk_zone_kmeans.pkl'
SCALER_PATH = BASE_DIR / 'ml' / 'risk_zone_scaler.pkl'
CLUSTERED_CSV_PATH = BASE_DIR / 'ml' / 'risk_zones_clustered.csv'
INFO_PATH = BASE_DIR / 'ml' / 'risk_zone_info.json'


def _extract_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with columns: State, Latitude, Longitude, Accidents."""
    has_lat = 'Latitude' in df.columns
    has_lon = 'Longitude' in df.columns

    if has_lat and has_lon:
        out = df[['Latitude', 'Longitude']].copy()
        out['State'] = df.get('State/UT/City', df.get('State', 'Unknown'))
        # If accident totals column exists, use it; else set 0
        acc_col = 'Grand Total - Total' if 'Grand Total - Total' in df.columns else None
        out['Accidents'] = df[acc_col] if acc_col else 0
        return out[['State', 'Latitude', 'Longitude', 'Accidents']]

    # Fallback: use state/UT centroids + totals
    if 'State/UT/City' not in df.columns:
        raise ValueError('Cannot infer state names. Expected column State/UT/City.')

    state_rows = df[df.get('Category', '') == 'State'].copy() if 'Category' in df.columns else df.copy()
    states = state_rows['State/UT/City'].astype(str).tolist()
    totals = state_rows['Grand Total - Total'].astype(float).tolist() if 'Grand Total - Total' in state_rows.columns else [0] * len(states)

    rows = []
    for s, t in zip(states, totals):
        lat, lon = STATE_COORDINATES.get(s, (20.5937, 78.9629))
        rows.append({'State': s, 'Latitude': lat, 'Longitude': lon, 'Accidents': int(t)})
    return pd.DataFrame(rows)


def train_kmeans(n_clusters: int = 3) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    locs = _extract_locations(df)

    # Feature matrix: lat, lon, log1p(accidents)
    X = locs[['Latitude', 'Longitude']].to_numpy(dtype=float)
    weights = np.log1p(locs['Accidents'].to_numpy(dtype=float)).reshape(-1, 1)
    features = np.hstack([X, weights])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    # Rank clusters by average accidents and map to risk levels
    cluster_avgs = {i: float(locs['Accidents'][clusters == i].mean()) for i in range(n_clusters)}
    ranked = sorted(cluster_avgs.items(), key=lambda kv: kv[1], reverse=True)
    risk_label = {}
    for rank, (cid, _) in enumerate(ranked):
        risk_label[cid] = 'High Risk' if rank == 0 else ('Low Risk' if rank == n_clusters - 1 else 'Medium Risk')

    out = locs.copy()
    out['Cluster'] = clusters
    out['Risk_Level'] = [risk_label[c] for c in clusters]

    # Persist artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    out.to_csv(CLUSTERED_CSV_PATH, index=False)
    with open(INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump({'cluster_avgs': cluster_avgs, 'risk_label': risk_label}, f, indent=2)

    return out


if __name__ == '__main__':
    df_out = train_kmeans(3)
    print('Trained K-Means. Saved to:', CLUSTERED_CSV_PATH)
    print(df_out.head())
