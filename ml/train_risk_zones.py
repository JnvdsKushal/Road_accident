import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Point
import os

# --- STEP 1: Load dataset ---
data = pd.read_csv('ml/accidents.csv')

# --- STEP 2: Basic cleaning ---
if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
    raise ValueError("Dataset must have 'Latitude' and 'Longitude' columns!")

data = data.dropna(subset=['Latitude', 'Longitude'])

# --- STEP 3: KMeans Clustering ---
X = data[['Latitude', 'Longitude']]
n_clusters = 5  # more clusters = more detailed zones
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# --- STEP 4: Assign risk levels dynamically based on cluster density ---
cluster_counts = data['Cluster'].value_counts()
max_count = cluster_counts.max()
risk_scores = {cluster: (count / max_count) for cluster, count in cluster_counts.items()}
data['Risk_Score'] = data['Cluster'].map(risk_scores)

# --- STEP 5: Convert to GeoDataFrame ---
geometry = [Point(xy) for xy in zip(data['Longitude'], data['Latitude'])]
geo_df = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

# --- STEP 6: Load India base map ---
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
india = world[world.name == 'India']

# --- STEP 7: Plot ---
plt.figure(figsize=(10, 10))
ax = india.plot(color='lightgrey', edgecolor='black')

# color gradient (green → yellow → red)
plt.scatter(
    geo_df['Longitude'],
    geo_df['Latitude'],
    c=geo_df['Risk_Score'],
    cmap='RdYlGn_r',  # reverse so high risk = red
    s=15,
    alpha=0.7
)

plt.title('Dynamic Accident Risk Zones in India', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.colorbar(label='Relative Risk Level')

# --- STEP 8: Save the map for frontend ---
os.makedirs('static/images', exist_ok=True)
output_path = 'static/images/india_dynamic_risk_map.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"✅ Dynamic risk map created: {output_path}")
