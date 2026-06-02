import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def build_road_graph(csv_path, num_nodes=200, k=5):
    df = pd.read_csv(csv_path)

    # Keep only required columns safely
    required_cols = [
        'Start_Lat', 'Start_Lng',
        'Temperature(F)', 'Humidity(%)',
        'Visibility(mi)', 'Severity'
    ]
    df = df[required_cols].dropna()

    # --- Spatial clustering ---
    coords = df[['Start_Lat', 'Start_Lng']].values
    kmeans = KMeans(n_clusters=num_nodes, random_state=42)
    df['node_id'] = kmeans.fit_predict(coords)

    # --- Aggregate node features ---
    node_features = df.groupby('node_id').mean()

    # --- Build edges (k-NN) ---
    centers = kmeans.cluster_centers_
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(centers)
    _, indices = nbrs.kneighbors(centers)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            edge_index.append([i, j])
            edge_index.append([j, i])

    edge_index = np.array(edge_index).T
    return node_features.values, edge_index
