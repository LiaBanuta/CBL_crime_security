# -*- coding: utf-8 -*-
"""
Hierarchical Burglary Forecasting (Updated Script with Contiguous Clusters of 13 & 4 LSOAs,
dropping the single missing LSOA code E01033585 to ensure a perfect shapefile match,
with medium≈401 nodes and coarse≈100 nodes, plus a COVID flag as an extra temporal feature,
and CPU‐only execution with normalized inputs and a smaller learning rate).

Levels:
  1. LSOA‐level (each individual LSOA; kNN graph on LSOA centroids)
  2. Level‐2: contiguous clusters of 13 LSOAs (each node ≈13‐LSOA polygon → ~401 nodes)
  3. Level‐3: contiguous clusters of 4 “medium” nodes (each node ≈4 of those ~401 → ~100 nodes)

We perform PCA on the raw LSOA static features to obtain 10 principal components per LSOA,
then Z‐score those components. We also log+Z‐score the lagged burglary counts so that all node
features (lag, 10 PCs, COVID flag) are roughly zero‐mean/unit‐variance. We train three parallel
GCN+LSTM pipelines (one per level) with a sum‐of‐MSE loss. We monitor both training and validation
MSE, and after full training we compute additional evaluation metrics (MAE, RMSE, R², Accuracy,
Precision, Recall) on the held‐out last 12 months. Finally, we plot next‐month predictions by LSOA centroid.

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score
)

import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

# Attempt to import PyTorch Geometric
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import add_self_loops, remove_self_loops
except ImportError:
    raise ImportError(
        "Please install torch-geometric and its dependencies:\n"
        "  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric"
    )

import collections

# Load burglary data
crime_csv_path = r"C:\Users\racsk\CBL_crime_security\CBL 2\final_data.csv"
print("Loading crime data from:", crime_csv_path)
crimes = pd.read_csv(crime_csv_path)
crimes.rename(columns={'LSOA code': 'lsoa_code'}, inplace=True)
crimes = crimes[crimes['Crime type'] == 'Burglary'].copy()
crimes['Month_dt']  = pd.to_datetime(crimes['Month'], format='%Y-%m')
crimes['Year']      = crimes['Month_dt'].dt.year
crimes['Month_num'] = crimes['Month_dt'].dt.month
crimes['lsoa_code'] = crimes['lsoa_code'].astype(str)
crime_lsoa_codes    = set(crimes['lsoa_code'].unique())
print(f"Number of unique LSOA codes in crime CSV: {len(crime_lsoa_codes)}")
shapefile_path = r"C:\Users\racsk\Downloads\Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-5236167991066794441\LSOA_2021_EW_BSC_V4.shp"
print("\nLoading LSOA shapefile from:", shapefile_path)
shp = gpd.read_file(shapefile_path)
if 'LSOA21CD' in shp.columns:
    shp.rename(columns={'LSOA21CD': 'lsoa_code'}, inplace=True)
elif 'LSOA11CD' in shp.columns:
    shp.rename(columns={'LSOA11CD': 'lsoa_code'}, inplace=True)
shp['lsoa_code'] = shp['lsoa_code'].astype(str)
shp_codes       = set(shp['lsoa_code'].unique())
print(f"Number of unique LSOA codes in shapefile:    {len(shp_codes)}")
missing_in_shp   = sorted(crime_lsoa_codes - shp_codes)
missing_in_crime = sorted(shp_codes       - crime_lsoa_codes)

print("\n→ Codes in crime CSV but NOT in shapefile (will be dropped):")
print(missing_in_shp, f"(total missing: {len(missing_in_shp)})")

print("\n→ Codes in shapefile but NOT in crime CSV (extra polygons):")
print(f"(total extra: {len(missing_in_crime)})")
if missing_in_shp:
    print(f"\nFiltering out {len(missing_in_shp)} missing LSOA(s) from crime data: {missing_in_shp}")
    crimes = crimes[~crimes['lsoa_code'].isin(set(missing_in_shp))].copy()
all_lsoa_codes = sorted(crimes['lsoa_code'].unique())
num_lsoas      = len(all_lsoa_codes)
print(f"\nAfter filtering, {num_lsoas} LSOA codes remain (perfectly matched to shapefile).")
agg_lsoa = (
    crimes
      .groupby(['Month_dt', 'lsoa_code'])
      .size()
      .reset_index(name='burglary_count')
      .rename(columns={'Month_dt': 'Month'})
)
all_months = pd.date_range(
    start=crimes['Month_dt'].min(),
    end=crimes['Month_dt'].max(),
    freq='MS'
)
multi_index = pd.MultiIndex.from_product(
    [all_months, all_lsoa_codes],
    names=['Month', 'lsoa_code']
)
df_full_lsoa = pd.DataFrame(index=multi_index).reset_index()
df_full_lsoa['Year']      = df_full_lsoa['Month'].dt.year
df_full_lsoa['Month_num'] = df_full_lsoa['Month'].dt.month

crime_lsoa_agg = pd.merge(
    df_full_lsoa,
    agg_lsoa[['Month', 'lsoa_code', 'burglary_count']],
    on=['Month', 'lsoa_code'],
    how='left'
)
crime_lsoa_agg['burglary_count'] = crime_lsoa_agg['burglary_count'].fillna(0).astype(int)

min_lon, max_lon = crimes['Longitude'].min(), crimes['Longitude'].max()
min_lat, max_lat = crimes['Latitude'].min(),  crimes['Latitude'].max()
covid_start = pd.Timestamp("2020-03-01")
lsoa_months = sorted(crime_lsoa_agg['Month'].unique())
covid_flags = np.array([1 if month >= covid_start else 0 for month in lsoa_months])
print(f"\nBuilt COVID flags for {len(lsoa_months)} months: first 12 flags = {covid_flags[:12]}")


#LSOA_data loading and cleaning
lsoa_csv_path = r"C:\Users\racsk\CBL_crime_security\CBL 2\lsoa_data.csv"
print("\nLoading LSOA static‐features data from:", lsoa_csv_path)
lsoa_full = pd.read_csv(
    lsoa_csv_path,
    sep=';',
    skiprows=1,
    low_memory=False
)
print(f"Successfully loaded LSOA dataframe.  Shape = {lsoa_full.shape}")
print("First 5 rows (before renaming):")
print(lsoa_full.head(5))
lsoa_full.rename(columns={'Unnamed: 0': 'lsoa_code',
                          'Unnamed: 1': 'lsoa_name'},
                 inplace=True)
lsoa_full['lsoa_code'] = lsoa_full['lsoa_code'].astype(str)
all_cols    = list(lsoa_full.columns)
static_cols = [c for c in all_cols if c not in ['lsoa_code', 'lsoa_name']]
static_df = pd.DataFrame({'lsoa_code': all_lsoa_codes})
static_df = pd.merge(
    static_df,
    lsoa_full[['lsoa_code'] + static_cols],
    on='lsoa_code',
    how='left'
)
for col in static_cols:
    static_df[col] = pd.to_numeric(static_df[col], errors='coerce')
    if static_df[col].notna().sum() == 0:
        static_df[col] = 0.0
    else:
        median_val = static_df[col].median()
        static_df[col] = static_df[col].fillna(median_val)

print("\nAfter filling, first 5 rows of static_df:")
print(static_df.head(5))
print("\nStatic columns (first 10):", static_cols[:10])

#PCA usage
n_pca_components = 10
static_matrix = static_df[static_cols].values  

print(f"\nPerforming PCA on static matrix of shape {static_matrix.shape} …")
pca = PCA(n_components=n_pca_components)
principal_components = pca.fit_transform(static_matrix)
print("PCA explained variance ratios (top 5):", pca.explained_variance_ratio_[:5])
pca_means = principal_components.mean(axis=0, keepdims=True)  
pca_stds  = principal_components.std(axis=0, keepdims=True)   
pca_stds[pca_stds == 0] = 1.0
principal_components = (principal_components - pca_means) / pca_stds
df_static_pca = pd.DataFrame(
    data=principal_components,
    index=static_df['lsoa_code'].values,
    columns=[f"PC{i+1}" for i in range(n_pca_components)]
)
df_static_pca.reset_index(inplace=True)
df_static_pca.rename(columns={'index': 'lsoa_code'}, inplace=True)

#Graph building
print("\nFiltering shapefile to only crime LSOAs …")
lsoa_shapes = shp[shp['lsoa_code'].isin(all_lsoa_codes)].copy()
lsoa_shapes = lsoa_shapes.reset_index(drop=True)
print(f"Filtered shapefile: {lsoa_shapes.shape[0]} polygons (should be ~{num_lsoas}).")
print("\nBuilding contiguity graph (Queen‐type) of LSOAs…")
G = nx.Graph()
for idx, row in lsoa_shapes.iterrows():
    G.add_node(row['lsoa_code'], geometry=row['geometry'])
sindex = lsoa_shapes.sindex

for idx, row in lsoa_shapes.iterrows():
    this_code = row['lsoa_code']
    geom = row['geometry']
    possible_matches_index = list(sindex.intersection(geom.bounds))
    for j in possible_matches_index:
        if j == idx:
            continue
        other_code = lsoa_shapes.at[j, 'lsoa_code']
        other_geom = lsoa_shapes.at[j, 'geometry']
        if geom.touches(other_geom) or geom.intersects(other_geom):
            G.add_edge(this_code, other_code)

print(f"Contiguity graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

#Setup the hierarchy
def build_contiguous_clusters(G_input, cluster_size=3):
    """
    Partition G_input’s nodes into contiguous clusters of exactly `cluster_size`, using BFS region‐growing.
    Returns a dict: { node_id -> cluster_id }.
    If a BFS from a seed cannot find `cluster_size` unassigned neighbors, it labels that seed as a “small cluster.”
    Any leftover nodes fewer than `cluster_size` at the end also become “small clusters.”
    """
    unassigned = set(G_input.nodes())
    cluster_assignment = dict()
    cluster_idx = 0

    while len(unassigned) >= cluster_size:
        seed = next(iter(unassigned))
        cluster_idx += 1
        this_cid = f"cluster_{cluster_size}_{cluster_idx:04d}"

        queue = collections.deque([seed])
        gathered = set([seed])

        while queue and len(gathered) < cluster_size:
            curr = queue.popleft()
            for nbr in G_input.neighbors(curr):
                if nbr in unassigned and nbr not in gathered:
                    gathered.add(nbr)
                    queue.append(nbr)
                    if len(gathered) == cluster_size:
                        break

        if len(gathered) == cluster_size:
            for node in gathered:
                cluster_assignment[node] = this_cid
                unassigned.remove(node)
        else:
            cluster_assignment[seed] = f"{this_cid}_small"
            unassigned.remove(seed)
            cluster_idx -= 1  

    for leftover in list(unassigned):
        cluster_idx += 1
        cid = f"cluster_{cluster_size}_{cluster_idx:04d}_small"
        cluster_assignment[leftover] = cid
        unassigned.remove(leftover)

    return cluster_assignment

print("\nBuilding contiguous clusters of size 13 (Medium level) …")
c13_map = build_contiguous_clusters(G, cluster_size=13)
print("Number of LSOAs assigned to clusters‐of‐13:", len(c13_map))
cluster13_to_lsos = collections.defaultdict(list)
for lsoa, c13 in c13_map.items():
    cluster13_to_lsos[c13].append(lsoa)

G_med = nx.Graph()
for c13_id, members in cluster13_to_lsos.items():
    G_med.add_node(c13_id)

for c13_id, members in cluster13_to_lsos.items():
    for lsoa in members:
        for nbr in G.neighbors(lsoa):
            nbr_c13 = c13_map.get(nbr)
            if nbr_c13 is not None and nbr_c13 != c13_id:
                G_med.add_edge(c13_id, nbr_c13)

num_c13 = len(cluster13_to_lsos)
print(f"Medium‐level contiguity graph has {num_c13} nodes and {G_med.number_of_edges()} edges.")

print("\nBuilding contiguous clusters of size 4 (Coarse level) …")
c4_map = build_contiguous_clusters(G_med, cluster_size=4)
print("Number of medium‐nodes assigned to clusters‐of‐4:", len(c4_map))
df_cluster13 = (
    pd.DataFrame.from_dict(c13_map, orient='index', columns=['cluster13_id'])
      .reset_index()
      .rename(columns={'index': 'lsoa_code'})
)
df_cluster4 = (
    pd.DataFrame.from_dict(c4_map, orient='index', columns=['cluster4_id'])
      .reset_index()
      .rename(columns={'index': 'cluster13_id'})
)
all_c13_ids = sorted(df_cluster13['cluster13_id'].unique().tolist())
all_c4_ids  = sorted(df_cluster4['cluster4_id'].unique().tolist())
num_c13     = len(all_c13_ids)  # should be ~401
num_c4      = len(all_c4_ids)   # should be ~100
print(f"Total clusters-of-13 (medium): {num_c13}, clusters-of-4 (coarse): {num_c4}")

crime_lsoa_agg = (
    crime_lsoa_agg
      .merge(df_cluster13, on='lsoa_code', how='left')
      .merge(df_cluster4, on='cluster13_id', how='left')
)
df_static_pca = df_static_pca.merge(df_cluster13, on='lsoa_code', how='left') \
                             .merge(df_cluster4, on='cluster13_id', how='left')
#time series data implementation
pivot_lsoa = crime_lsoa_agg.pivot(
    index='Month',
    columns='lsoa_code',
    values='burglary_count'
).fillna(0).sort_index()
counts_lsoa = pivot_lsoa[all_lsoa_codes].values  
records_c13 = []
for month in lsoa_months:
    sub = crime_lsoa_agg[crime_lsoa_agg['Month'] == month][['cluster13_id', 'burglary_count']]
    grouped = sub.groupby('cluster13_id')['burglary_count'].sum().reset_index()
    grouped['Month'] = month
    records_c13.append(grouped)

df_c13_agg = pd.concat(records_c13, axis=0)
multi_c13 = pd.MultiIndex.from_product([lsoa_months, all_c13_ids], names=['Month','cluster13_id'])
df_full_c13 = pd.DataFrame(index=multi_c13).reset_index().merge(
    df_c13_agg, on=['Month','cluster13_id'], how='left'
)
df_full_c13['burglary_count'] = df_full_c13['burglary_count'].fillna(0).astype(int)

pivot_c13 = df_full_c13.pivot(
    index='Month',
    columns='cluster13_id',
    values='burglary_count'
).fillna(0).sort_index()
counts_c13 = pivot_c13[all_c13_ids].values  
records_c4 = []
for month in lsoa_months:
    sub = crime_lsoa_agg[crime_lsoa_agg['Month'] == month][['cluster4_id', 'burglary_count']]
    grouped = sub.groupby('cluster4_id')['burglary_count'].sum().reset_index()
    grouped['Month'] = month
    records_c4.append(grouped)

df_c4_agg = pd.concat(records_c4, axis=0)
multi_c4 = pd.MultiIndex.from_product([lsoa_months, all_c4_ids], names=['Month','cluster4_id'])
df_full_c4 = pd.DataFrame(index=multi_c4).reset_index().merge(
    df_c4_agg, on=['Month','cluster4_id'], how='left'
)
df_full_c4['burglary_count'] = df_full_c4['burglary_count'].fillna(0).astype(int)

pivot_c4 = df_full_c4.pivot(
    index='Month',
    columns='cluster4_id',
    values='burglary_count'
).fillna(0).sort_index()
counts_c4 = pivot_c4[all_c4_ids].values  # shape = (#months, #clusters_of_4)



all_raw_lags   = counts_lsoa.flatten()  
all_log_lags   = np.log1p(all_raw_lags) 
global_lag_mean = all_log_lags.mean()   
global_lag_std  = all_log_lags.std()    
if global_lag_std == 0:
    global_lag_std = 1.0


#PCA usage on clusters
df_pca = df_static_pca[['lsoa_code'] + [f"PC{i+1}" for i in range(n_pca_components)]].copy()
df_pca_c13 = df_pca.merge(df_cluster13, on='lsoa_code', how='left')
static_c13_df = (
    df_pca_c13
      .groupby('cluster13_id')
      .agg({f"PC{i+1}": "mean" for i in range(n_pca_components)})
      .reset_index()
)
static_c13_df = static_c13_df.set_index('cluster13_id').loc[all_c13_ids]
static_c13_tensor = torch.tensor(static_c13_df.values, dtype=torch.float)
df_temp = df_static_pca[['lsoa_code'] + [f"PC{i+1}" for i in range(n_pca_components)] + ['cluster13_id']].copy()
df_temp = df_temp.merge(df_cluster4, on='cluster13_id', how='left')  # adds cluster4_id

static_c4_df = (
    df_temp
      .groupby('cluster4_id')
      .agg({f"PC{i+1}": "mean" for i in range(n_pca_components)})
      .reset_index()
)
static_c4_df = static_c4_df.set_index('cluster4_id').loc[all_c4_ids]
static_c4_tensor = torch.tensor(static_c4_df.values, dtype=torch.float)

static_lsoa_tensor = torch.tensor(principal_components, dtype=torch.float)


#Adjacency
centroid_geom = lsoa_shapes[['lsoa_code', 'geometry']].copy()
centroid_geom['centroid_x'] = centroid_geom['geometry'].centroid.x
centroid_geom['centroid_y'] = centroid_geom['geometry'].centroid.y

centroid_full = centroid_geom[['lsoa_code', 'centroid_x', 'centroid_y']].copy()
overall_x = centroid_full['centroid_x'].mean()
overall_y = centroid_full['centroid_y'].mean()
centroid_full['centroid_x'] = centroid_full['centroid_x'].fillna(overall_x)
centroid_full['centroid_y'] = centroid_full['centroid_y'].fillna(overall_y)

coords_matrix = centroid_full[['centroid_x', 'centroid_y']].values
knn = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(coords_matrix)
_, neighbors = knn.kneighbors(coords_matrix)

src_lsoa, dst_lsoa = [], []
for i in range(num_lsoas):
    for j in neighbors[i]:
        if j == i:
            continue
        src_lsoa.append(i)
        dst_lsoa.append(j)

edge_index_lsoa = np.array([src_lsoa, dst_lsoa], dtype=np.int64)
edge_index_lsoa, _ = remove_self_loops(torch.tensor(edge_index_lsoa))
edge_index_lsoa, _ = add_self_loops(edge_index_lsoa, num_nodes=num_lsoas)
edge_index_lsoa = edge_index_lsoa.numpy()

graph_lsoa = {
    'num_nodes': num_lsoas,
    'edge_index': torch.tensor(edge_index_lsoa, dtype=torch.long),
    'nodes': centroid_full['lsoa_code'].tolist()
}
print(f"\nLevel 1 (LSOA) graph: {num_lsoas} nodes, {edge_index_lsoa.shape[1]} edges")



src_c13, dst_c13 = [], []
for edge in G_med.edges():
    a, b = edge
    idx_a = all_c13_ids.index(a)
    idx_b = all_c13_ids.index(b)
    src_c13.append(idx_a)
    dst_c13.append(idx_b)

edge_index_c13 = np.array([src_c13 + dst_c13, dst_c13 + src_c13], dtype=np.int64)
edge_index_c13, _ = remove_self_loops(torch.tensor(edge_index_c13))
edge_index_c13, _ = add_self_loops(edge_index_c13, num_nodes=num_c13)
edge_index_c13 = edge_index_c13.numpy()

graph_c13 = {
    'num_nodes': num_c13,
    'edge_index': torch.tensor(edge_index_c13, dtype=torch.long),
    'nodes': all_c13_ids
}
print(f"Level 2 (clusters_of_13) graph: {num_c13} nodes, {edge_index_c13.shape[1]} edges")

cluster4_to_c13 = collections.defaultdict(list)
for c13, c4 in c4_map.items():
    cluster4_to_c13[c4].append(c13)

G_coarse = nx.Graph()
for c4_id in all_c4_ids:
    G_coarse.add_node(c4_id)

for c4_id, c13_list in cluster4_to_c13.items():
    for c13 in c13_list:
        for nbr_med in G_med.neighbors(c13):
            nbr_c4 = c4_map.get(nbr_med)
            if nbr_c4 is not None and nbr_c4 != c4_id:
                G_coarse.add_edge(c4_id, nbr_c4)

src_c4, dst_c4 = [], []
for edge in G_coarse.edges():
    a, b = edge
    idx_a = all_c4_ids.index(a)
    idx_b = all_c4_ids.index(b)
    src_c4.append(idx_a)
    dst_c4.append(idx_b)

edge_index_c4 = np.array([src_c4 + dst_c4, dst_c4 + src_c4], dtype=np.int64)
edge_index_c4, _ = remove_self_loops(torch.tensor(edge_index_c4))
edge_index_c4, _ = add_self_loops(edge_index_c4, num_nodes=num_c4)
edge_index_c4 = edge_index_c4.numpy()

graph_c4 = {
    'num_nodes': num_c4,
    'edge_index': torch.tensor(edge_index_c4, dtype=torch.long),
    'nodes': all_c4_ids
}
print(f"Level 3 (clusters_of_4) graph: {num_c4} nodes, {edge_index_c4.shape[1]} edges")



#Apply hierarchy
class HierarchicalBurglaryDataset(Dataset):
    """
    For each time index t in [T .. num_months-1], returns:
      data = {
        'lsoa':   { 'x': Tensor[num_lsoas, T, in_ch], 'edge_index': Tensor[2, E_lsoa] },
        'medium': { 'x': Tensor[num_c13,   T, in_ch], 'edge_index': Tensor[2, E_c13]   },
        'coarse': { 'x': Tensor[num_c4,    T, in_ch], 'edge_index': Tensor[2, E_c4]   }
      }
      target = {
        'lsoa':   Tensor[num_lsoas],
        'medium': Tensor[num_c13],
        'coarse': Tensor[num_c4]
      }
    Node feature dimension in_ch = 1 (normalized lag) + n_pca_components (static‐PCA = 10) + 1 (covid_flag) = 12.
    """
    def __init__(self,
                 counts_lsoa, counts_c13, counts_c4,
                 static_lsoa, static_c13, static_c4,
                 covid_flags,
                 edge_lsoa, edge_c13, edge_c4,
                 T=12):
        """
        counts_lsoa:   np.array [num_months, num_lsoas]
        counts_c13:    np.array [num_months, num_c13]
        counts_c4:     np.array [num_months, num_c4]
        static_lsoa:   torch.Tensor [num_lsoas,    num_static]  (10 dims, Z‐scored PCA)
        static_c13:    torch.Tensor [num_c13,     num_static]  (10 dims)
        static_c4:     torch.Tensor [num_c4,      num_static]  (10 dims)
        covid_flags:   np.array [num_months] of 0/1
        edge_lsoa:     torch.Tensor [2, E_lsoa]
        edge_c13:      torch.Tensor [2, E_c13]
        edge_c4:       torch.Tensor [2, E_c4]
        T: number of historical months to use
        """
        self.T = T
        self.edge_lsoa   = edge_lsoa
        self.edge_med    = edge_c13
        self.edge_coarse = edge_c4

        self.counts_lsoa   = counts_lsoa
        self.counts_med    = counts_c13
        self.counts_coarse = counts_c4

        self.static_lsoa   = static_lsoa
        self.static_med    = static_c13
        self.static_coarse = static_c4

        self.covid_flags = covid_flags

        self.num_lsoa   = counts_lsoa.shape[1]
        self.num_med    = counts_c13.shape[1]
        self.num_coarse = counts_c4.shape[1]
        self.num_static = static_lsoa.shape[1]  

        X_lsoa_seq   = []
        X_med_seq    = []
        X_coarse_seq = []
        Y_lsoa       = []
        Y_med        = []
        Y_coarse     = []

        num_months = counts_lsoa.shape[0]
        for t in range(T, num_months):
            covid_window = covid_flags[t-T:t]  

            hist_lsoa   = counts_lsoa[t-T:t, :]      
            target_lsoa = counts_lsoa[t, :]          
            seq_lsoa    = []
            for i_node in range(self.num_lsoa):
                raw_lag   = hist_lsoa[:, i_node]              
                log_lag   = np.log1p(raw_lag)                 
                norm_lag  = (log_lag - global_lag_mean) / global_lag_std
                lag_series = norm_lag.reshape(T, 1)           

                static_vec  = static_lsoa[i_node].unsqueeze(0).repeat(T, 1)  
                covid_vec   = covid_window.reshape(T, 1)                     
                node_seq    = np.concatenate([lag_series,
                                              static_vec.numpy(),
                                              covid_vec], axis=1)            
                seq_lsoa.append(node_seq)
            X_lsoa_seq.append(np.stack(seq_lsoa, axis=0))  
            Y_lsoa.append(target_lsoa)

            hist_med    = counts_c13[t-T:t, :]      
            target_med  = counts_c13[t, :]          
            seq_med     = []
            for i_node in range(self.num_med):
                raw_med    = hist_med[:, i_node]                   
                log_med    = np.log1p(raw_med)                     
                norm_med   = (log_med - global_lag_mean) / global_lag_std
                lag_series = norm_med.reshape(T, 1)                

                static_vec = static_c13[i_node].unsqueeze(0).repeat(T, 1)  
                covid_vec  = covid_window.reshape(T, 1)                     
                node_seq   = np.concatenate([lag_series,
                                              static_vec.numpy(),
                                              covid_vec], axis=1)            
                seq_med.append(node_seq)
            X_med_seq.append(np.stack(seq_med, axis=0)) 
            Y_med.append(target_med)

            hist_coarse   = counts_c4[t-T:t, :] 
            target_coarse = counts_c4[t, :]     
            seq_coarse    = []
            for i_node in range(self.num_coarse):
                raw_coarse  = hist_coarse[:, i_node]               
                log_coarse  = np.log1p(raw_coarse)                 
                norm_coarse = (log_coarse - global_lag_mean) / global_lag_std
                lag_series  = norm_coarse.reshape(T, 1)            

                static_vec  = static_c4[i_node].unsqueeze(0).repeat(T, 1)  
                covid_vec   = covid_window.reshape(T, 1)                     
                node_seq    = np.concatenate([lag_series,
                                              static_vec.numpy(),
                                              covid_vec], axis=1)            
                seq_coarse.append(node_seq)
            X_coarse_seq.append(np.stack(seq_coarse, axis=0))  
            Y_coarse.append(target_coarse)

        self.X_lsoa   = torch.tensor(np.stack(X_lsoa_seq, axis=0), dtype=torch.float)

        self.Y_lsoa   = torch.tensor(np.stack(Y_lsoa, axis=0), dtype=torch.float)


        self.X_med    = torch.tensor(np.stack(X_med_seq, axis=0), dtype=torch.float)

        self.Y_med    = torch.tensor(np.stack(Y_med, axis=0), dtype=torch.float)


        self.X_coarse = torch.tensor(np.stack(X_coarse_seq, axis=0), dtype=torch.float)
         
        self.Y_coarse = torch.tensor(np.stack(Y_coarse, axis=0), dtype=torch.float)


        self.num_samples = self.X_lsoa.shape[0]
        self.in_channels = 1 + self.num_static + 1  # = 12

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = {
            'lsoa':   {
                'x': self.X_lsoa[idx],          # [num_lsoas, T, 12]
                'edge_index': self.edge_lsoa    # [2, E_lsoa]
            },
            'medium': {
                'x': self.X_med[idx],           # [num_c13, T, 12]
                'edge_index': self.edge_med     # [2, E_c13]
            },
            'coarse': {
                'x': self.X_coarse[idx],        # [num_c4, T, 12]
                'edge_index': self.edge_coarse  # [2, E_c4]
            }
        }
        target = {
            'lsoa':   self.Y_lsoa[idx],        # [num_lsoas]
            'medium': self.Y_med[idx],         # [num_c13]
            'coarse': self.Y_coarse[idx]       # [num_c4]
        }
        return data, target



dataset = HierarchicalBurglaryDataset(
    counts_lsoa   = counts_lsoa,
    counts_c13    = counts_c13,
    counts_c4     = counts_c4,
    static_lsoa   = static_lsoa_tensor,
    static_c13    = static_c13_tensor,
    static_c4     = static_c4_tensor,
    covid_flags   = covid_flags,
    edge_lsoa     = graph_lsoa['edge_index'],
    edge_c13      = graph_c13['edge_index'],
    edge_c4       = graph_c4['edge_index'],
    T=12
)
print(f"\nPhase 11 complete: Hierarchical dataset with {len(dataset)} samples created.")


#Model HAGEN
class HierarchicalHAGEN(nn.Module):
    """
    Three‐level hierarchical GCN+LSTM:
      - Level 1: LSOA‐level GCN + LSTM
      - Level 2: clusters_of_13 GCN + LSTM
      - Level 3: clusters_of_4 GCN + LSTM
    """
    def __init__(self, 
                 num_lsoa, num_c13, num_c4,
                 in_channels, hidden_channels):
        super(HierarchicalHAGEN, self).__init__()
        self.num_lsoa   = num_lsoa
        self.num_med    = num_c13
        self.num_coarse = num_c4
        self.hidden_channels = hidden_channels

        # Level 1 (LSOA)
        self.gcn_lsoa     = GCNConv(in_channels, hidden_channels)
        self.dropout_lsoa = nn.Dropout(p=0.3)
        self.lstm_lsoa    = nn.LSTM(input_size=hidden_channels,
                                    hidden_size=hidden_channels,
                                    num_layers=1,
                                    batch_first=True)
        self.fc_lsoa      = nn.Linear(hidden_channels, 1)

        # Level 2 (clusters_of_13)
        self.gcn_med     = GCNConv(in_channels, hidden_channels)
        self.dropout_med = nn.Dropout(p=0.3)
        self.lstm_med    = nn.LSTM(input_size=hidden_channels,
                                   hidden_size=hidden_channels,
                                   num_layers=1,
                                   batch_first=True)
        self.fc_med      = nn.Linear(hidden_channels, 1)

        # Level 3 (clusters_of_4)
        self.gcn_coarse      = GCNConv(in_channels, hidden_channels)
        self.dropout_coarse  = nn.Dropout(p=0.3)
        self.lstm_coarse     = nn.LSTM(input_size=hidden_channels,
                                       hidden_size=hidden_channels,
                                       num_layers=1,
                                       batch_first=True)
        self.fc_coarse       = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        """
        data: dict with keys 'lsoa','medium','coarse'
          data[level]['x'] is [num_nodes_level, T, in_ch]
          data[level]['edge_index'] is [2, num_edges_level]
        Returns:
          preds: dict with keys 'lsoa','medium','coarse'; each is [num_nodes_level]
        """
        preds = {}

        x_seq     = data['lsoa']['x']               # [num_lsoas, T, in_ch]
        edge_idx  = data['lsoa']['edge_index']      # [2, E_lsoa]
        N, T_seq, in_ch = x_seq.size()

        gcn_out = []
        for t_i in range(T_seq):
            x_t  = x_seq[:, t_i, :]                    # [num_lsoas, in_ch]
            h_t  = self.gcn_lsoa(x_t, edge_idx)         # [num_lsoas, hidden]
            h_t  = F.relu(h_t)
            h_t  = self.dropout_lsoa(h_t)
            gcn_out.append(h_t.unsqueeze(1))            # [num_lsoas, 1, hidden]
        h_lsoa_seq = torch.cat(gcn_out, dim=1)          # [num_lsoas, T, hidden]

        lstm_out, (hn, _) = self.lstm_lsoa(h_lsoa_seq)   # hn[0] = [num_lsoas, hidden]
        final_lsoa   = hn[0]
        pred_lsoa    = self.fc_lsoa(final_lsoa).squeeze(-1)  # [num_lsoas]
        preds['lsoa'] = pred_lsoa

        x_seq     = data['medium']['x']                # [num_c13, T, in_ch]
        edge_idx  = data['medium']['edge_index']       # [2, E_c13]
        M, T_seq, in_ch = x_seq.size()

        gcn_out = []
        for t_i in range(T_seq):
            x_t = x_seq[:, t_i, :].float()            # [num_c13, in_ch]
            h_t = self.gcn_med(x_t, edge_idx)         # [num_c13, hidden]
            h_t = F.relu(h_t)
            h_t = self.dropout_med(h_t)
            gcn_out.append(h_t.unsqueeze(1))
        h_med_seq = torch.cat(gcn_out, dim=1)          # [num_c13, T, hidden]

        lstm_out, (hn, _) = self.lstm_med(h_med_seq)
        final_med   = hn[0]                            # [num_c13, hidden]
        pred_med    = self.fc_med(final_med).squeeze(-1)  # [num_c13]
        preds['medium'] = pred_med

        x_seq     = data['coarse']['x']                # [num_c4, T, in_ch]
        edge_idx  = data['coarse']['edge_index']       # [2, E_c4]
        C, T_seq, in_ch = x_seq.size()

        gcn_out = []
        for t_i in range(T_seq):
            x_t = x_seq[:, t_i, :].float()             # [num_c4, in_ch]
            h_t = self.gcn_coarse(x_t, edge_idx)       # [num_c4, hidden]
            h_t = F.relu(h_t)
            h_t = self.dropout_coarse(h_t)
            gcn_out.append(h_t.unsqueeze(1))
        h_coarse_seq = torch.cat(gcn_out, dim=1)       # [num_c4, T, hidden]

        lstm_out, (hn, _) = self.lstm_coarse(h_coarse_seq)
        final_coarse = hn[0]                           # [num_c4, hidden]
        pred_coarse  = self.fc_coarse(final_coarse).squeeze(-1)  # [num_c4]
        preds['coarse'] = pred_coarse

        return preds

print("\nPhase 13 complete: Defined HierarchicalHAGEN model.")


#training and evaluation
def train_hierarchical(model, dataset, num_epochs=30, lr=0.0005, batch_size=1):
    """
    Train the hierarchical model on the first (num_samples - test_size) samples,
    while monitoring validation MSE on the last test_size samples every 5 epochs.
    test_size=12 (last 12 months).
    Returns trained model.
    """
    total_samples = len(dataset)
    test_size     =12
    train_size    = total_samples - test_size

    train_indices = list(range(train_size))
    test_indices  = list(range(train_size, total_samples))

    train_subset  = torch.utils.data.Subset(dataset, train_indices)
    test_subset   = torch.utils.data.Subset(dataset, test_indices)

    train_loader  = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_subset, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss  = nn.MSELoss()

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        for data, target in train_loader:
            data_lsoa_x   = data['lsoa']['x'].squeeze(0)
            edge_lsoa     = data['lsoa']['edge_index'].squeeze(0)
            y_lsoa        = target['lsoa'].squeeze(0)

            data_med_x    = data['medium']['x'].squeeze(0)
            edge_med      = data['medium']['edge_index'].squeeze(0)
            y_med         = target['medium'].squeeze(0)

            data_coarse_x = data['coarse']['x'].squeeze(0)
            edge_coarse   = data['coarse']['edge_index'].squeeze(0)
            y_coarse      = target['coarse'].squeeze(0)

            optimizer.zero_grad()
            preds = model({
                'lsoa':   {'x': data_lsoa_x,   'edge_index': edge_lsoa},
                'medium': {'x': data_med_x,    'edge_index': edge_med},
                'coarse': {'x': data_coarse_x, 'edge_index': edge_coarse}
            })
            loss = (
                mse_loss(preds['lsoa'],   y_lsoa) +
                mse_loss(preds['medium'], y_med) +
                mse_loss(preds['coarse'], y_coarse)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} — Avg Train MSE = {avg_train_loss:.4f}", end='')

        if epoch % 5 == 0 or epoch == num_epochs:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for data, target in test_loader:
                    data_lsoa_x   = data['lsoa']['x'].squeeze(0)
                    edge_lsoa     = data['lsoa']['edge_index'].squeeze(0)
                    y_lsoa        = target['lsoa'].squeeze(0)

                    data_med_x    = data['medium']['x'].squeeze(0)
                    edge_med      = data['medium']['edge_index'].squeeze(0)
                    y_med         = target['medium'].squeeze(0)

                    data_coarse_x = data['coarse']['x'].squeeze(0)
                    edge_coarse   = data['coarse']['edge_index'].squeeze(0)
                    y_coarse      = target['coarse'].squeeze(0)

                    preds = model({
                        'lsoa':   {'x': data_lsoa_x,   'edge_index': edge_lsoa},
                        'medium': {'x': data_med_x,    'edge_index': edge_med},
                        'coarse': {'x': data_coarse_x, 'edge_index': edge_coarse}
                    })
                    val_loss = (
                        mse_loss(preds['lsoa'],   y_lsoa).item() +
                        mse_loss(preds['medium'], y_med).item() +
                        mse_loss(preds['coarse'], y_coarse).item()
                    )
                    val_losses.append(val_loss)

            avg_val_loss = np.mean(val_losses)
            print(f"  ∥  Val MSE = {avg_val_loss:.4f}")
        else:
            print()

    return model

def evaluate_hierarchical(model, test_subset):
    """
    After training, evaluate on test_subset; return MAE, RMSE, R2, Accuracy, Precision, Recall at LSOA level.
    """
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    model.eval()

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data_lsoa_x = data['lsoa']['x'].squeeze(0)
            edge_lsoa   = data['lsoa']['edge_index'].squeeze(0)
            y_lsoa      = target['lsoa'].squeeze(0)

            preds = model({
                'lsoa':   {'x': data_lsoa_x,   'edge_index': edge_lsoa},
                'medium': {'x': data['medium']['x'].squeeze(0),
                           'edge_index': data['medium']['edge_index'].squeeze(0)},
                'coarse': {'x': data['coarse']['x'].squeeze(0),
                           'edge_index': data['coarse']['edge_index'].squeeze(0)}
            })
            y_pred_lsoa = preds['lsoa']

            all_preds.append(y_pred_lsoa.numpy())
            all_targets.append(y_lsoa.numpy())

    all_preds   = np.stack(all_preds, axis=0)     
    all_targets = np.stack(all_targets, axis=0)   

    preds_flat   = all_preds.flatten()
    targets_flat = all_targets.flatten()

    mae      = mean_absolute_error(targets_flat, preds_flat)
    rmse     = np.sqrt(mean_squared_error(targets_flat, preds_flat))
    r2       = r2_score(targets_flat, preds_flat)

    bin_targets = (targets_flat > 0).astype(int)
    bin_preds   = (preds_flat > 0.5).astype(int)  

    accuracy  = accuracy_score(bin_targets, bin_preds)
    precision = precision_score(bin_targets, bin_preds, zero_division=0)
    recall    = recall_score(bin_targets, bin_preds, zero_division=0)

    return mae, rmse, r2, accuracy, precision, recall

#Execution
def save_predictions_to_csv(model,dataset,graph_lsoa,path="lsoa_predictions.csv"):
    data,_=dataset[len(dataset)-1]
    with torch.no_grad():
        out=model({'lsoa':{'x':data['lsoa']['x'],'edge_index':data['lsoa']['edge_index']},
                   'medium':{'x':data['medium']['x'],'edge_index':data['medium']['edge_index']},
                   'coarse':{'x':data['coarse']['x'],'edge_index':data['coarse']['edge_index']}})
    arr=out['lsoa'].cpu().numpy()
    df=pd.DataFrame({'lsoa_code':graph_lsoa['nodes'],'predicted_burglary_count':arr})
    df.to_csv(path,index=False); print(f"Saved predictions to {path}")
if __name__ == "__main__":


    #Hyperparameters
    in_channels   = 1 + n_pca_components + 1  
    hidden_dim    = 64
    num_epochs    = 50
    batch_size    = 1
    learning_rate = 0.0005  # smaller LR due to normalized inputs

    #Instantiate hierarchical model
    model = HierarchicalHAGEN(
        num_lsoa         = num_lsoas,
        num_c13          = num_c13,
        num_c4           = num_c4,
        in_channels      = in_channels,
        hidden_channels  = hidden_dim
    )

    # Train (with validation monitoring)
    print("\nStarting training of hierarchical model…")
    trained_model = train_hierarchical(
        model=model,
        dataset=dataset,
        num_epochs=num_epochs,
        lr=learning_rate,
        batch_size=batch_size
    )

    # Evaluate on test set (last 12 months)
    total_samples = len(dataset)
    test_size     = 12
    train_size    = total_samples - test_size
    test_indices  = list(range(train_size, total_samples))
    test_subset   = torch.utils.data.Subset(dataset, test_indices)

    print("\nEvaluating hierarchical model on test set (last 12 months)…")
    mae, rmse, r2, accuracy, precision, recall = evaluate_hierarchical(
        trained_model,
        test_subset
    )
    print("\n===== Final Evaluation Metrics on Test Set (last 12 months) =====")
    print(f"LSOA‐level MAE:       {mae:.4f}")
    print(f"LSOA‐level RMSE:      {rmse:.4f}")
    print(f"LSOA‐level R2 Score:  {r2:.4f}")
    print(f"LSOA‐level Accuracy:  {accuracy:.4f}")
    print(f"LSOA‐level Precision: {precision:.4f}")
    print(f"LSOA‐level Recall:    {recall:.4f}")


    last_idx = len(dataset) - 1
    data, _ = dataset[last_idx]
    with torch.no_grad():
        data_lsoa_x = data['lsoa']['x']      # [num_lsoas, T, in_ch]
        edge_lsoa   = data['lsoa']['edge_index']
        preds = trained_model({
            'lsoa':   {'x': data_lsoa_x, 'edge_index': edge_lsoa},
            'medium': {'x': data['medium']['x'], 'edge_index': data['medium']['edge_index']},
            'coarse': {'x': data['coarse']['x'], 'edge_index': data['coarse']['edge_index']}
        })
        next_month_preds_lsoa = preds['lsoa'].numpy()  # [num_lsoas]


    window_data, _ = dataset[-1]

    history = window_data['lsoa']['x'].clone()       
    edge_lsoa = window_data['lsoa']['edge_index']    


    last_month    = lsoa_months[-1]
    future_months = [last_month + pd.DateOffset(months=i) for i in range(1, 13)]


    year_preds = []
    for _ in range(12):
        model.eval()
        with torch.no_grad():
            preds = model({
                'lsoa':   {'x': history,   'edge_index': edge_lsoa},
                'medium': window_data['medium'],
                'coarse': window_data['coarse']
            })
        next_pred = preds['lsoa'].numpy()    
        year_preds.append(next_pred)


        history = torch.roll(history, shifts=-1, dims=1)
        history[:, -1, 0] = torch.from_numpy(next_pred)  # inject as new lag


    df_year = pd.DataFrame(
        data = np.stack(year_preds, axis=1),     # shape = (num_lsoas,12)
        index = all_lsoa_codes,
        columns = [m.strftime("%Y-%m") for m in future_months]
    ).T  # now rows=months, cols=lsoas
    df_year.index.name = 'Month'
    df_year.to_csv("nextyearpredictions.csv")
    print("Saved 12-month forecasts to nextyear_predictions.csv")

    # 4) plot total trend
    plt.figure(figsize=(8,4))
    df_year.sum(axis=1).plot(marker='o')
    plt.title("Total Predicted Burglaries: Next 12 Months")
    plt.xlabel("Month")
    plt.ylabel("Sum of LSOA Predictions")
    plt.grid(True)
    plt.show()


    longitudes = centroid_full['centroid_x'].values
    latitudes  = centroid_full['centroid_y'].values

    plt.figure(figsize=(8, 10))
    sc = plt.scatter(
        longitudes,
        latitudes,
        c=next_month_preds_lsoa,
        cmap='viridis',
        s=50,
        edgecolor='k'
    )
    plt.colorbar(sc, label='Predicted Burglary Count (Next Month)')
    plt.title("Next‐Month Burglary Predictions by LSOA Centroid")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()
