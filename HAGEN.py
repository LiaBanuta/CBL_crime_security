import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from shapely.geometry import box
from pathlib import Path
from torch_geometric.nn import GCNConv
from scipy.spatial.distance import cdist
from itertools import product
import copy

data_dir = Path("data")
crime_data = pd.read_csv(data_dir / "final_data.csv")
social_data = pd.read_csv(data_dir / "lsoa_data.csv", delimiter=';', low_memory=False)
lsoa_gdf = gpd.read_file(data_dir / "LSOA_2021_EW_BSC_V4.shp")

crime_data['LSOA code'] = crime_data['LSOA code'].str.strip().str.upper()
social_data.columns = [str(col).strip() for col in social_data.columns]
social_data = social_data.rename(columns={social_data.columns[0]: 'LSOA_code'})
social_data['LSOA_code'] = social_data['LSOA_code'].str.strip().str.upper()

lsoa_gdf = lsoa_gdf[lsoa_gdf['LSOA21CD'].isin(crime_data['LSOA code'].unique())]
lsoa_gdf = lsoa_gdf.to_crs(epsg=27700)

xmin, ymin, xmax, ymax = lsoa_gdf.total_bounds
cell_size = 500
grid_cells = [box(x0, y0, x0 + cell_size, y0 + cell_size)
              for x0 in np.arange(xmin, xmax, cell_size)
              for y0 in np.arange(ymin, ymax, cell_size)]
grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=lsoa_gdf.crs)

grid_with_lsoa = gpd.sjoin(grid, lsoa_gdf[['LSOA21CD', 'geometry']], how='inner', predicate='intersects')
grid_with_lsoa = grid_with_lsoa.rename(columns={'LSOA21CD': 'LSOA_code'})
grid_with_lsoa = grid_with_lsoa.drop(columns=[col for col in grid_with_lsoa.columns if 'index_right' in col])

crime_data['Month'] = pd.to_datetime(crime_data['Month'])
gridded_data = crime_data[crime_data['LSOA code'].isin(grid_with_lsoa['LSOA_code'])]
gridded_data = gridded_data.rename(columns={'LSOA code': 'LSOA_code'})
time_series = gridded_data.groupby(['Month', 'LSOA_code']).size().unstack(fill_value=0).sort_index()

Household_Language_columns = [col for col in social_data.columns if 'Household Language' in col]
house_price_columns = [col for col in social_data.columns if 'House Prices' in col]
Dwelling_type_columns = [col for col in social_data.columns if 'Dwelling type' in col]
Households_type_columns = [col for col in social_data.columns if 'Households' in col]
Population_type_columns = [col for col in social_data.columns if '2011 Census Population' in col]
social_columns =Household_Language_columns + house_price_columns + Dwelling_type_columns + Households_type_columns + Population_type_columns
social_data = social_data[['LSOA_code'] + social_columns]
social_data = social_data.set_index('LSOA_code').apply(pd.to_numeric, errors='coerce').fillna(0)
social_data = social_data.loc[time_series.columns.intersection(social_data.index)]
time_series = time_series[social_data.index]

scaler_crime = StandardScaler()
crime_scaled = scaler_crime.fit_transform(time_series.values)

scaler_social = StandardScaler()
social_scaled = scaler_social.fit_transform(social_data.values)

months = time_series.index.to_series().dt.month
month_sin = np.sin(2 * np.pi * months.values / 12).reshape(-1, 1)
month_cos = np.cos(2 * np.pi * months.values / 12).reshape(-1, 1)
temporal_features = np.hstack([month_sin, month_cos])

def create_dataset(data, social, temporal, look_back=36):
    X, y = [], []
    for i in range(look_back, data.shape[0]):
        for j in range(data.shape[1]):
            X.append(np.hstack([data[i-look_back:i, j].flatten(), social[j], temporal[i]]))
            y.append(data[i, j])
    return np.array(X), np.array(y)


lsoa_gdf = lsoa_gdf.set_index('LSOA21CD').loc[social_data.index]
centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in lsoa_gdf.geometry])
dist_matrix = cdist(centroids, centroids)
threshold = 1000

adjacency = (dist_matrix <= threshold) & (dist_matrix > 0)  # no self loops

row, col = np.where(adjacency)
edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
edge_weight = torch.tensor(dist_matrix[row, col], dtype=torch.float32)

class HAGEN_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers=5):
        super().__init__()
        assert num_layers >= 2, "num_layers should be at least 2"

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.residual = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = self.relu(h)
            h = self.dropout(h)
        h = self.convs[-1](h, edge_index, edge_weight=edge_weight)
        return h + self.residual(x)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()


def plot_predictions_vs_actuals(y_true, y_pred, title="Predictions vs Actuals"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    plt.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f}",
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.grid(True)
    plt.show()
param_grid = {
    'lr': [0.001, 0.0005],
    'hidden_dim': [32, 64],
    'dropout': [0.2, 0.3],
    'weight_decay': [0.0, 1e-4]
}

x = torch.tensor(social_scaled, dtype=torch.float32)

best_r2 = -np.inf
best_params = {}
best_gcn_model = None
best_mlp_model_state = None

print(f"Data shapes: social_scaled={social_scaled.shape}, crime_scaled={crime_scaled.shape}, temporal_features={temporal_features.shape}")

for lr, hidden_dim, dropout, weight_decay in product(*param_grid.values()):
    print(f"Training with params: lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}, weight_decay={weight_decay}")

    gcn_model = HAGEN_GCN(x.shape[1], hidden_dim, x.shape[1], dropout)
    gcn_model.eval()
    with torch.no_grad():
        gcn_features = gcn_model(x, edge_index, edge_weight=edge_weight).numpy()

    X, y = create_dataset(crime_scaled, gcn_features, temporal_features, look_back=12)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = MLP(X.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    patience = 10
    best_val_loss = np.inf
    patience_counter = 0
    best_model_state = None
    best_val_r2_for_params = -np.inf

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        train_preds = model(X_train_t)
        train_loss = criterion(train_preds, y_train_t)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_preds_np = val_preds.detach().cpu().numpy()
            y_val_np = y_val_t.detach().cpu().numpy()
            val_r2 = r2_score(y_val_np, val_preds_np)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if val_r2 > best_val_r2_for_params:
            best_val_r2_for_params = val_r2

    print(f"Params: lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}, weight_decay={weight_decay} -> Best Val R²: {best_val_r2_for_params:.4f}")

    if best_val_r2_for_params > best_r2:
        best_r2 = best_val_r2_for_params
        best_params = {'lr': lr, 'hidden_dim': hidden_dim, 'dropout': dropout, 'weight_decay': weight_decay}
        best_gcn_model = copy.deepcopy(gcn_model)
        best_mlp_model_state = best_model_state

print(f"\nBest params: {best_params} with test R²={best_r2:.4f}")

# --- Final training on full data ---
print("\nTraining final model with best params on full train+val")

gcn_model = best_gcn_model
gcn_model.eval()
with torch.no_grad():
    gcn_features = gcn_model(x, edge_index, edge_weight=edge_weight).numpy()

X_full, y_full = create_dataset(crime_scaled, gcn_features, temporal_features, look_back=12)

final_model = MLP(X_full.shape[1])
final_model.load_state_dict(best_mlp_model_state)
optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
criterion = nn.MSELoss()

X_full_t = torch.tensor(X_full, dtype=torch.float32)
y_full_t = torch.tensor(y_full, dtype=torch.float32)

patience = 10
best_loss = np.inf
patience_counter = 0

for epoch in range(100):
    final_model.train()
    optimizer.zero_grad()
    preds = final_model(X_full_t)
    loss = criterion(preds, y_full_t)
    loss.backward()
    optimizer.step()

    final_model.eval()
    with torch.no_grad():
        preds_np = preds.detach().cpu().numpy()
        y_np = y_full_t.detach().cpu().numpy()
        train_mae = mean_absolute_error(y_np, preds_np)
        train_r2 = r2_score(y_np, preds_np)

    print(f"Epoch {epoch+1:02d} - Train Loss: {loss.item():.4f} - Train R²: {train_r2:.4f} - Train MAE: {train_mae:.4f}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        torch.save(final_model.state_dict(), "final_model_checkpoint.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping final training at epoch {epoch+1}")
            break
``
final_model.load_state_dict(torch.load("final_model_checkpoint.pt"))
final_model.eval()
with torch.no_grad():
    preds = final_model(X_full_t).detach().cpu().numpy()

plot_predictions_vs_actuals(y_full, preds, title="Final Model Predictions vs Actuals")








