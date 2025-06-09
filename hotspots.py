import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterSampler
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import copy
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
import datetime 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
grid_with_lsoa = grid_with_lsoa.rename(columns={'LSOA21CD': 'LSOA_code'}).drop(columns=['index_right'])

crime_data['Month'] = pd.to_datetime(crime_data['Month'])
gridded_data = crime_data[crime_data['LSOA code'].isin(grid_with_lsoa['LSOA_code'])]
gridded_data = gridded_data.rename(columns={'LSOA code': 'LSOA_code'})
time_series = gridded_data.groupby(['Month', 'LSOA_code']).size().unstack(fill_value=0).sort_index()

cols = social_data.columns
relevant_cols = [col for col in cols if any(key in col for key in ['Dwelling type', 'Households', 'Household Language', '2011 Census Population', 'House Prices'])]
social_data = social_data[['LSOA_code'] + relevant_cols].set_index('LSOA_code').apply(pd.to_numeric, errors='coerce').fillna(0)
social_data = social_data.loc[time_series.columns.intersection(social_data.index)]
time_series = time_series[social_data.index]

scaler_social = StandardScaler()
social_scaled = scaler_social.fit_transform(social_data.values)

months = time_series.index.to_series().dt.month
month_sin = np.sin(2 * np.pi * months.values / 12).reshape(-1, 1)
month_cos = np.cos(2 * np.pi * months.values / 12).reshape(-1, 1)
temporal_features = np.hstack([month_sin, month_cos])

lsoa_gdf = lsoa_gdf.set_index('LSOA21CD').loc[social_data.index]
centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in lsoa_gdf.geometry])
dist_matrix = cdist(centroids, centroids)
adjacency = (dist_matrix <= 1000) & (dist_matrix > 0)
row, col = np.where(adjacency)
edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long).to(device)
edge_weight = torch.tensor(dist_matrix[row, col], dtype=torch.float32).to(device)
look_back = 36
crime_counts = time_series.values 

def create_dataset(crime, social, temporal, labels, look_back):
    n_time, n_lsoas = crime.shape
    feature_len = look_back + social.shape[1] + temporal.shape[1]
    X = np.zeros(((n_time - look_back) * n_lsoas, feature_len))
    y = np.zeros((n_time - look_back) * n_lsoas)
    idx = 0
    for t in range(look_back, n_time):
        for l in range(n_lsoas):
            features = np.concatenate([
                crime[t - look_back:t, l], 
                social[l],                  
                temporal[t]                
            ])
            X[idx] = features
            y[idx] = labels[t, l]
            idx += 1
    return X, y



class STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3, num_layers=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for conv in self.convs[:-1]:
            h = self.relu(conv(h, edge_index, edge_weight))
            h = self.dropout(h)
        h = self.sigmoid(self.convs[-1](h, edge_index, edge_weight))
        return h

param_dist = {
    'lr': np.logspace(-4, -2, 20),
    'hidden_dim': [32, 64, 128],
    'dropout': [0.0, 0.1, 0.3, 0.5],
    'weight_decay': np.logspace(-6, -3, 10)
}
percentiles = [ 50, 60, 70, 80]
thresholds = [ 0.5, 0.6, 0.7, 0.8,]
param_list = list(ParameterSampler(param_dist, n_iter=20, random_state=42))

best_f1 = -np.inf
best_acc = -np.inf
best_f1_info = None
best_acc_info = None
criterion = nn.BCELoss(reduction='none')

for params in param_list:
    best_combined_score_for_param = -np.inf
    best_percentile_for_param = None
    best_threshold_for_param = None
    best_acc_for_param = None
    best_f1_for_param = None

    for percentile in percentiles:
        hotspot_labels = (crime_counts >= np.percentile(crime_counts, percentile, axis=1, keepdims=True)).astype(int)
        y = hotspot_labels[look_back:].flatten()
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

        X, y_np = create_dataset(crime_counts, social_scaled, temporal_features, hotspot_labels, look_back)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        split = int(0.8 * len(X_t))
        X_train, X_val = X_t[:split], X_t[split:]
        y_train, y_val = y_t[:split], y_t[split:]

        class_weights = torch.tensor([1.0, (y_np == 0).sum() / (y_np == 1).sum()], dtype=torch.float32).to(device)
        loss_weights = class_weights[1] * y_t + class_weights[0] * (1 - y_t)
        w_train, w_val = loss_weights[:split], loss_weights[split:]

        model = STGCN(X.shape[1], params['hidden_dim'], 1, dropout=params['dropout']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            output_train = model(X_train, edge_index, edge_weight).squeeze()
            loss_train = (criterion(output_train, y_train) * w_train).mean()
            loss_train.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            output_val = model(X_val, edge_index, edge_weight).squeeze()
            val_preds = output_val.unsqueeze(1).repeat(1, len(thresholds))
            thresholds_tensor = torch.tensor(thresholds, device=device).reshape(1, -1)
            val_pred_bin = (val_preds >= thresholds_tensor).cpu().numpy()
            y_val_np = y_val.cpu().numpy().reshape(-1, 1)

            for i, threshold in enumerate(thresholds):
                preds = val_pred_bin[:, i]
                acc = accuracy_score(y_val_np, preds)
                f1 = f1_score(y_val_np, preds)
                mcc = matthews_corrcoef(y_val_np, preds)
                bal_acc = balanced_accuracy_score(y_val_np, preds)
                combined_score = (0.25 * f1) + (0.5 * mcc) + (0.15 * bal_acc) + (0.1 * acc)

 
                if combined_score > best_combined_score_for_param:
                    best_combined_score_for_param = combined_score
                    best_percentile_for_param = percentile
                    best_threshold_for_param = threshold
                    best_acc_for_param = acc
                    best_f1_for_param = f1

                # Track global bests
                if combined_score > best_f1:  # or other global metric
                    best_f1 = combined_score
                    best_f1_info = {
                        'params': params,
                        'percentile': percentile,
                        'threshold': threshold,
                        'model_state': copy.deepcopy(model.state_dict())
                    }

    print(f"Best for params {params} -> Percentile: {best_percentile_for_param}, Threshold: {best_threshold_for_param}, Accuracy: {best_acc_for_param:.4f}, F1: {best_f1_for_param:.4f}, Combined: {best_combined_score_for_param:.4f}")

print("\nBest F1 setup overall:")
print(best_f1_info['params'], f"Percentile: {best_f1_info['percentile']}, Threshold: {best_f1_info['threshold']}, Combined Score: {best_f1:.4f}")

def train_and_report(params, percentile, threshold, epochs=100, early_stop_patience=10):
    best_val_acc = -np.inf
    best_val_f1 = -np.inf
    best_val_mcc = -np.inf
    best_val_bal_acc = -np.inf
    best_val_precision = -np.inf
    best_val_recall = -np.inf

    print("Final training")

    hotspot_labels = (crime_counts >= np.percentile(crime_counts, percentile, axis=1, keepdims=True)).astype(int)
    y = hotspot_labels[look_back:].flatten()
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    X, y_np = create_dataset(crime_counts, social_scaled, temporal_features, hotspot_labels, look_back)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    split = int(0.8 * len(X_t))
    X_train, X_val = X_t[:split], X_t[split:]
    y_train, y_val = y_t[:split], y_t[split:]

    class_weights = torch.tensor([1.0, (y_np == 0).sum() / (y_np == 1).sum()], dtype=torch.float32).to(device)
    loss_weights = class_weights[1] * y_t + class_weights[0] * (1 - y_t)
    w_train, w_val = loss_weights[:split], loss_weights[split:]

    model = STGCN(X.shape[1], params['hidden_dim'], 1, dropout=params['dropout']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    best_val_loss = np.inf
    patience_counter = 0
    best_model = None

    best_train_acc = -np.inf
    best_train_f1 = -np.inf
    best_train_mcc = -np.inf
    best_train_bal_acc = -np.inf
    best_train_precision = -np.inf
    best_train_recall = -np.inf

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output_train = model(X_train, edge_index, edge_weight).squeeze()
        loss_train = (criterion(output_train, y_train) * w_train).mean()
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output_val = model(X_val, edge_index, edge_weight).squeeze()
            loss_val = (criterion(output_val, y_val) * w_val).mean().item()

            # Predictions
            preds_train = (output_train >= threshold).cpu().numpy()
            preds_val = (output_val >= threshold).cpu().numpy()
            y_train_np = y_train.cpu().numpy()
            y_val_np = y_val.cpu().numpy()

            # Train metrics
            acc_train = accuracy_score(y_train_np, preds_train)
            f1_train = f1_score(y_train_np, preds_train)
            mcc_train = matthews_corrcoef(y_train_np, preds_train)
            bal_acc_train = balanced_accuracy_score(y_train_np, preds_train)
            precision_train = precision_score(y_train_np, preds_train)
            recall_train = recall_score(y_train_np, preds_train)

            acc_val = accuracy_score(y_val_np, preds_val)
            f1_val = f1_score(y_val_np, preds_val)
            mcc_val = matthews_corrcoef(y_val_np, preds_val)
            bal_acc_val = balanced_accuracy_score(y_val_np, preds_val)
            precision_val = precision_score(y_val_np, preds_val)
            recall_val = recall_score(y_val_np, preds_val)

            if acc_val > best_val_acc:
                best_val_acc = acc_val
            if f1_val > best_val_f1:
                best_val_f1 = f1_val
            if mcc_val > best_val_mcc:
                best_val_mcc = mcc_val
            if bal_acc_val > best_val_bal_acc:
                best_val_bal_acc = bal_acc_val
            if precision_val > best_val_precision:
                best_val_precision = precision_val
            if recall_val > best_val_recall:
                best_val_recall = recall_val

            if acc_train > best_train_acc:
                best_train_acc = acc_train
            if f1_train > best_train_f1:
                best_train_f1 = f1_train
            if mcc_train > best_train_mcc:
                best_train_mcc = mcc_train
            if bal_acc_train > best_train_bal_acc:
                best_train_bal_acc = bal_acc_train
            if precision_train > best_train_precision:
                best_train_precision = precision_train
            if recall_train > best_train_recall:
                best_train_recall = recall_train

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {loss_train.item():.4f} "
                  f"Acc: {acc_train:.4f} F1: {f1_train:.4f} "
                  f"BalAcc: {bal_acc_train:.4f} MCC: {mcc_train:.4f} "
                  f"Precision: {precision_train:.4f} Recall: {recall_train:.4f} | "
                  f"Val Loss: {loss_val:.4f} "
                  f"Acc: {acc_val:.4f} F1: {f1_val:.4f} "
                  f"BalAcc: {bal_acc_val:.4f} MCC: {mcc_val:.4f} "
                  f"Precision: {precision_val:.4f} Recall: {recall_val:.4f}")

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
                best_model = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        output_val = model(X_val, edge_index, edge_weight).squeeze()
        preds_val = (output_val >= threshold).cpu().numpy()
        y_val_np = y_val.cpu().numpy()

        acc_val = accuracy_score(y_val_np, preds_val)
        f1_val = f1_score(y_val_np, preds_val)
        mcc_val = matthews_corrcoef(y_val_np, preds_val)
        bal_acc_val = balanced_accuracy_score(y_val_np, preds_val)
        precision_val = precision_score(y_val_np, preds_val)
        recall_val = recall_score(y_val_np, preds_val)

        print("\n Best metrics on validation set")
        print(f"Best Accuracy: {best_val_acc:.4f}")
        print(f"Best F1 Score: {best_val_f1:.4f}")
        print(f"Best MCC: {best_val_mcc:.4f}")
        print(f"Best Balanced Accuracy: {best_val_bal_acc:.4f}")
        print(f"Best Precision: {best_val_precision:.4f}")
        print(f"Best Recall: {best_val_recall:.4f}")

        print("\n Best metrics on training set")
        print(f"Best Accuracy: {best_train_acc:.4f}")
        print(f"Best F1 Score: {best_train_f1:.4f}")
        print(f"Best MCC: {best_train_mcc:.4f}")
        print(f"Best Balanced Accuracy: {best_train_bal_acc:.4f}")
        print(f"Best Precision: {best_train_precision:.4f}")
        print(f"Best Recall: {best_train_recall:.4f}")

    return model




best_params = best_f1_info['params']
best_percentile = best_f1_info['percentile']
best_threshold = best_f1_info['threshold']
final_model = train_and_report(best_params, best_percentile, best_threshold, epochs=100, early_stop_patience=10)

def generate_future_temporal_features(start_month, n_months=12):
    months = [(start_month + pd.DateOffset(months=i)).month for i in range(n_months)]
    month_sin = np.sin(2 * np.pi * np.array(months) / 12).reshape(-1, 1)
    month_cos = np.cos(2 * np.pi * np.array(months) / 12).reshape(-1, 1)
    return np.hstack([month_sin, month_cos])

def predict_future_months(model, crime_counts, social_scaled, temporal_features, look_back, n_months=12, threshold=0.5):
    model.eval()
    n_time, n_lsoas = crime_counts.shape
    last_time_index = n_time - 1
    last_observed_month = time_series.index[-1]
    future_start_month = last_observed_month + pd.DateOffset(months=1)
    future_temporal = generate_future_temporal_features(future_start_month, n_months)
    predictions = np.zeros((n_lsoas, n_months))


    for i in range(n_months):
        crime_input = crime_counts[(last_time_index - look_back + 1):(last_time_index + 1), :] 
        X_future = np.zeros((n_lsoas, look_back + social_scaled.shape[1] + 2))

        for lsoa_idx in range(n_lsoas):
            features = np.concatenate([
                crime_input[:, lsoa_idx],    
                social_scaled[lsoa_idx],    
                future_temporal[i]           
            ])
            X_future[lsoa_idx] = features

        X_t = torch.tensor(X_future, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(X_t, edge_index, edge_weight).squeeze().cpu().numpy()
            preds_bin = (output >= threshold).astype(int)
            predictions[:, i] = preds_bin

    return predictions


future_preds = predict_future_months(final_model, crime_counts, social_scaled, temporal_features, look_back, n_months=12, threshold=best_threshold)
lsoa_codes = social_data.index.tolist()
months_future = [(time_series.index[-1] + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(12)]
df_preds = pd.DataFrame(future_preds, index=lsoa_codes, columns=months_future)
output_path = data_dir / "future_crime_hotspot_predictions.csv"
df_preds.to_csv(output_path)

print(f"Future hotspot predictions saved to {output_path}")















