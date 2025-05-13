import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re

final_data = pd.read_csv("data/final_data.csv")
social_data = pd.read_excel('data/social_data_msoa.xlsx', sheet_name='iadatasheet1')
postcode_map = pd.read_csv("data/Online_ONS_Postcode_Directory_Live (6).csv", usecols=["LSOA21", "MSOA21"])

social_data.columns = [
    "_".join(filter(None, map(str.strip, map(str, col)))) if isinstance(col, tuple) else str(col).strip()
    for col in social_data.columns
]
social_data.columns = [re.sub(r'_Unnamed.*$', '', col) for col in social_data.columns]

social_data.dropna(how='all', inplace=True)
social_data.reset_index(drop=True, inplace=True)

social_data.rename(columns={social_data.columns[0]: 'MSOA_code', social_data.columns[1]: 'MSOA_name'}, inplace=True)
social_data.drop(index=0, inplace=True)

postcode_map = postcode_map.drop_duplicates()
final_data['LSOA code'] = final_data['LSOA code'].str.strip().str.upper()
postcode_map['LSOA21'] = postcode_map['LSOA21'].str.strip().str.upper()

final_data = final_data.merge(postcode_map, how="left", left_on="LSOA code", right_on="LSOA21")
final_data = final_data.dropna(subset=["MSOA21"])

final_data['MSOA21'] = final_data['MSOA21'].str.strip().str.upper()
social_data['MSOA_code'] = social_data['MSOA_code'].astype(str).str.strip().str.upper()

merged = final_data.merge(social_data, how="left", left_on="MSOA21", right_on="MSOA_code")

ethnicity_columns = [col for col in social_data.columns if 'Ethnic Group' in col]
religion_columns = [col for col in social_data.columns if 'Religion' in col]
house_price_columns = [col for col in social_data.columns if 'House Prices' in col]
house_income_columns =[col for col in social_data.columns if 'Household Income Estimates (2011/12)' in col]
social_data_features = merged[ethnicity_columns + religion_columns + house_price_columns + house_income_columns]

social_data_features = social_data_features.apply(pd.to_numeric, errors='coerce')
social_data_features = social_data_features.dropna()

crime_monthly = merged.groupby('Month').size().reset_index(name='Crime Count')
crime_data = crime_monthly['Crime Count'].values.reshape(-1, 1)

scaler = StandardScaler()
crime_data = scaler.fit_transform(crime_data)

def create_dataset(data, social_data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(np.hstack((data[i:(i + look_back), 0], social_data[i + look_back])))
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 50 
X, y = create_dataset(crime_data, social_data_features.values, look_back)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

model = LSTMModel(input_size=X_train.shape[2]) 
loss_function = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  

epochs = 100
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
train_r2 = []
test_r2 = []
train_mae_values = []  

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    train_pred = model(X_train).squeeze()
    train_loss = loss_function(train_pred, y_train)
    train_loss.backward()
    optimizer.step()

    train_pred_np = train_pred.detach().numpy().reshape(-1, 1)  
    y_train_np = y_train.detach().numpy().reshape(-1, 1)
    
    train_mae = mean_absolute_error(y_train_np, train_pred_np)
    train_r2_value = r2_score(y_train_np, train_pred_np)
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).squeeze()
        test_pred_np = test_pred.numpy().reshape(-1, 1)
        y_test_np = y_test.numpy().reshape(-1, 1)
        
        test_mae = mean_absolute_error(y_test_np, test_pred_np)
        test_r2_value = r2_score(y_test_np, test_pred_np)
        test_loss = loss_function(test_pred, y_test)  

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())  
    train_accuracies.append(train_mae)
    test_accuracies.append(test_mae)
    train_r2.append(train_r2_value)
    test_r2.append(test_r2_value)
    train_mae_values.append(train_mae) 

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.item():.6f} - Train MAE: {train_mae:.2f} - Test MAE: {test_mae:.2f} - Train R2: {train_r2_value:.2f} - Test R2: {test_r2_value:.2f} - Test Loss: {test_loss.item():.6f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label="Train Loss")
plt.plot(range(epochs), test_losses, label="Test Loss", color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Test Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_mae_values, label="Train MAE", color='blue')
plt.plot(range(epochs), test_accuracies, label="Test MAE", color='red')
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Train and Test MAE")
plt.legend()

plt.tight_layout()
plt.show()

metrics_df = pd.DataFrame({
    'Epoch': range(1, epochs + 1),
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Train MAE': train_mae_values,  
    'Test MAE': test_accuracies,
    'Train R2': train_r2,
    'Test R2': test_r2
})
metrics_df.to_csv("data/lstm_training_metrics.csv", index=False)

















