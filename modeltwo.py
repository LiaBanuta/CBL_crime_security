import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
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

# Extracting features (same as before)
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

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Train R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")

predictions_df = pd.DataFrame({
    'MSOA': merged['MSOA21'].iloc[train_size:].reset_index(drop=True),
    'Predicted Crime Next Month': y_test_pred
})

print(predictions_df)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(len(y_train)), y_train, label="Actual Train Data", color='blue')
plt.plot(range(len(y_train_pred)), y_train_pred, label="Predicted Train Data", color='red')
plt.title("Train Data vs Predicted Train Data")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(y_test)), y_test, label="Actual Test Data", color='blue')
plt.plot(range(len(y_test_pred)), y_test_pred, label="Predicted Test Data", color='red')
plt.title("Test Data vs Predicted Test Data")
plt.legend()

plt.tight_layout()
plt.show()


