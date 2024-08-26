import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('search_ads_data.csv')

# Preview the data
print(data.head())

# Handle missing values (if any)
data = data.ffill()

# Encode categorical variables
label_encoder = LabelEncoder()
data['ad_category'] = label_encoder.fit_transform(data['ad_category'])

# Feature scaling
scaler = StandardScaler()
data[['budget', 'clicks', 'impressions']] = scaler.fit_transform(data[['budget', 'clicks', 'impressions']])

# Split data into features and target variable
X = data.drop(columns=['conversion_rate'])
y = data['conversion_rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Script executed successfully!")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

import matplotlib.pyplot as plt

# Plot predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Conversion Rate")
plt.ylabel("Predicted Conversion Rate")
plt.title("Actual vs Predicted Conversion Rate")
plt.show()

import matplotlib.pyplot as plt

# Plot predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Conversion Rate")
plt.ylabel("Predicted Conversion Rate")
plt.title("Actual vs Predicted Conversion Rate")
plt.show()
