import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset

df = pd.read_csv(r"c:\Users\Loga\Downloads\Advertising.csv")

# Drop unnecessary columns
df = df.drop(['Product_ID'], axis=1, errors='ignore')

# Encode categorical variables
categorical_cols = ['Region', 'Product_Category', 'Advertising_Channel']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and target variable
X = df.drop(['Sales'], axis=1)
y = df['Sales']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Regression Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Feature Importance
feature_importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importance, color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Sales Prediction")
plt.show()

# Function to predict sales for new input
def predict_sales(new_data):
    new_df = pd.DataFrame([new_data])
    new_df = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)
    missing_cols = set(X.columns) - set(new_df.columns)
    for col in missing_cols:
        new_df[col] = 0
    new_df = new_df[X.columns]
    new_scaled = scaler.transform(new_df)
    prediction = model.predict(new_scaled)
    return prediction[0]

# Example input for prediction
new_input = {
    'Region': 'North',
    'Product_Category': 'Electronics',
    'Advertising_Channel': 'Social Media',
    'Ad_Spend': 5000,
    'Discount': 10,
    'Season': 'Winter'
}

predicted_sales = predict_sales(new_input)
print(f"Predicted Sales: {predicted_sales}")
