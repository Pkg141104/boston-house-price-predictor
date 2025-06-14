import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import pickle

# Load California housing data
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

# Features and Target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save predictions to CSV
df['Predicted'] = model.predict(X)
df.to_csv('prediction_data.csv', index=False)

print("✅ Model trained and saved to model.pkl and prediction_data.csv")
