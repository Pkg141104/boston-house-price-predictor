# train_model.py
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Select only 8 features for simplicity
selected_features = ['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'TAX', 'NOX', 'AGE', 'CRIM']
X_selected = X[selected_features]

# Train model
model = LinearRegression()
model.fit(X_selected, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
