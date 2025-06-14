from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.datasets import fetch_california_housing

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Get values from form and convert to float
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])[0]

        # Step 2: Load prediction dataset (for plotting)
        data = pd.read_csv('prediction_data.csv')

        # Step 3: Create histogram plot
        plt.figure(figsize=(6, 4))
        plt.hist(data['Predicted'], bins=30, alpha=0.7, label='All Predictions')
        plt.axvline(prediction, color='red', linewidth=2, label='Your Prediction')
        plt.title('Your Prediction vs Model Distribution')
        plt.xlabel('Predicted Price')
        plt.ylabel('Count')
        plt.legend()
        plot_path = os.path.join('static', 'prediction_plot.png')
        plt.savefig(plot_path)
        plt.close()

        # Step 4: Get most influential feature
        coef = model.coef_
        feature_names = fetch_california_housing().feature_names
        top_feature = feature_names[np.argmax(np.abs(coef))]

        # Step 5: Send results to HTML
        return render_template(
            'index.html',
            prediction_text=f"Predicted Median House Value: ${prediction:.2f}",
            top_feature=f"Top influencing feature: {top_feature}",
            image_file='prediction_plot.png'
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# To run on Render/Replit as well
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
