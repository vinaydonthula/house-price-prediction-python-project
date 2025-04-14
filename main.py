import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    year = request.form.get('Year')

    print(location, bhk, bath, sqft)

    try:
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)
    except (ValueError, TypeError):
        return "Invalid input values", 400

    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    try:
        prediction = pipe.predict(input)[0] * 1e5
        prediction = abs(prediction)
    except Exception as e:
        return f"Prediction error: {e}", 500

    if year:
        try:
            y = int(year)
            if y >= 2023:
                for i in range(2024, y + 1):
                    prediction *= 1.05
        except ValueError:
            return "Invalid year format", 400

    return str(np.round(prediction, 2))

if __name__ == "__main__":
    print("\nApp is running at: http://127.0.0.1:5001/")
    print("Press CTRL+C to stop.\n")
    app.run(debug=True, port=5001)
