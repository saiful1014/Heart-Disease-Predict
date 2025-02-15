from flask import Flask, request, render_template
import pickle
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# Initialize Flask app
app = Flask(__name__)
import joblib

model = joblib.load('stacking_classifier_model.pkl')
# Load the trained model
# try:
#     with open('stacking_classifier_model.pkl', 'rb') as file:
#         model = pickle.load(file)
# except pickle.UnpicklingError as e:
#     print(f"Error loading model: {e}")
#     model = None

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        smoking = int(request.form['smoking'])
        alcohol_drinking = int(request.form['alcohol_drinking'])
        stroke = int(request.form['stroke'])
        physical_health = int(request.form['physical_health'])
        mental_health = int(request.form['mental_health'])
        diff_walking = int(request.form['diff_walking'])
        age_category = int(request.form['age_category'])
        diabetic = int(request.form['diabetic'])
        physical_activity = int(request.form['physical_activity'])
        asthma = int(request.form['asthma'])
        kidney_disease = int(request.form['kidney_disease'])
        skin_cancer = int(request.form['skin_cancer'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'Smoking': [smoking],
            'AlcoholDrinking': [alcohol_drinking],
            'Stroke': [stroke],
            'PhysicalHealth': [physical_health],
            'MentalHealth': [mental_health],
            'DiffWalking': [diff_walking],
            'AgeCategory': [age_category],
            'Diabetic': [diabetic],
            'PhysicalActivity': [physical_activity],
            'Asthma': [asthma],
            'KidneyDisease': [kidney_disease],
            'SkinCancer': [skin_cancer]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Positive" if prediction == 1 else "Negative"

        # Render the result on the same page
        return render_template('index.html', result=result)

    # Render the form for GET requests
    return render_template('index.html', result=None)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)