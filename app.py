from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('stacking_classifier_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and model is not None:
        try:
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
            # Check age_category for predefined negative result
            
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
            

            # Print input data to console
            print("Input physical_health:", input_data)

            # Make prediction
            if age_category in [22, 23, 25, 35, 36, 37, 38, 39, 40]:
                prediction = 0
            else:
                prediction = model.predict(np.array(input_data))[0]
            result ="Yes" if prediction == 1 else "NO"
            print("prediction :", prediction)

        except Exception as e:
            result = f"Error: {e}"
        
        # Render the result on the same page
        return render_template('index.html', result=result)

    # Render the form for GET requests
    return render_template('index.html', result=None)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in production