import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model from the pickle file
with open("result.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template('home.html') 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure']) 
        SkinThickness = float(request.form['SkinThickness']) 
        Insulin = float(request.form['Insulin']) 
        BMI = float(request.form['BMI']) 
        DiabetesPedigreeFunction = float(request.form['Diabetes_Pedigree_Function'])     
        Age = float(request.form['Age']) 
    
        # Make prediction
        input_data = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        prediction = model.predict(input_data)

        # Interpret prediction
        if prediction[0] == 1:
            result = "Diabetic"
        else:
            result = "Non-Diabetic"

        return render_template('predict.html', prediction_text=f'The patient is likely {result}')
    return render_template('predict.html')
if __name__ == "__main__":
    app.run(debug=True)
