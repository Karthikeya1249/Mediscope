from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import csv

app = Flask(__name__)

# Load AI Models
model = joblib.load('model.pkl')
mlb = joblib.load('encoder.pkl')

# Load Hospital Database
try:
    hospital_df = pd.read_csv('hospitals.csv')
    CITIES_LIST = sorted(hospital_df['City'].unique())
except Exception as e:
    print(f"Error loading hospitals.csv: {e}")
    CITIES_LIST = []

CARE_ADVICE = {
    'Flu': ('Self-care', 'Rest, drink fluids, take paracetamol if needed.'),
    'Common Cold': ('Self-care', 'Stay warm, rest, and drink herbal tea.'),
    'Migraine': ('Self-care', 'Rest in a dark quiet room, apply cold compress.'),
    'Chickenpox': ('General Physician', 'Consult a doctor for antiviral medication.'),
    'Gastritis': ('General Physician', 'Avoid spicy food, consult GP for antacids.'),
    'Fungal Infection': ('General Physician', 'Keep area dry, consult doctor for antifungal cream.'),
    'Diabetes': ('Specialist', 'Consult an Endocrinologist for blood sugar management.'),
    'Heart Disease': ('Emergency', 'Seek immediate medical help (Cardiologist).'),
    'Pneumonia': ('Emergency', 'See a doctor immediately for chest X-ray and antibiotics.'),
    'Allergy': ('General Physician', 'Avoid allergens and take antihistamines.')
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    care_category = ""
    care_advice = ""
    user_name = ""
    hospital_rec = ""
    
    available_symptoms = sorted(mlb.classes_)

    if request.method == 'POST':
        try:
            user_name = request.form['name']
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            city = request.form['city']
            selected_symptoms = request.form.getlist('symptoms')

            # AI Prediction
            input_df = pd.DataFrame(0, index=[0], columns=['Age', 'Gender'] + list(mlb.classes_))
            input_df['Age'] = age
            input_df['Gender'] = gender
            for sym in selected_symptoms:
                if sym in input_df.columns:
                    input_df[sym] = 1

            predicted_disease = model.predict(input_df)[0]
            
            info = CARE_ADVICE.get(predicted_disease, ('General Physician', 'Consult a doctor.'))
            care_category = info[0]
            care_advice = info[1]
            prediction_text = predicted_disease

            # Hospital Logic
            if care_category != 'Self-care':
                match = hospital_df[(hospital_df['City'] == city) & (hospital_df['Disease'] == predicted_disease)]
                if not match.empty:
                    hospital_rec = match.iloc[0]['Hospital']
                else:
                    general_match = hospital_df[(hospital_df['City'] == city) & (hospital_df['Disease'] == 'General')]
                    if not general_match.empty:
                        hospital_rec = general_match.iloc[0]['Hospital'] + " (General)"
                    else:
                        hospital_rec = "No specific hospital data found."
            else:
                hospital_rec = "Home rest is recommended. No hospital visit required."

            # === NEW: SAVE TO PATIENT HISTORY CSV ===
            history_file = 'patient_history.csv'
            file_exists = os.path.isfile(history_file)
            
            with open(history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Name', 'Age', 'Gender', 'City', 'Disease']) # Header
                
                gender_text = 'Female' if gender == 1 else 'Male'
                writer.writerow([user_name, age, gender_text, city, predicted_disease])

        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template('index.html', 
                           symptoms=available_symptoms, 
                           cities=CITIES_LIST,
                           prediction=prediction_text,
                           care_category=care_category,
                           care_advice=care_advice,
                           hospital_rec=hospital_rec,
                           user_name=user_name)

# === NEW: ADMIN DASHBOARD ROUTE ===
@app.route('/dashboard')
def dashboard():
    history_file = 'patient_history.csv'
    
    if not os.path.exists(history_file):
        return "<h1>No Data Yet!</h1><p>Please go to the home page and make a prediction first.</p>"
    
    # Read the data we've been saving
    df = pd.read_csv(history_file)
    
    # Count how many of each disease
    disease_counts = df['Disease'].value_counts().to_dict()
    
    # Count how many patients from each city
    city_counts = df['City'].value_counts().to_dict()
    
    return render_template('dashboard.html', 
                           disease_labels=list(disease_counts.keys()), 
                           disease_data=list(disease_counts.values()),
                           city_labels=list(city_counts.keys()), 
                           city_data=list(city_counts.values()),
                           total_patients=len(df))

if __name__ == '__main__':
    app.run(debug=True, port=5000)