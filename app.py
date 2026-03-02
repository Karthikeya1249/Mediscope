from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from supabase import create_client, Client

app = Flask(__name__)

# === CONNECT TO SUPABASE ===
SUPABASE_URL = "https://ukwowjufiifjqyppkdjm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVrd293anVmaWlmanF5cHBrZGptIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzIyOTQ3NjEsImV4cCI6MjA4Nzg3MDc2MX0.UqQGmQt138c6uWgTgQ9iWLvnSJFdFeXIWY0dua78zgA"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

            # === SAVE TO SUPABASE ===
            gender_text = 'Female' if gender == 1 else 'Male'
            
            patient_data = {
                "name": user_name,
                "age": age,
                "gender": gender_text,
                "city": city,
                "disease": predicted_disease
            }
            
            supabase.table("patient_records").insert(patient_data).execute()

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

@app.route('/dashboard')
def dashboard():
    try:
        response = supabase.table("patient_records").select("*").execute()
        data = response.data
        
        if not data:
            return "<h1>No Data Yet!</h1><p>Please make a prediction first.</p>"
        
        df = pd.DataFrame(data)
        
        disease_counts = df['disease'].value_counts().to_dict()
        city_counts = df['city'].value_counts().to_dict()
        
        return render_template('dashboard.html', 
                               disease_labels=list(disease_counts.keys()), 
                               disease_data=list(disease_counts.values()),
                               city_labels=list(city_counts.keys()), 
                               city_data=list(city_counts.values()),
                               total_patients=len(df))
    except Exception as e:
        return f"<h1>Error loading dashboard:</h1><p>{e}</p>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)