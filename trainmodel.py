import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# === 1. DATASET (The Knowledge Base) ===
# We map Symptoms + Age + Gender to a Disease
df=pd.read_csv("data.csv")
df['Symptoms'] = df['Symptoms'].apply(lambda x: x.split(", "))

# === 2. PREPROCESS ===
# Convert list of symptoms into "One-Hot" columns (0 or 1)
mlb = MultiLabelBinarizer()
symptoms_encoded = mlb.fit_transform(df['Symptoms'])
symptoms_df = pd.DataFrame(symptoms_encoded, columns=mlb.classes_)

# Combine Age/Gender with the new Symptom columns
X = pd.concat([df[['Age', 'Gender']], symptoms_df], axis=1)
y = df['Disease']

# === 3. TRAIN ===
print("🧠 Training MediScope Model...")
model = DecisionTreeClassifier()
model.fit(X, y)

# === 4. SAVE ===
joblib.dump(model, 'model.pkl')
joblib.dump(mlb, 'encoder.pkl')
print("✅ Success! Model and Encoder saved.")