import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Load dataset ---
df = pd.read_csv("heart.csv")  

# Features & target
X = df.drop("condition", axis=1)
y = df["condition"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- Streamlit UI ---
st.set_page_config(page_title="Heart Disease Risk Assessment", page_icon="❤️", layout="wide")

# Sidebar for info
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("""
This app predicts **heart disease risk** based on patient details.
- Uses a **Random Forest Classifier** trained on heart disease dataset.
- Accuracy: **{:.2f}%**
""".format(acc*100))

st.sidebar.header("💡 Health Tips")
st.sidebar.write("""
- Exercise regularly 🏃‍♂️  
- Maintain healthy blood pressure 💓  
- Control cholesterol 🥗  
- Avoid smoking 🚭  
- Eat a balanced diet 🍎  
""")

st.markdown("<h1 style='text-align: center; color: red;'>❤️ Heart Disease Risk Assessment App</h1>", unsafe_allow_html=True)
st.markdown(f"<h4 style='text-align: center;'>Model Accuracy: <span style='color:green'>{acc*100:.2f}%</span></h4>", unsafe_allow_html=True)
st.write("---")

st.header("🩺 Enter Patient Details:")

# Columns layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type",
                      ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia Type", ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"])

# --- Map inputs to numeric values ---
sex_val = 0 if sex == "Female" else 1
cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
fbs_val = 0 if fbs == "No" else 1
restecg_val = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
exang_val = 0 if exang == "No" else 1
slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal_val = ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"].index(thal)

# --- Prediction ---
if st.button("Predict ❤️"):
    features = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                          thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.markdown("<h2 style='text-align: center; color: red;'>⚠️ Patient is at risk of Heart Disease</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: darkred;'>Consult a doctor for further evaluation.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: green;'>✅ Patient is Healthy</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: darkgreen;'>Keep following healthy habits!</p>", unsafe_allow_html=True)
