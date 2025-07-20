import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Attack Risk Prediction App 💓")

st.write("""
Enter the patient's medical details to predict if they are at risk of having a heart attack.
""")

# Define user inputs for each feature
def user_input_features():
    sex = st.selectbox("Sex", ["Female", "Male"])
    general_health = st.selectbox("General Health", ["Poor", "Fair", "Good", "Very good", "Excellent"])
    physical_activities = st.selectbox("Physical Activities", ["Yes", "No"])
    had_angina = st.selectbox("Had Angina", ["Yes", "No"])
    had_stroke = st.selectbox("Had Stroke", ["Yes", "No"])
    had_asthma = st.selectbox("Had Asthma", ["Yes", "No"])
    had_skin_cancer = st.selectbox("Had Skin Cancer", ["Yes", "No"])
    had_copd = st.selectbox("Had COPD", ["Yes", "No"])
    had_depressive_disorder = st.selectbox("Had Depressive Disorder", ["Yes", "No"])
    had_kidney_disease = st.selectbox("Had Kidney Disease", ["Yes", "No"])
    had_arthritis = st.selectbox("Had Arthritis", ["Yes", "No"])
    had_diabetes = st.selectbox("Had Diabetes", [
        "No",
        "Yes",
        "No, pre-diabetes or borderline diabetes",
        "Yes, but only during pregnancy (female)"
    ])
    age_category = st.selectbox("Age Category", [
        'Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 'Age 35 to 39', 'Age 40 to 44',
        'Age 45 to 49', 'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 'Age 65 to 69',
        'Age 70 to 74', 'Age 75 to 79', 'Age 80 or older'
    ])
    last_checkup = st.selectbox("Last Checkup Time", [
        'Within past year (anytime less than 12 months ago)',
        'Within past 2 years (1 year but less than 2 years ago)',
        'Within past 5 years (2 years but less than 5 years ago)',
        '5 or more years ago'
    ])
    removed_teeth = st.selectbox("Removed Teeth", ['None of them', '1 to 5', '6 or more, but not all', 'All'])
    smoker_status = st.selectbox("Smoker Status", [
        'Never smoked', 'Former smoker',
        'Current smoker - now smokes some days', 'Current smoker - now smokes every day'
    ])
    race = st.selectbox("Race/Ethnicity Category", ['White', 'Black', 'Hispanic', 'Asian', 'Other'])  # Customize based on your dataset
    alcohol_drinkers = st.selectbox("Alcohol Drinkers", ["Yes", "No"])
    chest_scan = st.selectbox("Chest Scan", ["Yes", "No"])
    deaf_or_hard_of_hearing = st.selectbox("Deaf or Hard of Hearing", ["Yes", "No"])
    difficulty_concentrating = st.selectbox("Difficulty Concentrating", ["Yes", "No"])
    difficulty_walking = st.selectbox("Difficulty Walking", ["Yes", "No"])
    difficulty_dressing_bathing = st.selectbox("Difficulty Dressing/Bathing", ["Yes", "No"])

    # Map inputs to encoded values (same as your preprocessing)
    data = {
        'Sex': 0 if sex == "Female" else 1,
        'PhysicalActivities': 1 if physical_activities == "Yes" else 0,
        'HadAngina': 1 if had_angina == "Yes" else 0,
        'HadStroke': 1 if had_stroke == "Yes" else 0,
        'HadAsthma': 1 if had_asthma == "Yes" else 0,
        'HadSkinCancer': 1 if had_skin_cancer == "Yes" else 0,
        'HadCOPD': 1 if had_copd == "Yes" else 0,
        'HadDepressiveDisorder': 1 if had_depressive_disorder == "Yes" else 0,
        'HadKidneyDisease': 1 if had_kidney_disease == "Yes" else 0,
        'HadArthritis': 1 if had_arthritis == "Yes" else 0,
        'HadDiabetes': {
            'No': 0,
            'Yes': 1,
            'No, pre-diabetes or borderline diabetes': 2,
            'Yes, but only during pregnancy (female)': 3
        }[had_diabetes],
        'GeneralHealth': {
            'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4
        }[general_health],
        'LastCheckupTime': {
            'Within past year (anytime less than 12 months ago)': 0,
            'Within past 2 years (1 year but less than 2 years ago)': 1,
            'Within past 5 years (2 years but less than 5 years ago)': 2,
            '5 or more years ago': 3
        }[last_checkup],
        'RemovedTeeth': {
            'None of them': 0, '1 to 5': 1, '6 or more, but not all': 2, 'All': 3
        }[removed_teeth],
        'SmokerStatus': {
            'Never smoked': 0,
            'Former smoker': 1,
            'Current smoker - now smokes some days': 2,
            'Current smoker - now smokes every day': 3
        }[smoker_status],
        'AgeCategory': {
            'Age 18 to 24': 0, 'Age 25 to 29': 1, 'Age 30 to 34': 2, 'Age 35 to 39': 3, 'Age 40 to 44': 4,
            'Age 45 to 49': 5, 'Age 50 to 54': 6, 'Age 55 to 59': 7, 'Age 60 to 64': 8, 'Age 65 to 69': 9,
            'Age 70 to 74': 10, 'Age 75 to 79': 11, 'Age 80 or older': 12
        }[age_category],
        'RaceEthnicityCategory': {
            'White': 0, 'Black': 1, 'Hispanic': 2, 'Asian': 3, 'Other': 4  # Adjust to match your label encoding
        }[race],
        'AlcoholDrinkers': 1 if alcohol_drinkers == "Yes" else 0,
        'ChestScan': 1 if chest_scan == "Yes" else 0,
        'DeafOrHardOfHearing': 1 if deaf_or_hard_of_hearing == "Yes" else 0,
        'DifficultyConcentrating': 1 if difficulty_concentrating == "Yes" else 0,
        'DifficultyWalking': 1 if difficulty_walking == "Yes" else 0,
        'DifficultyDressingBathing': 1 if difficulty_dressing_bathing == "Yes" else 0,
    }

    return pd.DataFrame([data])

# Get input data
input_df = user_input_features()

# Scale features
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Output
st.subheader("Prediction Result")
result = "🚨 At Risk of Heart Attack" if prediction[0] == 1 else "✅ Not at Risk"
st.write(f"**Prediction:** {result}")
st.write(f"**Probability of Risk:** {prediction_proba[0][1]:.2f}")