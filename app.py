import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the correct order of features
FEATURE_COLUMNS = [
    "Sex", "GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "LastCheckupTime",
    "PhysicalActivities", "SleepHours", "RemovedTeeth", "HadAngina", "HadStroke",
    "HadAsthma", "HadSkinCancer", "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease",
    "HadArthritis", "HadDiabetes", "DeafOrHardOfHearing", "DifficultyConcentrating",
    "DifficultyWalking", "DifficultyDressingBathing", "SmokerStatus", "ChestScan",
    "RaceEthnicityCategory", "AgeCategory", "BMI", "AlcoholDrinkers"
]

# Mapping dictionaries (based on how you encoded your features)
sex_map = {"Female": 0, "Male": 1}
last_checkup_map = {
    "Within past year (anytime less than 12 months ago)": 0,
    "Within past 2 years (1 year but less than 2 years ago)": 1,
    "Within past 5 years (2 years but less than 5 years ago)": 2,
    "5 or more years ago": 3
}
yes_no_map = {"No": 0, "Yes": 1}
removed_teeth_map = {
    "None of them": 0,
    "1 to 5": 1,
    "6 or more, but not all": 2,
    "All": 3
}
diabetes_map = {
    "No": 0,
    "Yes (other than pregnancy)": 1,
    "Yes, but only during pregnancy (female)": 2,
    "No, pre-diabetes or borderline diabetes": 3
}
smoker_map = {
    "Never smoked": 0,
    "Former smoker": 1,
    "Current smoker - now smokes some days": 2,
    "Current smoker - now smokes every day": 3
}
race_map = {
    "White only, Non-Hispanic": 0,
    "Black only, Non-Hispanic": 1,
    "Other race only, Non-Hispanic": 2,
    "Multiracial, Non-Hispanic": 3,
    "Hispanic": 4
}
age_map = {
    "Age 18 to 24": 0, "Age 25 to 29": 1, "Age 30 to 34": 2, "Age 35 to 39": 3,
    "Age 40 to 44": 4, "Age 45 to 49": 5, "Age 50 to 54": 6, "Age 55 to 59": 7,
    "Age 60 to 64": 8, "Age 65 to 69": 9, "Age 70 to 74": 10,
    "Age 75 to 79": 11, "Age 80 or older": 12
}

# Streamlit app
st.title("Cardiovascular Risk Calculator")
st.markdown("This app uses a logistic regression model to predict the risk of a cardiovascular event based on survey data.")

# Input form
with st.form("input_form"):
    Sex = st.radio("Sex", list(sex_map.keys()))
    GeneralHealth = st.slider("General Health (0 = Poor, 4 = Excellent)", 0, 4, 2)
    PhysicalHealthDays = st.slider("Good Physical Health (past 30 days)", 0, 30, 0)
    MentalHealthDays = st.slider("Good Mental Health (past 30 days)", 0, 30, 0)
    LastCheckupTime = st.radio("Last Checkup Time", list(last_checkup_map.keys()))
    PhysicalActivities = st.radio("Physical Activities", list(yes_no_map.keys()))
    SleepHours = st.slider("Sleep Hours", 0, 24, 7)
    RemovedTeeth = st.radio("Number of Teeth Removed", list(removed_teeth_map.keys()))
    HadAngina = st.radio("Had Angina", list(yes_no_map.keys()))
    HadStroke = st.radio("Had Stroke", list(yes_no_map.keys()))
    HadAsthma = st.radio("Had Asthma", list(yes_no_map.keys()))
    HadSkinCancer = st.radio("Had Skin Cancer", list(yes_no_map.keys()))
    HadCOPD = st.radio("Had COPD", list(yes_no_map.keys()))
    HadDepressiveDisorder = st.radio("Had Depressive Disorder", list(yes_no_map.keys()))
    HadKidneyDisease = st.radio("Had Kidney Disease", list(yes_no_map.keys()))
    HadArthritis = st.radio("Had Arthritis", list(yes_no_map.keys()))
    HadDiabetes = st.radio("Had Diabetes", list(diabetes_map.keys()))
    DeafOrHardOfHearing = st.radio("Deaf or Hard of Hearing", list(yes_no_map.keys()))
    DifficultyConcentrating = st.radio("Difficulty Concentrating", list(yes_no_map.keys()))
    DifficultyWalking = st.radio("Difficulty Walking", list(yes_no_map.keys()))
    DifficultyDressingBathing = st.radio("Difficulty Dressing/Bathing", list(yes_no_map.keys()))
    SmokerStatus = st.selectbox("Smoker Status", list(smoker_map.keys()))
    ChestScan = st.radio("Chest Scan", list(yes_no_map.keys()))
    RaceEthnicityCategory = st.selectbox("Race/Ethnicity", list(race_map.keys()))
    AgeCategory = st.selectbox("Age Category", list(age_map.keys()))
    BMI = st.slider("BMI", 10.0, 50.0, 22.0)
    AlcoholDrinkers = st.radio("Alcohol Drinkers", list(yes_no_map.keys()))

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode the inputs
    input_data = [
        sex_map[Sex],
        GeneralHealth,
        PhysicalHealthDays,
        MentalHealthDays,
        last_checkup_map[LastCheckupTime],
        yes_no_map[PhysicalActivities],
        SleepHours,
        removed_teeth_map[RemovedTeeth],
        yes_no_map[HadAngina],
        yes_no_map[HadStroke],
        yes_no_map[HadAsthma],
        yes_no_map[HadSkinCancer],
        yes_no_map[HadCOPD],
        yes_no_map[HadDepressiveDisorder],
        yes_no_map[HadKidneyDisease],
        yes_no_map[HadArthritis],
        diabetes_map[HadDiabetes],
        yes_no_map[DeafOrHardOfHearing],
        yes_no_map[DifficultyConcentrating],
        yes_no_map[DifficultyWalking],
        yes_no_map[DifficultyDressingBathing],
        smoker_map[SmokerStatus],
        yes_no_map[ChestScan],
        race_map[RaceEthnicityCategory],
        age_map[AgeCategory],
        BMI,
        yes_no_map[AlcoholDrinkers]
    ]

    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Results")
    st.write("Risk of Cardiovascular Disease:", "Yes, High Cardiovascular Risk" if prediction == 1 else "No, Low Cardiovascular Risk")
    st.write(f"Predicted Probability: {prediction_proba:.2%}")
