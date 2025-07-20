import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
with open("logreg_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the correct order of features
FEATURE_COLUMNS = [
    "Sex", "GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "LastCheckupTime",
    "PhysicalActivities", "SleepHours", "RemovedTeeth", "HadAngina",
    "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD", "HadDepressiveDisorder",
    "HadKidneyDisease", "HadArthritis", "HadDiabetes", "DeafOrHardOfHearing",
    "DifficultyConcentratingt", "DifficultyWalking", "DifficultyDressingBathing",
    "SmokerStatus", "ChestScan", "RaceEthnicityCategory", "AgeCategory", "BMI", "AlcoholDrinkers"
]

# Streamlit app
st.title("Cardiovascular Risk Calculator")
st.markdown("This app uses a logistic regression model to predict the risk of a cardiovascular event based on survey data.")

# Input form
with st.form("input_form"):
    Sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
    GeneralHealth = st.slider("General Health (1 = Excellent, 5 = Poor)", 1, 5, 3)
    PhysicalHealthDays = st.slider("Physical Health (past 30 days)", 0, 30, 0)
    MentalHealthDays = st.slider("Mental Health (past 30 days)", 0, 30, 0)
    LastCheckupTime = st.selectbox("Last Checkup Time", [1, 2, 3, 4])
    PhysicalActivities = st.selectbox("Physical Activities", [0, 1])
    SleepHours = st.slider("Sleep Hours", 0, 24, 7)
    RemovedTeeth = st.selectbox("Number of Teeth Removed", [0, 1, 2, 3, 4])
    HadAngina = st.selectbox("Had Angina", [0, 1])
    HadStroke = st.selectbox("Had Stroke", [0, 1])
    HadAsthma = st.selectbox("Had Asthma", [0, 1])
    HadSkinCancer = st.selectbox("Had Skin Cancer", [0, 1])
    HadCOPD = st.selectbox("Had COPD", [0, 1])
    HadDepressiveDisorder = st.selectbox("Had Depressive Disorder", [0, 1])
    HadKidneyDisease = st.selectbox("Had Kidney Disease", [0, 1])
    HadArthritis = st.selectbox("Had Arthritis", [0, 1])
    HadDiabetes = st.selectbox("Had Diabetes", [0, 1])
    DeafOrHardOfHearing = st.selectbox("Deaf or Hard of Hearing", [0, 1])
    DifficultyConcentratingt = st.selectbox("Difficulty Concentrating", [0, 1])
    DifficultyWalking = st.selectbox("Difficulty Walking", [0, 1])
    DifficultyDressingBathing = st.selectbox("Difficulty Dressing/Bathing", [0, 1])
    SmokerStatus = st.selectbox("Smoker Status", [0, 1, 2])
    ChestScan = st.selectbox("Chest Scan", [0, 1])
    RaceEthnicityCategory = st.selectbox("Race/Ethnicity", [0, 1, 2, 3, 4, 5, 6])
    AgeCategory = st.selectbox("Age Category", list(range(1, 14)))
    BMI = st.slider("BMI", 10.0, 50.0, 22.0)
    AlcoholDrinkers = st.selectbox("Alcohol Drinkers", [0, 1])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Collect input in correct order
    input_data = [
        Sex, GeneralHealth, PhysicalHealthDays, MentalHealthDays, LastCheckupTime,
        PhysicalActivities, SleepHours, RemovedTeeth, HadAngina,
        HadStroke, HadAsthma, HadSkinCancer, HadCOPD, HadDepressiveDisorder,
        HadKidneyDisease, HadArthritis, HadDiabetes, DeafOrHardOfHearing,
        DifficultyConcentratingt, DifficultyWalking, DifficultyDressingBathing,
        SmokerStatus, ChestScan, RaceEthnicityCategory, AgeCategory, BMI, AlcoholDrinkers
    ]

    # Convert to DataFrame with correct column names
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    # Display results
    st.subheader("Prediction Results")
    st.write("Risk of Cardiovascular Disease:", "💔 Yes" if prediction == 1 else "✅ No")
    st.write(f"Predicted Probability: {prediction_proba:.2%}")
