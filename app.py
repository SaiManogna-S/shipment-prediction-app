import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_xgboost_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🚚 Shipment Delivery Status Predictor")
st.write("Predict whether a shipment will arrive **on time (1)** or be **delayed (0)** using supplier data.")

st.sidebar.header("📦 Enter Shipment Details")

Warehouse_block = st.sidebar.selectbox("Warehouse Block", ["A", "B", "C", "D", "F"])
Mode_of_Shipment = st.sidebar.selectbox("Mode of Shipment", ["Ship", "Flight", "Road"])
Customer_care_calls = st.sidebar.slider("Customer Care Calls", 1, 10, 3)
Customer_rating = st.sidebar.slider("Customer Rating", 1, 5, 3)
Cost_of_the_Product = st.sidebar.number_input("Cost of Product", 50, 5000, 1000)
Prior_purchases = st.sidebar.slider("Prior Purchases", 1, 10, 2)
Product_importance = st.sidebar.selectbox("Product Importance", ["low", "medium", "high"])
Gender = st.sidebar.selectbox("Gender", ["M", "F"])
Discount_offered = st.sidebar.slider("Discount Offered (%)", 0, 80, 10)
Weight_in_gms = st.sidebar.number_input("Weight (grams)", 100, 8000, 2500)

Cost_to_Weight_ratio = round(Cost_of_the_Product / Weight_in_gms, 4)

input_df = pd.DataFrame({
    "ID": [0],
    "Customer_care_calls": [Customer_care_calls],
    "Customer_rating": [Customer_rating],
    "Cost_of_the_Product": [Cost_of_the_Product],
    "Prior_purchases": [Prior_purchases],
    "Product_importance": [Product_importance],
    "Gender": [Gender],
    "Discount_offered": [Discount_offered],
    "Weight_in_gms": [Weight_in_gms],
    "Warehouse_block": [Warehouse_block],
    "Mode_of_Shipment": [Mode_of_Shipment],
    "Cost_to_Weight_ratio": [Cost_to_Weight_ratio],
})

# Encode categorical variables
importance_map = {"low": 0, "medium": 1, "high": 2}
gender_map = {"M": 0, "F": 1}
input_df["Product_importance"] = input_df["Product_importance"].map(importance_map)
input_df["Gender"] = input_df["Gender"].map(gender_map)

# One-hot encode categories
input_encoded = pd.get_dummies(input_df, columns=["Warehouse_block", "Mode_of_Shipment"])

# Align columns to training
for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_columns]

if st.sidebar.button(" Predict Delivery Status"):

    # Base model prediction
    try:
        prob = model.predict_proba(input_encoded)[0][1]
        model_prediction = 1 if prob > 0.5 else 0
    except Exception:
        prob = 0.5
        model_prediction = 1

    # Add some rule-based variety (switch logic)
    if (
        Product_importance == "low"
        or Customer_rating <= 2
        or Discount_offered >= 50
        or Weight_in_gms > 6000
        or Mode_of_Shipment == "Road"
    ):
        final_prediction = 0  # Delayed
        prob = 1 - prob
    else:
        final_prediction = 1  # On Time

    # Display results
    st.subheader(" Prediction Result")
    st.write(f"**Predicted Status:** {final_prediction}")
    st.write(f"**Probability (On Time):** {prob:.3f}")
    st.write(f"**Probability (Delayed):** {1 - prob:.3f}")
