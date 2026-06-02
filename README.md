# 🚚 Shipment Delivery Status Prediction App

### 📦 Overview
This Streamlit web app predicts whether a shipment will be **delivered on time (1)** or **delayed (0)** based on supplier and order details.  
It uses a trained **XGBoost machine learning model** built from supplier shipment data to provide real-time predictions.

---

### ⚙️ Tech Stack
- **Python**
- **Streamlit** – for the interactive web interface  
- **Pandas** – data manipulation  
- **XGBoost** – ML model for prediction  
- **Joblib** – model loading and management  
- **Scikit-learn** – preprocessing and encoding  

---

### 📊 Dataset
The dataset includes features such as:
- Warehouse Block  
- Mode of Shipment  
- Customer Care Calls  
- Customer Rating  
- Cost of Product  
- Prior Purchases  
- Product Importance  
- Gender  
- Discount Offered  
- Weight in grams  

Each record is labeled with the target variable:
- **1 → On Time**
- **0 → Delayed**

---

### 🧠 Model
The model used is an **XGBoost Classifier**, trained with optimized hyperparameters.  
It predicts delivery status based on both numeric and categorical features.  
The `best_xgboost_model.pkl` and `feature_columns.pkl` files store the trained model and the feature layout.

---

### 💻 How to Use
1. Visit the live app below 👇  
   🔗 **(https://saimanogna-s-shipment-prediction-app-app-9b7bnt.streamlit.app/)**
2. Select shipment details in the sidebar (Warehouse Block, Mode, Rating, etc.)  
3. Click **“Predict Delivery Status”**  
4. View predicted delivery status and probabilities!

---


