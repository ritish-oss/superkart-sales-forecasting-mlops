import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="🛒", layout="wide")

MODEL_PATH = "models/best_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

st.title("🛒 SuperKart Sales Forecasting Application")
st.markdown("Automated MLOps Revenue Predictor across Weekly, Monthly, Quarterly, and Annual Horizons.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Product Attributes")
    prod_weight = st.number_input("Product Weight (kg)", min_value=1.0, max_value=50.0, value=12.5, step=0.1)
    prod_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    prod_area = st.slider("Allocated Area Ratio", min_value=0.001, max_value=0.400, value=0.050, step=0.005)
    prod_type = st.selectbox("Product Category", [
        "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
        "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
        "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"
    ])
    prod_mrp = st.number_input("Product MRP ($)", min_value=10.0, max_value=500.0, value=140.0, step=1.0)

with col2:
    st.subheader("🏬 Store Attributes")
    est_year = st.number_input("Establishment Year", min_value=1980, max_value=2026, value=2010, step=1)
    store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    city_type = st.selectbox("Store City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Food Mart", "Departmental Store", "Supermarket Type 1", "Supermarket Type 2"])

input_df = pd.DataFrame([{
    "Product_Weight": prod_weight,
    "Product_Sugar_Content": prod_sugar,
    "Product_Allocated_Area": prod_area,
    "Product_Type": prod_type,
    "Product_MRP": prod_mrp,
    "Store_Size": store_size,
    "Store_Location_City_Type": city_type,
    "Store_Type": store_type,
    "Store_Age": 2026 - est_year
}])

st.markdown("---")
if st.button("Generate Sales Forecast", type="primary"):
    if model is not None:
        pred_monthly = model.predict(input_df)[0]
        st.success(f"### Predicted Monthly Sales: ${pred_monthly:,.2f}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Weekly Estimated Sales", f"${(pred_monthly / 4.33):,.2f}")
        m2.metric("Quarterly Estimated Sales", f"${(pred_monthly * 3):,.2f}")
        m3.metric("Annual Estimated Sales", f"${(pred_monthly * 12):,.2f}")

        with st.expander("Inspected Preprocessed Input Features"):
            st.dataframe(input_df)
    else:
        st.error("Model file models/best_model.pkl not found!")
