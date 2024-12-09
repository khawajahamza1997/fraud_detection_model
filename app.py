import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load your pre-trained model
model = joblib.load('fraud_transaction_model.pkl')

# Custom CSS for consistent theming
css = """
    <style>
        /* Set the background color for the main area */
        .css-1d391kg { 
            background-color: #192841 !important; /* Dark blue for sidebar */
            color: white !important; /* White text in sidebar */
        }

        /* Set the background color for the page content */
        .css-1lcbmhc, .css-1a1i4j3 {
            background-color: #FFFFFF !important; /* White background for main content */
            color: #192841 !important; /* Dark text for main content */
        }

        /* Sidebar header */
        .css-1m6r2kl { 
            color: white !important; /* White text in sidebar header */
        }

        /* Feature Importance Bar Graph */
        .stSidebar .stBlock {
            background-color: #192841 !important; /* Blue background for sidebar block */
            color: white !important; /* White text */
        }

        /* Override button styles */
        .stButton>button {
            background-color: #192841 !important;
            color: white !important;
            border: none;
            padding: 10px 20px;
        }

        /* Modify other button styles */
        .stButton>button:hover {
            background-color: #0e1b2a !important;
        }

        /* Override the slider's color */
        .stSlider>div>div>div>input {
            background-color: #192841 !important;
        }

        /* Customize the form input field text */
        input, select, textarea {
            color: #192841 !important;
            background-color: #FFFFFF !important;
        }
    </style>
"""

# Inject CSS into the app
st.markdown(css, unsafe_allow_html=True)

# Sidebar with graphs
st.sidebar.header("Features With Maximum Contribution To Model Accuracy")

# Example: Feature Importance Graph
feature_importance = {
    "amount": 0.4,
    "old_balance": 0.15,
    "new_balance": 0.04,
    "day_of_week": 0.02,
    "hour_of_day": 0.08,
    "transaction_category": 0.2,
    "amount_type_interaction": 0.08,
}

# Plotting feature importance
fig, ax = plt.subplots(figsize=(5, 8))
sns.barplot(x=list(feature_importance.values()), y=list(feature_importance.keys()), ax=ax, palette="Blues_r")
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Feature Importance')
st.sidebar.pyplot(fig)

# Streamlit app title
st.title("Transaction Fraud Prediction App")
st.markdown("### Predict whether a transaction is fraudulent or legitimate")

# Input form for transaction details
with st.form("transaction_form"):
    st.header("Enter Transaction Details")
    
    # Step input with description
    step = st.slider("Select Step (Transaction Sequence)", min_value=1, max_value=100, value=1)
    
    # Numeric inputs for balances and amounts
    amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0, step=0.01)
    old_balance = st.number_input("Old Balance", min_value=0.0, value=0.0, step=0.01)
    new_balance = st.number_input("New Balance", min_value=0.0, value=0.0, step=0.01)
    merchant_old_balance = st.number_input("Merchant Old Balance", min_value=0.0, value=0.0, step=0.01)
    merchant_new_balance = st.number_input("Merchant New Balance", min_value=0.0, value=0.0, step=0.01)
    customer_id = st.number_input("Customer ID", min_value=0.0)
    merchant_id = st.number_input("Merchant ID", min_value=0.0)
    
    # Other transaction details
    isFlaggedFraud = st.checkbox("Transaction flagged as fraud?", value=False)
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    hour_of_day = st.slider("Hour of Day", min_value=0, max_value=23, value=12)
    
    # Modify transaction_category to allow "Other"
    transaction_category = st.selectbox(
        "Transaction Category", 
        ["Groceries", "Electronics", "Apparel", "Restaurants", "Travel", "Other"]
    )
    if transaction_category == "Other":
        transaction_category = st.text_input("Specify Other Transaction Category", "")
    
    avg_transaction_amount = st.number_input("Average Transaction Amount", min_value=0.0, value=0.0, step=0.01)
    transaction_count = st.number_input("Transaction Count", min_value=0, value=0, step=1)
    
    # amount_type_interaction as checkboxes
    amount_type_interaction = st.multiselect("Select Interaction Type(s)", ["High", "Medium", "Low"])
    
    # Transaction types as checkboxes
    type_CASH_IN = st.checkbox("Type: CASH_IN", value=False)
    type_CASH_OUT = st.checkbox("Type: CASH_OUT", value=False)
    type_DEBIT = st.checkbox("Type: DEBIT", value=False)
    type_PAYMENT = st.checkbox("Type: PAYMENT", value=False)
    type_TRANSFER = st.checkbox("Type: TRANSFER", value=False)
    
    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write("Form Submitted!")
