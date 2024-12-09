This repository contains a fraud detection model built to identify fraudulent financial transactions. It uses machine learning, specifically a Random Forest Classifier to predict whether a transaction is fraudulent or not based on various transaction features.

Project Overview
The project involves building a machine learning model to detect fraudulent transactions in a dataset. The model is trained using features like transaction type, amount, balances, and customer details. It also applies techniques such as SMOTEENN for handling imbalanced data.

Key Steps in the Project:
Data Preprocessing: Data is cleaned and transformed with feature engineering techniques.
Model Training: The model is trained using a Random Forest Classifier.
Evaluation: Model evaluation is done using cross-validation, confusion matrices, and ROC-AUC scores.
Deployment: A Streamlit app is created to interact with the model and make predictions.
Files in the Repository
Fraud Transaction Model.ipynb: Jupyter notebook containing the data analysis, preprocessing, feature engineering, model training, and evaluation.
app.py: Streamlit app file that serves the model for prediction and interacts with users.
Fraud Detection.csv: Dataset containing the transaction data used to train the model.
Fraud Transaction Model.py: Python file that replicates the model's functionality from the notebook for use in production.
fraud_transaction_model.pkl: Serialized machine learning model saved using joblib, ready for deployment.
Requirements
To run this project locally, you need the following libraries:

streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
You can install the necessary dependencies by running:

Copy code
pip install -r requirements.txt
How to Run the Streamlit App
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/khawajahamza1997/fraud_detection_model.git
cd fraud_detection_model
Install the required dependencies:

Copy code
pip install -r requirements.txt
Run the Streamlit app:

arduino
Copy code
streamlit run app.py
The app will open in your web browser, and you can interact with the fraud detection model.

How the Model Works
The fraud detection model is a Random Forest Classifier, which is trained to predict whether a transaction is fraudulent (isFraud column). The model is trained on features like:

amount: Transaction amount.
old_balance: The balance of the customer before the transaction.
new_balance: The balance of the customer after the transaction.
merchant_old_balance: The merchant's balance before the transaction.
merchant_new_balance: The merchant's balance after the transaction.
type: Type of transaction (e.g., CASH_IN, CASH_OUT, etc.)
Important Notes:
The dataset has been preprocessed to handle missing values, categorical data, and balance the classes using SMOTEENN.
The model uses cross-validation and hyperparameter tuning to achieve the best performance.
Model Evaluation
The model is evaluated using the following metrics:

ROC-AUC score: Measures the ability of the model to distinguish between classes.
Confusion Matrix: Provides a visualization of the model's classification performance.
Feature Importance: Displays which features are most important for making predictions.
Deployment
The Streamlit app provides an interactive interface to make predictions using the trained fraud detection model. You can deploy this app to Streamlit Cloud or any other platform that supports Python web applications.

License
This project is licensed under the MIT License.

