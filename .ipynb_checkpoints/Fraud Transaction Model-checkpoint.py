import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
import joblib

# Set the working directory to the project folder
import os
os.chdir('C:/Users/khawa/Models/Fraud Transaction Model')

# Load the dataset
df = pd.read_csv('Fraud Detection.csv')

# Reduce the dataset size for faster processing (99% of data is discarded)
df_reduced, _ = train_test_split(df, test_size=0.99, stratify=df['isFraud'], random_state=42)

# Rename columns for clarity and consistency in feature names
df_reduced.rename(columns={
    "nameOrig": "customer_id", 
    "nameDest": "merchant_id", 
    "oldbalanceOrg": "old_balance", 
    "newbalanceOrig": "new_balance",
    "oldbalanceDest": "merchant_old_balance",
    "newbalanceDest": "merchant_new_balance"
}, inplace=True)

# Exploratory Data Analysis (EDA): Visualizing the distribution of transaction types
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=df_reduced)
plt.title('Distribution of Transaction Types')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.show()

# Visualize the distribution of transaction amounts
plt.figure(figsize=(8, 6))
sns.histplot(df_reduced['amount'], kde=True, bins=200)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Feature Engineering
# Extract day of the week and hour of the day from the 'step' variable
df_reduced['day_of_week'] = (df_reduced['step'] // 24) % 7 + 1
df_reduced['hour_of_day'] = df_reduced['step'] % 24

# Categorize transactions based on amount
df_reduced['transaction_category'] = pd.cut(df_reduced['amount'], bins=[0, 50, 500, 5000, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])

# Calculate customer statistics like average transaction amount and transaction count
customer_stats = df_reduced.groupby("customer_id").agg(
    avg_transaction_amount=("amount","mean"),
    transaction_count=("customer_id","count")
).reset_index()

# Merge customer statistics with the original data
df_reduced = df_reduced.merge(customer_stats, on="customer_id", how="left")

# Create interaction feature between transaction amount and type
df_reduced['amount_type_interaction'] = df_reduced['amount'] * df_reduced['type'].factorize()[0]

# One-hot encode the 'type' feature (transaction type)
df_reduced = pd.get_dummies(df_reduced, columns=["type"])

# Data scaling: Convert the columns to numeric and fill missing values
df_reduced['amount'] = pd.to_numeric(df_reduced['amount'], errors='coerce').fillna(0)
df_reduced['old_balance'] = pd.to_numeric(df_reduced['old_balance'], errors='coerce').fillna(0)
df_reduced['new_balance'] = pd.to_numeric(df_reduced['new_balance'], errors='coerce').fillna(0)
df_reduced['merchant_old_balance'] = pd.to_numeric(df_reduced['merchant_old_balance'], errors='coerce').fillna(0)
df_reduced['merchant_new_balance'] = pd.to_numeric(df_reduced['merchant_new_balance'], errors='coerce').fillna(0)

# Apply log transformation to the amount and balance features to reduce skewness
df_reduced['log_amount'] = np.log1p(df_reduced['amount'])
df_reduced['log_old_balance'] = np.log1p(df_reduced['old_balance'])
df_reduced['log_new_balance'] = np.log1p(df_reduced['new_balance'])
df_reduced['log_merchant_old_balance'] = np.log1p(df_reduced['merchant_old_balance'])
df_reduced['log_merchant_new_balance'] = np.log1p(df_reduced['merchant_new_balance'])

# Encoding categorical variables (like customer_id, merchant_id, and transaction category)
le = LabelEncoder()
df_reduced[["customer_id", "merchant_id", 'transaction_category', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
            'type_PAYMENT', 'type_TRANSFER']] = df_reduced[["customer_id", "merchant_id", 'transaction_category', 
                                                            'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
                                                            'type_PAYMENT', 'type_TRANSFER']].apply(lambda col: le.fit_transform(col))

# Split data into features (X) and target variable (y)
X = df_reduced.drop("isFraud", axis=1)
y = df_reduced["isFraud"]

# Split data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTEENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbor) to handle class imbalance
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# GridSearchCV (Optional) - Search for optimal hyperparameters for the Random Forest model
param_grid = {'max_depth': [5, 10, 15], 'n_estimators': [50, 100, 150]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='average_precision')
grid_search.fit(X_train, y_train)

# Train the Random Forest model with the optimal hyperparameters
model = RandomForestClassifier(max_depth=15, n_estimators=150, class_weight="balanced")
model.fit(X_resampled, y_resampled)

# Evaluate the model's performance using ROC AUC score
test_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("Test set score:", test_score)

# Visualize feature importances (which features are most important for the model's decisions)
feature_importances = model.feature_importances_
features = X_train.columns

plt.figure(figsize=(12, 6))
plt.barh(features, feature_importances)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Confusion matrix to visualize model performance on test data
cm = confusion_matrix(y_test, model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model using joblib
joblib.dump(model, 'fraud_transaction_model.pkl')





