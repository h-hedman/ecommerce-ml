# Author: Hedman
# Project: Random Forest Classifier Example
# Date: 2025-02-24
# Description: A machine learning model using Random Forest to predict customer purchases on an e-commerce platform.
# --------------------------------------------------------------------------------------------------------------
# Project Aim:
# The aim of this study is to develop and evaluate a Random Forest Classifier to predict whether a customer 
# makes a purchase on an e-commerce platform based on demographic and behavioral factors. 
# Model outcomes are compared with LightGMB and XGBoost.
#
# Research Objectives:
# 1. Identify key predictors of customer purchases.
# 2. Assess the accuracy and performance of the model.
# 3. Understand the impact of customer age, session duration, and promotion engagement on purchase decisions.
#
# Hypothesis:
# H₀: There is no significant relationship between customer features and purchasing behavior.
# H₁: Customer attributes (age, session duration, and promotion engagement) significantly influence purchases.
# --------------------------------------------------------------------------------------------------------------
# Libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# --------------------------------------------------------------------------------------------------------------
# Set random seed for reproducibility
np.random.seed(888)
random.seed(42)
fake = Faker()

# Define the number of samples
num_samples = 1500

# Generate synthetic customer data using Faker
data = {
    'customer_id': [fake.unique.uuid4() for _ in range(num_samples)],  # Unique Customer ID
    'customer_age': np.random.randint(18, 70, num_samples),  # Age between 18 and 75
    'session_duration_minutes': np.random.randint(1, 60, num_samples),  # Time spent on site (1-60 minutes)
    'promotion_engagement': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),  # 40% click promotion
    'num_items_viewed': np.random.randint(1, 20, num_samples),  # Number of products viewed (1-20)
    'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], num_samples, p=[0.5, 0.3, 0.2]),  # 50% mobile users
    'cart_abandonment_rate': np.random.uniform(0, 1, num_samples),  # Cart abandonment rate (0-100%)
    'weekend_visit': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),  # 30% visits on weekends
    'user_tier': np.random.choice(['new', 'returning', 'premium'], num_samples, p=[0.4, 0.4, 0.2]),  # 40% new users
    'purchase_history': np.random.randint(0, 10, num_samples),  # Number of previous purchases (0-9)
    'total_spent': np.random.uniform(5, 500, num_samples),  # Total amount spent ($5 - $500)
    'customer_region': np.random.choice(['North America', 'Europe', 'Asia', 'Other'], num_samples, p=[0.4, 0.3, 0.2, 0.1]), # Region
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Generate "Purchase Made" based on realistic conditions
df['purchase_made'] = np.where(
    (df['promotion_engagement'] == 1) & 
    (df['num_items_viewed'] > 5) & 
    (df['cart_abandonment_rate'] < 0.5) & 
    (df['total_spent'] > 50), 1, 0
)

# One-Hot Encode Categorical Variables
df = pd.get_dummies(df, columns=['device_type', 'user_tier', 'customer_region'], drop_first=True)

# Drop customer_id (not a useful feature for prediction)
df = df.drop(columns=['customer_id'])

# Display new class distribution
print("Updated Class Distribution:\n", df['purchase_made'].value_counts())

# Save expanded dataset (Optional)
#df.to_csv("ecommerce_data_expanded.csv", index=False)
# --------------------------------------------------------------------------------------------------------------
# Train-Test Split (Stratified to maintain class balance)
feature_columns = [col for col in df.columns if col != 'purchase_made']
target_column = 'purchase_made'

features = df[feature_columns]  # Predictor variables
target = df[target_column]  # Target variable

train_features, test_features, train_labels, test_labels = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=91)

# Train the model on the training data
rf_model.fit(train_features, train_labels)

# Make predictions on the test set
predicted_purchases = rf_model.predict(test_features)
# --------------------------------------------------------------------------------------------------------------
# Evaluate Model Performance
accuracy = accuracy_score(test_labels, predicted_purchases)
conf_matrix = confusion_matrix(test_labels, predicted_purchases)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)

# Display classification report for detailed metrics
print("\nClassification Report:")
print(classification_report(test_labels, predicted_purchases))\
# --------------------------------------------------------------------------------------------------------------
# Feature Importance Analysis
feature_importance = rf_model.feature_importances_
features_list = feature_columns

# Plot feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance, y=features_list, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()
# --------------------------------------------------------------------------------------------------------------
# Increasing min_split_gain (e.g., to 0.1) makes the model more conservative about splitting.
lgbm_model = LGBMClassifier(random_state=91, min_split_gain=0.1)

# Train the LightGBM model on the training data.
lgbm_model.fit(train_features, train_labels)

# Make predictions on the test set using LightGBM.
lgbm_preds = lgbm_model.predict(test_features)

# Evaluate LightGBM performance.
lgbm_accuracy = accuracy_score(test_labels, lgbm_preds)
lgbm_conf_matrix = confusion_matrix(test_labels, lgbm_preds)
lgbm_classification_report = classification_report(test_labels, lgbm_preds)

print("\n----- LightGBM Performance with Increased min_split_gain -----")
print(f"LightGBM Accuracy: {lgbm_accuracy * 100:.2f}%")
print("LightGBM Confusion Matrix:")
print(lgbm_conf_matrix)
print("\nLightGBM Classification Report:")
print(lgbm_classification_report)
# --------------------------------------------------------------------------------------------------------------
# XGBoost Model Comparison
# Initialize the XGBoost model.
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=91)

# Train the XGBoost model on the training data.
xgb_model.fit(train_features, train_labels)

# Make predictions on the test set using XGBoost.
xgb_preds = xgb_model.predict(test_features)

# Evaluate XGBoost performance.
xgb_accuracy = accuracy_score(test_labels, xgb_preds)
xgb_conf_matrix = confusion_matrix(test_labels, xgb_preds)
xgb_classification_report = classification_report(test_labels, xgb_preds)

print("\n----- XGBoost Performance -----")
print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")
print("XGBoost Confusion Matrix:")
print(xgb_conf_matrix)
print("\nXGBoost Classification Report:")
print(xgb_classification_report)
# --------------------------------------------------------------------------------------------------------------
