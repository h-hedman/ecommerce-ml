# Author: Hedman
# Project: Prediction model of revenue conversion in e-commerce
# Date: 2025-02-20
# --------------------------------------------------------------------------------------------------------------
# DESCRIPTION: 
## This model predicts revenue conversion in e-commerce using synthetic data. 
## It leverages features such as impressions, clicks, and past conversions with a logistic regression framework. 
## Techniques like class resampling, cross-validation, threshold tuning, and advanced evaluation (ROC, calibration curves)
## are applied to optimize and assess performance.
#
# OBJECTIVE: 
## Develop and evaluate a predictive model that accurately forecasts revenue conversion, 
## addressing class imbalance and fine-tuning decision thresholds to improve prediction for both conversion and non-conversion outcomes.
#
# Hypothesis:
## H0: Impressions, clicks, and past conversion history do not significantly influence revenue conversion.
## H1: Higher impressions, increased clicks, and positive past conversion history significantly boost the likelihood of revenue conversion.
# --------------------------------------------------------------------------------------------------------------
# Libraries
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
# --------------------------------------------------------------------------------------------------------------
# Generate Synthetic Data
np.random.seed(91)  # Ensures reproducibility

num_samples = 2000
impressions = np.random.randint(100, 10000, size=num_samples)
clicks = np.random.randint(1, 500, size=num_samples)
past_conversions = np.random.randint(0, 2, size=num_samples)  # 0 or 1

# Generate labels based on clicks and past conversions
conversion_probability = 1 / (1 + np.exp(-0.0002 * (clicks * impressions) - 0.5 * past_conversions))
revenue_conversion = (conversion_probability > np.random.rand(num_samples)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Impressions': impressions,
    'Clicks': clicks,
    'Past_Conversions': past_conversions,
    'Revenue_Conversion': revenue_conversion
})

# Check Class Distribution Before Resampling
print("Original Class Distribution:\n", df['Revenue_Conversion'].value_counts())

# Handle Class Imbalance (Manually Resampling More "0s")
df_majority = df[df['Revenue_Conversion'] == 1]
df_minority = df[df['Revenue_Conversion'] == 0]

# Ensure at least 10 samples for class "0"
if len(df_minority) < 10:
    df_minority = resample(df_minority, replace=True, n_samples=10, random_state=42)

df_balanced = pd.concat([df_majority, df_minority])

# Prepare Data for Training
X = df_balanced[['Impressions', 'Clicks', 'Past_Conversions']]
y = df_balanced['Revenue_Conversion']
print("Balanced Class Distribution:\n", y.value_counts())

# Split Dataset with Stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize Features (done within the pipeline later)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # For initial evaluation if needed
X_test_scaled = scaler.transform(X_test)

# Compute Class Weights for Logistic Regression
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weights_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class Weights:", weights_dict)
# --------------------------------------------------------------------------------------------------------------
# Advanced Modeling with Pipeline and GridSearchCV
# Create a pipeline that includes polynomial feature expansion, scaling, and logistic regression
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(class_weight=weights_dict, solver='liblinear', max_iter=1000))
])

# Set up hyperparameter grid for logistic regression
param_grid = {
    'log_reg__C': [0.01, 0.1, 1, 10],
    'log_reg__penalty': ['l1', 'l2']
}
# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters from Grid Search:", grid_search.best_params_)

# Evaluate on test data
y_pred = grid_search.predict(X_test)
y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", 
            xticklabels=["No Conversion", "Conversion"], yticklabels=["No Conversion", "Conversion"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --------------------------------------------------------------------------------------------------------------
# Cross-Validation
cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# --------------------------------------------------------------------------------------------------------------
# Advanced Evaluation: ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.show()

# Advanced Evaluation: Calibration Curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration Curve")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Mean Predicted Value")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.show()
# --------------------------------------------------------------------------------------------------------------
# Tune decision threshold for better model performance
adjusted_threshold = 0.3
y_pred_adjusted = (y_pred_proba > adjusted_threshold).astype(int)
print("\nAdjusted Classification Report:\n", classification_report(y_test, y_pred_adjusted))
# --------------------------------------------------------------------------------------------------------------
