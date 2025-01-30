import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from pymongo import MongoClient

# Define feature names for better readability
FEATURE_NAMES = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

# Load dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Splitting features and labels
X = data[FEATURE_NAMES]
Y = data['DEATH_EVENT']

# Split data into train and test sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train logistic regression model
logit = LogisticRegression(max_iter=5000)
logit.fit(train_x, train_y)

# Save model to pickle
with open('logit_regression_model.pkl', 'wb') as model_file:
    pickle.dump(logit, model_file)

# Model predictions on test data
predictions = logit.predict(test_x)
score = logit.score(test_x, test_y)

# Print results
print("\nHeart Disease Failure Prediction and Classification")
print(f"Model Accuracy: {score:.2%}")

# Visualization
sns.set_theme()  # This will set the seaborn style

# Confusion matrix
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(test_y, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
probs = logit.predict_proba(test_x)[:, 1]
fpr, tpr, thresholds = roc_curve(test_y, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# User input prediction
print("\nEnter patient information:")
user_input = {}
for feature in FEATURE_NAMES:
    while True:
        try:
            value = float(input(f"Enter {feature}: "))
            user_input[feature] = value
            break
        except ValueError:
            print("Please enter a valid number")

# Convert user input to DataFrame for prediction
user_input_df = pd.DataFrame([user_input])
predicted_output = logit.predict(user_input_df)
prediction_prob = logit.predict_proba(user_input_df)[0]

# Print predicted output with probability
print("\nPrediction Results:")
print(f"Death Event Prediction: {'Yes' if predicted_output[0] == 1 else 'No'}")
print(f"Probability: {max(prediction_prob):.2%}")

