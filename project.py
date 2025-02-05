# Install required libraries (comment these out in PyCharm if libraries are already installed)
#!pip install imbalanced-learn scikit-learn seaborn matplotlib
# !pip install scikit-learn --upgrade # Upgrade scikit-learn if needed

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle  # For saving the models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load data from the local path (change the path to where your dataset is located)
data = pd.read_csv("Fraud.csv")

# Basic data information
print(data.head())
print(data.shape)
print(data.isna().sum())  # Check for missing values
print(data.duplicated().sum())  # Check for duplicates
data.info()  # Display info about data types and null values

# Balancing the dataset
data1 = data[data["isFraud"] == 1]
data0 = data[data["isFraud"] == 0][:8213]  # Sampling to balance
new_data = pd.concat([data0, data1], axis=0)

# Visualize "isFlaggedFraud" and "isFraud" distributions
plt.pie(new_data["isFlaggedFraud"].value_counts(), labels=new_data["isFlaggedFraud"].unique(), autopct="%.2f")
plt.title("Is Flagged Fraud Distribution")
plt.show()

plt.pie(new_data["isFraud"].value_counts(), labels=new_data["isFraud"].unique(), autopct="%.2f")
plt.title("Is Fraud Distribution")
plt.show()

print("Fraud Counts:\n", data["isFraud"].value_counts())
print(new_data.info())

# Drop unnecessary columns
new_data = new_data[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']]

# Encode categorical 'type' feature
lb = LabelEncoder()
new_data["type"] = lb.fit_transform(new_data["type"])

# Define features (X) and target (y)
X = new_data.iloc[:, :-1]
y = new_data.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Save the Random Forest model
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(model_rf, file)

# Display feature importances
importances = model_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Model Predictions for Random Forest
ypred_rf = model_rf.predict(X_test)
print("The Accuracy with Random Forest is:", accuracy_score(ypred_rf, y_test))

# Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, ypred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print Random Forest metrics
print("Random Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_test, ypred_rf):.2f}")
print(f"Precision: {precision_score(y_test, ypred_rf):.2f}")
print(f"Recall: {recall_score(y_test, ypred_rf):.2f}")
print(f"F1 Score: {f1_score(y_test, ypred_rf):.2f}")

# Train the K-Nearest Neighbors model
model_knn = KNeighborsClassifier(n_neighbors=2)
model_knn.fit(X_train, y_train)

# Save the KNN model
with open("knn_model.pkl", "wb") as file:
    pickle.dump(model_knn, file)

# Predictions with KNN
ypred_knn = model_knn.predict(X_test)
print("The Accuracy with KNN is:", accuracy_score(ypred_knn, y_test))

# Confusion Matrix for KNN
conf_matrix_knn = confusion_matrix(y_test, ypred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues')
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print KNN metrics
print("K-Nearest Neighbors Metrics:")
print(f"Accuracy: {accuracy_score(y_test, ypred_knn):.2f}")
print(f"Precision: {precision_score(y_test, ypred_knn):.2f}")
print(f"Recall: {recall_score(y_test, ypred_knn):.2f}")
print(f"F1 Score: {f1_score(y_test, ypred_knn):.2f}")

# ROC Curve for both models
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
y_prob_knn = model_knn.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_knn, tpr_knn, color='orange', label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid()
plt.show()
