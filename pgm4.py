# Logistic Regression ,Objective: Classify emails as spam or not using a Spam dataset.
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# 1. Load the dataset
file_path = "spam.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"'{file_path}' not found. Please make sure the file is in the correct directory.")

# Read the CSV
df = pd.read_csv(file_path)

# Check for the target column
if 'spam' not in df.columns:
    raise ValueError("The dataset must contain a 'spam' column as the target label (1 = spam, 0 = not spam).")

# Check for non-numeric features
if not all(df.drop('spam', axis=1).dtypes.apply(lambda dt: pd.api.types.is_numeric_dtype(dt))):
    raise ValueError("All features must be numeric. Please preprocess text or categorical data first.")

# Preview the dataset
print("Dataset Preview:")
print(df.head())

# 2. Prepare features and labels
X = df.drop('spam', axis=1)  # Features
y = df['spam']               # Target

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to avoid convergence warnings
model.fit(X_train_scaled, y_train)

# 6. Predictions
y_pred = model.predict(X_test_scaled)

# 7. Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
