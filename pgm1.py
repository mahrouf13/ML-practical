#1. Data Preprocessing:"Load a dataset, handle missing values, encode categorical data, and normalize/standardize features."
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample dataset
data = pd.DataFrame({
    'Age': [25, np.nan, 35, 40, 29],
    'Salary': [50000, 60000, np.nan, 80000, 52000],
    'Department': ['Sales', 'Engineering', 'HR', np.nan, 'Sales'],
    'Purchased': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

# Display original dataset
print("Original Dataset:\n")
print(data)

# Separate features and target
X = data.drop('Purchased', axis=1)
y = data['Purchased']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Define preprocessing for numerical data (imputation + standardization)
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical data (imputation + one-hot encoding)
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Fit and transform the features
X_processed = preprocessor.fit_transform(X)

# Display processed features
print("\nProcessed Features (after handling missing values, encoding, and scaling):\n")
print(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)

# Optionally, show the transformed feature names
encoded_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + encoded_feature_names.tolist()

print("\nProcessed Feature Names:")
print(all_feature_names)
