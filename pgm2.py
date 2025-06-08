# Import libraries
#Exploratory Data Analysis (EDA) "Visualize datasets using Matplotlib and Seaborn, identify trends, outliers, and correlations"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load dataset
df = sns.load_dataset('titanic')  # Built-in dataset from seaborn

# 1. Basic Information
print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# 2. Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Distribution Plots
# Age distribution
plt.figure()
sns.histplot(df['age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 4. Categorical Counts
# Class counts
plt.figure()
sns.countplot(x='class', data=df, palette='Set2')
plt.title('Passenger Class Distribution')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# 5. Boxplot for Outliers
# Age by Class
plt.figure()
sns.boxplot(x='class', y='age', data=df, palette='Set3')
plt.title('Age Distribution by Passenger Class')
plt.show()

# 6. Trends (if time series data is available, here just simulated)
# Survival by Age
plt.figure()
sns.lineplot(x='age', y='survived', data=df)
plt.title('Survival Trend by Age')
plt.xlabel('Age')
plt.ylabel('Survival Rate')
plt.show()

# 7. Correlation Heatmap
# Select only numerical columns for correlation
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numerical_cols].corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# 8. Scatter Plot (example: Fare vs Age)
plt.figure()
sns.scatterplot(x='age', y='fare', hue='survived', data=df, palette='deep')
plt.title('Fare vs Age (colored by Survival)')
plt.show()
