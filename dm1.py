import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/Vedant/Downloads/industry.csv")
print("First five rows of the DataFrame:")
print(df.head())
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nSummary statistics for numerical columns:")
print(df.describe())
df.fillna(df.median(numeric_only=True), inplace=True)
print("\nMissing values after filling:")
print(df.isnull().sum())
threshold = 50
filtered_df = df[df['numerical_column'] > threshold]
print("\nFirst five rows of the filtered DataFrame:")
print(filtered_df.head())
group_stats = df.groupby('categorical_column').agg(['mean', 'std'])
print("\nMean and standard deviation for each group:")
print(group_stats)
df['new_column'] = df['column1'] / df['column2']
print("\nDataFrame with the new column:")
print(df.head())
correlation_matrix = df.corr()
print("\nCorrelation matrix for numerical columns:")
print(correlation_matrix)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='column1', y='column2', data=df)
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('Scatter plot between Column 1 and Column 2')
plt.show()
