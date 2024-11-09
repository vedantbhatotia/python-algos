import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes  
diabetes = load_diabetes()  
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['TARGET'] = diabetes.target 
print(df.head()) #->1
print(df.isnull().sum()) #->2
df.describe()
df.fillna(df.median(), inplace=True)
threshold = 0.05
filtered_df = df[df['bmi'] > threshold]
print(filtered_df.head()) #->3
df['Target_Category'] = np.where(df['TARGET'] > 150, 'High Target', 'Low Target')

grouped_stats = df.groupby('Target_Category').agg(['mean', 'std'])
print(grouped_stats) #->4
# df_additional = df[['TARGET']].copy()
# df_additional['Additional_Info'] = np.random.rand(len(df))

# merged_df = pd.merge(df, df_additional, on='TARGET')
# print(merged_df.head())
