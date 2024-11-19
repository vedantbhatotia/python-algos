import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\Vedant\Downloads\housing.csv")
relevant_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 
                    'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']
df = df[relevant_columns]
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']
for column in binary_columns:
    df[column] = df[column].apply(lambda x: 1 if x == 'yes' else 0)
df = df.dropna()
X = df.drop(columns=['price'])
y = df['price']
X = (X - X.mean()) / X.std()
X = np.array(X)
y = np.array(y)
w = np.zeros(X.shape[1]) 
b = 0 

alpha = 0.01
iterations = 1000 
m = len(y) 
losses = []
for i in range(iterations):
    y_pred = np.dot(X, w) + b    
    loss = (1/m) * np.sum((y_pred - y) ** 2)
    losses.append(loss)    
    dw = (2/m) * np.dot(X.T, (y_pred - y))
    db = (2/m) * np.sum(y_pred - y)    
    w = w - alpha * dw
    b = b - alpha * db    


y_pred_final = np.dot(X, w) + b
plt.scatter(y, y_pred_final)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()
