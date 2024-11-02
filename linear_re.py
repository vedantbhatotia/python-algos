import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
df = pd.read_csv(r"C:\Users\Vedant\Downloads\ex1data1.csv")
print(df.head())
x = np.array(df.X)
y = np.array(df.Y)
w = 0
b = 0

alpha = 0.001
m = len(x)
iterations = 1000

for i in range(iterations):
    y_pred = x * w + b
    dw = -(2/len(x)) * np.sum(x * (y - y_pred))
    db = -(2/len(x)) * np.sum(y - y_pred)
    w = w - alpha * dw
    b = b - alpha * db
    if i % 100 == 0:
        loss = np.mean((y_pred - y) ** 2)
        print(f"Iteration {i}: Loss = {loss}, w = {w}, b = {b}")

plt.scatter(x,y)
plt.plot(x,w*x+b)
plt.show();