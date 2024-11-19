import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df0 = df[df.target == 0]  
df1 = df[df.target == 1] 
df2 = df[df.target == 2]

X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

print(f"Model Accuracy: {knn.score(X_test, y_test) * 100:.2f}%")

sample_prediction = knn.predict([[4.8, 3.0, 1.5, 0.3]])
print(f"Predicted class for [4.8, 3.0, 1.5, 0.3]: {iris.target_names[sample_prediction[0]]}")

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))
sn.heatmap(cm, annot=True, cmap='coolwarm', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix Heatmap")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(6, 4))
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green", marker='+', label='Setosa')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='.', label='Versicolor')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color="red", marker='x', label='Virginica')
plt.legend()
plt.title("Visualization of Predicted Classes")
plt.show()
