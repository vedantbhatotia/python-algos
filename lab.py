import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r"C:\Users\Vedant\Downloads\rename.csv")
X = df[['ABE', 'TJR', 'NV', 'DST', 'LMAC']]
y = df['OUT'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=['ABE', 'TJR', 'NV', 'DST', 'LMAC'], class_names=clf.classes_.astype(str), filled=True)
plt.title("Full Decision Tree")
plt.show()

clf_pruned = DecisionTreeClassifier(random_state=42, max_depth=5)
clf_pruned.fit(X_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(clf_pruned, feature_names=['ABE', 'TJR', 'NV', 'DST', 'LMAC'], class_names=clf_pruned.classes_.astype(str), filled=True)
plt.title("Pruned Decision Tree (Max Depth = 5)")
plt.show()

y_pred = clf_pruned.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


def predict_class(attribute_values):
    input_data = np.array(attribute_values).reshape(1, -1)
    prediction = clf_pruned.predict(input_data)
    return prediction[0]

example_input = [5, 2, 1, 0, 3]

predicted_class = predict_class(example_input)
print(f"Predicted class for input {example_input}: {predicted_class}")
