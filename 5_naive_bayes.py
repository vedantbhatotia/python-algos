import pandas as pd
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"C:\Users\Vedant\Downloads\titanic.csv")

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Fare'].fillna(df['Fare'].median(), inplace=True)

df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})


df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = df.drop(['Embarked'],axis=1)

y = df.Embarked


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=50)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(GaussianNB(),X,y,cv=10)

print("scores : ",scores)
print("average : ",np.average(scores))

model = GaussianNB()


model.fit(X_train,y_train)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,model.predict(X_test)))

print(classification_report(y_test,model.predict(X_test)))