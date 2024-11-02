import pandas as pd
df = pd.read_csv("C:/Users/Vedant/Downloads/titanic.csv")
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
target = df.Survived
inputs = df.drop('Survived',axis='columns')
dummies = pd.get_dummies(inputs.Sex)
inputs = pd.concat([inputs,dummies],axis='columns')
inputs.drop(['Sex','male'],axis='columns',inplace=True)
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)
model.predict(X_test[:10])
model.predict_proba(X_test[:10])
from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(),X_train, y_train, cv=5)