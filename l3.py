import pandas as pd
import numpy as np
import matplotlib.pyplot
import seaborn as sns  
data=pd.read_csv('C:/Users/KIKE/OneDrive/Desktop/New folder/titanic.csv')
print(data.head())
print(data.isnull().sum())

# data preprocessing
print('Median of age column %.2f'% (data["Age"]. median(skipna = True)))
print('Percent of missing records in cabin %.2f' %((data["Cabin"].isnull().sum()/data.shape[0])*100))
print('Most common boarding point : %s' % data['Embarked'].value_counts().idxmax())
data ["Age"].fillna(data["Age"]. median(skipna = True), inplace = True)
data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(), inplace = True)
data.drop('Cabin', axis = 1, inplace = True)
print(data.isnull().sum())


data.drop('PassengerId', axis = 1, inplace = True)
data.drop('Name', axis = 1, inplace = True)
data.drop('Ticket', axis = 1, inplace = True)
data["TravelAlone"] = np.where((data["SibSp"]+data[ "Parch"]) > 8, 0, 1)
data.drop('SibSp', axis = 1, inplace = True)
data.drop('Parch', axis = 1, inplace = True)
print(data.head())


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data["Sex"]=label_encoder.fit_transform(data["Sex"])
data["Embarked"] =label_encoder.fit_transform(data["Embarked"])
print(data.head())



X = data[["Pclass", "Sex", "Age", "Fare", "Embarked", "TravelAlone"]]
Y = data["Survived"]
from sklearn. model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 2)
from sklearn. linear_model import LogisticRegression
Ir_model = LogisticRegression()
Ir_model.fit(X_train, Y_train)
y_pred = Ir_model.predict(X_test)
print(y_pred) 
