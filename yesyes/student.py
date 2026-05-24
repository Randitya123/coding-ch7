import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("C:/Users/KIKE/OneDrive/Desktop/New folder/yesyes/student-mat.csv")
# Check the first few rows
print(data.head())
print(data.describe())
print(data.info())

#Encode all categorical columns using LabelEncoder 
from sklearn.preprocessing import LabelEncoder
Label_encoder=LabelEncoder()
categorical_cols=['school', 'sex', 'address','famsize', 'Pstatus', 'Mjob','reason', 'guardian','Fjob','schoolsup','famsup', 'paid','activities',
'nursery', 'higher', 'internet', 'romantic']# Add 'address' to this list


for col in categorical_cols:
    data[col] = Label_encoder.fit_transform(data[col])

data.drop(['G1','G2'],axis=1,inplace=True)

X=data.drop('G3',axis=1)
Y=data['G3']

from sklearn.model_selection import train_test_split
X_test, X_train, Y_test, Y_train=train_test_split(X,Y,test_size=0.2, random_state=5)
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X_train,Y_train)

y_pred=regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse=mean_squared_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print('Mean Squared Error',mse)
print('R-squared:',r2)

plt.figure(figsize=(10,5))
plt.scatter(range(len(Y_test)),Y_test, color="blue",label="Actual")
plt.scatter(range(len(y_pred)),y_pred, color="red",label="Predicted")
plt.title("Actual vs Predicted")
plt.xlabel("Student Index")
plt.ylabel("Final Grade(G3)")
plt.legend()
plt.show()