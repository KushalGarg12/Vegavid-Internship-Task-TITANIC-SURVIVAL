#importing libraries
import pandas as pd
import pickle
#importing dataset
df=pd.read_csv("titanic.csv")
df.head()
#remove unwanted columns 
df1=df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
df1.head()
#replacing missing values with mean
import math
value=math.floor(df1['Age'].mean())
df5=df1.fillna(value)
df5.head()
#making dummy variables of column SEX
dummy=pd.get_dummies(df.Sex)
dummy.head()

df2=pd.concat([df5,dummy],axis='columns')
df2.head()
df3=df2.drop(['Sex'],axis='columns')
df3.head()
in_var=df3.drop(['Survived'],axis='columns')
in_var.head()
target=df['Survived']
target.head()
#splitting training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(in_var,target,test_size=0.3,random_state=0)
x_test
#training and fitting data
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
model.score(x_test,y_test)
#repeating all the steps for the file test.csv for prediction
data=pd.read_csv("test.csv")
data.head()
data1=data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
data1.head()
value1=math.floor(data1['Age'].mean())
data2=data1.fillna(value1)
data2.head()
dummy1=pd.get_dummies(data2.Sex)
dummy1.head()
data2=pd.concat([data2,dummy1],axis='columns')
data2.head()
data2=data2.drop(['Sex'],axis='columns')
data2.head()
data2['Fare']=math.floor(data2['Fare'])

predictions=model.predict(data2)
#saving the pickle file for deploying 

pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))