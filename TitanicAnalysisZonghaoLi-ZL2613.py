## Titanic Analysis by Zonghao Li (ZL2613)

%matplotlib inline
# First import useful data handling libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning Algorithms
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Loading data 
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# Data checking and cleaning
train.info()

train.head()

train.describe()

#### Since PassengerId is just an id from 1 to 891 for each passenger. So we just ignore that column cause it would 
#### not affect our analysis.  
#### Most values for column Cabin were missing, and this variable should be closely related to Fare 
#### and Pclass, so we also delete that column. 

del train['Cabin']
del test['Cabin']
del train['PassengerId']

train.head()

train['Ticket'].describe()

train['Ticket'].head()

#### Variable Ticket has 681 Unique values and it contains both numeric values and Alphanumeric Values. So it would not 
#### help in the model. We could delete this column.

del train['Ticket']
del test['Ticket']

#### Now we want to check whether Pclass and Fare are highly correlated cause usually higher ticket class costs more.

pd.DataFrame.boxplot(train,column='Fare', by='Pclass')

#### We can see that Fare for Pclass1 is significantly higher than 2 and 3. But the difference between 2 and 3 are not
#### very significant. So I decide to keep both variables in our model. 

#### For variables SibSp and Parch, these are # of siblings / spouses aboard the Titanic and # of parents / children aboard 
#### the Titanic respectively. We could create a variable FamilyNumber to demonstrate the sum of these two.

train['FamilyNumber'] = train['SibSp'] + train['Parch']
test['FamilyNumber'] = test['SibSp'] + test['Parch']

####Change sex variable from 'male', 'female' into '1', '0' for modeling.
train['Sex'] = train['Sex'].map({'male':1,'female':0})
test['Sex'] = test['Sex'].map({'male':1,'female':0})

#### Now we deal with the names cause the information about social status in names could be an useful variable in 
#### out model. 
title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = test[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
test['Title'] = title['Title']

title = pd.DataFrame()
title[ 'Title' ] = train[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
train['Title'] = title['Title']

count= pd.crosstab(index=title['Title'],  columns="count")
count.sort_values(by='count',ascending=False)

#### Since Mr, Miss, Mrs, and Master took up almost all the count in title, we could groupup other rate titles into 'Rare'
train.ix[((train.Title!='Mr') & (train.Title!='Mrs') &(train.Title!='Miss') &(train.Title!='Master') ),'Title'] = 'Rare'
test.ix[((test.Title!='Mr') & (test.Title!='Mrs') &(test.Title!='Miss') &(test.Title!='Master') ),'Title'] = 'Rare'

## Filling missing values

train[train.Embarked.isnull()]

#### There are just 2 missing values of Embarked, we would just fill in that with the most common value
train.Embarked.fillna('S',inplace=True)
test.Embarked.fillna('S',inplace=True)

test[test.Fare.isnull()]

#### There are just 1 missing value of Fare in test dataset, we simply fill that with mean
test.Fare.fillna(test.Fare.mean(),inplace=True)

#### One important missing value we have to fill is age. We will not simply fill in the median or mean of the age column but use
#### other variables like Parch and Title to get more accurate information.
child = train[train['Age'] <12] [['Name','Age','Parch']]
g = sns.FacetGrid(child, col='Parch')
g.map(plt.hist, 'Age', bins=20)

#### We could see from this graph that it is almost impossible for an infant or little child to travel on Titanic along. 
#### So we could assume that if the passenger whose age is missing and traveling along is adult. We would use the 
#### mean age of the adults to fill in those missing value.
adult = train[train['Age'] > 12]
adultMean = adult.Age.mean()
child = train[train['Age'] <= 12]
childMean = child.Age.mean()
allMean = train.Age.mean()
train.Age.fillna(-1,inplace=True)
train.ix[((train.Age==-1) & (train.Parch==0) ),'Age'] = adultMean
#### After that we fill in mean age of children for those title are 'Master' which means boys and young men

train.ix[((train.Age==-1) & (train.Title =='Master') ),'Age'] = childMean

#### For the left we just fill up with mean age of all training group
train.ix[train.Age==-1,'Age'] = allMean

#### Do the same thing for test group
adult = test[test['Age'] > 12]
adultMean = adult.Age.mean()
child = test[test['Age'] <= 12]
childMean = child.Age.mean()
allMean = test.Age.mean()
test.Age.fillna(-1,inplace=True)
test.ix[((test.Age==-1) & (test.Parch==0) ),'Age'] = adultMean
#### After that we fill in mean age of children for those title are 'Master' which means boys and young men

test.ix[((test.Age==-1) & (test.Title =='Master') ),'Age'] = childMean

#### For the left we just fill up with mean age of all testing group
test.ix[test.Age==-1,'Age'] = allMean

#### Convert Title and Embarked to Integers
train['Title'] = train['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5})
test['Title'] = test['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5})
train['Embarked'] = train['Embarked'].map({'S':1,'C':2,'Q':3})
test['Embarked'] = test['Embarked'].map({'S':1,'C':2,'Q':3})

#### Finally we delete column Name, SibSp, and Parch
del train['Name']
del test['Name']
del train['SibSp']
del test['SibSp']
del train['Parch']
del test['Parch']
train.head()

## Applying Modeling

#### This question is the kind of question like Classification and regression problem. So we would try 
#### Random Forest, Logistic Regression, Decision Tree and KNN. 

train_X = train.drop('Survived',axis=1)
train_Y = train['Survived']
test_X  = test.drop("PassengerId", axis=1).copy()

#### First we try Random Forest Method
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(train_X,train_Y)
randomforestY = random_forest.predict(test_X)
random_forest.score(train_X,train_Y)

#### Now we try Logistic Regression
logreg = LogisticRegression()
logreg.fit(train_X, train_Y)
logregY = logreg.predict(test_X)
logreg.score(train_X, train_Y)

#### Try K Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_X, train_Y)
knnY = knn.predict(test_X)
round(knn.score(train_X, train_Y) * 100, 2)

#### Try Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_Y)
decisiontreeY = decision_tree.predict(test_X)
round(decision_tree.score(train_X, train_Y) * 100, 2)

#### After comparing the models we tried, the random forest model would fits our model best. And that is true based on the score
#### we get from Keggle. We scored 0.77511 and around the middle of the leader board. This analysis could be further analyzed 
#### if we dig more in the variable selection and filling missing data. For example, we could try divide age into groups since 
#### it is the age group that matters. A 2 year old and 3 year old would not make much difference in surviving. We could also
#### discover more about Cabin since the position you were would affect the surviving. 
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = randomforestY
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

nbconvert --to python Tianic
