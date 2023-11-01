# Ex-07-Feature-Selection
## Aim
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# Algorithm
### Step 1
Read the given data.
### Step 2
Clean the data set using data cleaning process.
### Step 3
Apply label encoding and ordinal encoding to necessary columns.
### Step 4
Apply Feature selection techniques to all the features of the data set.
### Step 5
Save the data to the file.

# Code
```python
import pandas as pd
from sklearn.feature_selection import SelectKBest,SelectFromModel,RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.feature_selection import chi2,f_regression,mutual_info_classif,SelectPercentile
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.metrics import accuracy_score

# reading titanic_dataset
data=pd.read_csv('titanic_dataset.csv')

# dropping unwanted columns and null rows
data=data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
data=data.dropna()

# separating the features and target variable
x=data.drop(['Survived'],axis=1)
y=data['Survived']

# feature encoding for Sex and Embarked column
le=LabelEncoder()
x['Sex']=le.fit_transform(x['Sex'])
x['Embarked']=le.fit_transform(x['Embarked'])

k=5
# selecting the best columns using filter
selector=SelectKBest(score_func=chi2,k=k)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features: ",end='')
print(selected_features)

# selecting the best columns using correlation coefficient
selector=SelectKBest(score_func=f_regression,k=k)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features: ",end='')
print(selected_features)

# selecting the best columns using mutual information
selector=SelectKBest(score_func=mutual_info_classif,k=k)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features: ",end='')
print(selected_features)

percentile=60
# selecting the best columns based on percentile using chi2
selector=SelectPercentile(score_func=chi2,percentile=percentile)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features: ",end='')
print(selected_features)

# selecting the best columns based on percentile using f_regression
selector=SelectPercentile(score_func=f_regression,percentile=percentile)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features: ",end='')
print(selected_features)

# selecting the best columns based on percentile using mutual_info_classif
selector=SelectPercentile(score_func=mutual_info_classif,percentile=percentile)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features: ",end='')
print(selected_features)

# reading carprice dataset
data=pd.read_csv("CarPrice.csv")

# dropping unwanted column
data=data.drop(columns=['car_ID','CarName'],axis=1)

# label encoding
le=LabelEncoder()
col=['fueltype','aspiration','carbody','drivewheel','enginelocation','enginetype','fuelsystem']
for index in col:
  data[index]=le.fit_transform(data[index])

# ordinal encoding
cylinder=['two','three','four','five','six','eight','twelve']
door=['two','four']
oe=OrdinalEncoder(categories=[door,cylinder])
data[['doornumber','cylindernumber']]=oe.fit_transform(data[['doornumber','cylindernumber']])

# separating features and target variables
x=data.drop(columns=['price'])
y=data[['price']]

# forward selection
model=LinearRegression()
sfm=SelectFromModel(model,threshold='mean')
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print('Selected features:',selected_features)

# backward elimination
model=LinearRegression()
rfe=RFE(model,n_features_to_select=len(x.columns)-18)
rfe.fit(x,y)
selected_features=x.columns[rfe.get_support()]
print('Selected features:',selected_features)

# exhaustive feature selection
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
efs=ExhaustiveFeatureSelector(model,min_features=3,max_features=4,scoring='r2')
efs=efs.fit(x_train,y_train)
selected_features=list(x.columns[list(efs.best_idx_)])
model.fit(x_train[selected_features],y_train)
y_predict=model.predict(x_test[selected_features])
print('Selected features:',selected_features)
```

# Output
## Titanic dataset
### Features selected using chi2 method and k best
![image](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex-07/assets/142985837/07a50ae3-c6b2-4a8b-9749-3f71aacad289)

### Features selected using correlation coefficient and k best
![image](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex-07/assets/142985837/464d3492-a904-493d-a420-1360edb98807)

### Features selected using mutual information and k best
![image](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex-07/assets/142985837/3e6ba1d3-75b5-43d3-8965-35c5ac2c18ca)

### Features selected using chi2 and percentile
![image](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex-07/assets/142985837/c43468ef-8af9-4afd-9773-ba3203b4b569)

### Features selected using f_regression and percentile
![image](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex-07/assets/142985837/9c28c0a4-f0e5-4d8f-9c34-d4d630bec803)

### Features selected using mutual_info_classif and percentile
![image](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex-07/assets/142985837/18520b3d-c9a9-4bb1-8bc5-04149d72e221)

## Car price dataset
### Features selected using forward selection
![image](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex-07/assets/142985837/5c5466ac-77a1-40e2-ba7f-a2a922513c85)

### Features selected using backward elimination
![image](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex-07/assets/142985837/2747f347-ea29-4fc7-8b3e-f57dd9bb5282)

### Features selected using exhaustive feature selection
![image](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex-07/assets/142985837/13bf3eb6-fc12-445f-9dcd-72989057e58c)

# Result
Thus, the give dataset is explored to select best features through various feature selection techniques and the features are printed.
