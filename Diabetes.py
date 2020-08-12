import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dia=pd.read_csv("C:\\Users\\svije\\Downloads\\datasets_228_482_diabetes.csv")
dia.shape
dia.columns                
dia.info()             
dia.describe()

#1.Data cleaning:
#a.identification of duplicate values
dia.duplicated()
dia.duplicated().sum()
#there are no duplicates

#b.identification of any missing values
dia.isnull().sum()
sns.heatmap(dia.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#there are no missing values
#2.Exploratory data Analysis
sns.set_style('whitegrid')
dia['Age'].hist(bins=30)
plt.xlabel('Age')
#pairplot givesthe idea of how features are interconnected with hue as outcome
sns.pairplot(dia,hue='Outcome',palette='bwr')

# lmplot shows the probability as a function of age and glucose
sns.set_style('whitegrid')
sns.lmplot('Age','Glucose',data=dia, hue='Outcome', 
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
# lmplot shows the probability as a function of age and bloodpressure
sns.set_style('whitegrid')
sns.lmplot('Age','BloodPressure',data=dia, hue='Outcome',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
# lmplot shows the probability as a function of age and bmi
sns.set_style('whitegrid')
sns.lmplot('Age','BMI',data=dia, hue='Outcome',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
# lmplot shows the probability as a function of age and DiabetesPedigreeFunction
sns.set_style('whitegrid')
sns.lmplot('Age','DiabetesPedigreeFunction',data=dia, hue='Outcome',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)                

#Use of histogram for identification of split                        
sns.set_style('whitegrid')
g = sns.FacetGrid(dia,hue="Outcome",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Age',bins=20,alpha=0.7)                
                
sns.set_style('whitegrid')
h = sns.FacetGrid(dia,hue="Outcome",palette='coolwarm',size=6,aspect=2)
h = h.map(plt.hist,'Glucose',bins=20,alpha=0.7)                 

sns.set_style('whitegrid')
i = sns.FacetGrid(dia,hue="Outcome",palette='coolwarm',size=6,aspect=2)
i = i.map(plt.hist,'BMI',bins=20,alpha=0.7)                  

#Model building
#a.splitting the data set in train and test data set import train test split from sklearn
from sklearn.model_selection import train_test_split                

X = dia[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dia['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=101)                
#splitting the data in ration of 70:30 for train and test

#b.fetting the model in train data set, to fit the model import logisticregression from sklearn
from sklearn.linear_model import LogisticRegression               
lgmodel = LogisticRegression()
lgmodel.fit(X_train,y_train)  

#c.Predictions
predictions = lgmodel.predict(X_test)
predictions             
                
#d.Evaluation of model using confusion matrix
# Create a classification report for the model
from sklearn.metrics import confusion_matrix
con_mat=confusion_matrix(y_test, predictions)
con_mat
# create heatmap
sns.heatmap(pd.DataFrame(con_mat), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#dimension of this matrix is 2*2 because this model is binary classification. 
#You have two classes 0 and 1. Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions. 
#In the output, 133 and 51 are actual predictions, and 30 and 17 are incorrect predictions.

#Accuracy check
from sklearn.metrics import classification_repor
print(classification_report(y_test,predictions))

#or this method we can use

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predictions))
#we are getting the accuracy of 79%               
print("Precision:",metrics.precision_score(y_test, predictions))
#we are getting the Precision of 75%             
print("Recall:",metrics.recall_score(y_test, predictions))
#we are getting the Precision of 63%

#ROC curve    
auc=metrics.roc_auc_score(y_test, predictions)

Conclusion AUC score 1 represents perfect classifier.
