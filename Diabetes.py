import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dia=pd.read_csv("C:\\Users\\svije\\Downloads\\datasets_228_482_diabetes.csv")
dia.shape
dia.columns                
dia.info()             
pd.options.display.max_columns= None
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

#2.Visualization and Exploratory data Analysis

#2.1 visualizing and checking the bivariate relationships using pairplot, heatmap

#a.pairplot gives the idea of how features are interconnected with hue as outcome
sns.pairplot(dia,hue='Outcome',palette='bwr')

#b.we can clearly see that there it is a non-linour relationship
#lets check using heatmap
dia.corr()
sns.heatmap(dia.corr(), annot=True,cmap='coolwarm')

#Pregnancis ,glucose and age has partial correlation of 0.54 and skinthickness and isuline has  0.44
sns.set_style('whitegrid')
dia[['Age','Outcome']].hist(bins=30)
plt.xlabel('Age v/s Outcome')

#lets check how many are diabitic and non diabitic
sns.distplot(dia)
dia['Outcome'].value_counts()
#out of 768 records 500 i,e 65% is non-diabitic(0) and 268 i,e are non diabitic(1)
sns.countplot(x=dia.Outcome)

#lets explore each feature to get the insights keeping Age dependent variable
# shows the probability of Pregnancies with age
sns.distplot(dia.Pregnancies)
sns.countplot(dia.Pregnancies)
sns.jointplot(x='Pregnancies',y='Age',data=dia)
sns.set_style('whitegrid')
sns.lmplot('Pregnancies','Age',data=dia, hue='Outcome', 
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

#Use of histogram for identification of split                        
sns.set_style('whitegrid')
g = sns.FacetGrid(dia,hue="Outcome",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Pregnancies',bins=20,alpha=0.7) 

#check any outliers in Pregnancies
sns.boxplot(x=dia.Outcome,y=dia.Pregnancies)

#summary outcome on Pregnancies
#1.Women whose age is in range of 38 and above pregnancies between are moslty diabitic 
#2.Women with pregnancies with age of 20 to 38 in 0 to 5 weeks will be non-diabitic
#3.There are some outliers

#Glucose
sns.distplot(dia.Glucose)
sns.jointplot(x='Glucose',y='Age',data=dia)

sns.set_style('whitegrid')
sns.lmplot('Glucose','Age',data=dia, hue='Outcome',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

sns.set_style('whitegrid')
g = sns.FacetGrid(dia,hue="Outcome",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Glucose',bins=20,alpha=0.7)

sns.boxplot(x=dia.Outcome,y=dia.Glucose)

#summary outcome on Glucose
#1.Women whose age is in range of 20 to 30 yrs pregnancies non-diabitc have low glucose level between 75 to 120
#2.Women whose age is in range of above 30 yrs pregnancies diabitc have high glucose level between  125 to 200
#3.There are some outliers


#BloodPressure
sns.distplot(dia.BloodPressure)
sns.jointplot(x='Glucose',y='Age',data=dia)

sns.set_style('whitegrid')
sns.lmplot('BloodPressure','Age',data=dia, hue='Outcome',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

sns.set_style('whitegrid')
g = sns.FacetGrid(dia,hue="Outcome",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'BloodPressure',bins=20,alpha=0.7)

sns.boxplot(x=dia.Outcome,y=dia.BloodPressure)

#1.Most non diabetic women seems to have nominal value b/w 60 to 80 and diabetic women seems to have high BP b/w 80 to 110. 
#2.There are some outliers

#SkinThickness         
sns.distplot(dia.SkinThickness)
sns.jointplot(x='SkinThickness',y='Age',data=dia)

sns.set_style('whitegrid')
sns.lmplot('SkinThickness','Age',data=dia, hue='Outcome',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

sns.set_style('whitegrid')
g = sns.FacetGrid(dia,hue="Outcome",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'SkinThickness',bins=20,alpha=0.7)

sns.boxplot(x=dia.Outcome,y=dia.SkinThickness)

#Insulin
sns.jointplot(x='Insulin',y='Age',data=dia)

sns.set_style('whitegrid')
sns.lmplot('Insulin','Age',data=dia, hue='Outcome',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

sns.set_style('whitegrid')
g = sns.FacetGrid(dia,hue="Outcome",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Insulin',bins=20,alpha=0.7)

sns.boxplot(x=dia.Outcome,y=dia.Insulin)

#Skinthickness and Insulin is not giving enoff information
#Few outliers in both

#BMI
sns.jointplot(x='BMI',y='Age',data=dia)

sns.set_style('whitegrid')
sns.lmplot('BMI','Age',data=dia, hue='Outcome',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

sns.set_style('whitegrid')
g = sns.FacetGrid(dia,hue="Outcome",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'BMI',bins=20,alpha=0.7)

sns.boxplot(x=dia.Outcome,y=dia.BMI)

#diabitic seems to have high BMI compaired to non-diabitic

#to handel outliers we follow IQR method
q3=dia.quantile(.75)
q1=dia.quantile(.25)
iqr=q3-q1
print(iqr)
dia1=dia[~((dia<(q1-1.5*iqr))|(dia>(q3+1.5*iqr))).any(axis=1)]
dia1.shape

#Model building
#a.splitting the data set in train and test data set import train test split from sklearn
from sklearn.model_selection import train_test_split                

X = dia1[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dia1['Outcome']
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
#In the output, 126 and 31 are actual predictions, and 22 and 13 are incorrect predictions.

#Accuracy check
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#or this method we can use

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predictions))
#we are getting the accuracy of 82%               
print("Precision:",metrics.precision_score(y_test, predictions))
#we are getting the Precision of 70%             
print("Recall:",metrics.recall_score(y_test, predictions))
#we are getting the Precision of 58%

#ROC curve    
auc=metrics.roc_auc_score(y_test, predictions)
auc
#Conclusion AUC score equal to 0.75 close to 1 represents perfect classifier.
