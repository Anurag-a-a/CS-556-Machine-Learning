#!/usr/bin/env python
# coding: utf-8

# ## **`Linear Regression`**

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#reading data from csv file
admission_cv = pd.read_csv('College_Admissions.csv')

#description of the data
admission_cv.describe()


# In[3]:


#Droping the columns which are not important (not considered)
admission_cv.drop(labels=['Serial No.','University Rating','Research'], axis=1, inplace=True)


# In[4]:


admission_cv


# In[5]:


#plotting covariance matrix to see the covariance among different colums
sns.heatmap(admission_cv.corr(), annot=True)


# In[6]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[7]:


X = admission_cv.iloc[:, :-1]
y = admission_cv.iloc[:, -1]
from sklearn.utils import shuffle
X, y = shuffle(X, y)

#spliting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 


# In[8]:


#Scalling the data
ss = StandardScaler()

#applying fit_transform to training data and transform to testing data as it won't change the mean and variance
X_train, X_test = ss.fit_transform(X_train), ss.transform(X_test) 
pd.DataFrame(X_test) # getting a glimpse


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


#applying linear regression
LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
y_pred_train = LR.predict(X_train)


# In[11]:


LR_score = LR.score(X_test,y_test) 


# In[12]:


n_train=len(X_train)  
n_test=len(X_test)

#calculated MSE of training data
train_MSE = 1/n_train*sum((y_train-y_pred_train)**2)
train_MSE


# In[13]:


#calculated MSE of testing data
test_MSE = 1/n_test*sum((y_test-y_pred)**2)  
test_MSE


# In[14]:


#coefficient of each considered column (the most affecting column is CGPA in prediction)
pd.DataFrame(index=X.columns.values, data=LR.coef_, columns = ['coefficient'])


# In[15]:


#visualizing coefficients
plt.figure(figsize=(8,6))
plt.bar(X.columns.values, LR.coef_)


# # Applying PCA for linear regression

# In[16]:


from sklearn.decomposition import PCA 


# In[17]:


y_pca = admission_cv['Chance of Admit ']
X_pca = admission_cv.iloc[:, :-1]


# In[18]:


#Variance of each column
pca_all = PCA()
X_PCA_all = pca_all.fit_transform(X_pca)
pca_all.explained_variance_ratio_   


# In[19]:


#chosing first two principle components as it explains almost 88% of data
pca = PCA(n_components=2)
pca.fit(X_pca)
X_PCA = pca.transform(X_pca)

X_PCA = pd.DataFrame(X_PCA, columns=['PC1', 'PC2'])


# In[20]:


#scatter plot of data using pca=2
plt.figure(figsize=(8, 6))
plt.scatter(X_PCA['PC1'], X_PCA['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot using pca=2')

PCA_max = np.argmax(X_PCA['PC1'])
PCA_min = np.argmin(X_PCA['PC1'])

print(PCA_max)
print(PCA_min)


# In[21]:


#Apply scalling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_pca = scaler.fit_transform(X_pca)


# In[22]:


pd.DataFrame(X_pca)


# In[23]:


#spliting in train and test data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, train_size=0.8, shuffle=True)


# In[24]:


#apply linear regression using principle components
model = LinearRegression()
model.fit(X_train_pca, y_train_pca)


# In[25]:


#model score
LR_PCA_score = model.score(X_test_pca, y_test_pca)


# In[26]:


#predicted values of training and testing 
pred_train_pca = model.predict(X_train_pca)
pred_test_pca = model.predict(X_test_pca)


# In[27]:


#calculating MSE of training data
n_train_pca=len(X_train_pca)  
n_test_pca=len(X_test_pca)

pca_train_MSE = 1/n_train_pca*sum((y_train_pca-pred_train_pca)**2)
pca_train_MSE


# In[28]:


#calculating MSE of testing data
pca_test_MSE = 1/n_test_pca*sum((y_test_pca-pred_test_pca)**2) 
pca_test_MSE


# In[29]:


#fitting regression line(decision boundry) in scatter plot
plt.figure(figsize=(8, 6))
plt.plot(pred_test_pca, y_test_pca, 'o')
plt.xlabel('Predicted value')
plt.plot([min(pred_test_pca), max(pred_test_pca)], [min(y_test_pca), max(y_test_pca)], color = 'black')
plt.ylabel('Actual value')


# In[30]:


print(LR_score,LR_PCA_score) 
print('appliying pca improves our model score') 


# In[31]:


y_pred #predicted values of testing data of Linear regression


# In[32]:


pred_test_pca #predicted values of testing data of Linear regression with PCA


# In[34]:


LR_predictions = pd.DataFrame(y_pred, columns=['LR_predictions']).to_csv('LR_predictions.csv')


# In[35]:


LR_PCA_predictions = pd.DataFrame(pred_test_pca, columns=['LR_PCA_predictions']).to_csv('LR_PCA_predictions.csv')

