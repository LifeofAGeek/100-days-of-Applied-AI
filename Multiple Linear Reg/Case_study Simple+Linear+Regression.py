#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression 

# In this example we will consider sales based on 'TV' marketing budget. 
# 
# In this notebook, we'll build a linear regression model to predict 'Sales' using 'TV' as the predictor variable.
# 

# ## Understanding the Data

# Let's start with the following steps:
# 
# 1. Importing data using the pandas library
# 2. Understanding the structure of the data

# In[9]:


import pandas as pd


# In[13]:


advertising = pd.read_csv(r"C:\Users\sethi\Documents\.ipynb_checkpoints\tvmarketing.csv")


# Now, let's check the structure of the advertising dataset.

# In[14]:


# Display the first 5 rows
advertising.head()


# In[15]:


# Display the last 5 rows
advertising.tail()


# In[16]:


# Let's check the columns
advertising.info()


# In[37]:


# Check the shape of the DataFrame (rows, columns)
advertising.shape


# In[38]:


# Let's look at some statistical information about the dataframe.
advertising.describe()


# # Visualising Data Using Seaborn

# In[19]:


# Conventional way to import seaborn
import seaborn as sns

# To visualise in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(advertising, x_vars=['TV'], y_vars='Sales',size=7, aspect=0.7, kind='scatter')


# # Perfroming Simple Linear Regression

# Equation of linear regression<br>
# $y = c + m_1x_1 + m_2x_2 + ... + m_nx_n$
# 
# -  $y$ is the response
# -  $c$ is the intercept
# -  $m_1$ is the coefficient for the first feature
# -  $m_n$ is the coefficient for the nth feature<br>
# 
# In our case:
# 
# $y = c + m_1 \times TV$
# 
# The $m$ values are called the model **coefficients** or **model parameters**.

# ### Generic Steps in Model Building using ```sklearn```
# 
# Before you read further, it is good to understand the generic structure of modeling using the scikit-learn library. Broadly, the steps to build any model can be divided as follows: 

# ## Preparing X and y
# 
# -  The scikit-learn library expects X (feature variable) and y (response variable) to be NumPy arrays.
# -  However, X can be a dataframe as Pandas is built over NumPy.

# In[21]:


# Putting feature variable to X
X = advertising['TV']

# Print the first 5 rows
X.head()


# In[22]:


# Putting response variable to y
y = advertising['Sales']

# Print the first 5 rows
y.head()


# ## Splitting Data into Training and Testing Sets

# In[43]:


#random_state is the seed used by the random number generator, it can be any integer.

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)


# In[44]:


print(type(X_train))
print(type(X_test))
print(type(y_train))
print(type(y_test))


# In[45]:


train_test_split   #Press Tab to auto-fill the code
#Press Tab+Shift to read the documentation


# In[46]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[47]:


#It is a general convention in scikit-learn that observations are rows, while features are columns. 
#This is needed only when you are using a single feature; in this case, 'TV'.

import numpy as np

X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]


# In[48]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ## Performing Linear Regression

# In[49]:


# import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lr = LinearRegression()

# Fit the model using lr.fit()
lr.fit(X_train, y_train)


# ## Coefficients Calculation

# In[50]:


# Print the intercept and coefficients
print(lr.intercept_)
print(lr.coef_)


# $y = 6.989 + 0.0464 \times TV $<br>
# 
# Now, let's use this equation to predict our sales.

# ## Predictions

# In[51]:


# Making predictions on the testing set
y_pred = lr.predict(X_test)


# In[52]:


type(y_pred)


# #### Computing RMSE and R^2 Values

# In[53]:


# Actual vs Predicted
import matplotlib.pyplot as plt
c = [i for i in range(1,61,1)]         # generating index 
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                       # Y-label


# In[54]:


# Error terms
c = [i for i in range(1,61,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label


# In[55]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)


# In[56]:


r_squared = r2_score(y_test, y_pred)


# In[57]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[58]:


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

