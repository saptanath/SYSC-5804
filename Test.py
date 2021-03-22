#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


flight = pd.read_csv('flights.csv', dtype='unicode')


# In[4]:


flight.head()


# In[5]:


flight.info


# In[6]:


par = pd.read_csv('parameters.csv', dtype='unicode')


# In[7]:


par


# In[8]:


get_ipython().system('pip install matplotlib')


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


flight 


# In[11]:


flight = flight[flight.route != 'A1']


# In[12]:


flight = flight[flight.route != 'A2']


# In[13]:


flight = flight[flight.route != 'A3']


# In[14]:


flight.columns[1:24]


# merge the data and time column 

# In[15]:


flight['date'] = pd.to_datetime(flight['date']+' '+flight['time_day'])


# In[16]:


flight.replace('25-50-100-25','100',inplace=True)


# encode the column that has non numerical values. 

# In[17]:


import seaborn as sns
flight['route'].unique()


# In[18]:


route_cat = {'R1':1,'R2':2,'R3':3,'R4':4,'R5':5,'R6':6,'R7':7,'H':8}
flight['route'] = flight['route'].apply(lambda x : route_cat.get(x,x))


# In[19]:


columns = flight.columns[1:25].tolist()
columns


# create column that has power values based on voltage and current calculation. 

# In[20]:


for col in columns:
    flight[col] = flight[col].astype('float')


# In[21]:


flight['power'] = flight['battery_voltage'] * flight['battery_current']


# In[22]:


columns.append('power')
columns


# In[23]:


for col in columns:
    flight[col] = flight[col].round(2)


# In[24]:


flight


# In[25]:


numeric = flight.columns[1:24].tolist()


# In[26]:


numeric.append('power')
numeric.append('altitude')
numeric


# In[27]:


target = 'power'
categories = flight.columns[1:25].tolist()


# In[28]:


for cat in categories:
    sns.scatterplot(x=flight[cat],y=flight[target])
    plt.gca().set_title(cat)
    plt.show()


# In[29]:


numeric


# In[30]:


plt.figure(figsize=(20,20))
sns.heatmap(flight[numeric].corr())


# rmove rows that has pwoer value equal to 0

# In[31]:


flight[(flight['power'] == 0.0)&(flight['power']< 1.0)]


# In[32]:


flight_0 = flight[flight.power != 0]


# In[33]:


flight_0[flight_0['power']<1]


# based on new data set plot correltation plots and heatmap. 

# In[34]:


for cat in categories:
    sns.scatterplot(x=flight_0[cat],y=flight_0[target])
    plt.gca().set_title(cat)
    plt.show()


# In[35]:


plt.figure(figsize=(20,20))
sns.heatmap(flight_0[numeric].corr())


# correlation between target (power) and wind_speed, velocity_y, velocity_z and position_z

# In[36]:


categories.append('route')
categories


# In[37]:


encode = ['route']


# create dataset with encoded columns 

# In[38]:


df_encoded = pd.get_dummies(flight_0[categories],columns=encode)


# In[39]:


df_encoded


# In[40]:


df_encoded = df_encoded.drop(['battery_voltage','battery_current'], axis=1)


# In[41]:


df_encoded 


# remove space in title of columns 

# In[42]:


space_remover_dict = dict(zip(df_encoded.columns,[name.replace(' ','') for name in df_encoded.columns]))
df_encoded = df_encoded.rename(space_remover_dict,axis=1)


# split data 80-10-10 split. 

# In[43]:


import numpy as np
from sklearn.model_selection import train_test_split
X = df_encoded
y = flight_0['power']
X_train, X_test, y_train, y_test = train_test_split(X,y.values,test_size=0.2,random_state=1)
X_train,X_cv,y_train,y_cv = train_test_split(X_train,y_train,test_size=0.2/0.8,random_state=1)


# normal linear regression model 

# In[44]:


import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

lr_pipe = make_pipeline(StandardScaler(),LinearRegression())
lr_pipe.fit(X_train.values,(y_train))


# In[45]:


lr_pipe.score(X_train,y_train)


# In[46]:


lr_pipe.score(X_cv,y_cv)


# In[47]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(lr_pipe.predict(X_train),y_train)


# In[48]:


mean_absolute_error(lr_pipe.predict(X_cv),y_cv)


# use ridge as a regularization model to imporve linear regression model 

# In[49]:


from sklearn.linear_model import Ridge

ridge = make_pipeline(StandardScaler(),Ridge())
ridge.fit(X_train.values,(y_train))


# In[50]:


mean_absolute_error(ridge.predict(X_train),y_train)


# In[51]:


mean_absolute_error(ridge.predict(X_cv),y_cv)


# find best alpha for the ridge model 

# In[52]:


from sklearn.linear_model import RidgeCV

sc = StandardScaler()
alphas = [5e-1,1e-1, 1, 10, 50, 100, 175, 250, 325, 500, 750, 1000, 2000, 3000]
ridge_cv = RidgeCV(alphas=alphas,  store_cv_values=True)
ridge_cv.fit(sc.fit_transform(X_train.astype(float)), (y_train))
best_alpha = ridge_cv.alpha_
best_alpha


# In[53]:


ridge = make_pipeline(StandardScaler(),Ridge(alpha=best_alpha))
ridge.fit(X_train.values,(y_train))


# In[54]:


mean_absolute_error(ridge.predict(X_train),y_train)


# In[55]:


mean_absolute_error(ridge.predict(X_cv),y_cv)


# In[56]:


from mlxtend.feature_selection import SequentialFeatureSelector


# In[57]:


from sklearn.model_selection import KFold


# use forward feature selection to find the bect features to improve the model 

# In[114]:


lr = LinearRegression()
sfs = SequentialFeatureSelector(lr, k_features=30, forward=True, floating=False, verbose=0, cv=KFold(5), n_jobs=-1)

sfs.fit(X_train, (y_train))


# In[115]:


plt.figure(figsize=(10,5))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score $R^2$")
plt.plot([subset for subset in sfs.subsets_], [sfs.subsets_[subset]['avg_score'] for subset in sfs.subsets_])
plt.show()


# In[116]:


features = list(sfs.subsets_[15]['feature_names'])
features


# In[97]:


lr_small = make_pipeline(StandardScaler(),LinearRegression())
lr_small.fit(X_train[features].values,(y_train))


# In[98]:


mean_absolute_error(lr_small.predict(X_train[features]),y_train)


# In[99]:


mean_absolute_error(lr_small.predict(X_cv[features]),y_cv)


# In[100]:


lr_small.score(X_train[features],y_train)


# In[109]:


mean_squared_error(lr_small.predict(X_train[features]),y_train,squared=False)


# since the model has not imporved, implement neural network model first as normal then with additional optimal features selected. 

# In[101]:


from sklearn.neural_network import MLPRegressor
mlp_regressor = make_pipeline(StandardScaler(),MLPRegressor(solver='adam', alpha=1e-10, random_state=1,max_iter=2000))


# In[66]:


mlp_regressor.fit(X_train,y_train)


# In[67]:


mean_absolute_error(mlp_regressor.predict(X_train),y_train)


# In[68]:


mean_absolute_error(mlp_regressor.predict(X_cv),y_cv)


# use the same neural network but with the features selected from forward feature selection. 

# In[102]:


mlp_regressor.fit(X_train[features],y_train)


# In[103]:


mean_absolute_error(mlp_regressor.predict(X_train[features]),y_train)


# In[104]:


mean_absolute_error(mlp_regressor.predict(X_cv[features]),y_cv)


# In[105]:


mlp_regressor.score(X_train[features],y_train)


# In[106]:


mlp_regressor.score(X_cv[features],y_cv)


# In[107]:


from sklearn.metrics import mean_squared_error
mean_squared_error(mlp_regressor.predict(X_train[features]),y_train,squared=False)


# In[108]:


mean_squared_error(mlp_regressor.predict(X_cv[features]),y_cv,squared=False)


# based on the good model use K-fold algorithm to validate that the model choosen is indeed good. 

# In[76]:


X_f = X[features]


# In[77]:


from sklearn.model_selection import KFold


# In[ ]:


# kfold = KFold(n_splits=3,random_state=10,shuffle=True)
# scores = []
# for train_index, test_index in kfold.split(X_f):
#     mlp_regressor = make_pipeline(StandardScaler(),MLPRegressor(solver='adam', alpha=0.1, random_state=1,
#                                                                 max_iter=2000))
#     X_train_k, X_test_k = X_f.iloc[train_index], X_f.iloc[test_index]
#     y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]
#     mlp_regressor.fit(X_train_k.values,y_train_k)
#     scores.append(mean_absolute_error(mlp_regressor.predict(X_test_k.values),y_test_k))


# In[ ]:


# scores


# In[78]:


alphas = [5e-1,1e-1, 1, 10, 50, 100, 175, 250, 325, 500, 750, 1000, 2000, 3000]
scores = []
for al in alphas:
    mlp_regressor = make_pipeline(StandardScaler(),MLPRegressor(solver='adam', alpha=al, random_state=1,max_iter=2000))
    mlp_regressor.fit(X_train[features],y_train)
    scores.append(mean_absolute_error(mlp_regressor.predict(X_test[features]),y_test))


# In[79]:


scores


# In[ ]:




