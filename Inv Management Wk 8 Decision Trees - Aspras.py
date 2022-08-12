#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Please  construct  a  new  dataset  by  either  adding  two  independent  variables  or  removing  two independent
#variables  from  finalsample.dta  dataset.  If  you  choose  to  add  two  independent variables, you could add 
#any two independent variables that you think help explain stock returns. If  you  choose  to  remove  two  
#independent  variables,  you  could  remove  any  two  independent variables that already exist in the 
#finalsample.dta dataset.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# In[2]:


dectreedata=pd.read_stata('/Users/jimmyaspras/Downloads/finalsample.dta')
dectreedata.columns


# In[3]:


dectreedata.sort_values(by=['datadate'], inplace=True)
dectreedata1=dectreedata[dectreedata['lagPrice2']>=5]#remove penny stocks
dectreedata1['Year']=dectreedata1['datadate'].dt.year
dectreedata1['Month']=dectreedata1['datadate'].dt.month
#set gvkey and datadate as the index
dectreedata1=dectreedata1.set_index(['gvkey','datadate'])
dectreedata1.head()


# In[4]:


#Split  your  new  dataset  into  training  and  testing  samples.  Testing  sample  should  include  data 
#with year>=2016. 
#
#Drop dvpspq and atq from the train/test data
#
train=dectreedata1[dectreedata1['Year']<2016]
X_train=train[['sic_2', 'lagRet2', 'lagVOL2',
       'lagPrice2', 'lagMV2', 'lagShareturnover2', 'lagRet2_sic', 'lagRet12',
       'lagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
       'lagRet12_max', 'lagRet12_sic', 'ceqq', 'cheq',
       'dlttq', 'epspiq', 'saleq', 'sp500_ret_d', 'nasdaq_ret_d',
       'r2000_ret_d', 'dollar_ret_d', 'VIX', 'yield_3m', 'yield_10y',
       'gdp_growth', 'Bull_ave', 'Bull_Bear', 'ret', 'debt', 'cash', 'sale',
       'BM', 'PE', 'div_p', 'loglagPrice2', 'loglagVOL12', 'loglagMV2',
       'logatq', 'loglagVOL2']]


# In[5]:


#Set return as the dependent training variable
Y_train=train[['ret']]


# In[6]:


#Set testing independent variables
test=dectreedata1[dectreedata1['Year']>=2016]
X_test=test[['sic_2', 'lagRet2', 'lagVOL2',
       'lagPrice2', 'lagMV2', 'lagShareturnover2', 'lagRet2_sic', 'lagRet12',
       'lagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
       'lagRet12_max', 'lagRet12_sic', 'ceqq', 'cheq',
       'dlttq', 'epspiq', 'saleq', 'sp500_ret_d', 'nasdaq_ret_d',
       'r2000_ret_d', 'dollar_ret_d', 'VIX', 'yield_3m', 'yield_10y',
       'gdp_growth', 'Bull_ave', 'Bull_Bear', 'ret', 'debt', 'cash', 'sale',
       'BM', 'PE', 'div_p', 'loglagPrice2', 'loglagVOL12', 'loglagMV2',
       'logatq', 'loglagVOL2']]


# In[7]:


#Set return as the dependent testing variable
Y_test=test[['ret']]


# In[8]:


#Calculate avg monthly risk free return
rf1=pd.read_excel("/Users/jimmyaspras/Downloads/Treasury bill.xlsx")
rf1['rf']=rf1['DGS3MO']/1200
rf2=rf1[['Date','rf']].dropna()
rf2['Year']=rf2['Date'].dt.year
rf2['Month']=rf2['Date'].dt.month
rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()


# In[9]:


#Import benchmark index return
indexret1=pd.read_stata("/Users/jimmyaspras/Downloads/Index return.dta")


# In[10]:


#Give a value for min_samples_leaf (you could pick any value) and train DecisionTreeRegressor using your
#new training sample. Use the trained model to predict returns based on your new testing sample. 
#Report the average return of the portfolio that consists of the 100 stocks with the highest predicted 
#returns in each year-month. Also, report the Sharpe ratio of the portfolio. 
DTree_m= DecisionTreeRegressor(min_samples_leaf=75)
DTree_m.fit(X_train,Y_train)


# In[11]:


Y_predict=pd.DataFrame(DTree_m.predict(X_test), columns=['Y_predict'])
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month


# In[12]:


#Rank stocks by return
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
stock_long2['datadate'].value_counts()


# In[13]:


#Calculate returns
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()


# In[14]:


#Merge rf and index
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()


# In[15]:


#The decicion tree model produces returns of 37.01% above the market. This is statistically significant with a
#p-value of 0


# In[16]:


#Calculate Sharpe ratio
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR


# In[17]:


#Give values for min_samples_leaf, n_estimators, and max_samples, and train RandomForestRegressor using your 
#new training sample. Use the trained model to predict returns based on your new testing sample. 
#Report the average return of the portfolio that consists of the 100 stocks with the highest predicted 
#returns in each year-month. Also, report the Sharpe ratio of the portfolio.
RFor_m= RandomForestRegressor(n_estimators=150, min_samples_leaf=150,bootstrap=True,max_samples=0.5,n_jobs=-1)
#n_estimators:The number of trees in the forest.
#bootstrap: whether use a different subsample of training sample to train each tree
#max_samples=0.5: randomly draw 50% of the training sample to train each tree
#n_jobs=-1 means using all CPU processors
RFor_m.fit(X_train,Y_train.values.ravel())


# In[18]:


#Predict returns
Y_predict=pd.DataFrame(RFor_m.predict(X_test), columns=['Y_predict'])
#Merge with actual returns
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month


# In[19]:


#Rank stocks
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100] #Select 100 best
stock_long2['datadate'].value_counts()


# In[20]:


#Calculate returns
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()


# In[21]:


#Merge rf and index
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()


# In[22]:


#The random forest decicion tree model produces returns of 36.99% above the market. This is statistically significant with a
#p-value of 0


# In[23]:


#Sharpe ratio
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR


# In[24]:


#Give    values    for    min_samples_leaf,    n_estimators,    and    max_samples,    and    train 
#ExtraTreesRegressor  using  your  new  training  sample.  Use  the  trained  model  to  predict  returns 
#based on your new testing sample. Report the average return of the portfolio that consists of the 100 
#stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of the portfolio.
ETree_m= ExtraTreesRegressor(n_estimators=150, min_samples_leaf=150, bootstrap=True,max_samples=0.5,n_jobs=-1)
ETree_m.fit(X_train,Y_train.values.ravel())


# In[25]:


#Predict returns
Y_predict=pd.DataFrame(ETree_m.predict(X_test), columns=['Y_predict'])
#Merge with actual
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month


# In[26]:


#Rank stocks
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
stock_long2['datadate'].value_counts()


# In[27]:


#Calculate returns
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()


# In[28]:


#Merge rf and index
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()


# In[29]:


#The extra trees regressor decicion tree model produces returns of 36.96% above the market. This is statistically significant with a
#p-value of 0


# In[30]:


#Sharpe ratio
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR


# In[31]:


#Give  values  for  min_samples_leaf  and  max_iter,  and  train  HistGradientBoostingRegressor using your 
#new training sample. Use the trained model to predict returns based on your new testing sample. Report the average
#return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. 
#Also, report the Sharpe ratio of the portfolio. 
GBR_m= HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=150, early_stopping='True')
#max_iter: The maximum number of iterations of the boosting process#early_stopping: If Yes, the algorithm use 
#internal cross-validation to determine max_iter 
GBR_m.fit(X_train,Y_train)


# In[32]:


#Predict returns
Y_predict=pd.DataFrame(GBR_m.predict(X_test), columns=['Y_predict']) 
#Merge with actual
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month


# In[33]:


#Rank stocks
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
stock_long2['datadate'].value_counts()


# In[34]:


#Calculate returns
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()


# In[35]:


#Merge rf and index
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()


# In[36]:


#The gradient boosting regressor decicion tree model produces returns of 36.72% above the market. This is statistically significant with a
#p-value of 0


# In[37]:


#Sharpe ratio
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR

