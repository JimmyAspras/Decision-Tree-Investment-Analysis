# Decision-Tree-Investment-Analysis

## Introuction

Can decision tree models be used to beat the market?

Data for this project was chosen and downloaded from Wharton Research Data Services: https://wrds-www.wharton.upenn.edu/. This was done as part of a course taken in Summer 2021 complete with prompts and analysis. Credit must be given to my professor, Dr. Wei Jiao, for much code and instruction included here.

This project follows a series of prompts to determine if a decision tree model can be refined to beat market return.

## Preparing the Data

Construct a new dataset by either adding two independent variables or removing two independent variables from finalsample.dta dataset.
If you choose to add two independent variables, you could add any two independent variables that you think help explain stock returns. If you choose to remove two independent variables, you could remove any two independent variables that already exist in the finalsample.dta dataset.

### Libraries
```python
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
```

### Importing and Working with the Data

```python
dectreedata=pd.read_stata('/Users/jimmyaspras/Downloads/finalsample.dta')
dectreedata.columns
```

Output:
```
Index(['gvkey', 'datadate', 'sic_2', 'lagdate', 'lagRet2', 'lagVOL2',
       'lagPrice2', 'lagMV2', 'lagShareturnover2', 'lagRet2_sic', 'lagRet12',
       'lagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
       'lagRet12_max', 'lagRet12_sic', 'lagdatadate', 'atq', 'ceqq', 'cheq',
       'dlttq', 'epspiq', 'saleq', 'dvpspq', 'sp500_ret_d', 'nasdaq_ret_d',
       'r2000_ret_d', 'dollar_ret_d', 'VIX', 'yield_3m', 'yield_10y',
       'gdp_growth', 'Bull_ave', 'Bull_Bear', 'ret', 'debt', 'cash', 'sale',
       'BM', 'PE', 'div_p', 'loglagPrice2', 'loglagVOL12', 'loglagMV2',
       'logatq', 'loglagVOL2'],
      dtype='object')
```

We want to sort the data in order by date and exclude penny stocks. We also create vectors for year and month for the analysis and set the stock key and datadate as the index.

```python
dectreedata.sort_values(by=['datadate'], inplace=True)
dectreedata1=dectreedata[dectreedata['lagPrice2']>=5]#remove penny stocks
dectreedata1['Year']=dectreedata1['datadate'].dt.year
dectreedata1['Month']=dectreedata1['datadate'].dt.month
#set gvkey and datadate as the index
dectreedata1=dectreedata1.set_index(['gvkey','datadate'])
dectreedata1.head()
```

Split your new dataset into training and testing samples. Testing sample should include data with year>=2016. I chose to drop dvpspq (dividends) and atq (total assets) from the train/test data to examine their effect on portfolio returns.

```python
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
 ```
 
Set return as the dependent training variable
```python
Y_train=train[['ret']]
```

Set testing independent variables
```python
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
```
Calculate avg monthly risk free return
```python
rf1=pd.read_excel("/Users/jimmyaspras/Downloads/Treasury bill.xlsx")
rf1['rf']=rf1['DGS3MO']/1200
rf2=rf1[['Date','rf']].dropna()
rf2['Year']=rf2['Date'].dt.year
rf2['Month']=rf2['Date'].dt.month
rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()
```
Import benchmark index return
```python
indexret1=pd.read_stata("/Users/jimmyaspras/Downloads/Index return.dta")
```
### Building the First Model - Decision Tree Regressor

**Give a value for min_samples_leaf (you could pick any value) and train DecisionTreeRegressor using your new training sample. Use the trained model to predict returns based on your new testing sample. Report the average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of the portfolio.**

Train the model
```python
DTree_m= DecisionTreeRegressor(min_samples_leaf=75)
DTree_m.fit(X_train,Y_train)
```

Predict returns
```python
Y_predict=pd.DataFrame(DTree_m.predict(X_test), columns=['Y_predict'])
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month
```

Rank stocks by return
```python
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
stock_long2['datadate'].value_counts()
```

output:
```python
2020-08-31    102
2021-02-28    101
2019-06-30    101
2016-07-31    101
2018-07-31    100
             ... 
2020-10-31     99
2020-05-31     99
2019-02-28     99
2016-05-31     99
2020-11-30     98
Name: datadate, Length: 64, dtype: int64
```

Calculate returns
```python
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()
```

Merge the risk free return rate and the index return to create benchmarks for the model return. Subtract to model return from each to determine model performance.
```python
#Merge rf and index
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
```

<img width="408" alt="image" src="https://user-images.githubusercontent.com/72087263/188276589-45367c6b-2475-4228-8be3-8427cb434f9f.png">

**The decicion tree model produces returns of 37.01% above the market. This is statistically significant with a p-value of 0.**

### Calculate Sharpe ratio
```python
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR
```

<img width="233" alt="image" src="https://user-images.githubusercontent.com/72087263/188282429-f926fdcf-56fa-402f-9fea-2e8b0212c14c.png">

### Give values for min_samples_leaf, n_estimators, and max_samples, and train RandomForestRegressor using your new training sample. Use the trained model to predict returns based on your new testing sample. Report the average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of the portfolio.
```python
RFor_m= RandomForestRegressor(n_estimators=150, min_samples_leaf=150,bootstrap=True,max_samples=0.5,n_jobs=-1)
#n_estimators:The number of trees in the forest.
#bootstrap: whether use a different subsample of training sample to train each tree
#max_samples=0.5: randomly draw 50% of the training sample to train each tree
#n_jobs=-1 means using all CPU processors
RFor_m.fit(X_train,Y_train.values.ravel())
```

Predict returns
```python
Y_predict=pd.DataFrame(RFor_m.predict(X_test), columns=['Y_predict'])
#Merge with actual returns
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month
```

Rank stocks
```python
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100] #Select 100 best
stock_long2['datadate'].value_counts()
```

Calculate returns
```python
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()
```

Merge rf and index
```python
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
```
<img width="399" alt="image" src="https://user-images.githubusercontent.com/72087263/188282527-c8edf27f-ef60-4984-a7ff-16a98cd799b0.png">

The random forest decicion tree model produces returns of 36.99% above the market. This is statistically significant with a p-value of 0

### Decision Tree Regressor Sharpe ratio

The Sharpe Ratio tells us the risk-adjusted rate of return of a portfolio. In other words, are the returns worth the risk? A higher Sharpe Ratio is desirable.

```python
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR
```

<img width="156" alt="image" src="https://user-images.githubusercontent.com/72087263/188282568-7e85d3af-3ada-4a37-b7aa-6f05675b53b7.png">

### Building the Second Model - Extra Trees Regressor

**Give values for min_samples_leaf, n_estimators, and max_samples, and train ExtraTreesRegressor using your new training sample. Use the trained model  to predict returns based on your new testing sample. Report the average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of the portfolio.**

As with the Decision Tree Regressor model, we follow the same steps to build the model, predict returns, and compare to benchmarks.

```python
ETree_m= ExtraTreesRegressor(n_estimators=150, min_samples_leaf=150, bootstrap=True,max_samples=0.5,n_jobs=-1)
ETree_m.fit(X_train,Y_train.values.ravel())
```

Predict returns
```python
Y_predict=pd.DataFrame(ETree_m.predict(X_test), columns=['Y_predict'])
#Merge with actual
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month
```

Rank stocks
```python
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
stock_long2['datadate'].value_counts()
```

Calculate returns
```python
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()
```

Merge rf and index
```python
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
```

<img width="397" alt="image" src="https://user-images.githubusercontent.com/72087263/188282705-3e3887c6-c911-45c6-b730-0a803a276f49.png">

**The extra trees regressor decicion tree model produces returns of 36.96% above the market. This is statistically significant with a p-value of 0.**

Sharpe ratio
```python
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR
```

<img width="148" alt="image" src="https://user-images.githubusercontent.com/72087263/188282735-4223c144-c23f-4ef2-9165-8b8bdf7c7677.png">

### Building the Third Model - Histogram-based Gradient Boosting Regression Tree

**Give values for min_samples_leaf and max_iter, and train HistGradientBoostingRegressor using your new training sample. Use the trained model to predict returns based on your new testing sample. Report the average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of the portfolio.**

As with the previous two models, we follow the same process for the Histogram-based Gradient Boosting Regression Tree.

```python
GBR_m= HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=150, early_stopping='True')
#max_iter: The maximum number of iterations of the boosting process#early_stopping: If Yes, the algorithm use 
#internal cross-validation to determine max_iter 
GBR_m.fit(X_train,Y_train)
```

Predict returns
```python
Y_predict=pd.DataFrame(GBR_m.predict(X_test), columns=['Y_predict']) 
#Merge with actual
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month
```

Rank stocks
```python
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
stock_long2['datadate'].value_counts()
```

Calculate returns
```python
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()
```

Merge rf and index
```python
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
```

<img width="396" alt="image" src="https://user-images.githubusercontent.com/72087263/188282806-3b2ee1d2-f115-4a3e-a3da-44d1f5564217.png">

**The gradient boosting regressor decicion tree model produces returns of 36.72% above the market. This is statistically significant with a p-value of 0.**

Sharpe ratio
```python
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR
```
<img width="146" alt="image" src="https://user-images.githubusercontent.com/72087263/188282852-32333732-be59-43dd-a862-27f6db505871.png">

## Conclusion

All three models produce statistically significant returns of around 37% above the market. They all have similar Sharpe Ratios around 8.4-8.5 as well. These decision tree models, therefore, are capable of producing returns greater than the market.
