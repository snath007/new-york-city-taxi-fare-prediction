# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:33:48 2021

@author: DELL
"""

import pandas as pd
# import vaex
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
# from pandas.tseries.holiday import *

# vaex_df = vaex.from_csv('E:/personal projects/new-york-city-taxi-fare-prediction/train.csv', convert = True, chunk_size = 5_000_000)

df = pd.read_csv('E:/personal projects/new-york-city-taxi-fare-prediction/train.csv', nrows = 1200000)


df.info()

########Checking for null values################

df.isnull().sum()

df['dropoff_longitude'].fillna(df['dropoff_longitude'].mean(), inplace = True) #replacing nan values with mean
df['dropoff_latitude'].fillna(df['dropoff_latitude'].mean(), inplace = True)

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])


########Creating additional features from pickup time features##############

df['day_type'] = 1

df['day_type'] = np.where((df['pickup_datetime'].dt.dayofweek==5)|(df['pickup_datetime'].dt.dayofweek==6),0,df['day_type'])

df['time_of_day'] = df['pickup_datetime'].dt.hour

# for i in sorted(df['time_of_day'].unique()):
#     print(i)

########Creating column for US federal holidays#############################

# cal = USFederalHolidayCalendar()

# df['holiday'] = np.where((df['pickup_datetime'].dt.day.isin(cal.holidays(start=df['pickup_datetime'].min(), end=df['pickup_datetime'].max()))),1,0)

# del df['holiday']

########Checking for outliers using box plots##############################

df.boxplot(['pickup_longitude','pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount'])

sns.boxplot(data = df)

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5*iqr
    fence_high = q3 + 1.5*iqr
    df_out = df_in.loc[(df_in[col_name]> fence_low) & (df_in[col_name] < fence_high)]
    return df_out

for col_name in ['pickup_longitude','pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']:
    df = remove_outlier(df, col_name)
    print('Outlier removed for '+str(col_name))
    
df.boxplot(['pickup_longitude','pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount'])


df_copy = df.copy()


########Checking if the data is skewed####################

from scipy.stats import skew


from sklearn.preprocessing import LabelEncoder


df.columns

features = ['pickup_longitude','pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'day_type', 'time_of_day', 'fare_amount']

for feature in features:
    print(skew(df[feature]))

le_pa = LabelEncoder()
le_ti = LabelEncoder()

df['passenger_count'] = le_pa.fit_transform(df['passenger_count'])
df['time_of_day'] = le_ti.fit_transform(df['time_of_day'])



#####Checking the correlation between the dependent and the independent variables#################

df.corr()

plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);

############Prep-ing the test data#####################################

y = df['fare_amount']
X = df.drop(columns='fare_amount')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



###Model fitting#########################################################

input_features = ['pickup_longitude', 'passenger_count', 'day_type', 'time_of_day']

##Linear Regression############################

import statsmodels.api as sm

X_train = sm.add_constant(X_train)

input_features.remove('dropoff_longitude')

model = sm.OLS(y_train, X_train[input_features]).fit()

lin_reg = model.predict(X_test[input_features])

###Checking linear regression assumptions####

##1. Multicollinarity with VIF score:
    
from statsmodels.stats.outliers_influence import variance_inflation_factor 

X.columns

vif_data = pd.DataFrame() 
vif_data["feature"] = ['pickup_longitude', 'passenger_count', 'day_type', 'time_of_day'] 
    
vif_data["VIF"] = [variance_inflation_factor(X[['pickup_longitude', 'passenger_count', 'day_type', 'time_of_day']].values, i) for i in range(len(vif_data["feature"]))] 
  
print(vif_data)

###dropped the insignifivant features based on VIF score####

##2. Normality of the residuals ###########

residual = y_test - lin_reg

sns.distplot(residual)

##3. Homoscedasticity

fig, ax = plt.subplots(figsize=(6,2.5))
ax.scatter(lin_reg, residual)

##4. No autocorrelation of residuals

import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(residual, lags=40 , alpha=0.05)
acf.show()

###Checking model parameters###

model.summary()

###Random Forest#############################

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train[input_features], y_train)

rand_for = regr.predict(X_test[input_features])

###K Nearest#####################

from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor()
neigh.fit(X_train[input_features], y_train)

KNN_res = neigh.predict(X_test[input_features])


#######Comparing MAPE for all 3 models################

from sklearn.metrics import mean_absolute_error

metrics = pd.DataFrame({'linear_regression': mean_absolute_error(lin_reg, y_test), 'random_forest': mean_absolute_error(rand_for, y_test), 'KNN': mean_absolute_error(KNN_res, y_test)}, index=([1]))



