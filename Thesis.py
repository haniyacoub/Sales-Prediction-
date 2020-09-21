#!/usr/bin/env python
# coding: utf-8

# In[2]:


from datetime import datetime, timedelta,date
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split


folder = r'\\smc-filer\divsede\APPS\EVA\notebooks\notebooks\users\Hani Yacoub\\'
name = 'Book1.xlsx'
Main_df = pd.read_excel(folder + name)

folder = r'\\smc-filer\divsede\APPS\EVA\notebooks\notebooks\users\Hani Yacoub\\'
name = 'Prices_ALL.xlsx'
Price_df = pd.read_excel(folder + name)

Price_df['Model_Name'] = Price_df['Model_Name'].drop_duplicates()
Price_df = Price_df.dropna()

folder = r'\\smc-filer\divsede\APPS\EVA\notebooks\notebooks\users\Hani Yacoub\\'
name = 'temp.xlsx'
temp = pd.read_excel(folder + name)

#Df for selected product category (air condition)
folder = r'\\smc-filer\divsede\APPS\EVA\notebooks\notebooks\users\Hani Yacoub\\'
name = 'air.xlsx'
air = pd.read_excel(folder + name)

Price_df['Price'] = Price_df['Price'].replace( ',' , '.',regex=True)

Price_df = Price_df.drop(3665)
Price_df = Price_df.drop(5346)
Price_df = Price_df.drop(6019)
Price_df = Price_df.drop(6191)
Price_df = Price_df.drop(6337)
Price_df = Price_df.drop(6376)
Price_df = Price_df.drop(6377)
Price_df = Price_df.drop(7707)
Price_df = Price_df.drop(7713)
Price_df = Price_df.drop(7716)

Price_df['Price'] = pd.to_numeric(Price_df['Price'])
Main_df = Main_df.merge(Price_df, left_on='ITEM_MODEL', right_on='Model_Name')
Main_df['INVOICE_DATE'] = Main_df['INVOICE_DATE'].astype('datetime64[ns]')

#Choose Duration (2016 - 2020)
Sub_Sec_df = Main_df[108599:201000]

#Plot_Df
#grouped = pd.DataFrame(Sub_Sec_df.reset_index().groupby('INVOICE_DATE')['Price'].sum())
#grouped.plot(figsize=(18,12))

#Choose which Client Class you want (Sole buyers, creditcards and online)
Filter_by_Class = Sub_Sec_df[(Sub_Sec_df['CLIENT_CLASSIFICATION'] == 1) | (Sub_Sec_df['CLIENT_CLASSIFICATION'] ==21 ) | (Sub_Sec_df['CLIENT_CLASSIFICATION'] == 65 )]

#This is used because, some values are very high and abnormal (1500 is an arbitury number)
Filter_by_Class = Filter_by_Class[Filter_by_Class['Price'] < 1500]

df_air_con = Filter_by_Class.reset_index().merge(air, left_on='ITEM_MODEL', right_on='KC-840E')
df_air_con = df_air_con.set_index('INVOICE_DATE').drop(('KC-840E'),axis=1)
df_air_con = df_air_con.reset_index()
df_air_con = df_air_con.reset_index().groupby(pd.Grouper(key="INVOICE_DATE", freq='D')).sum()
df_air_con = df_air_con.merge(temp, left_on='INVOICE_DATE', right_on='DATE')
df_air_con['TAVG'] = (df_air_con['TAVG'] - 32) /1.8
df_air_con['TAVG'] = df_air_con['TAVG'].round(2)


df_air_con['day_of_week'] = df_air_con['DATE'].dt.day_name()
df_air_con = df_air_con.drop((['index']),axis=1)

#Price Disturbution 
df_air_con['Price'].hist(figsize=(18,12))

#Price KDE
df_air_con['Price'].plot(kind='kde',figsize=(18,12))

df_air_con = df_air_con.drop((['level_0','BRANCH_CODE','CLIENT_CODE','CLIENT_CLASSIFICATION','TRANSACTION_CODE']),axis=1)

#Sliced_df (31-12-2015 - 13-07-2020)
df_air_con = df_air_con[:] #342 : 1962

#represent month in date DATE as its first day
df_air_con['DATE'] = df_air_con['DATE'].dt.year.astype('str') + '-' + df_air_con['DATE'].dt.month.astype('str') + '-01'
df_air_con['DATE'] = pd.to_datetime(df_air_con['DATE'])
#groupby date and sum the sales
df_air_con = df_air_con.groupby('DATE').Price.sum().reset_index()


df_air_con.set_index('DATE').plot()

#create a new dataframe to model the difference
df_diff = df_air_con.copy()
#add previous sales to the next row
df_diff['prev_sales'] = df_diff['Price'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['Price'] - df_diff['prev_sales'])

df_diff.set_index('DATE').plot()

#create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_sales'],axis=1)
#adding lags
for inc in range(1,13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)

# Import statsmodels.formula.api
import statsmodels.formula.api as smf
# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 +lag_6 + lag_7 + lag_8  + lag_9 + lag_10 + lag_11 + lag_12', data=df_supervised)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['Price','DATE'],axis=1)
#split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=1, shuffle=False)

y_pred = model.predict(X_test,batch_size=1)

#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print (np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_air_con[-7:].DATE)
act_sales = list(df_air_con[-7:].Price)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['DATE'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)
#for multistep prediction, replace act_sales with the predicted sales

#merge with actual sales dataframe
df_sales_pred = pd.merge(df_air_con,df_result,on='DATE',how='left')
#plot actual and predicted

df_sales_pred.set_index('DATE').plot(figsize=(18,12))


# In[3]:


Main_df


# In[4]:


df_air_con


# In[ ]:




