#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:35:06 2021

@authors: Juan Manuel Vargas and Mohamed A. Bahloul 
"""
#%% Libraries

import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RationalQuadratic, RBF, Matern,DotProduct, WhiteKernel,ConstantKernel
from scipy.integrate import simps
from scipy.stats import skew
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm

from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestRegressor

from scipy.integrate import simps
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import diags
from sklearn.linear_model import LinearRegression
import random
import statsmodels.api as sm
from sklearn.svm import SVR


#%% Functions
    # Importing thscsae libraries
def RMSE(y_pred_lr,y):
  return np.sqrt(np.sum(((y_pred_lr-y)**2/len(y_pred_lr))))



#%% Load 2D-SCSA features extracted

#Data characteristic definition

med_f='no'
norm=1


if snr!="no":
    snr=int(snr)



print('RUNNING FOR  SIGNAL_'+type_sig+'_'+type_wav+' norm='+str(norm)+' snr='+str(snr))

features=pd.read_csv('./Data/2D-Text_Spec'+type_wav+'_'+type_sig+'_SNR='+str(snr)+'_wins='+str(ws)+'/features_final.csv',header=None)
y=pd.read_csv("./Data/PWV_cf.csv",header=None)


X_f=features.values
PWV_cf=np.transpose(y.values)         
if med_f=='yes':        
    medical_data=pd.read_csv('./Data/pwdb_haemod_params.csv')    
    features_m=[' age [years]',' HR [bpm]',' SBP_b [mmHg]',' DBP_b [mmHg]',' MBP_b [mmHg]',' PP_b [mmHg]',' PWV_cf [m/s]']
    medical_features=medical_data[features_m]
    X_f=np.concatenate((X_f,medical_features),axis=1)


#%% Data pre-proccesing

X_train,X_test,PWV_cf_train,PWV_cf_test=train_test_split(X_f,PWV_cf, test_size=0.3, random_state=31)


 
# Data standarization

sc = StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)

X_test=sc.transform(X_test)

y_test=PWV_cf_test.reshape(-1,)



pickle.dump(sc, open('./Data/2D-Text_Spec'+type_wav+'_'+type_sig+'_SNR='+str(snr)+'_wins='+str(ws)+"/scaler_LR.pkl", "wb"))




#%% Random Forest Training and testing


# print('Strart classification using RF')
# #CV using random search
# # C parameter
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(2, 20, num = 10)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [40, 60, 80]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [20, 30, 40]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# # Create the random grid
# random_grid_rf = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rf = RandomForestRegressor()

# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid_rf, n_iter = 5, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# # Fit the model
# rf_random.fit(X_train,PWV_cf_train.reshape(-1,))



# # Test model
# y_pred_rf =rf_random.predict(X_test)





# RMSE_RF=RMSE(y_pred_rf,y_test)

# # Estimated vs measured
# m, b = np.polyfit(y_pred_rf, y_test,1)
# X = sm.add_constant(y_pred_rf)
# est = sm.OLS(y_test, X)
# est2 = est.fit()
# p_value =  est2.pvalues[1]

# r_squared = est2.rsquared

# RMSE_rf=[RMSE_RF]
# R2_rf=[r_squared]
# res_rf=[[RMSE_rf,R2_rf]]
# RF_result= pd.DataFrame(res_rf, columns = ['RMSE','R2'])


# savedata = [y_test, y_pred_rf]
# df_savedata = pd.DataFrame(savedata)

# y_pred_rf=list(y_pred_rf)
# y_test=list(y_test)



# pickle.dump(rf_random, open('./Data/2D-Text_Spec'+type_wav+'_'+type_sig+'_SNR='+str(snr)+'_wins='+str(ws)+"/model_RF.pkl", "wb"))





    

# print('Ending classification using RF')




# #%% Gradient Boost Regression Training and testing

# print('Strart classification using GB')

# # Loss function
# loss = ['ls', 'lad', 'huber']
# # Learning rate
# learning_rate = [0.01, 0.02, 0.05, 0.1]
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(2, 10, num = 5)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [40, 60, 80]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [20, 30, 40]

# # Create the random grid
# random_grid = {'loss': loss, 
#                'learning_rate': learning_rate,
#                'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                }
# # First create the base model to tune
# gb = GradientBoostingRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# GBR = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, n_iter = 2, cv = 5, random_state=42, n_jobs = -1)

# # Fit the model
# GBR.fit(X_train,PWV_cf_train.reshape(-1,))



# # Test model
# y_pred_gbr =GBR.predict(X_test)





# RMSE_gbr=RMSE(y_pred_gbr,y_test)

# # Estimated vs measured
# m, b = np.polyfit(y_pred_gbr, y_test,1)
# X = sm.add_constant(y_pred_gbr)
# est = sm.OLS(y_test, X)
# est2 = est.fit()
# p_value =  est2.pvalues[1]

# r_squared = est2.rsquared

# RMSE_gbr=[RMSE_gbr]
# R2_gbr=[r_squared]
# res_gbr=[[RMSE_gbr,R2_gbr]]
# gbr_result= pd.DataFrame(res_gbr, columns = ['RMSE','R2'])


# savedata = [y_test, y_pred_gbr]
# df_savedata = pd.DataFrame(savedata)

# y_pred_gbr=list(y_pred_gbr)
# y_test=list(y_test)


# pickle.dump(GBR, open('./Data/2D-Text_Spec'+type_wav+'_'+type_sig+'_SNR='+str(snr)+'_wins='+str(ws)+"/model_GB.pkl", "wb"))



# print('Ending classification using GB')




# #%% Gaussian Regression Training and testing


# print('Strart classification using GR')

# random_grid = {'kernel':[
# 1.0 * RBF(1, (1, 1)) + 1.0 * Matern(2, (2, 2),2.5),DotProduct() + WhiteKernel(),ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")], 
#                }
# gr = GaussianProcessRegressor()
# # search across 100 different combinations, and use all available cores
# gr= RandomizedSearchCV(estimator = gr, param_distributions = random_grid, n_iter = 2, cv = 5, random_state=42)

# gr.fit(X_train,PWV_cf_train.reshape(-1,))




# y_pred_gr =gr.predict(X_test)






# RMSE_gr=RMSE(y_pred_gr,y_test)

# # Estimated vs measured
# m, b = np.polyfit(y_pred_gr, y_test,1)
# X = sm.add_constant(y_pred_gr)
# est = sm.OLS(y_test, X)
# est2 = est.fit()
# p_value =  est2.pvalues[1]

# r_squared = est2.rsquared

# RMSE_gr=[RMSE_gr]
# R2_gr=[r_squared]
# res_gr=[[RMSE_gr,R2_gr]]
# gr_result= pd.DataFrame(res_gr, columns = ['RMSE','R2'])


# savedata = [y_test, y_pred_gr]
# df_savedata = pd.DataFrame(savedata)

# y_pred_gr=list(y_pred_gr)
# y_test=list(y_test)




# pickle.dump(gr, open('./Data/2D-Text_Spec'+type_wav+'_'+type_sig+'_SNR='+str(snr)+'_wins='+str(ws)+"/model_GR.pkl", "wb"))





# print('Ending classification using GR')


#%% MLP Training and testing

print('Strart classification using MLP')
#CV using random search
# C parameter
random_grid= [{'solver':['lbfgs','lbfgs']}]


clf=MLPRegressor(solver='lbfgs',activation='tanh', alpha=0.437, learning_rate='adaptive',hidden_layer_sizes=[15,11])
model_mlp =RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 2, cv = 5, random_state=42, n_jobs = -1)
     


model_mlp.fit(X_train,PWV_cf_train.reshape(-1,))



y_pred_mlp =model_mlp.predict(X_test)






RMSE_mlp=RMSE(y_pred_mlp,y_test)

# Estimated vs measured
m, b = np.polyfit(y_pred_mlp, y_test,1)
X = sm.add_constant(y_pred_mlp)
est = sm.OLS(y_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]

r_squared = est2.rsquared

RMSE_mlp=[RMSE_mlp]
R2_mlp=[r_squared]
res_mlp=[[RMSE_mlp,R2_mlp]]
mlp_result= pd.DataFrame(res_mlp, columns = ['RMSE','R2'])


savedata = [y_test, y_pred_mlp]
df_savedata = pd.DataFrame(savedata)

y_pred_mlp=list(y_pred_mlp)
y_test=list(y_test)


pickle.dump(model_mlp, open('./Data/2D-Text_Spec'+type_wav+'_'+type_sig+'_SNR='+str(snr)+'_wins='+str(ws)+"/model_MLP.pkl", "wb"))



print('Ending classification using MLP')

#%% Linear regression Training and testing

# print("Strart classification using LR")

# # Set hyper-parameter space
# hyper_params = [{"fit_intercept":[True,False]}]

# # Create linear regression model 
# lm = LinearRegression()
# # Create RandomSearchCV() with 5-fold cross-validation
# model_cv = RandomizedSearchCV(estimator = lm,param_distributions=hyper_params,n_iter=2,cv = 5,random_state=42)  

# # Fit the model
# model_cv.fit(X_train,PWV_cf_train.reshape(-1,))



# # Test model
# y_pred_lr =model_cv.predict(X_test)



# RMSE_LR=RMSE(y_pred_lr,y_test)

# # Estimated vs measured
# m, b = np.polyfit(y_pred_lr, y_test,1)
# X = sm.add_constant(y_pred_lr)
# est = sm.OLS(y_test, X)
# est2 = est.fit()
# p_value =  est2.pvalues[1]

# r_squared_LR = est2.rsquared

# RMSE_lr=[RMSE_LR]
# R2_lr=[r_squared_LR]
# y_pred_lr=list(y_pred_lr)
# y_test=list(y_test)

# pickle.dump(model_cv, open('./Data/2D-Text_Spec'+type_wav+'_'+type_sig+'_SNR='+str(snr)+'_wins='+str(ws)+"/model_LR.pkl", "wb"))


# print("Ending classification using LR")

#%%

#SVR
print('Strart classification using SVR')
#CV using random search
# C parameter
C= [int(x) for x in np.linspace(100, 400, num = 5)]

# Create the random grid
random_grid = {'C':C, 
               }
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
svr= SVR()

SVR_rs = RandomizedSearchCV(estimator = svr, param_distributions = random_grid, n_iter = 5, cv = 5, random_state=42, n_jobs = -1)


# predict with the best parameters from random search

SVR_rs.fit(X_train,PWV_cf_train.reshape(-1,))



y_pred_svr = SVR_rs.predict(X_test)


RMSE_svr=RMSE(y_pred_svr,y_test)

# Estimated vs measured
m, b = np.polyfit(y_pred_svr, y_test,1)
X = sm.add_constant(y_pred_svr)
est = sm.OLS(y_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]

r_squared_svr = est2.rsquared

RMSE_svr=[RMSE_svr]
R2_svr=[r_squared_svr]
y_pred_svr=list(y_pred_svr)
y_test=list(y_test)

pickle.dump(SVR_rs, open('./Data/2D-Text_Spec'+type_wav+'_'+type_sig+'_SNR='+str(snr)+'_wins='+str(ws)+"/model_SVR.pkl", "wb"))




print('Ending classification using SVR')



#%% Vec


# RMSE=[RMSE_rf,RMSE_gbr,RMSE_gr,RMSE_mlp,RMSE_lr,RMSE_svr]
# R2=[R2_rf,R2_gbr,R2_gr,R2_mlp,R2_lr,R2_svr]
# y_pred=[list(y_pred_rf),list(y_pred_gbr),list(y_pred_gr),list(y_pred_mlp),list(y_pred_lr),list(y_pred_svr)]
# y_test=list(y_test)

RMSE=np.asarray([RMSE_mlp,RMSE_svr])
R2=np.asarray([R2_mlp,R2_svr])
y_pred=[np.asarray(y_pred_mlp).reshape(-1,1),np.asarray(y_pred_svr).reshape(-1,1)]
y_test=np.asarray(y_test).reshape(-1,1)
