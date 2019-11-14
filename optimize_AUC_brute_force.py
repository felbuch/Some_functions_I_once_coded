####################################
#By Felipe Buchbinder - 11/14¹2019
#To my friends, with love
####################################


import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score as auc


def sigmoid(z):
    '''An auxiliary function. The logistic function'''
    return(1.0/(1+np.exp(-z)))

def logit(x, w):
    '''An auxiliary function. Returns the odds or, equivalently,
    the scalar product <w,x> = w_1*x_1 + w_2*x_2 + ...'''
    return(x.dot(w))

def rcoef(inf_lim, sup_lim, d = 6):
    '''An auxiliary function. Generates a random vector to serve as possible
    coefficients for the logistic regressions to be assessed'''
    U = npr.rand(d)
    spread = sup_lim - inf_lim
    r = inf_lim + spread * U
    return(r)


def brute_optimize_AUC(x, y, NSim = 1000, min_coef_value = -10, max_coef_value = 10):
    '''This is our main function.
    It generates NSim logistic regressions and calculates their AUC based on the data
    whose independent variables are given in x and whose dependent variable is given in y.
    Both x and y are pandas dataframes.
    Only coefficients between min_coef_value and max_coef_value will be tested,
    so the AUC is maximum within this region.
    Hence, it is safer to use standardized values for x and y,
    as this will keep all coefficients in the same scale.
    '''
    best_auc = 0.0
    best_w = 0.0

    for sim in range(NSim):
        w_sim = rcoef(min_coef_value, max_coef_value)
        y_pred = x_train.apply(logit, w = w_sim, axis = 1)
        auc_sim = auc(y_train, y_pred)
        if auc_sim > best_auc:
            best_auc = auc_sim
            best_w = w_sim
        else:
            pass

    output = {'best_auc': best_auc, 'weights_of_best_model': best_w}
    return(output)

#Sample code
train = pd.read_excel("Train.xlsx")
y_train = train['default']
x_train = train[['age','y_emp','y_add','inc','cc_debt','auto_debt']]
x_train = scale(x_train)
x_train = pd.DataFrame(x_train)

opt = brute_optimize_AUC(x_train, y_train, 10000, -20,20)
print(opt)
