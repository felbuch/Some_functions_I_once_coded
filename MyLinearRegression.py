#Defines a Linear Regression class from scratch
import numpy as np


class MyLinearModel():

'''This is an OLS multiple linear regression class.
It can fit regressions both with and without an intercept.
If omit_intercept = True, the intercept is omited,
and if it is False, the intercept is included.
By default, the intercept IS included.

All matrices passed to the class are numpy arrays (not pandas dataframes!)

'''

    def __init__(self, omit_intercept = False):
        self.X = None
        self.Y = None
        self.b = None
        self.omit_intercept = omit_intercept
        pass

    def add_intercept_column(self, Matrix):
        '''This is an auxiliary function, for internal use.
        Given a matrix Matrix, it adds a column of 'ones'
        to allow for estimating the intercept of the regression model.'''
        if self.omit_intercept:
            #If we do not want to estimate and intercept, return the matrix unchanged
            return(Matrix)
        else:
            #If we do want to estimate an intercept, add a column of ones to the matrix
            N = len(Matrix) #How many rows does the extra column have
            X_with_ones = np.c_[np.ones(N),Matrix] #Add a column of ones to the inputed matrix
            return(X_with_ones)

    def fit(self, X, Y):
        '''Fits a regression model.
        Covariates are given in dataframe'''
        #Store inputed variables as attributes of the Self
        self.X = X
        self.X = self.add_intercept_column(self.X)
        self.Y = Y
        #Define intermediary matrices for convenience
        #and to simplify notation
        Xt = np.transpose(self.X) #Transpose of X
        Q = np.linalg.inv(Xt @ self.X) #(X'X)^{-1}
        self.b = Q @ Xt @Y #b = [(X'X)^-1]X'y
        return(self.b)

    def predict(self, new_X):
        '''Predicts the value of y on a new dataset, new_X.'''
        new_X = self.add_intercept_column(new_X)
        return new_X @ self.b

    def coef(self):
        '''Returns the model's coefficients.
        The coefficients could also have been obtained as the attribute .b'''
        return self.b

    def residuals(self):
        '''Calculates the residuals of the regression model.
        Residuals are in-sample.'''
        yhat = self.X @ self.b
        return self.Y - yhat

    def stdev_residuals(self):
        '''Returns the standard deviation of the residuals of the fitted model'''
        e = self.residuals()
        n = self.X.shape[0] #Number of observations
        p = self.X.shape[1] #Number of estimated parameters
        sigma2 = e.dot(np.transpose(e)) / (n-p) #Variance of residuals
        return(np.sqrt(sigma2)) #Square root of variance of residuals gives the standard deviation of residuals

    def coef_cov_mat(self):
        '''Covariance matrix of the coefficients.
        This function makes use of the formula
        cov(b) = [(X'X)^{-1}] sigma2.'''
        Xt = np.transpose(self.X)
        Q = np.linalg.inv(Xt @ self.X)
        sigma2 = self.stdev_residuals() ** 2
        cov = Q * sigma2
        return(cov)

    def coef_standard_errors(self):
        '''Coefficient's standard errors.
        The square root of the diagonal of the covariance matrix of the coefficients'''
        cov = self.coef_cov_mat()
        s2 = np.diag(cov)
        s = np.sqrt(s2)
        return(s)

    def R2(self):
        '''(Multivariate) R2 of the regression model (in-sample).'''
        SSR = self.stdev_residuals() ** 2 #Sum of squares of residuals
        SST = np.std(self.Y) ** 2 #Total sum of squares
        return(1 - (SSR / SST))


#Homework tests
random_X = np.random.normal(size = [100,2])
random_Y =  2 * random_X[:,0] + random_X[:,1] + np.random.normal(314,1,100)
new_random_X = np.random.normal(size = [100,2])
random_regression = MyLinearModel()
random_regression.fit(random_X, random_Y)
random_regression.predict(new_random_X)[0:5]
random_regression.residuals()[0:5]

random_regression.stdev_residuals()

random_regression.coef_cov_mat()
random_regression.coef_standard_errors()

random_regression.R2()

import pandas as pd
import statsmodels.formula.api as smfa
data = pd.DataFrame({'y':random_Y, 'x_1': random_X[:,0], 'x_2': random_X[:,1]})
statsmodels_ols = smfa.ols('y ~ x_1 + x_2', data = data).fit()
statsmodels_ols.summary()


#THE STANDARD ERRORS MATCH!!!!!!!!!!!!!!!!!!!!!!!!!!!!! :D :D :D :D :D :D :D :D  :D :D :D :D :D
# :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D
# :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D
#AND SO DOES THE R2!!!!!!!!!!!!!!!!!!!! :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D
# :D :D :D :D :D :D :D :D :D :D :D :D :D:D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D
# :D :D :D :D :D :D :D :D :D :D :D :D :D:D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D
# :D :D :D :D :D :D :D :D :D :D :D :D :D:D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D
# :D :D :D :D :D :D :D :D :D :D :D :D :D:D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D :D


yhat = random_regression.predict(random_X)

import matplotlib.pyplot as plt
plt.scatter(random_Y, yhat)
plt.show()

#Recall results
random_regression.b
#Now same thing ommiting intercept
reg_no_intercept = MyLinearModel(omit_intercept = True)
reg_no_intercept.fit(random_X, random_Y)
