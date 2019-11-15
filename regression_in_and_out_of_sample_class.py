import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from patsy import dmatrices

class egger_regression():

    '''Defines a class object which will give me
    what I need to do the regressions in prof. Daniel Egger's
    5th assignment at Duke University'''

    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.lm = LinearRegression()
        self.train_yhat = None
        self.test_that = None
        self.rmse_values = None
        self.r2_values = None


    def patsyfy(self, formula, train, test):
        self.train_y, self.train_x = dmatrices(formula, train)
        self.test_y, self.test_x = dmatrices(formula, test)
        pass

    def fit(self):
        self.lm.fit(self.train_x, self.train_y)
        pass

    def predict(self):
        self.train_yhat = self.lm.predict(self.train_x)
        self.test_yhat = self.lm.predict(self.test_x)
        pass

    def rmse(self):
        in_sample = np.sqrt(mse(self.train_y, self.train_yhat))
        out_of_sample = np.sqrt(mse(self.test_y, self.test_yhat))
        self.rmse_values = {'in_sample': in_sample, 'out_of_sample': out_of_sample}
        return(self.rmse_values)

    def r2(self):
        in_sample = r2_score(self.train_y, self.train_yhat)
        out_of_sample = r2_score(self.test_y, self.test_yhat)
        self.r2_values = {'in_sample': in_sample, 'out_of_sample': out_of_sample}
        return(self.r2_values)

    def optimal_threshold(self):

        yhat = sorted(self.train_yhat)
        profit = list()
        for i in range(len(yhat)):
            decision = np.array([-1]*i + [+1]*(len(yhat)-i))
            profit.append(decision.dot(yhat))
        output = {'optimal_threshold': yhat[np.argmax(profit)],
        'maximum_profit': profit[np.argmax(profit)]/len(yhat)}
        return(output)


#Example
#Homework - From Data to Decision, by prof. Daniel Egger @ Duke University
os.chdir('C:\\Users\\Felipe\\Desktop\\Duke MIDS\\From Data to Decision\\HW5')
data = pd.read_excel('data_PL.xlsx')
train = data.iloc[:200,]
test = data.iloc[201:,]
reg = egger_regression()
reg.patsyfy('PL ~ age + y_emp + y_add + inc + cc_debt + auto_debt', train, test)
reg.fit()
reg.predict()
reg.rmse()
reg.r2()
reg.optimal_threshold()
