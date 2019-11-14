#Find threshold which minimizes costs

import numpy as np
import pandas as pd

def cost_of_threshold(threshold, true_y, predicted_probs, C_FP, C_FN):
    '''An auxiliary function.
    Calculates the cost (of false positives and false negatives) of a classification model
    if a certain threshold is used. '''
    predicted_y = [1 if p > threshold else 0 for p in predicted_probs]

    df = pd.DataFrame({'y': true_y, 'yhat': predicted_y})
    FP = len(df[(df['y'] == 0) & (df['yhat']==1)])
    FN = len(df[(df['y'] == 1) & (df['yhat']==0)])

    cost = FP * C_FP + FN * C_FN
    return(cost)

def optimize_threshold(true_y, predicted_probs, C_FP, C_FN, mesh = 100):
    '''Finds the best value for the threshold (i.e. the one which incurs in the minimum total cost,
    considering both the costs of false positives and the opportunity costs of false negatives).
    true_y is the true value of y, which is either 0 or 1.
    predicted_probs are the predicted probabilities, as yielded by scikit learn's predict_proba method
    C_FP is the cost of a false positive
    C_FN is the cost of a false negative
    Mesh is the number of thresholds to be tested. The [0,1] range will be partitioned in _mesh_ intervals
    of equal size. Hence, if mesh = 100, we will try threshold 0, 0.01, 0.02, 0.03 etc.

    '''
    best_cost = np.Inf
    best_threshold = np.Inf

    for i in range(0,mesh):
        thresh = i/mesh
        cost = cost_of_threshold(thresh)
        if cost < best_cost:
            best_cost = cost
            best_threshold = thresh

    return({'best_threshold': best_threshold, 'best_cost' : best_cost})
