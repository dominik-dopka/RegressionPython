# Importing the libraries
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from enum import Enum
class Model(Enum):
    Ridge = 0
    Lasso = 1
    ElasticNet = 2

def check_errors():
    #error if wrong number of arguments
    if len(sys.argv) != 2:
        print("Models: Ridge, Lasso, ElasticNet")
        sys.exit("evaluate.py <model>")

    try:
        modelEnum = Model[sys.argv[1]]
    except KeyError:
        print("Wrong model! \nChoose from: Ridge, Lasso, Elastic Net, SVM")
        sys.exit("evaluate.py <model>")
        
    return modelEnum

def evaluateModel(modelEnum):
    # Used to evaluate optimal configuration
    
    # Importing the dataset
    dataset = pd.read_csv('Temperature_Revenue.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    
    # define model evaluation method
    from sklearn.model_selection import RepeatedKFold
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # define grid
    grid = dict()

    if modelEnum == Model.Ridge:
        from sklearn.linear_model import Ridge
        model = Ridge()
        grid['alpha'] = np.arange(0, 1, 0.01)
        print("Ridge Model:")
        
    elif modelEnum == Model.Lasso:
        from sklearn.linear_model import Lasso
        model = Lasso()
        grid['alpha'] = np.arange(0, 1, 0.01)
        print("Lasso Model:")
        
    elif modelEnum == Model.ElasticNet:
        from sklearn.linear_model import ElasticNet
        model = ElasticNet()
        grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
        grid['l1_ratio'] = np.arange(0, 1, 0.01)
        print("Elastic Net Model:")
        
    # define search
    from sklearn.model_selection import GridSearchCV
    search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X, y)
    # summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    
if __name__ == "__main__":
    modelEnum = check_errors()
    
    evaluateModel(modelEnum)
            