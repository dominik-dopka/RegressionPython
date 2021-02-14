# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd     

def setupModels():
    # Importing the dataset
    # dataset = pd.read_csv('Temperature_Revenue.csv')
    # dataset = pd.read_csv('Salary_Data.csv')
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 1].values
    
    # generate regression dataset
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=200, n_features=1, noise=20.0)
    # print(X)
    # print(y)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)
    
    # Linear Models
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()
    
    from sklearn.linear_model import Ridge
    ridgeModel = Ridge(alpha=0.99)
    
    from sklearn.linear_model import Lasso
    lassoModel = Lasso(alpha=0.38)
    
    from sklearn.linear_model import ElasticNet
    elasticNetModel = ElasticNet(alpha=0.1, l1_ratio=0.86)
    
    # Nonlinear Models
    from sklearn.tree import DecisionTreeRegressor
    decisionTree_model = DecisionTreeRegressor()
    
    from sklearn.neighbors import KNeighborsRegressor
    nearestNeighborsModel = KNeighborsRegressor()
    
    from sklearn.svm import SVR
    from sklearn import preprocessing
    from sklearn.pipeline import make_pipeline
       
    svrModel = make_pipeline(
        preprocessing.StandardScaler(),
        SVR(kernel='rbf', epsilon=0.01, C=100, gamma = 0.01)
    )
    
    # https://stackoverflow.com/questions/29819428/normalization-or-standardization-data-input-for-svm-scikitlearn
    
    # Linear Train
    linear_model.fit(X_train, y_train)
    ridgeModel.fit(X_train, y_train)
    lassoModel.fit(X_train, y_train)
    elasticNetModel.fit(X_train, y_train)
    
    # Nonlinear Training
    decisionTree_model.fit(X_train, y_train)
    nearestNeighborsModel.fit(X_train, y_train)
    svrModel.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred_linear = linear_model.predict(X_test)
    y_pred_ridge = ridgeModel.predict(X_test)
    y_pred_lasso = lassoModel.predict(X_test)
    y_pred_elasticNet = elasticNetModel.predict(X_test)
    
    y_pred_decisionTree = decisionTree_model.predict(X_test)
    y_pred_nearestNeighbors = nearestNeighborsModel.predict(X_test)
    y_pred_svr = svrModel.predict(X_test)
    
    return X_test, y_test, y_pred_linear, y_pred_ridge, y_pred_lasso, y_pred_elasticNet, y_pred_decisionTree, y_pred_nearestNeighbors, y_pred_svr
    
def displayLinearPlot(X_test, y_test, y_pred_linear, y_pred_ridge, y_pred_lasso, y_pred_elasticNet):
    fig = plt.figure('Linear Regression')

    plt.subplot(221)
    plt.title('Linear Regression')
    plt.scatter(X_test, y_test, color = 'red')
    plt.scatter(X_test, y_pred_linear, color = 'green')
    
    plt.subplot(222)
    plt.title('Ridge Regression')
    plt.scatter(X_test, y_test, color = 'red')
    plt.scatter(X_test, y_pred_ridge, color = 'green')
    
    plt.subplot(223)
    plt.title('Lasso Regression')
    plt.scatter(X_test, y_test, color = 'red')
    plt.scatter(X_test, y_pred_lasso, color = 'green')
    
    plt.subplot(224)
    plt.title('Elastic Net Regression')
    plt.scatter(X_test, y_test, color = 'red')
    plt.scatter(X_test, y_pred_elasticNet, color = 'green')
    
    plt.show()
    
def displayNonlinearPlot(X_test, y_test, y_pred_decisionTree, y_pred_nearestNeighbors, y_pred_svr):
    fig = plt.figure('Nonlinear Regression')

    plt.subplot(131)
    plt.title('Decision Tree Regression')
    plt.scatter(X_test, y_test, color = 'red')
    plt.scatter(X_test, y_pred_decisionTree, color = 'green')
    
    plt.subplot(132)
    plt.title('K-Nearest Neighbors Regression')
    plt.scatter(X_test, y_test, color = 'red')
    plt.scatter(X_test, y_pred_nearestNeighbors, color = 'green')
    
    plt.subplot(133)
    plt.title('Support Vector Machines Regression')
    plt.scatter(X_test, y_test, color = 'red')
    plt.scatter(X_test, y_pred_svr, color = 'green')
    
    plt.show()
    
def score(y_test, y_pred):

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
    
    score_mse = mean_squared_error(y_test, y_pred)
    score_r2 = r2_score(y_test, y_pred)
    score_mae = mean_absolute_error(y_test, y_pred)
    score_me = max_error(y_test, y_pred)
    
    scores = [score_mse, score_r2, score_mae, score_me]
    
    print('Mean squared error: %.2f' % score_mse)   
    print('Coefficient of determination: %.2f' % score_r2) 
    print('Mean absolute error: %.2f' % score_mae)
    print('Max error: %.2f' % score_me)
    
    return scores
    
def displayScorePlot(score_list):
    score_mse, score_r2, score_mae, score_me = []

    for score in score_list:
        score_mse.append(score[0])
        score_r2.append(score[1])
        score_mae.append(score[2])
        score_me.append(score[3])
        
        # print('\nMean squared error: %.2f' % score[0])   
        # print('Coefficient of determination: %.2f' % score[1]) 
        # print('Mean absolute error: %.2f' % score[2])
        # print('Max error: %.2f' % score[3])

if __name__ == "__main__":

    X_test, y_test, y_pred_linear, y_pred_ridge, y_pred_lasso, y_pred_elasticNet, y_pred_decisionTree, y_pred_nearestNeighbors, y_pred_svr = setupModels()
    
    print("Linear Model:")
    score_linear = score(y_test, y_pred_linear)
    
    print("\nRidge Model:")
    score_ridge = score(y_test, y_pred_ridge)
    
    print("\nLasso Model:")
    score_lasso = score(y_test, y_pred_lasso)
    
    print("\nElastic Net Model:")
    score_elasticNet = score(y_test, y_pred_elasticNet)
    
    print("\nDecision Tree Model:")
    score_decisionTree = score(y_test, y_pred_decisionTree)
    
    print("\nK-Nearest Neighbors Model:")
    score_nearestNeighbors = score(y_test, y_pred_nearestNeighbors)
    
    print("\nSupport Vector Machines Model:")
    score_svm = score(y_test, y_pred_svr)
    
    score_list = [score_linear, score_ridge, score_lasso, score_elasticNet, score_decisionTree, score_nearestNeighbors, score_svm]
    # print(score_list)
    
    # displayScorePlot(score_list)
    
    displayLinearPlot(X_test, y_test, y_pred_linear, y_pred_ridge, y_pred_lasso, y_pred_elasticNet)
    displayNonlinearPlot(X_test, y_test, y_pred_decisionTree, y_pred_nearestNeighbors, y_pred_svr)
        