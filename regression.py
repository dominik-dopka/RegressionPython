import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def initialLinearSetup(samples = 1000, features = 1):
    from sklearn.datasets import make_regression
    
    x, y = make_regression(n_samples=samples, n_features=features, noise=5.0)
    df = pd.DataFrame(data=x)
    
    from sklearn.model_selection import train_test_split
    train_data = train_test_split(x, y, test_size = 0.25)
    
    if features != 1:
        new_array = []
        i = 0
        while i < features:
            new_array = []
            j = 0
            
            while j < len(train_data[2]):
                new_array.append(train_data[0][j][i])
                j += 1
            
            plt.scatter(new_array, train_data[2], marker='.')
            i += 1
            
    else:
        plt.scatter(train_data[0], train_data[2], marker='.')

    plt.title('Look of linear problem')
    plt.show()
    
    return df, x, y, train_data
    
def initialCurveSetup(samples = 300):    
    x = 10 * np.random.RandomState(1).rand(samples)
    x = np.sort(x)
    y = 2 * x - 5 + np.random.RandomState(1).randn(samples)
    y = np.sin(x)
    
    x = x.reshape(-1, 1)
    
    df = pd.DataFrame(data=x)
    
    from sklearn.model_selection import train_test_split
    train_data = train_test_split(x, y, test_size = 0.25)
    
    plt.scatter(train_data[0], train_data[2], marker='.')
    plt.title('Look of sinusoid problem')
    plt.show()
    
    return df, x, y, train_data
    
def initialExampleDataSetup():
    data = pd.read_csv("movie_metadata.csv")
    
    # print(data)
    
    df = pd.DataFrame(data)
    x = df.dtypes[df.dtypes!='object'].index
    x = df[x]
    x = x.fillna(0)
    
    y = x['imdb_score']
    x.drop(['imdb_score'],axis=1,inplace=True)
    
    from sklearn.model_selection import train_test_split
    train_data = train_test_split(x, y, test_size = 0.05)
    
    for column in x:
        plt.scatter(x[column], y, marker='.')
    
    plt.title('Look of example data problem')
    plt.show()
    
    return df, x, y, train_data
    
def residualPlot(y_predict, title):
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    preds = pd.DataFrame({"preds":y_predict, "true":y_train})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds.plot(x = "preds", y = "residuals",kind = "scatter", marker='.')
    # plt.plot(y, y, linestyle=':')
    plt.axhline(y=0, color='black', linestyle='-', linewidth = 1)
    plt.title(title)
    plt.show()
    
def score(y_predict, name):
    from sklearn.metrics import mean_squared_error, r2_score, max_error
    
    score_mse = mean_squared_error(y_train, y_predict)
    score_r2 = r2_score(y_train, y_predict)
    score_me = max_error(y_train, y_predict)
    
    score_data.at[name, 'Mean squared error'] = score_mse
    score_data.at[name, 'Coefficient of determination'] = score_r2
    score_data.at[name, 'Max error'] = score_me
    
def scorePlot():
    score_data['Mean squared error'].plot(kind='bar')
    plt.title('Mean squared error')
    plt.show()
    
    score_data['Coefficient of determination'].plot(kind='bar')
    plt.title('Coefficient of determination')
    plt.show()
    
    score_data['Max error'].plot(kind='bar')
    plt.title('Max error')
    plt.show()
    
def linearRegression():
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    print(x_train)
    print(y_train)
    
    model.fit(x_train,y_train)
    y_predict = model.predict(x_train)
    
    residualPlot(y_predict, "Residual plot in Linear Regressor")
    score(y_predict, "Linear")
    
def ridgeRegression():   
    from sklearn.linear_model import Ridge
    model = Ridge()
    model.fit(x_train,y_train)
    y_predict = model.predict(x_train)
    
    residualPlot(y_predict, "Residual plot in Ridge Regressor")
    score(y_predict, "Ridge")
    
def lassoRegression():
    from sklearn.linear_model import Lasso
    # model = Lasso()
    model = Lasso(alpha=0.38)
    model.fit(x_train,y_train)
    y_predict = model.predict(x_train)
    
    residualPlot(y_predict, "Residual plot in Lasso Regressor")
    score(y_predict, "Lasso")
    
def elasticNet():
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=0.1, l1_ratio=0.86)
    # model = ElasticNet()
    model.fit(x_train,y_train)
    y_predict = model.predict(x_train)
    
    residualPlot(y_predict, "Residual plot in Elastic Net Regressor")
    score(y_predict, "Elastic Net")
    
def decisionTreeRegression():   
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(max_depth = 5)
    # model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)
    
    residualPlot(y_predict, "Residual plot in Decision Tree Regressor")
    score(y_predict, "Decision Tree")
    
def kNeighborsRegression():
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)
    
    residualPlot(y_predict, "Residual plot in K-Neighbors Regressor")
    score(y_predict, "K-Neighbors")
    
def svmRegression(isCurve):
    from sklearn.svm import SVR
    from sklearn import preprocessing
    from sklearn.pipeline import make_pipeline
    if isCurve == True:
        model = SVR()
    else:
        model = make_pipeline(
            preprocessing.StandardScaler(),
            SVR(kernel='rbf', epsilon=0.01, C=100, gamma = 0.01)
        )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)
    
    residualPlot(y_predict, "Residual plot in SVM Regressor")
    score(y_predict, "SVM")
    

    
if __name__ == "__main__":
    #Linear regression data
    # df, x, y, train_data = initialLinearSetup()
    
    # Multiple linear regression data
    # df, x, y, train_data = initialLinearSetup(1000, 10)
    
    #Curve data
    # df, x, y, train_data = initialCurveSetup()
    
    #Example data
    df, x, y, train_data = initialExampleDataSetup() 
    
    
    x_train, x_test, y_train, y_test = train_data
    
    col = ['Mean squared error', 'Coefficient of determination', 'Max error']
    models = ['Linear', 'Ridge', 'Lasso', 'Elastic Net', 'Decision Tree', 'K-Neighbors', 'SVM'] 
    score_data = pd.DataFrame(columns = col, index = models)
    
    linearRegression()
    ridgeRegression()
    lassoRegression()
    elasticNet()
    decisionTreeRegression()
    kNeighborsRegression()
    
    # Performs better for linear
    svmRegression(False)
    
    # Performs better for curve
    # svmRegression(True)
    
    scorePlot()