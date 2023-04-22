from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from bayes_opt import BayesianOptimization

def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

class GBR():
    def fit(self,X,y):
        gb_model = GradientBoostingRegressor(random_state=42)

        # Set up a search space for hyperparameters
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'learning_rate' : [0.1],
            'subsample': [1.0]
        }

        # Set up a 5-fold cross-validation object
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Use random search to find the best hyperparameters
        grid_search = RandomizedSearchCV(gb_model, param_distributions = param_grid, cv=kfold, scoring='neg_mean_absolute_error', n_iter = 5)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        return best_model
    
    def predict(self,model, X):
        y = model.predict(X)
        y = y.tolist()
        return y
    
class MLR():
    def fit(self,X,y):
        reg = LinearRegression().fit(X, y)
        return reg
    
    def predict(self,model, X):
        y = model.predict(X)
        y = y.tolist()
        return y 
    
class LASSO():
    def fit(self,X,y):
        param_grid = {'alpha': np.logspace(-10, 10, 100)}
        lasso = Lasso(max_iter=10000, tol=1e-4)
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        grid_search = GridSearchCV(lasso, param_grid, scoring=mae_scorer, cv=5)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_['alpha']

        best_lasso = Lasso(alpha=best_params)
        bestfit = best_lasso.fit(X, y)
        return bestfit
    def predict(self, model, X):
        y = model.predict(X)
        y = y.tolist()
        return y 
    
class RidgeReg():
    def fit(self,X,y):
        ridge_model = Ridge(max_iter=10000, tol=1e-4, solver='saga')
        # Set up a grid of hyperparameters to search over
        param_grid = {'alpha': np.logspace(-10, 10, 100)}

        # Set up a 5-fold cross-validation object
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Use cross-validation to find the best hyperparameters
        grid_search = GridSearchCV(ridge_model, param_grid, cv=kfold, scoring='neg_mean_absolute_error')
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        return best_model
    def predict(self, model, X):
        y = model.predict(X)
        y = y.tolist()
        return y
    
class KRR():
    def fit(self,X,y):
        kr_model = KernelRidge()

        # Set up a search space for hyperparameters
        param_dist = {
            'kernel': ['linear', 'rbf', 'poly'],
            'alpha': np.logspace(-5, 5, 10),
            'gamma': np.logspace(-5, 5, 10)
        }

        # Set up a 5-fold cross-validation object
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Use random search to find the best hyperparameters
        random_search = RandomizedSearchCV(kr_model, param_distributions=param_dist, cv=kfold, n_iter=10, scoring='neg_mean_absolute_error', random_state=42)
        random_search.fit(X, y)

        best_model = random_search.best_estimator_
        return best_model
    
    def predict(self, model, X):
        y = model.predict(X)
        y = y.tolist()
        return y
    
class GPR():

    def __init__(self, df, target) -> None:
        self.df = df
        self.target = target
        X = df.drop(target, axis=1)  
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test  

    def fit(self,X,y):

        def objective_func(length_scale, noise_level):
            kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
            gp = GaussianProcessRegressor(kernel=kernel)
            gp.fit(self.X_train, self.y_train)
            y_pred = gp.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            return -mae
        
        # Define the search space
        bounds = {
            'length_scale': (0.001, 1000),
            'noise_level': (1e-10, 100)
        }

        # Initialize the Bayesian optimization
        bo = BayesianOptimization(objective_func, bounds)

        # Run the optimization
        bo.maximize(init_points=5, n_iter=5)

        # Get the optimal hyperparameters
        best_params = bo.max['params']

        kernel = RBF(length_scale=best_params['length_scale']) + WhiteKernel(noise_level=best_params['noise_level'])
        gp = GaussianProcessRegressor(kernel=kernel)
        bestfit = gp.fit(X, y)
        return bestfit
    
    def predict(self, model, X):
            y = model.predict(X)
            y = y.tolist()
            return y    

class RFR():

    def fit(self, X, y):
        rf_model = RandomForestRegressor(random_state=42)

        # Set up a search space for hyperparameters
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        # Set up a 5-fold cross-validation object
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Use random search to find the best hyperparameters
        grid_search = RandomizedSearchCV(rf_model, param_distributions = param_grid, cv=kfold, scoring='neg_mean_absolute_error', n_iter = 5)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        return best_model
    
    def predict(self, model, X):
            y = model.predict(X)
            y = y.tolist()
            return y