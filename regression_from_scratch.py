import numpy as np

class OLS:
    def __init__(self,fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        self.beta = None

    def add_intercept(self, X):
        return np.column_stack((np.ones(X.shape[0]), X))

    def fit(self, X, y):

        X = np.asarray(X,dtype=float)
        y = np.asarray(y,dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            X_design = self.add_intercept(X)
        else:
            X_design = X

        XtX = X_design.T @ X_design
        Xty = X_design.T @ y

        self.beta = np.linalg.solve(XtX, Xty)

        if self.fit_intercept:
            self.intercept = self.beta[0]
            self.coefficients = self.beta[1:]

        return self

    def predict(self, X):

        X = np.asarray(X,dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            X_design = self.add_intercept(X)
        else:
            X_design = X

        return X_design @ self.beta

    def mse(self,X,y):
        y_pred = self.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        return mse

    def rmse(self,X,y):
        mse = self.mse(X,y)
        rmse = np.sqrt(mse)
        return rmse

    def r_2(self,X,y):
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        SSE = np.sum((y - y_pred)**2)
        SST = np.sum((y - y_mean)**2)
        r_2 = 1 - (SSE/SST)
        return r_2

    def adjusted_r_squared(self, X, y):
        n = len(y)
        if self.fit_intercept:
            p = len(self.coefficients) + 1
        else:
            p = len(self.coefficients)

        r2 = self.r_2(X, y)

        return 1 - (1 - r2) * (n - 1) / (n - p)

