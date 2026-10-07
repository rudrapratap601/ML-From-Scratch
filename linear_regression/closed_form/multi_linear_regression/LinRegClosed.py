import numpy as np

class LinRegClosed:

    def __init__(self):

        self.__coef_ = None
        self.__intercept_ = None

    def get_coef_(self):
        return self.__coef_

    def get_intercept_(self):
        return self.__intercept_
    
    def fit(self, X_train, Y_train):

        X_train = np.asarray(X_train, dtype = float)
        Y_train = np.asarray(Y_train, dtype = float)

        if X_train.ndim == 1:
            X_train = X_train.repeat(-1, 1)

        # Validate input dimensions
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 1D or 2D array.")

        if Y_train.ndim != 1:
            raise ValueError("Y_train must be a 1D array.")

        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError(
                "X_train and Y_train must have "
                "the same number of samples."
            )

        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty.")

        X_train = np.insert(X_train, 0, 1, axis = 1)

        X_betas = (np.linalg.inv(X_train.T @ X_train)) @ X_train.T @ Y_train

        self.__coef_ = X_betas[1:]
        self.__intercept_ = X_betas[0]

    def predict(self, X_test):

        # Check whether model is trained
        if self.__coef_ is None:
            raise ValueError(
                "Model is not trained. Call fit() first."
            )

        X_test = np.asarray(X_test, dtype=float)

        # Handle a single feature
        if X_test.ndim == 1:
            if len(self.__coef_) != 1:
                raise ValueError(
                    "For multiple features, provide "
                    "X_test as a 2D array."
                )

            X_test = X_test.reshape(-1, 1)

        if X_test.ndim != 2:
            raise ValueError("X_test must be a 1D or 2D array.")

        if X_test.shape[1] != len(self.__coef_):
            raise ValueError(
                "Number of features does not match "
                "the training data."
            )

        Y_pred = X_test @ self.__coef_ + self.__intercept_

        return Y_pred

    def mae(self, Y_test, Y_pred):

        Y_test = np.asarray(Y_test)
        Y_pred = np.asarray(Y_pred)

        if Y_test.shape[0] != Y_pred.shape[0]:
            raise ValueError (f"Y_test and Y_pred must have same length. Got Y_test: {Y_test.shape[0]}, Y_pred: {Y_pred.shape[0]}")

        n = Y_test.shape[0]
        num = 0

        for i in range(n):
            num = num + abs(Y_test[i] - Y_pred[i])

        return num / n

    def mse(self, Y_test, Y_pred):
            Y_test = np.asarray(Y_test)
            Y_pred = np.asarray(Y_pred)
    
            if Y_test.shape[0] != Y_pred.shape[0]:
                raise ValueError(f"Y_test and Y_pred must have same length. Got Y_test: {Y_test.shape[0]}, Y_pred: {Y_pred.shape[0]}")
    
            n = Y_test.shape[0]
            num = 0
    
            for i in range(n):
                num += (Y_test[i] - Y_pred[i]) ** 2
    
            return num / n
    
    def rmse(self, Y_test, Y_pred):

        Y_test = np.asarray(Y_test)
        Y_pred = np.asarray(Y_pred)

        if Y_test.shape[0] != Y_pred.shape[0]:
            raise ValueError (f"Y_test and Y_pred must have same length. Got Y_test: {Y_test.shape[0]}, Y_pred: {Y_pred.shape[0]}")
        
        return np.sqrt(self.mse(Y_test, Y_pred))

    def r2_score(self, Y_test, Y_pred):

        Y_test = np.asarray(Y_test)
        Y_pred = np.asarray(Y_pred)

        if Y_test.shape[0] != Y_pred.shape[0]:
            raise ValueError (f"Y_test and Y_pred must have same length. Got Y_test: {Y_test.shape[0]}, Y_pred: {Y_pred.shape[0]}")

        ssr = 0
        ssm = 0

        n = Y_test.shape[0]

        Y_mean = Y_test.mean()

        for i in range(n):
            ssr += (Y_test[i] - Y_pred[i]) ** 2
            ssm += (Y_test[i] - Y_mean) ** 2

        if ssm == 0:
            raise ValueError("Cannot calculate R2 score: all Y_test values are identical")

        return 1 - (ssr / ssm)

    def adj_r2_score(self, X_test, r2):

        X_test = np.asarray(X_test)

        if X_test.ndim == 1:
            n = X_test.shape[0]
            k = 1

        elif X_test.ndim == 2:
            n = X_test.shape[0]
            k = X_test.shape[1]

        else:
            raise ValueError("X_test must be a 1D or 2D array.")

        if n <= k + 1:
            raise ValueError(
                f"Not enough samples for Adjusted R2: "
                f"need n > {k + 1}, but got n = {n}"
            )

        return 1 - ((1 - r2) * (n - 1)) / (n - k - 1)


    