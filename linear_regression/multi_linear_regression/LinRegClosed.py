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


    