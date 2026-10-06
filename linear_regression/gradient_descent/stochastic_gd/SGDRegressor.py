import numpy as np


class SGDRegressor:

    def __init__(self, epochs=500, lr=0.01, tolerance=1e-4, scale=True):
        self.epochs = epochs
        self.lr = lr
        self.tolerance = tolerance
        self.scale = scale

        self.__coef_ = None
        self.__intercept_ = None

        self.mean_ = None
        self.std_ = None

        self.n_iterations_ = 0
        self.loss_ = None


    def get_intercept_(self):
        return self.__intercept_


    def get_coef_(self):
        return self.__coef_


    def fit(self, X_train, Y_train):

        X = np.asarray(X_train, dtype=float)
        Y = np.asarray(Y_train, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        m, n = X.shape

        if self.scale:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)

            if np.any(self.std_ == 0):
                raise ValueError(
                    "X contains a feature with zero standard deviation"
                )

            X = (X - self.mean_) / self.std_

        self.__intercept_ = 0.0
        self.__coef_ = np.ones(n)

        self.n_iterations_ = 0

        previous_loss = np.inf

        for _ in range(self.epochs):

            indices = np.random.permutation(m)

            for idx in indices:

                Y_pred = (
                    X[idx] @ self.__coef_ + self.__intercept_
                )

                error = Y[idx] - Y_pred

                b_slope = -2 * error
                m_slope = -2 * error * X[idx]

                self.__intercept_ -= self.lr * b_slope
                self.__coef_ -= self.lr * m_slope

            Y_pred_all = X @ self.__coef_ + self.__intercept_

            self.loss_ = np.mean(
                (Y - Y_pred_all) ** 2
            )

            self.n_iterations_ += 1

            if abs(previous_loss - self.loss_) < self.tolerance:
                break

            previous_loss = self.loss_

        return self


    def predict(self, X_test):

        X = np.asarray(X_test, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.scale:
            X = (X - self.mean_) / self.std_

        return (
            X @ self.__coef_
            + self.__intercept_
        )