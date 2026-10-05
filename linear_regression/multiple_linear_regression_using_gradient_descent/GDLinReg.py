import numpy as np


class GDLinReg:

    def __init__(
        self,
        epochs=1000,
        lr=0.01,
        tolerance=1e-8,
        scale=True
    ):
        self.epochs = epochs
        self.lr = lr
        self.tolerance = tolerance
        self.scale = scale

        self.__coef_ = None
        self.__intercept_ = None

        self.mean_ = None
        self.std_ = None

        self.n_iterations_ = 0
        self.gradient_norm_ = None


    def get_coef_(self):
        return self.__coef_


    def get_intercept_(self):
        return self.__intercept_


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
                    "X contains a feature with zero standard deviation."
                )

            X = (X - self.mean_) / self.std_

        self.__intercept_ = 0.0
        self.__coef_ = np.ones(n)

        self.n_iterations_ = 0

        for _ in range(self.epochs):

            Y_pred = X @ self.__coef_ + self.__intercept_


            error = Y - Y_pred


            b_slope = (-2 / m) * np.sum(error)
            m_slope = (-2 / m) * (X.T @ error)

            self.gradient_norm_ = max(
                np.max(np.abs(m_slope)),
                abs(b_slope)
            )

            if self.gradient_norm_ < self.tolerance:
                break

            self.__intercept_ -= self.lr * b_slope
            self.__coef_ -= self.lr * m_slope

            self.n_iterations_ += 1

        return self


    def predict(self, X_test):

        X = np.asarray(X_test, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.scale:
            X = (X - self.mean_) / self.std_

        return X @ self.__coef_ + self.__intercept_