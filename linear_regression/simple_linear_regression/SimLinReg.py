import matplotlib.pyplot as plt
import numpy as np

class SimLinReg:

    def __init__(self):
        self.__coefficient = None
        self.__intercept = None

    def get_coefficient(self):
        return self._coefficient

    def get_intercept(self):
        return self.__intercept

    def fit(self, X_train, Y_train):
        X_train = np.asarray(X_train).flatten()
        Y_train = np.asarray(Y_train).flatten()

        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError(f"X_train and Y_train must have same length. Got X_train: {X_train.shape[0]}, Y_train: {Y_train.shape[0]}")

        if X_train.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples to fit. Got {X_train.shape[0]}")

        num = 0
        den = 0

        x_mean = X_train.mean()
        y_mean = Y_train.mean()

        for i in range(X_train.shape[0]):
            num += (X_train[i] - x_mean) * (Y_train[i] - y_mean)
            den += (X_train[i] - x_mean) ** 2

        if den == 0:
            raise ValueError("Cannot fit model: all X values are identical")

        self.__coefficient = num / den
        self.__intercept = y_mean - self.__coefficient * x_mean
        return self

    def predict(self, X_test):
        if self.__coefficient is None or self.__intercept is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' first.")
        X_test = np.asarray(X_test)
        return self.__coefficient * X_test + self.__intercept

    def get_coefficient(self):
        return self.__coefficient

    def get_intercept(self):
        return self.__intercept

    def mae(self, Y_test, Y_pred):
        Y_test = np.asarray(Y_test)
        Y_pred = np.asarray(Y_pred)

        if Y_test.shape[0] != Y_pred.shape[0]:
            raise ValueError(f"Y_test and Y_pred must have same length. Got Y_test: {Y_test.shape[0]}, Y_pred: {Y_pred.shape[0]}")

        n = Y_test.shape[0]
        num = 0

        for i in range(n):
            num += abs(Y_test[i] - Y_pred[i])

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

        return np.sqrt(self.mse(Y_test, Y_pred))

    def r2_score(self, Y_test, Y_pred):
        Y_test = np.asarray(Y_test)
        Y_pred = np.asarray(Y_pred)

        if Y_test.shape[0] != Y_pred.shape[0]:
            raise ValueError(f"Y_test and Y_pred must have same length. Got Y_test: {Y_test.shape[0]}, Y_pred: {Y_pred.shape[0]}")

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
        n = X_test.shape[0]
        k = X_test.shape[1] if X_test.ndim > 1 else 1

        if n - 1 - k <= 0:
            raise ValueError(f"Not enough samples for Adjusted R2: need n > {k + 1}, but got n = {n}")

        return 1 - ((1 - r2) * (n - 1)) / (n - 1 - k)

    def show_best_fit_line(self, X, Y, X_train, x_label="Feature", y_label="Target", title="Best Fit Line"):
        if self.__coefficient is None or self.__intercept is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' first.")

        X = np.asarray(X).flatten()
        Y = np.asarray(Y).flatten()
        X_train = np.asarray(X_train).flatten()

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have same length. Got X: {X.shape[0]}, Y: {Y.shape[0]}")

        plt.scatter(X, Y, alpha=0.7, label='Data points')

        # Sort X_train to avoid zigzag line
        X_train_sorted = np.sort(X_train)
        plt.plot(X_train_sorted, self.predict(X_train_sorted), color='red', linewidth=2, label='Best fit line')

        plt.title(title, fontweight='bold')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

