import numpy as np


class GDSimlin:

    def __init__(self, epochs=1000, ln=0.01, tolerance=1e-8, scale=True):

        self.epochs = epochs
        self.ln = ln
        self.tolerance = tolerance
        self.scale = scale

        self.__coef_ = None
        self.__intercept_ = None

        self.loss_history = []
        self.n_iterations_ = 0

        self.mean_ = None
        self.std_ = None

    def get_coef_(self):
        return self.__coef_

    def get_intercept_(self):
        return self.__intercept_

    def fit(self, X_train, Y_train):

        # 1. Convert inputs to NumPy arrays
        X = np.asarray(X_train, dtype=float).ravel()
        Y = np.asarray(Y_train, dtype=float).ravel()

        # 2. Input validation
        if len(X) == 0:
            raise ValueError("Training data cannot be empty")

        if len(X) != len(Y):
            raise ValueError("X and Y must have equal lengths")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError("Training data must contain finite values")

        if self.epochs <= 0 or self.ln <= 0 or self.tolerance < 0:
            raise ValueError("Invalid training hyperparameters")

        n = len(X)

        # 3. Feature standardization
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)

        if self.std_ == 0:
            raise ValueError("X must contain varying values")

        if self.scale:
            X = (X - self.mean_) / self.std_

        # 4. Initialize parameters
        m = 1.0
        b = 0.0

        self.loss_history = []
        self.n_iterations_ = 0

        # 5. Batch Gradient Descent
        for epoch in range(self.epochs):

            # Predictions using ALL training samples
            y_pred = m * X + b

            # Calculate errors
            error = Y - y_pred

            # Calculate MSE gradients
            m_slope = (-2 / n) * np.sum(error * X)
            b_slope = (-2 / n) * np.sum(error)

            # Early stopping based on gradient magnitude
            if max(abs(m_slope), abs(b_slope)) < self.tolerance:
                break

            # Update parameters simultaneously
            m -= self.ln * m_slope
            b -= self.ln * b_slope

            # Detect numerical divergence
            if not np.isfinite(m) or not np.isfinite(b):
                raise FloatingPointError(
                    "Gradient Descent diverged. Reduce learning rate."
                )

            # Calculate updated loss
            y_pred_new = m * X + b

            loss = np.mean((Y - y_pred_new) ** 2)

            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Loss diverged. Reduce learning rate."
                )

            self.loss_history.append(loss)
            self.n_iterations_ += 1

        # 6. Convert coefficients back to original scale
        if self.scale:
            self.__coef_ = m / self.std_
            self.__intercept_ = b - self.__coef_ * self.mean_
        else:
            self.__coef_ = m
            self.__intercept_ = b

        return self

    def predict(self, X_test):

        if self.__coef_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X_test, dtype=float)

        return self.__coef_ * X + self.__intercept_