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

        # Model parameters
        self.__coef_ = None
        self.__intercept_ = None

        # Scaling parameters
        self.mean_ = None
        self.std_ = None

        # Training information
        self.n_iterations_ = 0
        self.gradient_norm_ = None


    def get_coef_(self):
        return self.__coef_


    def get_intercept_(self):
        return self.__intercept_


    def fit(self, X_train, Y_train):

        # -----------------------------
        # 1. Convert data to NumPy
        # -----------------------------
        X = np.asarray(X_train, dtype=float)
        Y = np.asarray(Y_train, dtype=float)

        # Make sure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        m, n = X.shape

        # -----------------------------
        # 2. Feature Scaling
        # -----------------------------
        if self.scale:

            # Calculate mean of every feature
            self.mean_ = np.mean(X, axis=0)

            # Calculate standard deviation of every feature
            self.std_ = np.std(X, axis=0)

            # Check for constant features
            if np.any(self.std_ == 0):
                raise ValueError(
                    "X contains a feature with zero standard deviation."
                )

            # Standardization
            X = (X - self.mean_) / self.std_

        # -----------------------------
        # 3. Initialize parameters
        # -----------------------------
        self.__intercept_ = 0.0
        self.__coef_ = np.ones(n)

        self.n_iterations_ = 0

        # -----------------------------
        # 4. Gradient Descent
        # -----------------------------
        for _ in range(self.epochs):

            # Prediction
            Y_pred = X @ self.__coef_ + self.__intercept_

            # Error
            error = Y - Y_pred

            # Gradients of MSE
            b_slope = (-2 / m) * np.sum(error)
            m_slope = (-2 / m) * (X.T @ error)

            # -------------------------
            # 5. Convergence Check
            # -------------------------
            self.gradient_norm_ = max(
                np.max(np.abs(m_slope)),
                abs(b_slope)
            )

            if self.gradient_norm_ < self.tolerance:
                break

            # -------------------------
            # 6. Update parameters
            # -------------------------
            self.__intercept_ -= self.lr * b_slope
            self.__coef_ -= self.lr * m_slope

            self.n_iterations_ += 1

        return self


    def predict(self, X_test):

        X = np.asarray(X_test, dtype=float)

        # Make sure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # IMPORTANT:
        # Apply the SAME scaling learned from training data
        if self.scale:
            X = (X - self.mean_) / self.std_

        return X @ self.__coef_ + self.__intercept_