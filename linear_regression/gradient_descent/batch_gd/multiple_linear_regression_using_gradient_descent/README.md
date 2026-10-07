# 📐 Multiple Linear Regression using Batch Gradient Descent

A complete implementation of **Multiple Linear Regression using Batch Gradient Descent** built from scratch using only Python and NumPy — no scikit-learn for the model itself.

---

## 📌 Overview

This project demonstrates how Multiple Linear Regression works when coefficients are learned iteratively using **Batch Gradient Descent** (BGD), an optimization algorithm that updates parameters using the entire training dataset in each iteration.

- Multiple coefficients learned through iterative optimization
- Batch Gradient Descent with configurable learning rate and convergence criteria
- Feature scaling (standardization) for faster convergence
- Early stopping based on gradient norm
- Supports any number of features

### Key Difference from Closed-Form Solution

Unlike the [closed-form OLS solution](../../../multi_linear_regression) which computes `β = (XᵀX)⁻¹Xᵀy` in one step, Gradient Descent:
- **Iteratively** updates parameters to minimize the loss function
- Requires tuning the **learning rate** (`lr`)
- Avoids computing matrix inverses (more efficient for high-dimensional data)
- Demonstrates the **optimization process** step-by-step

---

## 🧮 The Mathematics Behind It

### 1. Multiple Linear Regression Equation

The model predicts the target using multiple features:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ
```

In vectorized form:

```
ŷ = Xβ + β₀
```

Where:

- `ŷ` = predicted values (n × 1)
- `X` = feature matrix (n × k)
- `β` = coefficient vector (k × 1)
- `β₀` = intercept (scalar)

---

### 2. Loss Function: Mean Squared Error (MSE)

We want to minimize the MSE loss:

```
              1   n
L(β₀, β) = ——— Σ (yᵢ - ŷᵢ)²
              n  i=1
```

Where:
- `n` = number of training samples
- `yᵢ` = actual value
- `ŷᵢ = Xᵢβ + β₀` = predicted value

---

### 3. Gradient Descent Update Rules

To minimize the loss, we compute the **partial derivatives** (gradients) with respect to each parameter:

#### Gradient with respect to the coefficient vector (β):

```
∂L      -2
—— = ———— Xᵀ(y - ŷ)
∂β      n
```

#### Gradient with respect to the intercept (β₀):

```
∂L      -2   n
——— = ———— Σ (yᵢ - ŷᵢ)
∂β₀     n   i=1
```

**Implementation (vectorized):**

```python
# Compute predictions using ALL training samples
Y_pred = X @ self.__coef_ + self.__intercept_

# Calculate error vector
error = Y - Y_pred

# Calculate gradients
b_slope = (-2 / m) * np.sum(error)
m_slope = (-2 / m) * (X.T @ error)
```

---

### 4. Parameter Updates

Parameters are updated simultaneously using the learning rate `α`:

```
β₀_new = β₀_old - α · (∂L/∂β₀)
β_new = β_old - α · (∂L/∂β)
```

**Implementation:**

```python
self.__intercept_ -= self.lr * b_slope
self.__coef_ -= self.lr * m_slope
```

---

### 5. Feature Scaling (Standardization)

To ensure fast and stable convergence across features with different scales, standardization is applied:

```
         X - μ
X_scaled = ————
           σ
```

Where:
- `μ` = mean vector of features (computed column-wise)
- `σ` = standard deviation vector of features

**Implementation:**

```python
if self.scale:
    self.mean_ = np.mean(X, axis=0)  # Column-wise mean
    self.std_ = np.std(X, axis=0)    # Column-wise std
    
    if np.any(self.std_ == 0):
        raise ValueError("X contains a feature with zero standard deviation.")
    
    X = (X - self.mean_) / self.std_
```

**Note:** The learned coefficients remain in the scaled space. For predictions, test data is scaled using the **training set's** `mean_` and `std_`.

---

### 6. Convergence Criterion

Training stops early if the **maximum absolute gradient** (gradient norm) becomes smaller than the tolerance:

```
max(|∂L/∂β₀|, ||∂L/∂β||∞) < tolerance
```

Where `||∂L/∂β||∞` is the infinity norm (maximum absolute value) of the gradient vector.

**Implementation:**

```python
self.gradient_norm_ = max(
    np.max(np.abs(m_slope)),
    abs(b_slope)
)

if self.gradient_norm_ < self.tolerance:
    break
```

---

## 🛠️ Class Structure: `GDLinReg`

### Constructor Parameters

| Parameter   | Default | Description                                       |
| ----------- | ------- | ------------------------------------------------- |
| `epochs`    | 1000    | Maximum number of iterations                      |
| `lr`        | 0.01    | Learning rate (step size)                         |
| `tolerance` | 1e-8    | Early stopping threshold for gradient norm        |
| `scale`     | True    | Whether to standardize features before training   |

### Attributes (Private)

- `__coef_`: The coefficient vector for all features (in scaled space if `scale=True`)
- `__intercept_`: The y-intercept (bias term)

### Public Attributes

- `n_iterations_`: Actual number of iterations performed
- `gradient_norm_`: Final gradient norm at stopping
- `mean_`, `std_`: Scaling parameters (if `scale=True`)

### Methods

| Method                 | Description                             |
| ---------------------- | --------------------------------------- |
| `fit(X_train, Y_train)` | Train the model using Batch Gradient Descent |
| `predict(X_test)`      | Generate predictions for test data      |
| `get_coef_()`          | Return the coefficient vector (read-only) |
| `get_intercept_()`     | Return the intercept (read-only)        |

---

## 🚀 Usage Example

```python
import numpy as np
from GDLinReg import GDLinReg
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Load data
X, Y = load_diabetes(return_X_y=True)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Train model with Batch Gradient Descent
model = GDLinReg(epochs=1000, lr=0.01, tolerance=1e-8, scale=True)
model.fit(X_train, Y_train)

# Get learned parameters
print(f"Intercept: {model.get_intercept_():.2f}")
print(f"Coefficients:\n{model.get_coef_()}")
print(f"Iterations: {model.n_iterations_}")
print(f"Final gradient norm: {model.gradient_norm_:.2e}")

# Make predictions
Y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(f"MSE: {mean_squared_error(Y_test, Y_pred):.2f}")
print(f"R²: {r2_score(Y_test, Y_pred):.2f}")
```

---

## 🔒 Design Features

### Input Validation

- Reshapes 1D input to 2D (`X.reshape(-1, 1)`)
- Ensures feature matrix is 2D
- Checks for features with zero standard deviation
- Validates model is fitted before prediction
- Provides clear error messages

### Numerical Stability

- Uses vectorized operations for efficiency
- Checks for numerical issues during scaling
- Gradient norm tracking for convergence monitoring

### Encapsulation

- Private attributes (`__coef_`, `__intercept_`) prevent external modification
- Read-only access via getter methods
- Protects model integrity after training

### Robustness

- Converts inputs to numpy arrays automatically
- Works with Python lists, numpy arrays, and pandas DataFrames
- Handles both 1D and multi-dimensional feature arrays

---

## 📊 Batch Gradient Descent vs Closed-Form

| Aspect                | Batch Gradient Descent | Closed-Form OLS |
| --------------------- | ---------------------- | --------------- |
| **Computation**       | Iterative              | One-step        |
| **Speed (n < 10k)**   | Slower                | Faster          |
| **Speed (n > 100k)**  | Faster                | Very slow (O(k³) for matrix inverse) |
| **Memory**            | O(nk)                  | O(k²) for (XᵀX) |
| **Learning Rate**     | Required               | Not needed      |
| **Convergence**       | Gradual                | Immediate       |
| **High dimensions**   | Scales well            | Expensive       |

---

## 📂 Files

```
multiple_linear_regression_using_gradient_descent/
│
├── README.md                                              # This file
├── GDLinReg.py                                            # Batch Gradient Descent implementation
└── multiple_linear_regression_using_gradient_descent.ipynb # Jupyter notebook with full workflow
```

---

## 🎯 Key Takeaways

1. **Batch Gradient Descent** computes gradients using the full training set, ensuring stable and deterministic updates
2. **Feature scaling** is essential when features have different magnitudes — without it, gradient descent can oscillate or diverge
3. **Vectorization** (`X.T @ error`) makes the algorithm efficient even with many features
4. The **gradient norm** (max absolute gradient) is a better stopping criterion for multi-dimensional problems than loss change
5. Unlike the closed-form solution, BGD **avoids matrix inversion**, making it suitable for high-dimensional data

---

## 📚 References

- [Gradient Descent Optimization](https://en.wikipedia.org/wiki/Gradient_descent)
- [Batch vs Stochastic Gradient Descent](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)
- [Feature Scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- [Linear Regression Optimization](https://en.wikipedia.org/wiki/Linear_regression#Optimization)

---

**Built with ❤️ and pure mathematics**  
_Part of the [ML-From-Scratch](https://github.com/rudrapratap601/ML-From-Scratch) project_
