# 📐 Simple Linear Regression using Batch Gradient Descent

A complete implementation of **Simple Linear Regression using Batch Gradient Descent** built from scratch using only Python and NumPy — no scikit-learn for the model itself.

---

## 📌 Overview

This project demonstrates how Simple Linear Regression works when coefficients are learned iteratively using **Batch Gradient Descent** (BGD), an optimization algorithm that updates parameters using the entire training dataset in each iteration.

- Coefficient and intercept learned through iterative optimization
- Batch Gradient Descent with configurable learning rate and convergence criteria
- Feature scaling (standardization) for faster convergence
- Early stopping based on gradient magnitude
- Loss history tracking

### Key Difference from Closed-Form Solution

Unlike the [closed-form OLS solution](../../../simple_linear_regression) which computes coefficients in one step, Gradient Descent:
- **Iteratively** updates parameters to minimize the loss function
- Requires tuning the **learning rate** (`lr`)
- Can handle **larger datasets** more efficiently
- Demonstrates the **optimization process** visually

---

## 🧮 The Mathematics Behind It

### 1. Linear Regression Equation

The goal is to find the best-fit line:

```
ŷ = mx + b
```

Where:

- `ŷ` = predicted value
- `m` = coefficient (slope)
- `x` = input feature
- `b` = intercept (bias)

---

### 2. Loss Function: Mean Squared Error (MSE)

We want to minimize the MSE loss:

```
         1   n
L(m, b) = ——— Σ (yᵢ - ŷᵢ)²
         n  i=1
```

Where:
- `n` = number of training samples
- `yᵢ` = actual value
- `ŷᵢ = mx_i + b` = predicted value

---

### 3. Gradient Descent Update Rules

To minimize the loss, we compute the **partial derivatives** (gradients) with respect to `m` and `b`:

#### Gradient with respect to the coefficient (m):

```
∂L      -2   n
—— = ———— Σ (yᵢ - ŷᵢ) · xᵢ
∂m      n   i=1
```

#### Gradient with respect to the intercept (b):

```
∂L      -2   n
—— = ———— Σ (yᵢ - ŷᵢ)
∂b      n   i=1
```

**Implementation:**

```python
# Compute predictions using ALL training samples
y_pred = m * X + b

# Calculate errors
error = Y - y_pred

# Calculate gradients
m_slope = (-2 / n) * np.sum(error * X)
b_slope = (-2 / n) * np.sum(error)
```

---

### 4. Parameter Updates

Parameters are updated simultaneously using the learning rate `α`:

```
m_new = m_old - α · (∂L/∂m)
b_new = b_old - α · (∂L/∂b)
```

**Implementation:**

```python
m -= self.ln * m_slope
b -= self.ln * b_slope
```

---

### 5. Feature Scaling (Standardization)

To ensure fast and stable convergence, features are standardized:

```
       x - x̄
x_scaled = ————
        σ_x
```

Where:
- `x̄` = mean of feature
- `σ_x` = standard deviation of feature

**After training**, coefficients are transformed back to the original scale:

```
m_original = m_scaled / σ_x
b_original = b_scaled - m_original · x̄
```

**Implementation:**

```python
if self.scale:
    self.mean_ = np.mean(X)
    self.std_ = np.std(X)
    X = (X - self.mean_) / self.std_

# ... training happens on scaled data ...

# Transform back to original scale
if self.scale:
    self.__coef_ = m / self.std_
    self.__intercept_ = b - self.__coef_ * self.mean_
```

---

### 6. Convergence Criterion

Training stops early if the gradient magnitude becomes smaller than a tolerance threshold:

```
max(|∂L/∂m|, |∂L/∂b|) < tolerance
```

This indicates the loss function has reached a (near) minimum.

---

## 🛠️ Class Structure: `GDSimlin`

### Constructor Parameters

| Parameter   | Default | Description                                       |
| ----------- | ------- | ------------------------------------------------- |
| `epochs`    | 1000    | Maximum number of iterations                      |
| `ln`        | 0.01    | Learning rate (step size)                         |
| `tolerance` | 1e-8    | Early stopping threshold for gradient magnitude   |
| `scale`     | True    | Whether to standardize features before training   |

### Attributes (Private)

- `__coef_`: The slope of the regression line (in original scale)
- `__intercept_`: The y-intercept of the regression line (in original scale)

### Public Attributes

- `loss_history`: List of MSE values at each epoch
- `n_iterations_`: Actual number of iterations performed
- `mean_`, `std_`: Scaling parameters (if `scale=True`)

### Methods

| Method                 | Description                             |
| ---------------------- | --------------------------------------- |
| `fit(X_train, Y_train)` | Train the model using Batch Gradient Descent |
| `predict(X_test)`      | Generate predictions for test data      |
| `get_coef_()`          | Return the coefficient (read-only)      |
| `get_intercept_()`     | Return the intercept (read-only)        |

---

## 🚀 Usage Example

```python
from GDSimLin import GDSimlin
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('placement_SReg.csv')
X = df['cgpa'].values
Y = df['package'].values

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Train model with Batch Gradient Descent
model = GDSimlin(epochs=1000, ln=0.01, tolerance=1e-8, scale=True)
model.fit(X_train, Y_train)

# Get learned parameters
print(f"Coefficient: {model.get_coef_():.4f}")
print(f"Intercept: {model.get_intercept_():.4f}")
print(f"Iterations: {model.n_iterations_}")

# Make predictions
Y_pred = model.predict(X_test)

# Visualize convergence
plt.plot(model.loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Batch Gradient Descent Convergence')
plt.grid(True)
plt.show()
```

---

## 🔒 Design Features

### Input Validation

- Checks for non-empty training data
- Ensures `X` and `Y` have equal lengths
- Validates all values are finite (no NaN or Inf)
- Checks for zero standard deviation (constant feature)
- Validates hyperparameters are positive
- Prevents prediction on an unfitted model

### Numerical Stability

- Detects divergence (NaN or Inf in parameters)
- Raises `FloatingPointError` with actionable message
- Tracks loss at each iteration for debugging

### Encapsulation

- Private attributes (`__coef_`, `__intercept_`) prevent external modification
- Read-only access via getter methods
- Protects model integrity after training

### Robustness

- Converts inputs to numpy arrays automatically
- Works with Python lists, numpy arrays, and pandas Series
- Provides clear, descriptive error messages

---

## 📊 Batch Gradient Descent vs Closed-Form

| Aspect                | Batch Gradient Descent | Closed-Form OLS |
| --------------------- | ---------------------- | --------------- |
| **Computation**       | Iterative              | One-step        |
| **Speed (small data)** | Slower                | Faster          |
| **Speed (large data)** | Faster                | Slower (O(n³))  |
| **Learning Rate**     | Required               | Not needed      |
| **Convergence**       | Gradual                | Immediate       |
| **Visualization**     | Can plot loss curve    | N/A             |
| **Memory**            | Efficient              | Requires matrix inversion |

---

## 📂 Files

```
simple_linear_regression_using_gradient_descent/
│
├── README.md                                      # This file
├── GDSimLin.py                                    # Batch Gradient Descent implementation
└── simple_linear_reg_using_gradient_descent.ipynb # Jupyter notebook with full workflow
```

---

## 🎯 Key Takeaways

1. **Batch Gradient Descent** uses the entire training set in each iteration, computing the "true" gradient direction
2. The **learning rate** controls the step size — too large causes divergence, too small slows convergence
3. **Feature scaling** is crucial for gradient descent to converge quickly when features have different magnitudes
4. **Early stopping** based on gradient magnitude prevents unnecessary iterations after convergence
5. Unlike stochastic variants, BGD produces **smooth, deterministic convergence** but can be slower on large datasets

---

## 📚 References

- [Gradient Descent Optimization](https://en.wikipedia.org/wiki/Gradient_descent)
- [Batch vs Stochastic vs Mini-batch Gradient Descent](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)
- [Feature Scaling and Normalization](https://en.wikipedia.org/wiki/Feature_scaling)

---

**Built with ❤️ and pure mathematics**  
_Part of the [ML-From-Scratch](https://github.com/rudrapratap601/ML-From-Scratch) project_
