# 📐 Stochastic Gradient Descent Regressor

A complete implementation of **Linear Regression using Stochastic Gradient Descent (SGD)** built from scratch using only Python and NumPy — no scikit-learn for the model itself.

---

## 📌 Overview

This project demonstrates how Linear Regression works when coefficients are learned iteratively using **Stochastic Gradient Descent** (SGD), an optimization algorithm that updates parameters using **one random sample at a time** rather than the entire dataset.

- Coefficients updated with each individual training sample
- Faster iterations and lower memory usage
- Random shuffling of data at each epoch
- Loss computed on the full dataset after each epoch for monitoring
- Feature scaling for stable convergence

### Key Difference from Batch Gradient Descent

| Aspect                | Stochastic GD (SGD)    | Batch GD (BGD) |
| --------------------- | ---------------------- | -------------- |
| **Samples per update** | 1 (single sample)     | n (all samples) |
| **Gradient**          | Noisy approximation    | True gradient  |
| **Convergence**       | Oscillates around minimum | Smooth descent |
| **Speed per epoch**   | Faster                 | Slower         |
| **Memory**            | Very low               | Higher         |
| **Use case**          | Large datasets         | Small-medium datasets |

---

## 🧮 The Mathematics Behind It

### 1. Linear Regression Equation

The model predicts the target using multiple features:

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

We want to minimize:

```
              1   n
L(β₀, β) = ——— Σ (yᵢ - ŷᵢ)²
              n  i=1
```

But in **SGD**, we optimize one sample at a time using the **instantaneous loss**:

```
L_i(β₀, β) = (yᵢ - ŷᵢ)²
```

---

### 3. Stochastic Gradient Descent Update Rules

For **each individual sample** `(xᵢ, yᵢ)`, we compute the gradient:

#### Gradient with respect to the coefficient vector (β):

```
∂L_i
——— = -2(yᵢ - ŷᵢ) · xᵢ
∂β
```

#### Gradient with respect to the intercept (β₀):

```
∂L_i
——— = -2(yᵢ - ŷᵢ)
∂β₀
```

**Implementation (per-sample update):**

```python
# Loop through randomly shuffled training samples
for idx in indices:
    # Prediction for single sample
    Y_pred = X[idx] @ self.__coef_ + self.__intercept_
    
    # Error for single sample
    error = Y[idx] - Y_pred
    
    # Gradients (no averaging, just for this sample)
    b_slope = -2 * error
    m_slope = -2 * error * X[idx]
    
    # Immediate update
    self.__intercept_ -= self.lr * b_slope
    self.__coef_ -= self.lr * m_slope
```

**Key difference:** No division by `n` — each sample's gradient is used directly.

---

### 4. Random Shuffling

At the start of each epoch, the dataset is randomly shuffled:

```python
indices = np.random.permutation(m)
```

This ensures:
- Each sample is seen once per epoch
- Different update order across epochs
- Better exploration of the parameter space

---

### 5. Feature Scaling (Standardization)

Like Batch GD, SGD benefits greatly from feature scaling:

```
         X - μ
X_scaled = ————
           σ
```

Where:
- `μ` = mean vector of features
- `σ` = standard deviation vector of features

**Implementation:**

```python
if self.scale:
    self.mean_ = np.mean(X, axis=0)
    self.std_ = np.std(X, axis=0)
    
    if np.any(self.std_ == 0):
        raise ValueError("X contains a feature with zero standard deviation")
    
    X = (X - self.mean_) / self.std_
```

---

### 6. Convergence Criterion

After each epoch, the full dataset loss is computed:

```python
Y_pred_all = X @ self.__coef_ + self.__intercept_
self.loss_ = np.mean((Y - Y_pred_all) ** 2)
```

Training stops early if the **loss change** is below the tolerance:

```python
if abs(previous_loss - self.loss_) < self.tolerance:
    break
```

---

## 🛠️ Class Structure: `SGDRegressor`

### Constructor Parameters

| Parameter   | Default | Description                                       |
| ----------- | ------- | ------------------------------------------------- |
| `epochs`    | 500     | Maximum number of epochs (full passes through data) |
| `lr`        | 0.01    | Learning rate (step size)                         |
| `tolerance` | 1e-4    | Early stopping threshold for loss change          |
| `scale`     | True    | Whether to standardize features before training   |

### Attributes (Private)

- `__coef_`: The coefficient vector for all features
- `__intercept_`: The y-intercept (bias term)

### Public Attributes

- `n_iterations_`: Actual number of epochs performed
- `loss_`: Final MSE loss on the full training set
- `mean_`, `std_`: Scaling parameters (if `scale=True`)

### Methods

| Method                 | Description                             |
| ---------------------- | --------------------------------------- |
| `fit(X_train, Y_train)` | Train the model using Stochastic Gradient Descent |
| `predict(X_test)`      | Generate predictions for test data      |
| `get_coef_()`          | Return the coefficient vector (read-only) |
| `get_intercept_()`     | Return the intercept (read-only)        |

---

## 🚀 Usage Example

```python
import numpy as np
from SGDRegressor import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Load data
X, Y = load_diabetes(return_X_y=True)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Train model with Stochastic Gradient Descent
model = SGDRegressor(epochs=500, lr=0.01, tolerance=1e-4, scale=True)
model.fit(X_train, Y_train)

# Get learned parameters
print(f"Intercept: {model.get_intercept_():.2f}")
print(f"Coefficients:\n{model.get_coef_()}")
print(f"Epochs: {model.n_iterations_}")
print(f"Final loss: {model.loss_:.2f}")

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
- Checks for features with zero standard deviation
- Validates model is fitted before prediction
- Provides clear error messages

### Randomization

- Shuffles training data at the start of each epoch using `np.random.permutation`
- Ensures each sample is used exactly once per epoch
- Different order across epochs improves convergence

### Encapsulation

- Private attributes (`__coef_`, `__intercept_`) prevent external modification
- Read-only access via getter methods
- Protects model integrity after training

### Robustness

- Converts inputs to numpy arrays automatically
- Works with Python lists, numpy arrays, and pandas DataFrames
- Handles both 1D and multi-dimensional feature arrays

---

## 📊 Comparison: SGD vs Batch GD vs Mini-Batch GD

| Aspect                | Stochastic GD | Batch GD | Mini-Batch GD |
| --------------------- | ------------- | -------- | ------------- |
| **Samples per update** | 1            | n (all)  | b (batch_size) |
| **Updates per epoch** | n            | 1        | n / b         |
| **Gradient quality**  | Noisy        | Exact    | Balanced      |
| **Convergence path**  | Oscillates   | Smooth   | Moderately smooth |
| **Memory usage**      | Lowest       | Highest  | Medium        |
| **Speed per epoch**   | Fast         | Slow     | Medium        |
| **Best for**          | Huge datasets | Small-medium | General purpose |
| **Escape local minima** | Yes (noise helps) | No | Sometimes |

---

## 🎨 Visualization

The loss curve in SGD typically shows:
- **Rapid initial drop** as parameters move toward the minimum
- **Noisy oscillations** as single-sample gradients vary
- **Slower final convergence** compared to batch methods

Example:

```
Loss
 │
 │╲
 │ ╲
 │  ╲  ╱╲
 │   ╲╱  ╲╱╲  ← Noisy, oscillating descent
 │        ╲ ╱╲╱╲
 │         ╲╱   ╲╱╲
 │              ╲╱
 └────────────────────> Epoch
```

---

## 📂 Files

```
stochastic_gd/
│
├── README.md                  # This file
├── SGDRegressor.py            # Stochastic Gradient Descent implementation
└── sgd_regressor_test.ipynb   # Jupyter notebook with full workflow
```

---

## 🎯 Key Takeaways

1. **Stochastic Gradient Descent** updates parameters **after each sample**, making it much faster per epoch than Batch GD
2. The **noisy gradient** approximation can help escape shallow local minima but causes oscillation around the optimum
3. **Random shuffling** is critical — without it, the model may overfit to the data order
4. **Lower learning rate** is often needed compared to Batch GD to prevent excessive oscillation
5. **Per-epoch loss monitoring** (not per-sample) provides meaningful convergence tracking
6. SGD is the foundation of **modern deep learning optimizers** (Adam, RMSprop, etc. are SGD variants)

---

## 📚 References

- [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- [Batch vs Stochastic vs Mini-batch Gradient Descent](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)
- [SGD in Practice](https://leon.bottou.org/projects/sgd)

---

**Built with ❤️ and pure mathematics**  
_Part of the [ML-From-Scratch](https://github.com/rudrapratap601/ML-From-Scratch) project_
