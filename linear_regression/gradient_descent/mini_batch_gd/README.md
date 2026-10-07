# 📐 Mini-Batch Gradient Descent Regressor

A complete implementation of **Linear Regression using Mini-Batch Gradient Descent** built from scratch using only Python and NumPy — no scikit-learn for the model itself.

---

## 📌 Overview

This project demonstrates how Linear Regression works when coefficients are learned iteratively using **Mini-Batch Gradient Descent** (MBGD), a hybrid optimization algorithm that updates parameters using **small random batches** of the training data.

- Coefficients updated with mini-batches rather than single samples or the full dataset
- Configurable batch size for balancing speed and stability
- Random shuffling and batch sampling at each epoch
- Loss computed on the full dataset after each epoch
- Feature scaling for stable convergence

### Mini-Batch GD: The Best of Both Worlds

Mini-Batch Gradient Descent combines the advantages of both Batch and Stochastic GD:

| Aspect                | Stochastic GD | **Mini-Batch GD** | Batch GD |
| --------------------- | ------------- | ----------------- | -------- |
| **Samples per update** | 1            | **b (batch_size)** | n (all)  |
| **Convergence**       | Very noisy    | **Balanced**      | Smooth   |
| **Speed**             | Fast          | **Fast**          | Slow     |
| **Memory**            | Lowest        | **Low**           | High     |
| **Parallelization**   | Poor          | **Excellent**     | Good     |
| **GPU utilization**   | Poor          | **Excellent**     | Good     |

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

In **Mini-Batch GD**, we optimize using a **random subset** (mini-batch) of size `b`:

```
                 1   b
L_batch(β₀, β) = ——— Σ (yᵢ - ŷᵢ)²
                 b  i=1
```

---

### 3. Mini-Batch Gradient Descent Update Rules

For **each mini-batch** of samples, we compute the average gradient:

#### Gradient with respect to the coefficient vector (β):

```
∂L_batch      -2
———————— = ———— X_batch^T (y_batch - ŷ_batch)
   ∂β         b
```

#### Gradient with respect to the intercept (β₀):

```
∂L_batch      -2   b
———————— = ———— Σ (yᵢ - ŷᵢ)
  ∂β₀         b  i=1
```

**Implementation (mini-batch update):**

```python
for start in range(0, m, self.batch_size):
    # Extract mini-batch
    batch_idx = indices[start:start + self.batch_size]
    X_batch = X[batch_idx]
    Y_batch = Y[batch_idx]
    
    batch_m = len(batch_idx)
    
    # Predictions for the batch
    Y_pred = X_batch @ self.__coef_ + self.__intercept_
    
    # Error for the batch
    error = Y_batch - Y_pred
    
    # Gradients (averaged over batch)
    m_slope = (-2 / batch_m) * (X_batch.T @ error)
    b_slope = (-2 / batch_m) * np.sum(error)
    
    # Update parameters
    self.__intercept_ -= self.lr * b_slope
    self.__coef_ -= self.lr * m_slope
```

---

### 4. Random Shuffling and Batching

At the start of each epoch:

1. **Shuffle** the dataset indices
2. **Partition** into mini-batches of size `batch_size`

```python
indices = np.random.permutation(m)

for start in range(0, m, self.batch_size):
    batch_idx = indices[start:start + self.batch_size]
    # ... process batch
```

This ensures:
- Each sample is seen once per epoch
- Different batch compositions across epochs
- Better exploration of the parameter space

---

### 5. Feature Scaling (Standardization)

Like other gradient descent variants, MBGD benefits from feature scaling:

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

After each epoch (all mini-batches processed), the full dataset loss is computed:

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

## 🛠️ Class Structure: `MBGDRegressor`

### Constructor Parameters

| Parameter    | Default | Description                                       |
| ------------ | ------- | ------------------------------------------------- |
| `batch_size` | —       | **Required.** Number of samples per mini-batch   |
| `epochs`     | 500     | Maximum number of epochs (full passes through data) |
| `lr`         | 0.01    | Learning rate (step size)                         |
| `tolerance`  | 1e-4    | Early stopping threshold for loss change          |
| `scale`      | True    | Whether to standardize features before training   |

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
| `fit(X_train, Y_train)` | Train the model using Mini-Batch Gradient Descent |
| `predict(X_test)`      | Generate predictions for test data      |
| `get_coef_()`          | Return the coefficient vector (read-only) |
| `get_intercept_()`     | Return the intercept (read-only)        |

---

## 🚀 Usage Example

```python
import numpy as np
from MBGDRegressor import MBGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Load data
X, Y = load_diabetes(return_X_y=True)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Train model with Mini-Batch Gradient Descent
# batch_size is a required parameter
model = MBGDRegressor(batch_size=32, epochs=500, lr=0.01, tolerance=1e-4, scale=True)
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
- Validates `batch_size > 0`
- Checks for features with zero standard deviation
- Validates model is fitted before prediction
- Provides clear error messages

### Batching Logic

- Handles the **last batch** correctly — if `n` is not divisible by `batch_size`, the final batch will be smaller
- Shuffles indices (not data) for memory efficiency
- Each sample appears exactly once per epoch

### Encapsulation

- Private attributes (`__coef_`, `__intercept_`) prevent external modification
- Read-only access via getter methods
- Protects model integrity after training

### Robustness

- Converts inputs to numpy arrays automatically
- Works with Python lists, numpy arrays, and pandas DataFrames
- Handles both 1D and multi-dimensional feature arrays

---

## 📊 Choosing the Right Batch Size

| Batch Size | Name              | Pros                              | Cons                              |
| ---------- | ----------------- | --------------------------------- | --------------------------------- |
| **1**      | Stochastic GD     | Fast updates, escapes local minima | Very noisy, poor GPU utilization  |
| **32-128** | **Small mini-batch** | **Good balance, fast, stable** | —                                 |
| **256-512** | Large mini-batch | Smoother convergence, better GPU use | Slower updates, may get stuck  |
| **n**      | Batch GD          | Smooth, deterministic             | Slow, high memory                 |

**Common choices:**
- **32** — Good default for small-medium datasets
- **64** — Popular in practice, good GPU utilization
- **128-256** — For larger datasets (10k+ samples)

---

## 📈 Performance Characteristics

### Updates per Epoch

For a dataset with `n = 1000` samples:

| Method       | Batch Size | Updates per Epoch |
| ------------ | ---------- | ----------------- |
| Batch GD     | 1000       | 1                 |
| Mini-Batch GD | 32        | 31                |
| Mini-Batch GD | 100       | 10                |
| Stochastic GD | 1         | 1000              |

**More updates per epoch** → faster initial learning but more computation per epoch.

---

### Convergence Path

```
Loss
 │
 │╲
 │ ╲___              ← Batch GD (smooth)
 │  ╲  ╱╲╱╲
 │   ╲╱    ╲╱╲       ← Mini-Batch GD (balanced)
 │        ╲ ╱╲╱╲╱╲
 │         ╲╱      ╲ ← Stochastic GD (noisy)
 │
 └────────────────────> Epoch
```

Mini-Batch GD strikes a balance between smooth convergence and computational efficiency.

---

## 📂 Files

```
mini_batch_gd/
│
├── README.md                    # This file
├── MBGDRegressor.py             # Mini-Batch Gradient Descent implementation
└── mbgd_regressor_test.ipynb    # Jupyter notebook with full workflow
```

---

## 🎯 Key Takeaways

1. **Mini-Batch Gradient Descent** is the **most practical** variant for real-world applications
2. It combines the **speed of SGD** with the **stability of Batch GD**
3. **Batch size** is a critical hyperparameter — common values are 32, 64, or 128
4. **Vectorized mini-batch operations** leverage modern hardware (GPUs, SIMD) efficiently
5. **Random shuffling** is essential — it ensures each epoch presents data in a different order
6. Mini-Batch GD is the foundation of **all modern deep learning frameworks** (PyTorch, TensorFlow)

---

## 📚 References

- [Mini-Batch Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method)
- [Batch vs Stochastic vs Mini-batch Gradient Descent](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)
- [Practical Recommendations for Gradient-Based Training](https://arxiv.org/abs/1206.5533)

---

**Built with ❤️ and pure mathematics**  
_Part of the [ML-From-Scratch](https://github.com/rudrapratap601/ML-From-Scratch) project_
