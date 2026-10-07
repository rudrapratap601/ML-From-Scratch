# 📐 Closed-Form Linear Regression (Ordinary Least Squares)

A complete collection of **Closed-Form (Analytical) Linear Regression** implementations built from scratch using only **Python** and **NumPy** — no scikit-learn for model training.

---

## 📌 Overview

The **Closed-Form** (or analytical) approach solves the Linear Regression optimization problem by finding an exact, closed-form mathematical expression for the parameter values that minimize the Sum of Squared Residuals (SSR).

Unlike iterative optimization (e.g., Gradient Descent), closed-form methods compute optimal parameters in a **single calculation step** without:
- Setting a learning rate ($\alpha$)
- Choosing the number of epochs
- Tuning convergence thresholds or early stopping
- Risk of oscillation or divergence

---

## 🗂️ Submodules

| Module | Method | Features | Implementation | Documentation |
| :--- | :--- | :--- | :--- | :--- |
| **Simple Linear Regression** | Least Squares Formula | 1 Feature ($x$) | [`SimLinReg.py`](simple_linear_regression/SimLinReg.py) | [View README](simple_linear_regression/README.md) |
| **Multiple Linear Regression** | Normal Equation | $k$ Features ($X$) | [`LinRegClosed.py`](multi_linear_regression/LinRegClosed.py) | [View README](multi_linear_regression/README.md) |

---

## 🧮 The Mathematical Foundations

### 1. Simple Linear Regression (1D Feature)

For a single predictor variable, the best-fit line equation is:

$$\hat{y} = mx + b$$

The parameters $m$ (slope) and $b$ (intercept) are solved directly using the sample covariance and variance:

$$m = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$$

$$b = \bar{y} - m\bar{x}$$

**Implementation:**
```python
x_mean = X_train.mean()
y_mean = Y_train.mean()

num = np.sum((X_train - x_mean) * (Y_train - y_mean))
den = np.sum((X_train - x_mean) ** 2)

coefficient = num / den
intercept = y_mean - coefficient * x_mean
```

---

### 2. Multiple Linear Regression (Normal Equation)

When dealing with $k$ features, the linear equation expands to:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k$$

In matrix form, by augmenting $X$ with an initial column of $1$s for the intercept $\beta_0$:

$$\hat{y} = X\beta$$

The Sum of Squared Errors (SSE) is:

$$L(\beta) = (y - X\beta)^T(y - X\beta)$$

Setting the vector derivative $\nabla_\beta L(\beta) = 0$ yields the **Normal Equation**:

$$\beta = (X^T X)^{-1} X^T y$$

**Implementation:**
```python
# Augment X with column of 1s
X_aug = np.insert(X_train, 0, 1, axis=1)

# Solve in one analytical step
betas = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ Y_train

intercept = betas[0]
coefficients = betas[1:]
```

---

## 📊 Comparison: Simple vs Multiple Closed-Form

| Aspect | Simple Linear Regression | Multiple Linear Regression (Normal Eq) |
| :--- | :--- | :--- |
| **Input Shape** | 1D vector ($n \times 1$) | 2D matrix ($n \times k$) |
| **Computation** | Scalar summations | Matrix multiplication & inversion |
| **Time Complexity** | $O(n)$ | $O(n k^2 + k^3)$ |
| **Memory Complexity** | $O(1)$ auxiliary | $O(k^2)$ for Gram matrix $(X^TX)$ |
| **Invertibility Concern** | Only if $\text{Var}(X) = 0$ | Requires $X^TX$ to be non-singular |
| **Interpretability** | Direct slope ($m$) & intercept ($b$) | Feature weight vector ($\beta$) |

---

## ⚖️ When to Use Closed-Form vs Gradient Descent

```
                        Number of Features (k)
                 Small (k < 10,000)      Large (k > 10,000)
               ┌───────────────────────┬───────────────────────┐
  Small (n)    │  ✅ Closed-Form OLS   │  ⚠️ Gradient Descent  │
Dataset        │  (Fastest & Exact)    │  (Avoids (XᵀX)⁻¹)     │
Size           ├───────────────────────┼───────────────────────┤
  Large (n)    │  ✅ Closed-Form OLS   │  ✅ Gradient Descent  │
               │  (Parallelizable)     │  (Mini-Batch / SGD)   │
               └───────────────────────┴───────────────────────┘
```

### ✅ Advantages of Closed-Form Solutions
- **Exact Global Minimum**: Guarantees the optimal solution without approximation error.
- **Zero Hyperparameters**: No learning rate tuning, decay schedules, or epoch count needed.
- **Deterministic**: Always produces the same result on the same dataset.

### ⚠️ Limitations
- **Matrix Inversion Complexity**: Computing $(X^T X)^{-1}$ takes $O(k^3)$ time — scales poorly with many features ($k > 10,000$).
- **Singularity**: If features are collinear (multicollinear) or $k > n$, $X^TX$ is non-invertible.
- **Out-of-Core Data**: Requires the full dataset in memory for the matrix product.

---

## 🚀 Quick Usage Example

```python
import numpy as np
from multi_linear_regression.LinRegClosed import LinRegClosed
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# 1. Load data
X, Y = load_diabetes(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2. Train closed-form model (single-step calculation)
model = LinRegClosed()
model.fit(X_train, Y_train)

# 3. Retrieve analytical parameters
print("Intercept:", model.get_intercept_())
print("Coefficients:", model.get_coef_())

# 4. Predict
y_pred = model.predict(X_test)
print(f"R² Score: {model.r2_score(Y_test, y_pred):.4f}")
```

---

## 📂 Subdirectory Structure

```
closed_form/
│
├── README.md                          # This file
│
├── simple_linear_regression/
│   ├── README.md                      # Simple OLS Documentation
│   ├── SimLinReg.py                   # Least squares implementation
│   └── simple_linear_regression.ipynb # Interactive notebook
│
└── multi_linear_regression/
    ├── README.md                      # Multiple OLS Documentation
    ├── LinRegClosed.py                # Normal Equation implementation
    └── multi_linear_regression.ipynb  # Interactive notebook
```

---

## 📚 References

- [Ordinary Least Squares — Wikipedia](https://en.wikipedia.org/wiki/Ordinary_least_squares)
- [The Normal Equation — Stanford CS229](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [Linear Algebra and Learning from Data by Gilbert Strang](https://math.mit.edu/~gs/learningfromdata/)

---

**Built with ❤️ and pure mathematics**  
_Part of the [ML-From-Scratch](https://github.com/rudrapratap601/ML-From-Scratch) project_
