# 📈 Linear Regression — From Scratch

A comprehensive collection of **Linear Regression** implementations built completely from the ground up using only **Python** and **NumPy** — no high-level machine learning libraries for the algorithms.

---

## 📌 Overview

Linear Regression is one of the most fundamental algorithms in supervised machine learning. It models the relationship between independent input features ($X$) and a continuous target variable ($y$) by fitting a linear equation to observed data.

This directory explores both major mathematical approaches to solving linear regression problems:
1. **Closed-Form Solutions (Analytical / OLS)** — Exact mathematical solutions computed directly in one step.
2. **Iterative Optimization (Gradient Descent)** — Optimization algorithms that iteratively refine parameters to minimize the loss function.

---

## 🗂️ Module Directory

| Module | Approach | Equation / Method | Features | Implementation | Documentation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Simple Linear Regression** | Closed-Form (OLS) | Least Squares Method | Single (1D) | [`SimLinReg.py`](simple_linear_regression/SimLinReg.py) | [View README](simple_linear_regression/README.md) |
| **Multiple Linear Regression** | Closed-Form (OLS) | Normal Equation: $\beta = (X^TX)^{-1}X^Ty$ | Multiple ($k$D) | [`LinRegClosed.py`](multi_linear_regression/LinRegClosed.py) | [View README](multi_linear_regression/README.md) |
| **Batch GD (Simple)** | Iterative (Batch) | Full dataset gradient: $\frac{\partial L}{\partial m}, \frac{\partial L}{\partial b}$ | Single (1D) | [`GDSimLin.py`](gradient_descent/batch_gd/simple_linear_regression_using_gradient_descent/GDSimLin.py) | [View README](gradient_descent/batch_gd/simple_linear_regression_using_gradient_descent/README.md) |
| **Batch GD (Multiple)** | Iterative (Batch) | Vectorized gradient: $-\frac{2}{m}X^T(y - \hat{y})$ | Multiple ($k$D) | [`GDLinReg.py`](gradient_descent/batch_gd/multiple_linear_regression_using_gradient_descent/GDLinReg.py) | [View README](gradient_descent/batch_gd/multiple_linear_regression_using_gradient_descent/README.md) |
| **Stochastic GD (SGD)** | Iterative (Sample) | Single-sample update: $-2(y_i - \hat{y}_i)x_i$ | Multiple ($k$D) | [`SGDRegressor.py`](gradient_descent/stochastic_gd/SGDRegressor.py) | [View README](gradient_descent/stochastic_gd/README.md) |
| **Mini-Batch GD (MBGD)**| Iterative (Batch) | Subset gradient over batch size $b$ | Multiple ($k$D) | [`MBGDRegressor.py`](gradient_descent/mini_batch_gd/MBGDRegressor.py) | [View README](gradient_descent/mini_batch_gd/README.md) |

---

## 🧮 Mathematical Paradigms Compared

### 1. Closed-Form (Ordinary Least Squares)

The closed-form approach directly computes the global minimum of the Mean Squared Error (MSE) loss function using calculus and linear algebra without iterative loops.

* **Simple (1D) OLS:**
  $$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad b = \bar{y} - m\bar{x}$$

* **Multiple OLS (Normal Equation):**
  $$\beta = (X^T X)^{-1} X^T y$$

**Pros:** No hyperparameters to tune (no learning rate, no epochs); exact analytical solution.  
**Cons:** Computing the matrix inverse $(X^T X)^{-1}$ takes $O(k^3)$ time complexity, making it computationally prohibitive for datasets with many features.

---

### 2. Gradient Descent Optimization

Gradient Descent iteratively updates model parameters by taking steps proportional to the negative of the gradient of the loss function:

$$\theta_{\text{new}} = \theta_{\text{old}} - \alpha \nabla L(\theta)$$

Where $\alpha$ is the learning rate and $\nabla L(\theta)$ is the gradient vector.

#### The Three Gradient Descent Variants

```
       Batch GD (BGD)              Mini-Batch GD (MBGD)            Stochastic GD (SGD)
 ┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
 │ Uses ALL n samples      │   │ Uses batch of b samples │   │ Uses 1 sample           │
 │ Smooth, steady descent  │   │ Balanced & practical    │   │ Fast, noisy updates     │
 │ 1 update per epoch      │   │ n/b updates per epoch   │   │ n updates per epoch     │
 └─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
```

| Property | Batch Gradient Descent | Mini-Batch Gradient Descent | Stochastic Gradient Descent |
| :--- | :--- | :--- | :--- |
| **Batch Size ($b$)** | $n$ (entire dataset) | $1 < b < n$ (e.g., 32, 64) | $1$ (single sample) |
| **Updates per Epoch** | $1$ | $\lceil n / b \rceil$ | $n$ |
| **Gradient Stability** | Exact & deterministic | Moderate variance | High variance (noisy) |
| **Memory Footprint** | Highest | Low (scales with $b$) | Lowest |
| **Vectorization / GPU** | Good | Excellent | Poor |
| **Best Suited For** | Small to medium datasets | Medium to massive datasets | Streaming / huge datasets |

---

## 📊 Summary of Evaluation Metrics

All models are evaluated using standard regression performance metrics implemented from scratch:

| Metric | Formula | Description |
| :--- | :--- | :--- |
| **MAE** | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ | Mean Absolute Error — average magnitude of errors in original units |
| **MSE** | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | Mean Squared Error — penalizes large errors heavily |
| **RMSE** | $\sqrt{\text{MSE}}$ | Root Mean Squared Error — square root of MSE in target units |
| **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Proportion of target variance explained by features |
| **Adjusted $R^2$**| $1 - \frac{(1 - R^2)(n - 1)}{n - k - 1}$ | $R^2$ penalized for the number of predictors ($k$) |

---

## 🚀 Quick Usage Guide

### 1. Closed-Form Multiple Linear Regression
```python
from multi_linear_regression.LinRegClosed import LinRegClosed

model = LinRegClosed()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print(f"Intercept: {model.get_intercept_()}, Coefficients: {model.get_coef_()}")
```

### 2. Batch Gradient Descent
```python
from gradient_descent.batch_gd.multiple_linear_regression_using_gradient_descent.GDLinReg import GDLinReg

model = GDLinReg(epochs=1000, lr=0.01, tolerance=1e-8, scale=True)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
```

### 3. Stochastic Gradient Descent
```python
from gradient_descent.stochastic_gd.SGDRegressor import SGDRegressor

model = SGDRegressor(epochs=500, lr=0.01, tolerance=1e-4, scale=True)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
```

### 4. Mini-Batch Gradient Descent
```python
from gradient_descent.mini_batch_gd.MBGDRegressor import MBGDRegressor

model = MBGDRegressor(batch_size=32, epochs=500, lr=0.01, tolerance=1e-4, scale=True)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
```

---

## 🔒 Shared Implementation Highlights

Across all implementations in this module, the following design principles are strictly followed:

- **Pure NumPy Math**: No scikit-learn estimators are used for model training or inference.
- **Robust Feature Standardization**: Automatic scaling and zero-variance detection to guarantee numerical stability.
- **Encapsulation**: Strict private attributes (`__coef_`, `__intercept_`) with read-only getter functions.
- **Input Validation**: Shape compatibility checks, non-finite value detection, and informative error handling.
- **Scikit-Learn Verification**: Outputs are verified against scikit-learn counterparts (`LinearRegression`, `SGDRegressor`) for absolute correctness.

---

## 📂 Directory Structure

```
linear_regression/
│
├── README.md                                             # This file
├── data/
│   └── placement_SReg.csv                                # Placement dataset (CGPA vs Package)
│
├── simple_linear_regression/
│   ├── README.md                                         # Simple OLS Documentation
│   ├── SimLinReg.py                                      # Simple Linear Regression (Least Squares)
│   └── simple_linear_regression.ipynb                    # Full analysis notebook
│
├── multi_linear_regression/
│   ├── README.md                                         # Multiple OLS Documentation
│   ├── LinRegClosed.py                                   # Multiple Linear Regression (Normal Eq)
│   └── multi_linear_regression.ipynb                     # Full analysis notebook
│
└── gradient_descent/
    ├── batch_gd/
    │   ├── simple_linear_regression_using_gradient_descent/
    │   │   ├── README.md                                 # Simple BGD Documentation
    │   │   ├── GDSimLin.py                               # Simple BGD Regressor
    │   │   └── simple_linear_reg_using_gradient_descent.ipynb
    │   └── multiple_linear_regression_using_gradient_descent/
    │       ├── README.md                                 # Multiple BGD Documentation
    │       ├── GDLinReg.py                               # Multiple BGD Regressor
    │       └── multiple_linear_regression_using_gradient_descent.ipynb
    │
    ├── mini_batch_gd/
    │   ├── README.md                                     # Mini-Batch GD Documentation
    │   ├── MBGDRegressor.py                              # Mini-Batch GD Regressor
    │   └── mbgd_regressor_test.ipynb                     # Full analysis notebook
    │
    └── stochastic_gd/
        ├── README.md                                     # Stochastic GD Documentation
        ├── SGDRegressor.py                               # Stochastic GD Regressor
        └── sgd_regressor_test.ipynb                      # Full analysis notebook
```

---

## 📚 References & Further Reading

- [Ordinary Least Squares (Wikipedia)](https://en.wikipedia.org/wiki/Ordinary_least_squares)
- [Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
- [An Overview of Gradient Descent Algorithms by Sebastian Ruder](https://arxiv.org/abs/1609.04747)
- [Stanford CS229: Linear Regression & Gradient Descent](https://cs229.stanford.edu/main_notes.pdf)

---

**Built with ❤️ and pure mathematics**  
_Part of the [ML-From-Scratch](https://github.com/rudrapratap601/ML-From-Scratch) project_
