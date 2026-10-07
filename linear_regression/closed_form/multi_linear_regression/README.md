# 📐 Multiple Linear Regression - From Scratch

A complete implementation of **Multiple Linear Regression** built from the ground up using only Python and NumPy — no scikit-learn for the model itself. The coefficients are computed using the **Ordinary Least Squares (OLS) Closed-Form Solution**.

---

## 📌 Overview

This project demonstrates how Multiple Linear Regression works by implementing every component manually:

- Coefficient and intercept calculation using the **OLS Normal Equation** (closed-form)
- Prediction generation for multiple features
- Evaluation metrics (MAE, MSE, RMSE, R², Adjusted R²)
- Verification against scikit-learn's `LinearRegression` — results match exactly

### Dataset: Diabetes Prediction

- **Source**: `sklearn.datasets.load_diabetes`
- **Features**: 10 baseline variables (age, sex, BMI, blood pressure, and 6 blood serum measurements)
- **Target**: A quantitative measure of disease progression one year after baseline
- **Samples**: 442 patient records

---

## 🧮 The Mathematics Behind It

### 1. Multiple Linear Regression Equation

The model extends simple linear regression to multiple features:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ
```

In matrix form:

```
ŷ = Xβ
```

Where:

- `ŷ` = predicted values (n × 1)
- `X` = feature matrix augmented with a column of 1s (n × (k+1))
- `β` = coefficient vector including intercept (k+1 × 1)

---

### 2. OLS Closed-Form Solution (Normal Equation)

The optimal coefficients that minimize the sum of squared residuals are found analytically:

```
β = (XᵀX)⁻¹ Xᵀy
```

Where:

- `Xᵀ` = transpose of the augmented feature matrix
- `(XᵀX)⁻¹` = inverse of the Gram matrix
- `y` = target vector

**Implementation:**

```python
# Augment X with a column of 1s for the intercept
X_train = np.insert(X_train, 0, 1, axis=1)

# Compute all coefficients in one step
X_betas = (np.linalg.inv(X_train.T @ X_train)) @ X_train.T @ Y_train

# Separate intercept and feature coefficients
self.__intercept_ = X_betas[0]
self.__coef_ = X_betas[1:]
```

---

### 3. Making Predictions

Once the coefficients are learned, predictions are computed via a dot product:

```
ŷ = X_test · β + β₀
```

**Implementation:**

```python
def predict(self, X_test):
    Y_pred = X_test @ self.__coef_ + self.__intercept_
    return Y_pred
```

---

## 📊 Evaluation Metrics

### 1. Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual values:

```
         1   n
MAE = ——— Σ |yᵢ - ŷᵢ|
         n  i=1
```

**Implementation:**

```python
n = Y_test.shape[0]
num = 0

for i in range(n):
    num += abs(Y_test[i] - Y_pred[i])

return num / n
```

---

### 2. Mean Squared Error (MSE)

Penalizes larger errors more heavily by squaring them:

```
         1   n
MSE = ——— Σ (yᵢ - ŷᵢ)²
         n  i=1
```

**Implementation:**

```python
n = Y_test.shape[0]
num = 0

for i in range(n):
    num += (Y_test[i] - Y_pred[i]) ** 2

return num / n
```

---

### 3. Root Mean Squared Error (RMSE)

The square root of MSE, bringing the error back to the original unit:

```
RMSE = √MSE
```

**Implementation:**

```python
return np.sqrt(self.mse(Y_test, Y_pred))
```

---

### 4. R² Score (Coefficient of Determination)

Measures how well the model fits the data (0 to 1, where 1 is perfect):

```
       SSR      Σ(yᵢ - ŷᵢ)²
R² = 1 - ——— = 1 - ———————————
       SST      Σ(yᵢ - ȳ)²
```

Where:

- `SSR` = Sum of Squared Residuals
- `SST` = Total Sum of Squares

**Implementation:**

```python
ssr = 0
ssm = 0
n = Y_test.shape[0]
Y_mean = Y_test.mean()

for i in range(n):
    ssr += (Y_test[i] - Y_pred[i]) ** 2
    ssm += (Y_test[i] - Y_mean) ** 2

return 1 - (ssr / ssm)
```

---

### 5. Adjusted R² Score

Adjusted R² accounts for the number of predictors, penalizing unnecessary features:

```
              (1 - R²)(n - 1)
Adj R² = 1 - ——————————————————
                n - k - 1
```

Where:

- `n` = number of samples
- `k` = number of features

**Implementation:**

```python
n = X_test.shape[0]
k = X_test.shape[1]

return 1 - ((1 - r2) * (n - 1)) / (n - k - 1)
```

---

## 🛠️ Class Structure: `LinRegClosed`

### Attributes (Private)

- `__coef_`: The coefficient vector for all features
- `__intercept_`: The y-intercept (bias term)

### Methods

| Method                    | Description                              |
| ------------------------- | ---------------------------------------- |
| `fit(X_train, Y_train)`  | Train the model using the Normal Equation |
| `predict(X_test)`        | Generate predictions for test data       |
| `get_coef_()`            | Return the coefficient vector (read-only) |
| `get_intercept_()`       | Return the intercept (read-only)         |
| `mae(Y_test, Y_pred)`    | Calculate Mean Absolute Error            |
| `mse(Y_test, Y_pred)`    | Calculate Mean Squared Error             |
| `rmse(Y_test, Y_pred)`   | Calculate Root Mean Squared Error        |
| `r2_score(Y_test, Y_pred)` | Calculate R² Score                     |
| `adj_r2_score(X_test, r2)` | Calculate Adjusted R² Score            |

---

## 🚀 Usage Example

```python
import numpy as np
from LinRegClosed import LinRegClosed
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Load data
X, Y = load_diabetes(return_X_y=True)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
model = LinRegClosed()
model.fit(X_train, Y_train)

# Get parameters
print(f"Intercept: {model.get_intercept_()}")
print(f"Coefficients: {model.get_coef_()}")

# Make predictions
Y_pred = model.predict(X_test)

# Evaluate
r2 = model.r2_score(Y_test, Y_pred)
print(f"MAE:          {model.mae(Y_test, Y_pred):.2f}")
print(f"MSE:          {model.mse(Y_test, Y_pred):.2f}")
print(f"RMSE:         {model.rmse(Y_test, Y_pred):.2f}")
print(f"R²:           {r2:.2f}")
print(f"Adjusted R²:  {model.adj_r2_score(X_test, r2):.2f}")
```

---

## 📈 Results

For the Diabetes dataset with 80/20 train-test split (`random_state=2`):

| Metric          | From Scratch | scikit-learn |
| --------------- | :----------: | :----------: |
| **MAE**         |    45.21     |    45.21     |
| **MSE**         |   3094.46    |   3094.46    |
| **RMSE**        |    55.63     |    55.63     |
| **R²**          |     0.44     |     0.44     |
| **Adjusted R²** |     0.37     |     0.37     |

✅ **Predictions match scikit-learn's `LinearRegression` exactly**, confirming the correctness of the from-scratch implementation.

---

## 🔒 Design Features

### Input Validation

- Checks that `X_train` is 1D or 2D and `Y_train` is 1D
- Ensures matching number of samples between `X` and `Y`
- Prevents prediction on an unfitted model
- Validates feature count at prediction time matches training data
- Rejects empty training data

### Encapsulation

- Private attributes (`__coef_`, `__intercept_`) prevent external modification
- Read-only access via getter methods
- Protects model integrity after training

### Robustness

- Converts inputs to numpy arrays automatically
- Works with Python lists, numpy arrays, and pandas Series/DataFrames
- Provides clear, descriptive error messages

---

## 📂 Files

```
multi_linear_regression/
│
├── README.md                          # This file
├── LinRegClosed.py                    # OLS closed-form implementation class
└── multi_linear_regression.ipynb      # Jupyter notebook with full workflow
```

---

## 🎯 Key Takeaways

1. The **Normal Equation** `β = (XᵀX)⁻¹ Xᵀy` gives the exact optimal solution in one step — no iterative optimization needed
2. Unlike gradient descent, the closed-form solution **doesn't require a learning rate** or convergence tuning
3. The trade-off: computing `(XᵀX)⁻¹` is O(k³), so this approach becomes expensive for very high-dimensional data (large `k`)
4. **Adjusted R²** is crucial for multiple regression — it penalizes adding features that don't improve the model, unlike plain R²
5. Loop-based metric implementations help you understand the underlying math before switching to vectorized operations

---

## 📚 References

- [OLS Normal Equation](https://en.wikipedia.org/wiki/Ordinary_least_squares#Matrix/vector_formulation)
- [Multiple Linear Regression](https://en.wikipedia.org/wiki/Linear_regression#Multiple_linear_regression)
- [Coefficient of Determination (R²)](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [Adjusted R²](https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2)

---

**Built with ❤️ and pure mathematics**  
_Part of the [ML-From-Scratch](https://github.com/rudrapratap601/ML-From-Scratch) project_
