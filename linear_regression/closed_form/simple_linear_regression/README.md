# 📐 Simple Linear Regression - From Scratch

A complete implementation of **Simple Linear Regression** built from the ground up using only Python, NumPy, and Matplotlib — no scikit-learn for the model itself.

---

## 📌 Overview

This project demonstrates how Simple Linear Regression works by implementing every component manually:

- Coefficient and intercept calculation using the Least Squares Method
- Prediction generation
- Evaluation metrics (MAE, MSE, RMSE, R², Adjusted R²)
- Best-fit line visualization

### Dataset: Student Placement Prediction

- **Features**: CGPA (X)
- **Target**: Package in LPA (Y)
- **Samples**: 200 student records

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
- `b` = intercept

---

### 2. Calculating the Coefficient (Slope)

The coefficient `m` is calculated using the **Least Squares Method**:

```
       Σ(xᵢ - x̄)(yᵢ - ȳ)
m = ————————————————————
       Σ(xᵢ - x̄)²
```

Where:

- `xᵢ, yᵢ` = individual data points
- `x̄, ȳ` = mean of x and y
- `Σ` = summation over all data points

**Implementation:**

```python
num = 0
den = 0

x_mean = X_train.mean()
y_mean = Y_train.mean()

for i in range(X_train.shape[0]):
    num += (X_train[i] - x_mean) * (Y_train[i] - y_mean)
    den += (X_train[i] - x_mean) ** 2

coefficient = num / den
```

---

### 3. Calculating the Intercept

Once we have the coefficient, we calculate the intercept:

```
b = ȳ - m·x̄
```

**Implementation:**

```python
intercept = y_mean - coefficient * x_mean
```

---

### 4. Making Predictions

To predict for new data points:

```
ŷ = m·x + b
```

**Implementation:**

```python
def predict(self, X_test):
    return self.__coefficient * X_test + self.__intercept
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
sst = 0
n = Y_test.shape[0]
Y_mean = Y_test.mean()

for i in range(n):
    ssr += (Y_test[i] - Y_pred[i]) ** 2
    sst += (Y_test[i] - Y_mean) ** 2

return 1 - (ssr / sst)
```

---

### 5. Adjusted R² Score

Adjusted R² accounts for the number of predictors in the model:

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
k = X_test.shape[1] if X_test.ndim > 1 else 1

return 1 - ((1 - r2) * (n - 1)) / (n - 1 - k)
```

---

## 🛠️ Class Structure: `SimLinReg`

### Attributes (Private)

- `__coefficient`: The slope of the regression line
- `__intercept`: The y-intercept of the regression line

### Methods

| Method                                   | Description                         |
| ---------------------------------------- | ----------------------------------- |
| `fit(X_train, Y_train)`                  | Train the model using training data |
| `predict(X_test)`                        | Generate predictions for test data  |
| `get_coefficient()`                      | Return the coefficient (read-only)  |
| `get_intercept()`                        | Return the intercept (read-only)    |
| `mae(Y_test, Y_pred)`                    | Calculate Mean Absolute Error       |
| `mse(Y_test, Y_pred)`                    | Calculate Mean Squared Error        |
| `rmse(Y_test, Y_pred)`                   | Calculate Root Mean Squared Error   |
| `r2_score(Y_test, Y_pred)`               | Calculate R² Score                  |
| `adj_r2_score(X_test, r2)`               | Calculate Adjusted R² Score         |
| `show_best_fit_line(X, Y, X_train, ...)` | Visualize the regression line       |

---

## 🚀 Usage Example

```python
from SimLinReg import SimLinReg
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('placement_SReg.csv')
X = df['cgpa'].values
Y = df['package'].values

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
model = SimLinReg()
model.fit(X_train, Y_train)

# Get parameters
print(f"Coefficient: {model.get_coefficient()}")
print(f"Intercept: {model.get_intercept()}")

# Make predictions
Y_pred = model.predict(X_test)

# Evaluate
r2 = model.r2_score(Y_test, Y_pred)
print(f"MAE: {model.mae(Y_test, Y_pred):.4f}")
print(f"MSE: {model.mse(Y_test, Y_pred):.4f}")
print(f"RMSE: {model.rmse(Y_test, Y_pred):.4f}")
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {model.adj_r2_score(X_test, r2):.4f}")

# Visualize
model.show_best_fit_line(X, Y, X_train, x_label='CGPA', y_label='Package (LPA)', title='Student Placement Prediction')
```

---

## 📈 Results

For the placement dataset with 80/20 train-test split:

| Metric          | Value |
| --------------- | ----- |
| **MAE**         | 0.29  |
| **MSE**         | 0.12  |
| **RMSE**        | 0.35  |
| **R²**          | 0.78  |
| **Adjusted R²** | 0.77  |

The model explains **78%** of the variance in the package (salary) based on CGPA.

---

## 🔒 Design Features

### Input Validation

- Checks for matching array lengths
- Ensures minimum sample size
- Prevents division by zero
- Validates model is fitted before prediction

### Encapsulation

- Private attributes (`__coefficient`, `__intercept`) prevent external modification
- Read-only access via getter methods
- Protects model integrity after training

### Robustness

- Converts inputs to numpy arrays automatically
- Works with Python lists, numpy arrays, and pandas Series
- Provides clear error messages

---

## 📂 Files

```
simple_linear_regression/
│
├── README.md                          # This file
├── SimLinReg.py                       # Implementation of Simple Linear Regression
├── simple_linear_regression.ipynb     # Jupyter notebook with full workflow
└── placement_SReg.csv                 # Dataset (CGPA vs Package)
```

---

## 🎯 Key Takeaways

1. **Least Squares Method** minimizes the sum of squared residuals to find the best-fit line
2. The **coefficient** represents the rate of change (for every 1 unit increase in X, Y changes by `m`)
3. The **intercept** is the predicted value when X = 0
4. **R²** tells us how much variance in Y is explained by X
5. Loop-based implementations help understand the underlying math before optimizing with vectorization

---

## 📚 References

- [Understanding Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression)
- [Least Squares Method](https://en.wikipedia.org/wiki/Least_squares)
- [Coefficient of Determination (R²)](https://en.wikipedia.org/wiki/Coefficient_of_determination)

---

**Built with ❤️ and pure mathematics**  
_Part of the [ML-From-Scratch](https://github.com/rudrapratap601/ML-From-Scratch) project_
