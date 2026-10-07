<div align="center">

# 🧠 ML-From-Scratch

**Bare-bones implementations of machine learning models & algorithms — built from the ground up for deep understanding.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-ff69b4?style=for-the-badge)](#contributing)

<br/>

<img src="https://img.shields.io/badge/No_sklearn.fit()_here-⚙️_Pure_Math-black?style=for-the-badge" alt="Pure Math" />

---

_Every model is coded using only Python, NumPy & Pandas — no high-level ML library shortcuts._
_The goal is to understand the **math**, not just the API._

</div>

<br/>

## 📌 About

Most ML courses teach you to call `model.fit()` and move on.  
This repo takes the opposite approach — **every algorithm is implemented from scratch** using fundamental libraries so you can see exactly what happens under the hood.

Each project includes:

- 📐 **Mathematical derivation** of the algorithm
- 🧪 **From-scratch implementation** in a Jupyter Notebook
- 📊 **Evaluation metrics** computed manually (MAE, MSE, RMSE, R², Adjusted R²)
- 📈 **Visualizations** to validate the model's performance

<br/>

## 🗂️ Projects

|  #  | Model | Method | Path | Status |
| :-: | :--- | :--- | :--- | :-----: |
|  1  | **Linear Regression** | Regression | [`linear_regression`](linear_regression/) | ✅ Done |
|  2 | **Logistic Regression** | Classification | _coming soon_ | 🔜 |
|  3 | **K-Nearest Neighbors** | Non-parametric | _coming soon_ | 🔜 |
|  4 | **Decision Tree** | Tree-based | _coming soon_ | 🔜 |
|  5 | **Support Vector Machine** | Maximum Margin | _coming soon_ | 🔜 |
|  6 | **K-Means Clustering** | Unsupervised | _coming soon_ | 🔜 |
|  7 | **Principal Component Analysis** | Dimensionality Reduction | _coming soon_ | 🔜 |

> 🚧 _New models are added regularly — star the repo to stay updated!_

<br/>

## 🛠️ Tech Stack

<table>
  <tr>
    <td align="center" width="140">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="48" /><br/>
      <b>Python 3.10+</b><br/>
      <sub>Core language</sub>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="48" /><br/>
      <b>NumPy</b><br/>
      <sub>Linear algebra</sub>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="48" /><br/>
      <b>Pandas</b><br/>
      <sub>Data handling</sub>
    </td>
    <td align="center" width="140">
      <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" width="48" /><br/>
      <b>Matplotlib</b><br/>
      <sub>Visualization</sub>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" width="48" /><br/>
      <b>Jupyter</b><br/>
      <sub>Notebooks</sub>
    </td>
  </tr>
</table>

> **Why no scikit-learn?** — The whole point is to **not** use it.  
> `sklearn` is used _only_ for `train_test_split` and data loading utilities — never for the model itself.

<br/>

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.10
```

### Installation

```bash
# Clone the repo
git clone https://github.com/rudrapratap601/ML-From-Scratch.git
cd ML-From-Scratch

# Install dependencies
pip install numpy pandas matplotlib jupyter scikit-learn

# Launch Jupyter
jupyter notebook
```

<br/>

## 📂 Repo Structure

```
ML-From-Scratch/
│
├── LICENSE
├── README.md
├── .gitignore
│
└── linear_regression/                            # 6 complete implementations
    ├── README.md                                 # Linear Regression overview
    ├── data/
    │   └── placement_SReg.csv                    # Sample dataset
    │
    ├── closed_form/                              # Analytical solutions (OLS)
    │   ├── README.md                             # Closed-form methods overview
    │   ├── simple_linear_regression/
    │   │   ├── README.md                         # Documentation & Math
    │   │   ├── SimLinReg.py                      # Least Squares implementation
    │   │   └── simple_linear_regression.ipynb    # Full workflow notebook
    │   └── multi_linear_regression/
    │       ├── README.md                         # Documentation & Math
    │       ├── LinRegClosed.py                   # Normal Equation implementation
    │       └── multi_linear_regression.ipynb     # Full workflow notebook
    │
    └── gradient_descent/                         # Iterative optimization
        ├── README.md                             # Gradient Descent overview
        ├── batch_gd/
        │   ├── simple_linear_regression_using_gradient_descent/
        │   │   ├── README.md
        │   │   ├── GDSimLin.py
        │   │   └── simple_linear_reg_using_gradient_descent.ipynb
        │   └── multiple_linear_regression_using_gradient_descent/
        │       ├── README.md
        │       ├── GDLinReg.py
        │       └── multiple_linear_regression_using_gradient_descent.ipynb
        ├── stochastic_gd/
        │   ├── README.md
        │   ├── SGDRegressor.py
        │   └── sgd_regressor_test.ipynb
        └── mini_batch_gd/
            ├── README.md
            ├── MBGDRegressor.py
            └── mbgd_regressor_test.ipynb
```

<br/>

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

<br/>

<div align="center">

---

**Built with ❤️ and pure math**

⭐ _If this repo helped you understand ML better, give it a star!_ ⭐

</div>
