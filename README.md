# Forecasting Corporate Strategy Using Quantum and Classical Machine Learning Models
### A Comparative Approach

**MSc Thesis — Financial Engineering & Management**
Karlsruhe Institute of Technology (KIT) | HECTOR School of Engineering and Management

**Author:** Kaviya Gopal (Matr. Nr. 2571237)
**Supervisors:** Prof. Dr. Martin Ruckes · Prof. Dr. Marliese Uhrig-Homburg
**Issued:** September 2025 | **Submitted:** March 2026

---

## Overview

This repository contains the full implementation of my MSc thesis, which investigates whether **quantum machine learning (QML) models can improve the forecasting of corporate Free Cash Flow (FCF)** relative to classical and deep learning approaches — and whether any observed improvements translate into **economically meaningful valuation outcomes**.

The core research question:

> *Can quantum machine learning models better forecast corporate cash flows than classical models, and do such differences lead to economically meaningful advantages?*

Rather than limiting analysis to statistical error metrics, this study embeds model-generated forecasts into a **Discounted Cash Flow (DCF) valuation framework** under multiple economic scenarios — assessing the strategic consequences of model choice, not just its predictive accuracy.

---

## Key Findings

- **Increased model complexity does not guarantee economic reliability.** Quantum and deep learning models capture nonlinear relationships but exhibit higher sensitivity to noise and limited data, leading to unstable valuation outcomes.
- **Kernel-based quantum models (QSVM) perform most consistently**, producing comparatively stable valuation results across economic scenarios.
- **Forecasting errors are nonlinearly amplified in DCF models** — small inaccuracies can materially affect implied firm values and strategic signals, regardless of which model produced them.
- **QML shows conceptual promise but currently offers limited practical advantage** over well-structured classical models for corporate FCF forecasting in NISQ-era hardware conditions.

---

## Models Implemented

| Model | Type | Library |
|-------|------|---------|
| Linear Regression (OLS) | Classical baseline | `scikit-learn` |
| Long Short-Term Memory (LSTM) | Deep learning | `TensorFlow / Keras` |
| Quantum Support Vector Machine (QSVM) | Quantum–classical hybrid | `Qiskit` + `scikit-learn` |
| Quantum Neural Network (QNN) | Quantum–classical hybrid | `Qiskit` |

All models were trained and evaluated **sector-wise** across S&P 500 industry segments to reduce cross-sector heterogeneity and allow meaningful comparisons.

---

## Dataset

- **Universe:** Selected S&P 500 firms, segmented by sector (Information Technology, Financials, Energy, Health Care, and others)
- **Frequency:** Quarterly financial observations over a 20-year horizon
- **Firm-level features:** Net operating cash flows, investing cash flows, cash & equivalents, revenue, operating income, total liabilities, total assets
- **Macroeconomic features:** GDP growth, interest rates, S&P 500 growth
- **Target variable:** Free Cash Flow (FCF), in millions USD
- **Source:** [Financial Modeling Prep (FMP)](https://site.financialmodelingprep.com/) for firm-level data; public economic databases for macro variables

> All data was cleaned (missing values removed), standardised via z-score normalisation, and split on an **80/20 time-based train–test split** to prevent look-ahead bias.

---

## Repository Structure

```
├── data/
│   ├── raw/                  # Raw financial data from FMP (not committed — see data/README.md)
│   ├── processed/            # Cleaned, normalised, sector-segmented datasets
│   └── macroeconomic/        # GDP, interest rate, S&P 500 time series
│
├── models/
│   ├── linear_regression/    # OLS baseline — sector-wise implementation
│   ├── lstm/                 # LSTM architecture, training loop, hyperparameters
│   ├── qsvm/                 # Quantum kernel construction, QSVM regression
│   └── qnn/                  # Parameterised quantum circuits, QNN regressor
│
├── evaluation/
│   ├── statistical/          # MAE, RMSE, R² — model-wise and sector-wise
│   ├── heatmaps/             # MAE × Sectors, RMSE × Sectors, R² × Sectors
│   └── out_of_sample/        # Predicted vs. realised FCF diagnostics
│
├── valuation/
│   ├── dcf_framework/        # DCF model with scenario-based WACC and growth assumptions
│   ├── firm_valuation/       # Implied firm values per model, per company
│   └── dispersion_analysis/  # Z-score normalised DCF heatmaps, within-firm relative bias
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_dcf_valuation.ipynb
│
├── thesis/
│   └── ThesisReport.pdf      # Full submitted thesis
│
├── requirements.txt
└── README.md
```

---

## Methodology Summary

### 1. Data Pipeline
Firm-level quarterly financials were sourced from FMP, merged with macro indicators, cleaned for missing values, and segmented by GICS sector. Features were z-score normalised per sector prior to modelling. A strict chronological 80/20 split was applied universally.

### 2. Model Architectures

**Linear Regression** — OLS baseline. Coefficients estimated by minimising residual sum of squares. Interpretable and robust; serves as the primary benchmark.

**LSTM** — Recurrent architecture with forget, input, cell, and output gates modelling temporal dependencies via Backpropagation Through Time (BPTT). Input reshaped to 3D tensors (samples × timesteps × features); trained with Adam optimiser.

**QSVM** — Classical input features encoded into quantum states via parameterised rotation gates. Quantum kernel matrix computed from squared inner products of quantum state vectors (simulated via Qiskit statevector simulator). SVR applied with precomputed quantum kernel.

**QNN** — Variational quantum circuit with parameterised rotation gates and entangling layers. Trained via gradient-based optimisation of quantum circuit parameters. Predictions produced by measuring expectation values of quantum observables.

### 3. Evaluation Framework
- **Statistical metrics:** MAE, RMSE, R² — computed per model per sector
- **Out-of-sample diagnostics:** Predicted vs. realised FCF plots across all sectors
- **Cross-model comparison:** Sector-wise heatmaps, metric-wise rankings, model performance hierarchy

### 4. Valuation Integration
Model-generated FCF forecasts were fed into a DCF framework under multiple economic scenarios (baseline, optimistic, stress). Implied firm values were computed and compared across models to assess the **economic consequences of forecasting differences** — including the forecast amplification effect, where small prediction errors compound nonlinearly through terminal value calculations.

---

## Evaluation Results (Summary)

| Model | Relative Forecasting Stability | DCF Valuation Stability | Best Sectors |
|-------|-------------------------------|------------------------|--------------|
| Linear Regression | High | High | Financials, Utilities |
| LSTM | Medium | Low–Medium | IT (large datasets) |
| QSVM | Medium–High | **Most stable across scenarios** | Energy, Health Care |
| QNN | Low–Medium | Low | Limited |

> Full sector-wise heatmaps, out-of-sample diagnostics, and firm-level valuation dispersion charts are available in `evaluation/` and `valuation/`.

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- IBM Qiskit (for quantum model simulation)
- TensorFlow 2.x

### Install dependencies

```bash
git clone https://github.com/kaviyagopal/fcf-quantum-forecasting.git
cd fcf-quantum-forecasting
pip install -r requirements.txt
```

### Key dependencies

```
pandas
numpy
scikit-learn
tensorflow
qiskit
qiskit-machine-learning
matplotlib
seaborn
jupyter
```

> **Note on quantum simulation:** QSVM and QNN models are run on Qiskit's classical statevector simulator. No real quantum hardware access is required to reproduce results.

---

## How to Reproduce

1. **Data:** Follow `data/README.md` to pull firm-level data from FMP (free API key required). Macro data files are included in `data/macroeconomic/`.
2. **Pipeline:** Run `notebooks/01_data_pipeline.ipynb` to clean, normalise, and segment the dataset.
3. **Training:** Run `notebooks/03_model_training.ipynb` — models are trained sector-wise; runtime for QNN is significantly longer (~several hours on CPU).
4. **Evaluation:** Run `notebooks/04_evaluation.ipynb` to reproduce all statistical metrics and heatmaps.
5. **Valuation:** Run `notebooks/05_dcf_valuation.ipynb` to reproduce DCF-based firm valuation and dispersion analysis.

---

## Academic Context

This thesis contributes to the emerging **quantum finance literature** by providing an economically grounded, empirically validated evaluation of QML models in a corporate finance context — moving beyond market-level (stock price) applications that dominate existing QML research. It is one of the few studies to:

- Apply QML directly to **firm-level Free Cash Flow** rather than market returns
- Evaluate performance through a **valuation lens** (DCF) rather than statistical accuracy alone
- Demonstrate the **forecast amplification effect** — the nonlinear sensitivity of firm value to forecasting errors across model types

---

## Citation

If you reference this work, please cite as:

```
Gopal, K. (2026). Forecasting Corporate Strategy Using Quantum and Classical Machine
Learning Models: A Comparative Approach. Master's Thesis, MSc Financial Engineering
& Management, Karlsruhe Institute of Technology (KIT).
```

---

## Contact

**Kaviya Gopal**
MSc Financial Engineering & Management — KIT
[LinkedIn](https://linkedin.com/in/kaviyagopal) · [Email](mailto:kaviya.gopal@email.com)

---

*This repository is shared for academic and portfolio purposes. Data sourced from Financial Modeling Prep (FMP) is subject to their terms of use.*
