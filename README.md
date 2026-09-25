# 📊 Telco Customer Churn Prediction & Risk Scoring Engine

> An end-to-end machine learning system designed to predict telecommunication customer churn. Features custom Scikit-Learn preprocessing pipelines, multi-model benchmarking (Random Forest, XGBoost, CatBoost, AdaBoost, KNN), and an interactive web dashboard for real-time customer risk assessment.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telecochurnmlproject-kf8jspjzdkalmcb6pziylr.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Pipeline](https://img.shields.io/badge/CI-Automated%20Tests-brightgreen?logo=githubactions)](https://github.com/gautamp599-mickey/teleco_churn_ML_project/actions)

---

## 🚀 Live Demo

Access the interactive web dashboard for real-time customer risk scoring and feature exploration:

👉 **[Interactive Churn Prediction Web App](https://telecochurnmlproject-kf8jspjzdkalmcb6pziylr.streamlit.app/)**

---

## 📌 Problem & Business Context

In the telecommunications industry, customer acquisition costs significantly outpace retention costs. Losing subscribers not only drains recurring revenue but also increases long-term customer acquisition overhead.

The objective of this project is to detect early churn indicators to enable timely retention interventions. The modeling strategy prioritizes **Recall** to minimize **False Negatives** (customers who churn without being detected) while maintaining a balanced **$F_1$-Score** to prevent over-allocating costly retention incentives to satisfied customers.

---

## 📊 Model Evaluation & Benchmarks

Five classification models were evaluated on unseen holdout data using **stratified 80/20 train-test splits** and **5-fold cross-validation grid searches**:

| Model | Test $F_1$-Score | Test Recall | Operational Assessment |
| :--- | :---: | :---: | :--- |
| **Random Forest (Tuned)** | **0.673** | **0.800** | **Production Champion:** Captures 80% of actual churn events with strong precision balance. |
| **AdaBoost** | 0.612 | 0.683 | Solid baseline performance, but lower overall capture rate. |
| **XGBoost** | 0.608 | 0.730 | Solid recall achieved via `scale_pos_weight`, but lower $F_1$ balance. |
| **CatBoost** | 0.601 | 0.741 | Strong native handling of high-cardinality categorical signals. |
| **K-Nearest Neighbors (KNN)** | 0.534 | 0.582 | Sensitive to sparsity and distance distortion across encoded categorical spaces. |

### 🔍 Key Feature Drivers (Model Interpretability)

Global feature importance extracted from the production Random Forest pipeline identified the primary operational churn levers:

* **Contract Type:** The single strongest predictor of churn volatility. Customers on month-to-month arrangements exhibit substantially higher flight risk than those on 1- or 2-year contracts.
* **Product Stickiness & Add-ons:** The absence of key tech support and security add-ons (`OnlineSecurity`, `TechSupport`) strongly correlates with churn events.
* **Billing Friction:** Electronic check payment methods and elevated monthly charges represent primary customer friction points.
* **Demographic Neutrality:** Demographic attributes (such as gender) exhibited near-zero predictive importance, confirming that retention programs should target contractual structure and product engagement rather than demographics.

---

## 🏗️ Architecture & System Design

```text
Raw Customer Input (Single Instance / Batch CSV)
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         Pipeline Preprocessing (src/pipeline_utils.py)  │
│                                                         │
│  ├─ Custom Feature Engineering                          │
│  │   (Total_addons, Services, Tenure groupings)         │
│  ├─ Numerical Pipeline: Mean Imputation + StandardScaler│
│  ├─ Ordinal Pipeline: Categorical Order Encoding        │
│  └─ Nominal Pipeline: Imputation + One-Hot Encoding     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Tuned Production Classifier                │
│                                                         │
│  Random Forest (max_depth=5, class_weight='balanced')   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Inference & Delivery                  │
│                                                         │
│  ├─ CLI Batch/Single Scoring (src/predict.py)           │
│  └─ Real-Time Risk Probability Dashboard (app.py)       │
└─────────────────────────────────────────────────────────┘
```

### 🛠️ Tech Stack Breakdown

* **Core Runtime & Data:** Python 3.10+, Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-Learn, XGBoost, CatBoost
* **Serving & Frontend:** Streamlit
* **Code Quality & Testing:** Pytest, GitHub Actions CI

---

## 📁 Repository Structure

```text
teleco_churn_ML_project/
├── .devcontainer/                  # Development container configuration
├── .github/
│   └── workflows/
│       └── ci.yml                  # Automated CI workflow running Pytest
├── notebooks/                      # Exploratory research and experimentation
│   ├── 01_eda_teleco.ipynb         # Data auditing, cleaning, and EDA
│   ├── 02_pipeline.ipynb          # Custom transformer and pipeline validation
│   └── 03_Train_test.ipynb         # Model benchmarking, CV, and tuning
├── src/                            # Production source modules
│   ├── __init__.py
│   ├── pipeline_utils.py           # Custom Scikit-Learn transformers
│   ├── predict.py                  # CLI entry point for batch/single inference
│   └── train.py                    # Production model training & pipeline serialization
├── tests/                          # Automated test suite
│   ├── __init__.py
│   └── test_pipeline.py            # Unit tests for transformations and schema integrity
├── app.py                          # Interactive Streamlit web application
├── .gitignore
├── LICENSE                         # MIT License
├── README.md
└── requirements.txt                # Pinned production dependencies
```

---

## 🛠️ Setup & Quickstart Guide

### Prerequisites

* Python 3.10 or higher
* Virtual environment tool (`venv` or `conda`)

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/gautamp599-mickey/teleco_churn_ML_project.git
cd teleco_churn_ML_project

# Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests

Run the test suite to verify data transformers, schema encodings, and pipeline integrity:

```bash
pytest tests/ -v
```

### 3. Train & Serialize the Model Pipeline

Execute the training script to fit the preprocessing pipeline and the champion Random Forest model:

```bash
python -m src.train
```

### 4. Run CLI Inference

Score single-instance or sample customer inputs via the command line interface:

```bash
python -m src.predict
```

### 5. Launch the Streamlit Web Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to simulate customer profiles and inspect real-time risk scores.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.