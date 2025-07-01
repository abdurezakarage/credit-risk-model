# Credit Risk Model

## Project Structure
```
credit-risk-model/
├── data/           
│   ├── raw/       
│   └── processed/
├── notebooks/      # Jupyter notebooks
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── feature.ipynb       # Feature Engineering
│   ├── proxy.ipynb         # Proxy Target Variable Analysis
│   ├── train.ipynb         # Model Training
│   └── mlruns/             # MLflow experiment tracking
├── src/           # Source code
│   ├── __init__.py
│   ├── data_processing.py      # Data preprocessing utilities
│   ├── feature_engineerig.py   # Feature engineering functions
│   ├── proxy_target_variable.py # Proxy variable creation
│   └── train.py                # Model training pipeline
├── tests/         # Test files
│   └── __init__.py
├── venv/          # Virtual environment
├── .github/       # GitHub workflows and configurations
├── .gitignore     # Git ignore rules
├── requirements.txt
└── README.md
```

## Influence of Basel II on Model Interpretability

The **Basel II Capital Accord** emphasizes risk-sensitive capital allocation, requiring financial institutions to measure credit risk more accurately and justify the internal models used for regulatory capital estimation. This regulatory framework mandates transparency, auditability, and stress testing capabilities.

### Model Requirements

As a result, models used for credit scoring must be:

- **Interpretable**: Decision-makers and regulators need to understand how inputs drive outputs
- **Well-documented**: Model development, assumptions, limitations, and validation procedures must be traceable

---

## Need for Proxy Default Variables and Associated Risks

In the absence of a clear "default" indicator, analysts must construct proxy variables, such as:

- **90+ days past due (DPD)**
- **Charge-offs**
- **Collection referrals**

This is essential for supervised learning, where labeled outcomes are required to train and validate models.

### Business Risks

However, business risks include:

- **Label leakage**: If the proxy doesn't accurately represent true default, the model may misclassify risk
- **Bias and fairness issues**: Proxies may systematically disadvantage certain customer segments (e.g., informal income earners in developing economies)
- **Non-compliance**: Misaligned definitions with regulatory standards can result in penalties or invalidation of risk-weighted assets (RWAs)

---

## Model Complexity vs. Regulatory Compliance

In regulated financial environments, particularly under the **Basel II Accord** and supervisory expectations, a careful balance must be struck between model complexity and regulatory compliance.

### Interpretable Models

**Examples**: Logistic regression with Weight of Evidence (WoE) encoding

**Advantages**:
- ✅ High transparency
- ✅ Clear understanding of input variable effects
- ✅ Easier regulatory justification
- ✅ Lower operational costs
- ✅ Simpler fairness and bias auditing

**Disadvantages**:
- ❌ Moderate performance
- ❌ May underfit complex or non-linear data patterns

### Complex Models

**Examples**: Gradient Boosting, XGBoost

**Advantages**:
- ✅ Higher predictive performance
- ✅ Captures intricate data interactions

**Disadvantages**:
- ❌ Often function as black boxes
- ❌ Require model explainers (e.g., SHAP) for transparency
- ❌ Higher operational costs
- ❌ Greater difficulty in demonstrating fairness
- ❌ Regulatory compliance challenges

---

## Hybrid Modeling Strategy

To address these trade-offs, many institutions are adopting **hybrid modeling strategies**:

- **Complex models** for initial screening or feature engineering
- **Interpretable models** for final decision-making

This approach enables organizations to benefit from enhanced performance without compromising regulatory transparency and accountability.

---

## Getting Started

1. **Clone the repository**
2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---







