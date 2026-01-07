# Insurance Agent Fund Misappropriation Detection  
**Shin Kong Life Insurance Case Study**

## Overview
This project develops an **explainable machine learning system** to detect potential fund misappropriation behaviors among insurance agents.  
The goal is to support **internal control, risk management, and ESG-oriented corporate governance**, particularly under highly imbalanced real-world data conditions.

---

## Motivation
Cases where insurance agents illegally transfer policyholders’ funds into personal accounts pose significant **financial, legal, and reputational risks**.

This project aims to:
- Detect high-risk agent behaviors using machine learning
- Prioritize **high recall** to avoid missing misappropriation cases
- Provide **interpretable risk signals** aligned with internal audit logic
- Strengthen corporate governance under ESG frameworks

---

## Dataset
- Proprietary transactional data provided by **Shin Kong Life Insurance**
- Includes agent demographics, employment status, policy behaviors, and performance indicators
- **Severely imbalanced classification problem**

| Split | Normal (Class 0) | Misappropriation (Class 1) | Fraud Rate |
|------|------------------:|---------------------------:|-----------:|
| Train | 2,785 | 28 | 1.00% |
| Test  | 1,196 | 10 | 0.83% |

---

## Data Preprocessing
- Removed features with **>80% missing values**
- Imputed missing values for selected financial indicators (filled with 0 when appropriate)
- Numerical features scaled using **Min–Max Scaling**
- Categorical variables encoded with **One-Hot Encoding**
- Train-test split: **70% / 30%** (`random_state = 1234`)

---

## Models Evaluated
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  

Hyperparameter tuning was conducted using **Grid Search**.

---

## Model Performance
Due to extreme class imbalance, **Recall** was prioritized to maximize detection of risky agents.

| Model | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| Logistic Regression | 0.50 | 1.00 | 0.10 |
| SVM | 0.06 | 1.00 | 0.12 |
| Random Forest | 0.05 | 1.00 | 0.10 |
| XGBoost | 0.05 | 1.00 | 0.10 |

> High recall ensures potential misappropriation cases are not overlooked, even at the cost of precision.

---

## Feature Importance & Risk Indicators
Feature importance analysis identified several interpretable risk drivers:
- Agent job level (e.g., Assistant Manager, Deputy Manager, District Manager)
- Employment status changes (resignation, suspension, termination)
- Abnormal monthly performance patterns
- Policy reminder return rates
- Frequency of canceled endorsement restrictions on customer checks
- Policy status anomalies (lapse, suspension, automatic premium loan usage, high paid-up ratio)

These factors align closely with real-world audit and compliance practices.

---

## Audit & Monitoring Recommendations
A multi-level audit mechanism is proposed based on model insights:

### High-Risk Agent Profiles
- Mid-to-senior level agents  
- Tenure ≥ 2 years and age ≥ 40  
- Agents with recent contract termination or suspension  

### Suggested Review Frequency
- **Quarterly**: performance anomalies, reminder returns, check-related indicators  
- **Semi-Annual**: remittance cancellations, address inconsistency rates  
- **Annual**: frequent policy surrender/loan behavior, agent outstanding balances  

---

## Key Contributions
- Machine learning applied to **real-world, highly imbalanced insurance data**
- Recall-focused modeling for operational risk detection
- Interpretable feature importance supporting internal control decisions
- Practical audit recommendations bridging analytics and governance

---

## Future Work
- Integrate **anomaly detection** techniques
- Explore **deep learning models** for emerging fraud patterns
- Continuously retrain models with updated transactional data

---

## Tech Stack
- Python  
- Scikit-learn  
- XGBoost  
- Pandas / NumPy  
