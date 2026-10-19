# Credit Card Fraud Detection: A Hybrid Stacking Approach

This project implements a high-recall machine learning pipeline to detect fraudulent credit card transactions. Utilizing a dataset of over 284,000 transactions, the model addresses extreme class imbalance (0.17% fraud) through a combination of custom feature engineering and a **Professional Hybrid Stacking Ensemble**.

## Key Achievements
* **Recall:** **82%** (Successfully identifying 8 out of 10 fraudulent attempts).
* **Accuracy:** **99.72%** (Maintained via robust scaling and stratification).
* **Baseline Uplift:** Achieved an **AUPRC of 0.2941**, which is ~170x better than the random baseline (0.0017).
* **Data Integrity:** Implemented **Git LFS** to manage large-scale datasets (284k+ rows) within version control.
---

## Theory
## 1. Credit Card Fraud Detection as an Imbalanced Classification Problem

Credit card fraud detection is a **binary classification problem** where:
- **0 → Legitimate Transaction**
- **1 → Fraudulent Transaction**

A major challenge in this domain is **extreme class imbalance**:
- Fraud cases ≈ 0.17%
- Legitimate cases ≈ 99.83%

## 2. Precision–Recall Trade-off

Two critical metrics:

**Recall (Sensitivity):**
Recall = TP / (TP + FN)  
Measures how many actual fraud cases are correctly detected.

**Precision:**
Precision = TP / (TP + FP)  
Measures how many predicted fraud cases are truly fraud.

### Importance

- High Recall → Fewer frauds missed (important)
- High Precision → Fewer false alarms

In fraud detection:
> Recall is prioritized over Precision because missing fraud is more costly than false alerts.
---
## 3. AUPRC (Area Under Precision-Recall Curve)

AUPRC is a key metric for imbalanced classification.

- Focuses on performance of the positive (fraud) class
- More informative than ROC-AUC in skewed datasets

Baseline AUPRC:
AUPRC ≈ Fraud Rate ≈ 0.0017
Model Performance:
AUPRC = 0.2941

This represents a ~170× improvement over random guessing.
---
## 4. Feature Engineering Theory

### a) Time-Based Risk Modeling

Fraudulent transactions often follow temporal patterns:
- Higher likelihood during low-activity hours (e.g., 2 AM – 5 AM)
Instead of using raw timestamps:
- Time is transformed into categorical risk levels:
  - High Risk
  - Medium Risk
  - Low Risk
This improves model interpretability and predictive power.

---

### b) Zero-Amount Transaction Detection
Fraudsters often perform "verification pings":
- Transactions with amount = 0
- Used to test whether a card is active

Feature:

Is_Zero_Amount = 1 if Amount == 0 else 0

This acts as a strong fraud indicator.

---
### c) Robust Scaling

Standard scaling:
z = (x - mean) / standard deviation
Fails when data contains extreme outliers.

Robust scaling:
z = (x - median) / IQR

Advantages:
- Resistant to outliers
- Preserves distribution of financial data

---

## 5. SMOTE (Synthetic Minority Oversampling Technique)

SMOTE handles class imbalance by generating synthetic samples.

Formula:
x_new = x_i + λ(x_nn - x_i)

Where:
- x_i = minority sample
- x_nn = nearest neighbor
- λ ∈ [0,1]


### Benefits
- Reduces overfitting compared to duplication
- Improves minority class learning

### Important Rule
SMOTE must be applied **only on training data** to prevent data leakage.

## 6. Ensemble Learning and Stacking

Single models often fail due to bias or variance limitations.

### Stacking Architecture

- **Level 0 (Base Models):**
  - XGBoost → Captures complex non-linear relationships
  - Random Forest → Provides stability and handles structured features

- **Level 1 (Meta-Learner):**
  - Logistic Regression → Combines predictions from base models

### Working Principle

1. Base models generate prediction probabilities
2. Meta-model learns how to best combine them
3. Final prediction is produced

### Advantages
- Improves generalization
- Combines strengths of multiple algorithms

---
## 7. Pipeline Integrity and Data Leakage Prevention

A proper machine learning pipeline ensures:
- Consistent preprocessing
- No leakage of test data into training

Correct workflow:
1. Split dataset (Train/Test)
2. Apply SMOTE only on training data
3. Train models
4. Evaluate on test set
---
## 8. Stratified Sampling

Stratified splitting maintains class distribution:

Fraud / Total ≈ constant across train and test sets
### Importance
- Ensures realistic evaluation
- Prevents biased performance metrics

## Data Engineering & Preprocessing
To improve the signal-to-noise ratio, the following professional preprocessing steps were implemented:

1.  **Cyclical Time Risk Analysis:** Raw timestamps were converted into a **"Time Risk"** feature. This binned transactions into High, Medium, and Low risk periods (e.g., 2 AM – 5 AM), providing a significant predictive signal for late-night fraud patterns.
2.  **Verification "Ping" Detection:** Identified and flagged **$0.00 transactions**. Analysis revealed that over **5%** of total fraud cases involved these "account pings," which are now explicitly tracked as a binary feature (`Is_Zero_Amount`).
3.  **Outlier-Resistant Scaling:** Used `RobustScaler` (Median/IQR) instead of StandardScaler to handle the extreme range in transaction amounts (Max: $25,691 vs. Median: $22) without squashing the feature space.
4.  **Deduplication:** Identified and removed **1,081 duplicate records** to prevent data leakage and ensure an honest, generalized evaluation of model performance.

---
## Model Architecture: The Hybrid Stack
A single model often struggles with the precision-recall trade-off in highly imbalanced domains. This project utilizes a **StackingClassifier** to blend the strengths of multiple algorithms:

* **Level 0 (The Specialists):**
    * **XGBoost:** Optimized for non-linear patterns and complex interactions in the PCA-transformed features ($V1-V28$).
    * **Random Forest:** Utilized for its stability and ability to handle categorical signals like `Time Risk` and `Is_Zero_Amount`.
* **Level 1 (The Meta-Learner):**
    * **Logistic Regression:** Acts as a final judge, weighing the probability outputs of the base models to produce a final classification.

> **Balancing Strategy:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) strictly within a scikit-learn `Pipeline` to ensure that synthetic data never leaks into the test set.

---
## Performance Metrics
The model was evaluated using a **Stratified 80/20 split** to maintain the fraud ratio in the test set.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Recall (Class 1)** | **0.82** | Critical for catching the majority of fraudulent activity. |
| **Precision (Class 1)** | **0.36** | Balanced to maintain a functional user experience (False Positive management). |
| **Accuracy** | **99.72%** | High overall correctness across both classes. |
| **AUPRC** | **0.2941** | The primary KPI for imbalanced classification. |

---
## How to Run
1. Ensure Git LFS is installed: `git lfs install`
2. Install dependencies: 
   ```bash
   pip install pandas xgboost imbalanced-learn scikit-learn joblib
   ```
---
## Summary

This project is grounded in strong theoretical principles:

- Handles extreme class imbalance using SMOTE and PR-based metrics
- Uses feature engineering to extract meaningful behavioral signals
- Applies robust scaling for financial data stability
- Implements stacking ensemble to improve predictive performance
- Maintains strict pipeline discipline to avoid data leakage
