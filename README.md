<h1 align="center">Bad Debt Prediction</h1>

<p align="center">
  Predicting high-risk borrowers to reduce credit default losses using Machine Learning
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Pandas-Data-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NumPy-Array-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MLflow-Tracking-purple?style=for-the-badge" />
</p>

---

## Problem Statement

Credit Business Operating **Buy Now, Pay Later (BNPL)** faces a trade-off between **revenue growth** and **credit risk**. Approving customers without **structured risk assessment** leads to **payment defaults**, causing **bad debts** and **financial losses**. This lack of **predictive evaluation** impacts **cash flow**, **profitability**, and **risk management**.

This project builds a **machine learning classification model** to label customers as **Good (0) / Bad (1)**, enabling **data-driven credit decisions**.

---

## Business Impact

- Achieved **60% recall** on bad customers, catching 3 in 5 defaulters before credit approval — reducing reactive collections and financial exposure
- Demonstrated potential to reduce bad-debt exposure from **₹1M to ~₹0.4M** through model-driven risk decisioning (scenario-based estimate based on recall performance)
- Prioritised **KS (34%), Gini (0.48), and Recall** over accuracy, reflecting the true business cost of approving a high-risk customer
- Implemented **PSI/CSI monitoring** to detect shifts in customer behaviour and feature distributions, enabling timely recalibration before model performance degrades


---

## Data

Real client data under NDA — ~100K customers with 99 features.

Features include:
• Demographic (age, income, etc.)  
• Credit behavior (utilization, delinquency, repayment history)  
• Bureau scores  

⚠️ Dataset cannot be shared due to confidentiality constraints.

---
  
## 🔹 Solution Approach

<details>
<summary><b>1 — Data Preprocessing & EDA</b></summary>

- Cleaned missing values, removed duplicates, and validated financial variables for reliable analysis
- EDA across repayment behaviour, delinquency trends, and outliers using histograms, box plots, and correlation analysis

**Insight:** Credit score, repayment behaviour, and delinquency features were the strongest default predictors.

</details>

<details>
<summary><b>2 — Feature Engineering</b></summary>

- Compared CR21 vs CR22 bureau scores — selected CR22 for stronger good/bad customer separation
- Applied WoE binning and IV ranking to retain the most predictive, risk-aligned features with monotonic relationships

**Insight:** WoE + IV transformed raw noisy data into interpretable, risk-aligned features — improving both performance and explainability.

</details>

<details>
<summary><b>3 — Class Imbalance Handling</b></summary>

- Under-sampling tested first — caused information loss
- SMOTE-Tomek applied: synthetic minority oversampling + Tomek link removal to clean boundary overlap

**Insight:** SMOTE-Tomek improved bad customer recall — the critical metric for preventing financial loss.

</details>

<details>
<summary><b>4 — Model Selection</b></summary>

Trained and compared four models on train vs test performance to detect overfitting:

| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Bagging ensemble |
| XGBoost | Boosting |
| CatBoost | Boosting + categoricals |

**Insight:** Random Forest selected — best recall balance and stable generalisation with SMOTE-Tomek.

</details>

<details>
<summary><b>5 — Model Evaluation</b></summary>

### 🔹 Metric Interpretation

| Metric | Interpretation |
|---|---|
| ROC-AUC (0.74) | The model has a good ability to distinguish between good and bad customers, meaning it ranks risky customers higher than safe ones most of the time |
| Gini (0.48) | Indicates moderate discriminatory power; the model is effective in separating defaulters from non-defaulters, which is acceptable for credit risk models |
| KS Statistic (34%) | Shows strong separation between good and bad customers; values above 30% are generally considered good in credit risk modeling |
| Recall – Bad Customers (60%) | The model correctly identifies 60% of actual defaulters, helping reduce financial losses by catching high-risk customers early |

**Insight:** The model is optimised for **higher recall on bad customers**, ensuring risky applicants are flagged even at the cost of some false positives.

</details>

<details>
<summary><b>6 — Model Performance Visuals</b></summary>

### 🔹 Confusion Matrix
![Confusion Matrix](path/to/confusion_matrix.png)

👉 Shows classification breakdown:
- True Positives → Correct defaulters
- False Negatives → Missed defaulters (highest risk)

---

### 🔹 ROC Curve
![ROC Curve](path/to/roc_curve.png)

👉 Visualises model’s ability to distinguish between good and bad customers across thresholds.

---

### 🔹 Gini Interpretation
![Gini Curve](path/to/gini_curve.png)

👉 Derived from ROC; reflects model’s discriminatory power in credit risk evaluation.

---

### 🔹 Feature Importance
![Feature Importance](path/to/feature_importance.png)

👉 Highlights key drivers of default risk used by the model.

</details>

<details>
<summary><b>6 — PSI & CSI Monitoring</b></summary>

| Index | Value | Status |
|---|---|---|
| PSI | 0.39 | 🔴 Significant drift (> 0.25) |
| CSI | Low | 🟢 Features stable |

High PSI indicates a shift in customer data distribution, while low CSI confirms that feature importance remains stable.  

👉 This suggests **concept drift** — customer behaviour has changed, not the underlying features.  

**Action:** PSI alerts (> 0.25), WoE recalibration, and periodic model retraining.

</details>

---


### 🔹 **Project Architecture**


![Project Architecture](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/ml_bdb.svg)

-----


## 🔹 How the System Works

1. User enters customer details in the Streamlit app
2. The app sends input data to the trained model (`model.joblib`)
3. The model predicts:

   * **0 → Good Customer (Low Risk)**
   * **1 → Bad Customer (High Risk)**
4. PSI monitors whether new input data differs from training data

---


---

## 🔹 How to Run Locally

```bash id="h7xj07"
git clone https://github.com/Sreevarshan-fin/Bad_Debt_Prediction.git
cd bad-debt-prediction

pip install -r requirements.txt
streamlit run app/app.py
```

👉 Open browser: http://localhost:8501

---

## 💡 How to Use the App

1. Open the Streamlit app (local or deployed link)
2. Enter customer details
3. Click **Predict**

👉 The model returns customer risk classification (Good / Bad)



---

## 🔹 **Project Structure**

```
bad-debt-prediction/
│
├── app/
│   └── app.py                     # Streamlit deployment           
│
├── models/
│   └── model.joblib               # Trained model
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering & Model Training & Evaluation.ipynb
│   ├── 03_PSI and CSI.ipynb
|
├── requirements.txt
├── README.md
└── .gitignore
```
---
## 🔹 Tech Stack

* **Programming:** Python
* **Machine Learning:** Scikit-learn, XGBoost, CatBoost
* **Data Processing & Analysis:** Pandas, NumPy
* **Model Tracking & Deployment:** MLflow, AWS SageMaker, Streamlit
* **Monitoring & Risk Analytics:** WoE/IV, PSI, CSI
  
----

## 🔹 Challenges

- **Severe class imbalance** — bad customers were a tiny minority, requiring careful resampling strategy selection and metric prioritisation
- **Misleading accuracy** — shifted evaluation entirely toward recall, KS, and Gini to reflect true business risk
- **Feature selection complexity** — noisy, correlated, and leakage-prone variables addressed using WoE/IV filtering and stability checks
- **Recall vs precision trade-off** — SMOTE-Tomek improved bad customer detection but increased overfitting risk in some models, requiring careful validation

---

## 🔹 Future Improvements

* Integrate **Evidently AI** for automated monitoring of data drift, model performance, and data quality in production.

* Implement **A/B testing** to compare multiple models in real-world scenarios and select the best-performing model based on business metrics.

* Introduce a **dynamic decision threshold** based on business risk appetite instead of a fixed cutoff.

* Build a **feedback loop from actual repayment/default outcomes** to continuously improve model performance over time.


---

## 👉 **Proof of Work - Detailed Section**

#### 🔹 **Experiment Tracking & Model Lifecycle Setup (AWS MLflow)**


- **MLflow Tracking Server on AWS EC2**

MLflow tracking server hosted on AWS EC2 to log experiments, metrics, and artifacts centrally.

<img width="1852" height="143" alt="EC2_Instance" src="https://github.com/user-attachments/assets/3dd85c69-89f4-490c-962a-a569ce3b2807" /> 

------------------

-  **Experiment Run Tracking**

 Multiple model runs tracked with parameters and performance metrics to enable reproducible model comparison.
 

<img width="1917" height="893" alt="Experimental_Tracking_Table_EC2" src="https://github.com/user-attachments/assets/886ff470-703b-4e5a-bb10-3dbdcfce65b1" />


------------------------------------------
 
<img width="1193" height="512" alt="Comparision_2" src="https://github.com/user-attachments/assets/b72fcd6f-df1a-4e6c-aa33-12c1145eadc3" />


-----------------------------


- **Model Registry**


<img width="1911" height="837" alt="Model_Register" src="https://github.com/user-attachments/assets/f246e200-9457-45bd-9828-1f98526183ca" />


------------

## 🔹**Deployment**
