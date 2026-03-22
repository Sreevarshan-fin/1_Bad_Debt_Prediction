# Bad Debt Prediction

![Python](https://img.shields.io/badge/Python-3.11-black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-black)
![MLflow](https://img.shields.io/badge/MLflow-black)
![AWS EC2](https://img.shields.io/badge/AWS_EC2-black)
![Streamlit](https://img.shields.io/badge/Streamlit-black)
![XGBoost](https://img.shields.io/badge/XGBoost-black)
![CatBoost](https://img.shields.io/badge/CatBoost-black)
![Random Forest](https://img.shields.io/badge/Random_Forest-black)
![SMOTE-Tomek](https://img.shields.io/badge/SMOTE--Tomek-black)
![WoE/IV](https://img.shields.io/badge/WoE_/_IV-black)
![PSI/CSI](https://img.shields.io/badge/PSI_/_CSI-black)

---

### 🔹 **Overview**

This project builds a data-driven system to predict bad-debt customers before credit approval. It handles class imbalance using WoE–IV feature engineering and SMOTE-Tomek resampling, and compares models (Logistic Regression, Random Forest, XGBoost, CatBoost) using ROC-AUC, KS, and Gini. The solution includes MLflow tracking, a Streamlit app for real-time scoring, and PSI/CSI monitoring to ensure model stability and detect drift.

[![Open Streamlit App](https://img.shields.io/badge/Open%20App-Streamlit-red?logo=streamlit)](https://1baddebtprediction-h6ntjamopchs3yrgmzjhwf.streamlit.app/)
---

### 🔹 **Business Problem** 

Credit-based businesses enable “buy now, pay later” models, increasing sales but introducing repayment risk. Some high-risk customers get approved due to non-risk-driven decisions, leading to bad debt and financial loss.

👉 Goal: Predict Good (0) vs Bad (1) customers before approval to improve credit decisions.

---

## 🔹 Why This Problem Is Hard

| Challenge | Description |
|---|---|
| Class imbalance | Very few customers default — high accuracy can still miss risky cases entirely |
| Feature stability | Predictive features may not hold on new or future customer data |
| Business trade-offs | Missing bad customers causes direct losses; rejecting good ones reduces revenue opportunity |
| Threshold selection | Choosing the right cut-off requires balancing recall vs precision based on business cost |

---

### 🔹 Business Impact (Scenario)

* Achieved ~60% recall for high-risk borrowers, improving early identification of potential defaulters before credit approval
* Demonstrated potential to reduce bad-debt exposure from **₹1M to ~₹0.4M** based on model-driven risk detection (scenario-based estimate)
* Improved credit decisioning by prioritizing **risk-focused metrics (KS, Gini, Recall)** over traditional accuracy
* Enabled continuous model monitoring using **PSI/CSI**, supporting early detection of behavior shifts and timely **model recalibration**.

----
  
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

| Metric | Value | What It Measures |
|---|---|---|
| ROC-AUC | 0.74 | Overall ranking performance |
| Gini | 0.48 | Discriminatory power |
| KS Statistic | 34% | Good vs bad separation |
| Recall (bad) | 60% | High-risk customer detection |

**Insight:** Recall-focused evaluation ensures risky customers are flagged before approval.

</details>

<details>
<summary><b>6 — PSI & CSI Monitoring</b></summary>

| Index | Value | Status |
|---|---|---|
| PSI | 0.39 | 🔴 > 0.25 — significant drift |
| CSI | Low | 🟢 Features stable |

**Diagnosis:** High PSI + low CSI = concept drift — customer behaviour changed, not the features.

**Actions:** PSI > 0.25 alerts set · WoE bins recalibrated · Periodic retraining recommended

</details>


---

### 🔹 **Project Architecture**


![Credit Risk Flow](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/ml_bdb.svg)

-----

### 🔹 **Data Source**

**This project uses real client data under a Non-Disclosure Agreement, so raw dataset details cannot be disclosed.**

The working dataset contains about 100K customers and 99 features, covering demographic, behavioral, credit, and bureau attributes. This master dataset was used for all analysis and modeling.



![Credit Risk Flow](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/bdb_data_source.svg)

----------------

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

--------------

## 🔹 Challenges

- **Severe class imbalance** — bad customers were a tiny minority, requiring careful resampling strategy selection and metric prioritisation
- **Misleading accuracy** — shifted evaluation entirely toward recall, KS, and Gini to reflect true business risk
- **Feature selection complexity** — noisy, correlated, and leakage-prone variables addressed using WoE/IV filtering and stability checks
- **Recall vs precision trade-off** — SMOTE-Tomek improved bad customer detection but increased overfitting risk in some models, requiring careful validation

---

## 🔹 Future Improvements

- Build an **audit-ready scoring history** so credit decisions can be reviewed and explained over time
- Implement **automated drift alerts** with retraining triggers when PSI exceeds threshold
- **Recalibrate WoE bins** periodically as customer behaviour patterns shift
- Explore **scorecard development** using logistic regression on WoE features for full regulatory interpretability


---

## 👉 **Proof of Work - Detalied Section**

#### 🔹 **Experiment Tracking & Model Lifecycle Setup (AWS MLflow)**


- **MLflow Tracking Server on AWS EC2**

MLflow tracking server hosted on AWS EC2 to log experiments, metrics, and artifacts centrally.

<img width="1852" height="143" alt="EC2_Instance" src="https://github.com/user-attachments/assets/3dd85c69-89f4-490c-962a-a569ce3b2807" />


-  **Experiment Run Tracking**

 Multiple model runs tracked with parameters and performance metrics to enable reproducible model comparison.
 

<img width="1917" height="893" alt="Experimental_Tracking_Table_EC2" src="https://github.com/user-attachments/assets/886ff470-703b-4e5a-bb10-3dbdcfce65b1" />

 
<img width="1193" height="512" alt="Comparision_2" src="https://github.com/user-attachments/assets/b72fcd6f-df1a-4e6c-aa33-12c1145eadc3" />




- **Model Registry**


<img width="1911" height="837" alt="Model_Register" src="https://github.com/user-attachments/assets/f246e200-9457-45bd-9828-1f98526183ca" />


------------

-----

