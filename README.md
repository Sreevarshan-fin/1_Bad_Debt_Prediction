# **Bad Debt Prediction**

### 🔹 **Overview**

This project demonstrates a data-driven approach to predicting potential bad-debt customers before credit approval. It addresses key real-world challenges such as class imbalance and risk-focused evaluation using WoE–IV feature engineering and SMOTE-Tomek resampling. Multiple machine learning models (Logistic Regression, Random Forest, XGBoost, CatBoost) are compared using ROC-AUC, KS, and Gini to prioritize detection of risky borrowers. The workflow includes MLflow experiment tracking, a Streamlit scoring interface for real-time predictions, and PSI/CSI monitoring to evaluate model stability and detect drift over time.

[![Open Streamlit App](https://img.shields.io/badge/Open%20App-Streamlit-red?logo=streamlit)](https://1baddebtprediction-h6ntjamopchs3yrgmzjhwf.streamlit.app/)
---

### 🔹 **Business Problem** 

Credit-based businesses enable “buy now, pay later” models, increasing sales but introducing repayment risk. Some high-risk customers get approved due to non-risk-driven decisions, leading to bad debt and financial loss.

👉 Goal: Predict Good (0) vs Bad (1) customers before approval to improve credit decisions.

---

### 🔹 **Why This Problem Is Hard**

* **Class imbalance:** Only a small percentage of customers default, so a model can achieve high accuracy while still missing many risky borrowers.
* **Feature stability:** Some variables may appear predictive during training but may not remain reliable when applied to new customers or future data.
* **Business trade-offs:** Missing a bad customer leads to direct financial loss, while wrongly rejecting a good customer only results in lost opportunity, making model evaluation and threshold decisions more complex.


---
## 🔹 Solution Approach

####  1) Data Preprocessing and EDA

* Performed **data cleaning** by handling missing values using appropriate imputation techniques (mean/median for numerical, mode for categorical variables).
* Removed **duplicate records** to avoid bias in model training.
* Conducted **data validation checks** to ensure consistency in financial and credit-related variables.
* Performed **Exploratory Data Analysis (EDA)** to understand variable distributions, detect outliers, and identify relationships between features and the target variable.
* Used **univariate and bivariate analysis** (histograms, box plots, correlation analysis) to study customer behavior patterns.

**Insight:**
EDA revealed that **credit score variables and repayment-related features strongly influence default behavior**, making them critical for modeling.

---

#### 2) Feature Engineering

* Compared key credit variables (**NO_SCORE_CR21 vs SCORE_CR22**) using distribution analysis and box plots.
* Selected **SCORE_CR22** due to better separation between good and bad customers.
* Applied **Weight of Evidence (WoE) binning** to both numerical and categorical variables to transform features into risk-based representations.
* Calculated **Information Value (IV)** to measure predictive strength and select important features.
* Ensured **monotonic relationship** between features and target variable for better model interpretability.

**Insight:**
WoE + IV helped convert raw variables into **risk-aligned features**, improving both interpretability and predictive performance.

---

#### 3) Class Imbalance Handling

* Identified significant **class imbalance** (majority: good customers, minority: bad customers).
* Initially applied **under-sampling**, which reduced imbalance but caused **loss of important information**.
* Implemented **SMOTE-Tomek (oversampling + noise removal)** to generate synthetic minority samples and clean overlapping data points.

**Insight:**
SMOTE-Tomek improved **recall for bad customers**, which is critical since failing to detect risky customers leads to financial loss.

---

#### 4) Model Selection

* Trained multiple machine learning models:

  * Logistic Regression (baseline, interpretable)
  * Random Forest (bagging-based ensemble)
  * XGBoost (boosting-based model)
  * CatBoost (handles categorical features efficiently)

* Compared models using **train vs test performance** to detect overfitting.

* Focused on **generalization ability and stability across datasets**.

**Insight:**

* Under-sampling caused instability in most models.
* SMOTE-Tomek improved performance, especially for **ensemble models (Random Forest, XGBoost)**.
* **Random Forest** was selected due to stable performance and better recall balance.

---

#### 5) Model Evaluation

* Evaluated models using multiple metrics:

  * **ROC-AUC (~0.74):** Measures overall ranking performance
  * **Gini (~0.48):** Indicates model discriminatory power
  * **KS Statistic (~34%):** Measures separation between good and bad customers
  * **Recall (Bad Customers):** Priority metric for business

* Analyzed **confusion matrix** to understand classification errors.

**Insight:**

* The model shows **good separation capability** between risky and non-risky customers.
* Focus on **recall ensures higher detection of bad customers**, aligning with business goals.

---

#### 6) PSI and CSI (Model Stability)

* Applied **Population Stability Index (PSI)** on model scores to detect changes in data distribution over time.

* Applied **Characteristic Stability Index (CSI)** to monitor feature-level distribution changes.

* Evaluated model on **Out-of-Time (OOT) validation dataset**.

* Observed:

  * **PSI = 0.39 (> 0.25)** → significant drift
  * **CSI low** → features stable

**Insight:**

* High PSI with low CSI indicates **concept drift (change in customer behavior)** rather than feature drift.

**Actions Taken:**

* Set threshold-based alerts (PSI > 0.25)
* Recommended **periodic model retraining**
* Recalibrated WoE bins on new data
* Continuous monitoring of model performance

---




### 🔹 Business Impact (Scenario)

* In a scenario with **₹1 million potential bad-debt exposure**, traditional approval methods may fail to detect risky applicants early.
* The model **detects ~60% of high-risk borrowers**, enabling preventive actions such as rejection, manual review, or stricter credit terms.
* Early risk identification could **reduce potential losses from ₹1 million to approximately ₹0.4million** through proactive credit controls.
* **PSI and CSI monitoring** help detect shifts in customer behavior and score distributions, supporting timely model recalibration and retraining.

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
## 🔹 Future Improvements

* Maintain an audit-ready scoring history and performance tracking setup so credit decisions can be reviewed and explained over time.

* Although features are stable, score distribution drift indicates the need for stronger monitoring, drift alerts, recalibration, and periodic retraining to keep risk decisions accurate as data patterns change.

----------

### 🔹 **Challenges**

* Handling severe **class imbalance**, where bad customers were a small minority — required careful metric selection and resampling strategy testing.
* Avoiding misleading accuracy and shifting evaluation toward **bad-class recall, KS, and Gini**.
* **Feature selection** was challenging due to noisy, correlated, and leakage-prone variables — addressed using WoE/IV and stability checks.
* Balancing recall vs precision when using SMOTE-Tomek — improved detection but increased overfitting in some models.

---


-----
  
## 🔹 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![AWS SageMaker](https://img.shields.io/badge/AWS%20SageMaker-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

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



---


- 🔹 **KS statistics**

| Decile | Minimum_Probability | Maximum_Probability | Bad | Good | Bad Rate | Good Rate | Cum Bad | Cum Good | Cum Bad Rate | Cum Good Rate | KS |
|--------|---------------------|---------------------|-----|------|----------|-----------|---------|----------|--------------|---------------|------|
| 0 | 0.011 | 0.123 | 30 | 2066 | 1.431 | 98.569 | 30 | 2066 | 1.834 | 10.691 | 8.858 |
| 1 | 0.123 | 0.184 | 51 | 2045 | 2.433 | 97.567 | 81 | 4111 | 4.951 | 21.274 | 16.323 |
| 2 | 0.184 | 0.267 | 64 | 2032 | 3.053 | 96.947 | 145 | 6143 | 8.863 | 31.789 | 22.926 |
| 3 | 0.267 | 0.355 | 84 | 2012 | 4.008 | 95.992 | 229 | 8155 | 13.998 | 42.201 | 28.204 |
| 4 | 0.355 | 0.420 | 98 | 1998 | 4.676 | 95.324 | 327 | 10153 | 19.988 | 52.541 | 32.553 |
| 5 | 0.420 | 0.455 | 143 | 1953 | 6.823 | 93.177 | 470 | 12106 | 28.729 | 62.647 | 33.919 |
| 6 | 0.455 | 0.499 | 174 | 1922 | 8.302 | 91.698 | 644 | 14028 | 39.364 | 72.594 | 33.229 |
| 7 | 0.499 | 0.531 | 189 | 1907 | 9.017 | 90.983 | 833 | 15935 | 50.917 | 82.462 | 31.545 |
| 8 | 0.531 | 0.600 | 255 | 1841 | 12.166 | 87.834 | 1088 | 17776 | 66.504 | 91.989 | 25.486 |
| 9 | 0.600 | 0.960 | 548 | 1548 | 26.145 | 73.855 | 1636 | 19324 | 100.000 | 100.000 | 0.000 |


-------------

### 🔹 Monitoring and Data Drift 

**Note :** This project demonstrates model monitoring and drift detection using PSI, CSI. Since real production data was not available, drift analysis was performed using **holdout test predictions** to simulate monitoring conditions.

**PSI (Population Stability Index)**

| prob_range | train_count | train_% | test_count | test_% | A-B | ln(A/B) | PSI |
|------------|-------------|---------|------------|--------|-----|---------|------|
| 0.011854 - 0.168551 | 11150 | 10.000000 | 3760 | 17.938931 | -7.938931 | -0.584388 | 0.046394 |
| 0.168551 - 0.305608 | 11150 | 10.000000 | 3443 | 16.426527 | -6.426527 | -0.496312 | 0.031896 |
| 0.305608 - 0.410361 | 11150 | 10.000000 | 2835 | 13.525763 | -3.525763 | -0.302011 | 0.010648 |
| 0.410361 - 0.455815 | 11150 | 10.000000 | 2571 | 12.266221 | -2.266221 | -0.204264 | 0.004629 |
| 0.455815 - 0.502793 | 11150 | 10.000000 | 2277 | 10.863550 | -0.863550 | -0.082828 | 0.000715 |
| 0.502793 - 0.533260 | 11181 | 10.027803 | 2021 | 9.642176 | 0.385627 | 0.039215 | 0.000151 |
| 0.533260 - 0.589532 | 11119 | 9.972197 | 1761 | 8.401718 | 1.570480 | 0.171365 | 0.002691 |
| 0.589532 - 0.691033 | 11150 | 10.000000 | 1349 | 6.436069 | 3.563931 | 0.440667 | 0.015705 |
| 0.691033 - 0.831891 | 11150 | 10.000000 | 759 | 3.621183 | 6.378817 | 1.015784 | 0.064795 |
| 0.831891 - 0.995385 | 11150 | 10.000000 | 184 | 0.877863 | 9.122137 | 2.432850 | 0.221928 |





**Summary :**

Calculated Population Stability Index (PSI) by binning prediction probabilities and comparing train vs test distributions. Total PSI = 0.3996, which indicates significant data drift (> 0.25 threshold). This suggests the score distribution has shifted and the model should be monitored closely and considered for recalibration or retraining.

--------------------

**CSI (Characteristic Stability Index)**

| Feature | Train Count | Test Count | CSI |
|--------|------------:|-----------:|-----:|
| Active Credit Cards | 62912 | 20960 | 0.000017 |
| Applicant Age | 62912 | 20960 | 0.000134 |
| Bureau Default | 62912 | 20960 | 0.000080 |
| Bureau Enquiries (Last 12 Months) | 62912 | 20960 | 0.000357 |
| Credit Card Payment Failures | 62912 | 20960 | 0.000051 |
| Credit Score | 62912 | 20960 | 0.000000 |
| Derogatory | 62912 | 20960 | 0.000000 |
| Document Type | 62912 | 20960 | 0.000097 |
| Employment Status | 62912 | 20960 | 0.001132 |
| Late_Payment_30DPD_Last_12M | 62912 | 20960 | 0.000000 |
| Late_Payment_30DPD_Last_24M | 62912 | 20960 | 0.000000 |
| Long_Term_Payment_Delinquency_Count | 62912 | 20960 | 0.000012 |
| Occupation Type | 62912 | 20960 | 0.001164 |
| Open Defaults | 62912 | 20960 | 0.000000 |
| Recent Payment Irregularity | 62912 | 20960 | 0.000000 |
| Residential Status | 62912 | 20960 | 0.000119 |
| Score Card | 62912 | 20960 | 0.000063 |
| Total Historical Defaults | 62912 | 20960 | 0.000000 |



**Summary:**

Total Characteristic Stability Index ≈ 0.0032, indicating negligible feature drift between train and test datasets. Feature distributions are highly stable and within safe monitoring limits.

--------
   
