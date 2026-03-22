# Bad Debt Prediction

### 🔹 **Overview**

This project builds a data-driven system to predict bad-debt customers before credit approval. It handles class imbalance using WoE–IV feature engineering and SMOTE-Tomek resampling, and compares models (Logistic Regression, Random Forest, XGBoost, CatBoost) using ROC-AUC, KS, and Gini with a focus on risk detection. The solution includes MLflow tracking, a Streamlit app for real-time scoring, and PSI/CSI monitoring to ensure model stability and detect drift.

[![Open Streamlit App](https://img.shields.io/badge/Open%20App-Streamlit-red?logo=streamlit)](https://1baddebtprediction-h6ntjamopchs3yrgmzjhwf.streamlit.app/)
---

### 🔹 **Business Problem** 

Credit-based businesses enable “buy now, pay later” models, increasing sales but introducing repayment risk. Some high-risk customers get approved due to non-risk-driven decisions, leading to bad debt and financial loss.

👉 Goal: Predict Good (0) vs Bad (1) customers before approval to improve credit decisions.

---

### 🔹 **Why This Problem Is Hard**

* **Class imbalance:** Few customers default, so high accuracy can still miss risky cases.
* **Feature stability:** Predictive features may not hold on new or future data.
* **Business trade-offs:** Missing bad customers causes losses, while rejecting good ones impacts opportunity—making evaluation and threshold selection critical.

---
## 🔹 Solution Approach

#### 1) Data Preprocessing and EDA

* Improved data quality by handling missing and inconsistent values, enabling reliable analysis of repayment behavior and delinquency patterns.
* Removed duplicate records to prevent bias in identifying true customer risk signals.
* Validated key financial and credit variables to ensure consistency in delinquency and repayment tracking.
* Conducted EDA to analyze repayment behavior, identify delinquency trends, detect outliers, and uncover key drivers of default risk.
* Used **univariate and bivariate analysis** (histograms, box plots, correlation analysis) to study customer behavior patterns.

**Insight:**
Credit score, repayment behavior, and delinquency features emerged as strong predictors of default, clearly distinguishing high-risk customers from low-risk segments.

---

#### 2) Feature Engineering

* Compared credit score variables provided by two bureaus (**CR21 vs CR22**) using distribution analysis and box plots.
* Selected **CR22 score** as it showed clearer separation between good and bad customers, making it more predictive.
* Applied Weight of Evidence (WoE) binning to transform variables into risk-aligned features.
* Used Information Value (IV) to identify and retain the most predictive features.
* Ensured a monotonic relationship between features and default risk for better interpretability.

**Insight:**
CR22 outperformed CR21 in capturing customer risk, and WoE + IV transformed raw data into structured, risk-aligned features, improving both model interpretability and predictive power.


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

* Achieved ~60% recall for high-risk borrowers, improving early identification of potential defaulters before credit approval
* Demonstrated potential to reduce bad-debt exposure from **₹1M to ~₹0.4M** based on model-driven risk detection (scenario-based estimate)
* Improved credit decisioning by prioritizing **risk-focused metrics (KS, Gini, Recall)** over traditional accuracy
* Enabled continuous model monitoring using **PSI/CSI**, supporting early detection of behavior shifts and timely **model recalibration**.

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


-----
  
## 🔹 Tech Stack

<img width="1406" height="324" alt="image" src="https://github.com/user-attachments/assets/74b7508d-8965-4fc9-90ef-81cda46f0fc2" />




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

