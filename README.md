<h1 align="center">Bad Debt Prediction</h1>

<p align="center">
  <b>🚀 End-to-End Credit Risk Modeling System to Detect High-Risk Borrowers and Reduce Financial Losses</b>
</p>

<p align="center">
  Designed an end-to-end machine learning pipeline with risk-based feature engineering,
  recall-optimized modeling, and drift monitoring.
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS%20S3-569A31?style=for-the-badge&logo=amazonaws&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS%20SageMaker-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
  <img src="https://img.shields.io/badge/EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/WoE-6A0DAD?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/IV-6A0DAD?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PSI-D32F2F?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/CSI-D32F2F?style=for-the-badge"/>
</p>


---
## 🚀 Key Highlights

- Built an end-to-end **credit risk classification system** to identify high-risk borrowers and enable data-driven lending decisions  
- Achieved **60% recall**, successfully detecting the majority of potential defaulters before credit approval  
- Reduced estimated bad-debt exposure from **₹1M to ~₹0.4M**, improving financial risk control  
- Engineered predictive features using **WoE and IV**, enhancing model interpretability and risk separation  
- Handled severe class imbalance using **SMOTE-Tomek**, improving minority class detection without information loss  
- Selected **Random Forest** for its strong generalisation and optimal balance between recall and precision  
- Evaluated performance using **ROC-AUC, Gini, and KS**, aligning with industry-standard credit risk evaluation metrics  
- Implemented **PSI (0.39), CSI monitoring, and Out-of-Time (OOT) validation** to detect data drift and ensure production reliability  
- Designed a scalable real-time inference system using **AWS SageMaker, FastAPI, and Streamlit**  
- Enabled experiment tracking and model lifecycle management using **MLflow on AWS EC2**

> 💡 This project leverages real-world credit data to predict bad debt risk, using industry-standard credit risk modeling techniques followed in fintech and banking systems.

-----------------

![Project Architecture](assets/projectworkflow.gif)


----

## 🔎 Problem Statement

Credit Business Operating **Buy Now, Pay Later (BNPL)** faces a trade-off between **revenue growth** and **credit risk**. Approving customers without **structured risk assessment** leads to **payment defaults**, causing **bad debts** and **financial losses**. This lack of **predictive evaluation** impacts **cash flow**, **profitability**, and **risk management**.

This project builds a **machine learning classification model** to label customers as **Good (0) / Bad (1)**, enabling **data-driven credit decisions**.

----

## 📊 Data Overview

Real-world credit dataset (~100K customers, 99 features) collected under NDA, structured based on key risk dimensions:

* **Customer Behaviour**
* **Credit Behaviour**
* **Credit Bureau Data**

Includes credit bureau scores from two providers — **CR21** and **CR22** — enabling comparative analysis of their effectiveness in identifying high-risk customers.

⚠️ Dataset cannot be shared due to confidentiality constraints.

----

## ⚙️ Solution Approach

<details>
<summary><b>1. Data Preprocessing & EDA</b></summary>

Handled **missing values**, removed **duplicates**, and validated **financial variables** to ensure data consistency and reliability.
Performed **exploratory data analysis (EDA)** to analyse **repayment behaviour**, **delinquency trends**, and **outliers** using statistical plots and correlation analysis.

👉  **Insight:** **Credit score**, **repayment behaviour**, and **delinquency patterns** showed strong differentiation between **defaulters** and **non-defaulters**.

</details>

---

<details>
<summary><b>2. Feature Engineering</b></summary>

Compared **CR21** and **CR22** bureau scores using **box plots**, analysing **median separation**, **distribution spread**, and **outliers** between good and bad customers.
**CR22** showed clearer separation with reduced overlap, making it a more reliable predictor of default risk.

Applied **Weight of Evidence (WoE)** binning to transform variables into **monotonic risk-based categories**, improving interpretability and alignment with credit risk behaviour.

Performed **feature selection using Information Value (IV)**:

* **IV < 0.02** → Weak (**removed**)
* **0.02 – 0.1** → Medium
* **0.1 – 0.3** → Strong
* **> 0.3** → Very strong

Removed **highly correlated features** to avoid **multicollinearity** and improve **model stability**.

👉  **Insight:** **CR22 + WoE + IV filtering** significantly improved **class separation**, **interpretability**, and overall **model performance**.

</details>

---


<details>
<summary><b>3. Class Imbalance Handling</b></summary>


**To address severe class imbalance, two strategies were evaluated:**


🔹**Attempt 1: Under-Sampling**

Reduced the majority class to balance the dataset.

**Observations:**
- Loss of critical information due to removal of majority samples  
- Poor model generalisation, especially in precision  
- Overfitting observed across multiple models  
- ROC-AUC remained moderate (~0.74–0.84), but imbalance in precision-recall reduced reliability  

👉 **Conclusion:** Under-sampling failed to capture full data distribution and degraded performance.

---

🔹 **Attempt 2: SMOTE-Tomek**

Applied SMOTE for synthetic minority generation and Tomek Links for noise removal.

**Improvements:**
- Better class representation without information loss  
- Reduction in class overlap and noise  
- More consistent model performance  

**Model Comparison (Test Performance)**

| Model | Recall | Precision | AUC | Overfitting | Insight |
|------|--------|----------|-----|------------|--------|
| Logistic Regression | 0.91 | 0.09 | 0.69 | Yes | High recall but excessive false positives |
| CatBoost | 0.82 | 0.10 | 0.70 | Yes | Strong recall but unstable |
| XGBoost | 0.47 | 0.15 | 0.70 | No | Stable but lower detection |
| **Random Forest** | **0.61** | **0.16** | **0.74** | **No** | ✅ Best trade-off between detection and stability |


👉 **Final Choice:** SMOTE-Tomek retained as the optimal resampling strategy  

👉 **Decision Logic:** Prioritised a model that balances **risk detection (high recall)** with **stability and generalisation**, rather than maximising accuracy  

👉 **Insight:** Enabled reliable identification of **high-risk customers** while maintaining **robust and consistent model performance**

</details>

---

<details>
<summary><b>4. Model Selection</b></summary>

Evaluated multiple models to identify the most reliable performer on balanced data.

**Final Model:** Random Forest — selected for its consistent performance and ability to generalise well without overfitting.

👉 **Insight:** Ensemble tree-based models effectively capture complex, non-linear relationships, leading to stable predictions on unseen data.

</details>

---

<details>
<summary><b>5. Model Evaluation</b></summary>

Evaluated using key **credit risk metrics**:

* **ROC-AUC (0.74)** → Good class discrimination
* **Gini (0.48)** → Moderate predictive power
* **KS (34%)** → Strong separation
* **Recall (60%)** → Majority of defaulters identified

Maintained consistent performance across **train**, **test**, and **OOT datasets**.
Threshold tuning (e.g., **0.3**) used to prioritise **risk detection**.

👉  **Insight:** Model is optimised for **high recall**, ensuring early detection of **risky customers**.

</details>

---

<details>
<summary><b>6. Performance Analysis</b></summary>

Focused on **classification errors** and **feature contribution**.
Special attention on **False Negatives**, as they represent the highest **financial risk**.
Used **feature importance** to identify key drivers of default.

👉 **Insight:** Strong **risk separation** and clear **driver identification** validate model reliability.

</details>

---

<details>
<summary><b>7. PSI & CSI Monitoring</b></summary>


Used **PSI** and **CSI** with **Out-of-Time (OOT) validation** to track data drift and ensure model stability.

* **PSI (0.39)** → Significant shift in data distribution
* **CSI (Stable)** → Feature importance remains consistent

Currently, drift is monitored using **PSI and CSI with OOT validation**; future enhancements will extend this to **real-time production monitoring** using tools like **Evidently AI**.

👉  **Insight:** Indicates **concept drift**, requiring **monitoring**, **recalibration**, and **periodic retraining**.

</details>

---

## 📈 Business Impact & Decision Framework

- Achieved **60% recall**, identifying **3 out of 5 defaulters** before credit approval, enabling early risk detection  
- Reduced estimated bad-debt exposure from **₹1M to ~₹0.4M**, improving portfolio risk control
- 
### Decision Strategy
- Model optimized for **high recall** to prioritize detection of risky customers  
- Threshold (~0.3) tuned to minimize false negatives and reduce financial loss  

### Business Trade-Off
- Accepts a controlled increase in false positives (manual review effort)  
- Significantly reduces bad debt risk from undetected defaulters  

### Decision Enablement
- High-risk customers: Reject or approve with stricter terms (higher interest, lower limits)  
- Low-risk customers: Fast-track approvals with better credit offers  

### Model Reliability
- Evaluated using **KS (34%)**, **Gini (0.48)**, and **ROC-AUC** aligned with credit risk standards  
- Implemented **PSI/CSI monitoring with OOT validation** to ensure stability under data drift  

Ensures data-driven, risk-aware, and scalable lending decisions.

---


## 🧰 Tech Stack

* **Programming & Libraries:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost, CatBoost, Random Forest
* **Experiment Tracking & Deployment:** MLflow, AWS SageMaker, Streamlit
* **Monitoring & Risk Analytics:** PSI, CSI
* **Feature Engineering & Techniques:** WoE, Information Value (IV), SMOTE-Tomek, OOT Validation
* **Evaluation Techniques:** KS Statistic, Gini Coefficient, ROC-AUC, with prioritisation of Recall to ensure effective detection of high-risk customers



----


## ⚡ Challenges

- **Severe class imbalance** — bad customers were a tiny minority, requiring careful resampling strategy selection and metric prioritisation
- **Misleading accuracy** — shifted evaluation entirely toward recall, KS, and Gini to reflect true business risk
- **Feature selection complexity** — noisy, correlated, and leakage-prone variables addressed using WoE/IV filtering and stability checks
- **Recall vs precision trade-off** — SMOTE-Tomek improved bad customer detection but increased overfitting risk in some models, requiring careful validation

---

##  🚀 Future Improvements

* Integrate **Evidently AI** for automated monitoring of **data drift, model performance, and data quality** in production.
  *(Currently, drift is monitored using **PSI and CSI with OOT validation**; future work will extend this to real-time production data.)*

* Implement **A/B testing** to compare multiple models in real-world scenarios and select the best-performing model based on **business metrics**

* Introduce a **dynamic decision threshold** based on **business risk appetite**, replacing a fixed cutoff

* Build a **feedback loop** using actual repayment/default outcomes to continuously improve model performance over time


---

## 🔬 Experiment Tracking & Model Lifecycle (MLflow on AWS)

### 🔹 MLflow Tracking Server (AWS EC2)

Deployed an **MLflow Tracking Server on AWS EC2** to centrally log experiments, parameters, metrics, and artifacts.

![MLflow EC2](assets/EC2_Instance.png)

---

### 🔹 Experiment Run Tracking

Tracked multiple model runs with **parameters and performance metrics**, enabling reproducible comparison and model selection.

![Experiment Tracking](assets/Experimental_Tracking_Table_EC2.png)

![Model Comparison](assets/Comparision_2.png)

---

### 🔹 Model Registry

Registered and versioned models using **MLflow Model Registry** for structured lifecycle management and deployment readiness.

![Model Registry](assets/Model_Register.png)


-------------------

## 🌩️ Deployment

### SageMaker Deployment (Design & Implementation)

Designed a scalable model deployment pipeline using AWS SageMaker for real-time inference.

* Uploaded trained model artifacts to Amazon S3
* Developed custom inference script for prediction handling
* Configured SageMaker model and deployed a real-time endpoint
* Tested endpoint locally using SageMaker SDK

**Note:** Full deployment was designed and partially implemented; continuous hosting was not maintained due to cost constraints.

---

## 🏗️ System Architecture

```text id="f9t2b8"
User
  ↓
Streamlit App (EC2)  →  UI & Input Handling
  ↓
FastAPI Backend     →  Validation & API Processing
  ↓
SageMaker Endpoint  →  Real-time Inference
  ↓
Prediction Response →  Returned to UI


Amazon S3 → Model Storage & Versioning (used during deployment)
```

---

### Architecture Overview

This architecture enables scalable real-time predictions by separating key components:

* **Application Layer (Streamlit on EC2):** Handles user interaction
* **Backend Layer (FastAPI):** Validates and processes requests
* **Model Layer (SageMaker):** Serves the model via real-time endpoint
* **Storage Layer (S3):** Stores model artifacts and supports versioning

**Predictions are generated in real time and returned to the UI for display.**

-------

⚡ **Streamlit UI**


![UI](assets/streamlit_ui.png)

-----------


