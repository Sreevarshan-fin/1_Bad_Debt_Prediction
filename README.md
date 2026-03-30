<h1 align="center">Bad Debt Prediction</h1>

<p align="center">
  <b>Predicting high-risk borrowers to minimize bad debt and financial losses.</b>
</p>

<p align="center">
  Designed an end-to-end machine learning pipeline with risk-based feature engineering,
  recall-optimized modeling, and drift monitoring.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Pandas-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MLflow-black?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AWS%20SageMaker-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FastAPI-teal?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/WoE-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/IV-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PSI-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/CSI-red?style=for-the-badge"/>
</p>


---


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

To address severe class imbalance, two approaches were evaluated:

---

### 🔹 Attempt 1: Under-Sampling

Reduced majority class size to balance the dataset.

**Key Observations:**
- Significant **information loss** due to removal of majority samples  
- Models showed **poor generalisation**, especially in precision  
- **Overfitting observed** in most models  
- Despite reasonable **ROC-AUC (~0.74–0.84)**, imbalance in recall and precision made models unreliable  

👉 **Conclusion:** Under-sampling degraded model performance and failed to capture full data patterns.

---

### 🔹 Attempt 2: SMOTE-Tomek (Final Approach)

Applied **SMOTE (synthetic minority generation)** + **Tomek Links (noise removal)**.

**Key Improvements:**
- **Recall significantly improved** across models  
- Better **class balance without losing information**  
- Reduced **noise and class overlap**  
- Improved **generalisation**, especially in tree-based models  

**Model Comparison (Test Performance)**

| Model | Recall | Precision | AUC | Overfitting | Insight |
|------|--------|----------|-----|------------|--------|
| Logistic Regression | 0.91 | 0.09 | 0.69 | Yes | High recall but poor precision → many false positives |
| CatBoost | 0.82 | 0.10 | 0.70 | Yes | Overfitting despite strong recall |
| XGBoost | 0.47 | 0.15 | 0.70 | No | Stable but lower recall |
| **Random Forest** | **0.61** | **0.16** | **0.74** | **No** | ✅ Best balance of recall + stability |


👉 **Final Choice:** SMOTE-Tomek retained as the optimal resampling strategy  

👉 **Insight:** Enabled effective detection of **defaulters (high recall)** while maintaining **model stability and avoiding information loss**

</details>
---

<details>
<summary><b>4. Model Selection</b></summary>

Trained multiple models: **Logistic Regression**, **Random Forest**, **XGBoost**, and **CatBoost**.
Evaluated based on **generalisation**, **recall**, and ability to detect **high-risk customers**.

**Final Model:** **Random Forest** (best balance of **recall + stability**)

👉  **Insight:** **Ensemble models** captured complex patterns while maintaining **robust performance**.

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

## 📈 Business Impact

* Achieved **60% recall** on high-risk customers, identifying **3 out of 5 defaulters** before credit approval
* Reduced potential bad-debt exposure from **₹1M to ~₹0.4M** through model-driven risk decisions *(scenario-based estimate)*
* Prioritised **KS (34%)**, **Gini (0.48)**, and **Recall** over accuracy, aligning with real-world **credit risk cost considerations**
* Implemented **PSI/CSI monitoring** to detect data drift, enabling **proactive recalibration** and sustained model performance


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

## 🌐 **Deployment**
