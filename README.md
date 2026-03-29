<h1 align="center">Bad Debt Prediction</h1>

<p align="center">
  Detecting High-Risk Borrowers Preventing Bad Debt and Financial Losses
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

--


## 🔹 Solution Approach

<details>
<summary><b>1. Data Preprocessing & EDA</b></summary>

Handled **missing values**, removed **duplicates**, and validated **financial variables** to ensure data consistency and reliability.
Performed **exploratory data analysis (EDA)** to analyse **repayment behaviour**, **delinquency trends**, and **outliers** using statistical plots and correlation analysis.

**Insight:** **Credit score**, **repayment behaviour**, and **delinquency patterns** showed strong differentiation between **defaulters** and **non-defaulters**.

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

**Insight:** **CR22 + WoE + IV filtering** significantly improved **class separation**, **interpretability**, and overall **model performance**.

</details>

---

<details>
<summary><b>3. Class Imbalance Handling</b></summary>

Initial **under-sampling** led to **information loss**.
Implemented **SMOTE-Tomek** to balance the dataset using **synthetic sampling + noise removal**.

**Insight:** Improved detection of **defaulters**, increasing **recall** and reducing **missed high-risk customers**.

</details>

---

<details>
<summary><b>4. Model Selection</b></summary>

Trained multiple models: **Logistic Regression**, **Random Forest**, **XGBoost**, and **CatBoost**.
Evaluated based on **generalisation**, **recall**, and ability to detect **high-risk customers**.

**Final Model:** **Random Forest** (best balance of **recall + stability**)

**Insight:** **Ensemble models** captured complex patterns while maintaining **robust performance**.

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

**Insight:** Model is optimised for **high recall**, ensuring early detection of **risky customers**.

</details>

---

<details>
<summary><b>6. Performance Analysis</b></summary>

Focused on **classification errors** and **feature contribution**.
Special attention on **False Negatives**, as they represent the highest **financial risk**.
Used **feature importance** to identify key drivers of default.

**Insight:** Strong **risk separation** and clear **driver identification** validate model reliability.

</details>

---

<details>
<summary><b>7. PSI & CSI Monitoring</b></summary>

Used **PSI** and **CSI** along with **Out-of-Time (OOT) testing** to monitor model stability.

* **PSI (0.39)** → Significant **data drift**
* **CSI (Stable)** → Consistent feature importance

**Insight:** Indicates **concept drift**, requiring **monitoring**, **recalibration**, and **periodic retraining**.

</details>

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
