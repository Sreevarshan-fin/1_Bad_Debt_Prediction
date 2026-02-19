# **Bad Debt Prediction**

Built a bad-debt prediction system to identify high-risk credit applicants before approval using WoE–IV, SMOTE, and ensemble models, evaluated via ROC-AUC and KS. Enabled explainable risk bands with MLflow tracking and PSI/CSI monitoring for production readiness.



[![Launch App](https://img.shields.io/badge/Streamlit-App-DC2626?logo=streamlit&logoColor=white&style=for-the-badge)](https://1baddebtprediction-c87rnuzn44s9dqhzpyndxg.streamlit.app/)



---

### **Business Problem** 

Credit-based businesses let customers buy now and pay later through EMI or postpaid models. This helps grow sales, but it also brings repayment risk. Most customers pay on time, but some delay or default, and when dues can’t be recovered, they turn into bad debt that directly affects revenue and cash flow leads financial loss.

Companies already collect customer details, repayment history, and bureau data. But credit approvals are not always fully risk-driven. Because of that, some high-risk customers still get approved, leading to avoidable losses.

This project focuses on predicting whether a customer is likely to be Good or Bad before approval, so decisions can be more risk-aware.

---

### **Why This Problem Is Hard**

The biggest challenge is class imbalance — bad customers are a small part of the population. Because of this, accuracy alone can be misleading. A model can show high accuracy while still missing most risky customers.

Another issue is feature stability. Some variables look predictive during training but don’t remain reliable in real portfolios. Also, credit decisions must be explainable and aligned with policy — not just statistically strong.

From a business standpoint, missing a bad customer is more costly than wrongly flagging a good one — so model trade-offs matter.

---

### **Solution Approach**

Developed an imbalanced credit-risk model to predict good vs. bad customers, prioritizing early risk detection and ranking over raw accuracy using WoE–IV feature engineering, SMOTE-Tomek balancing, and multi-model benchmarking (LR, RF, XGBoost, CatBoost) with MLflow tracking. Chose Random Forest for its stable KS/Gini and consistent bad-customer recall, and delivered explainable Low/Medium/High risk bands to strengthen credit decision-making.

---

### **Project Architecture**

![Project Architecture](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/ml_bdb.svg?v=1)



### **Data Source**

**This project uses real client data under a Non-Disclosure Agreement, so raw dataset details cannot be disclosed.**

The working dataset contains about 92K customers and 99 features, covering demographic, behavioral, credit, and bureau attributes. This master dataset was used for all analysis and modeling.

![Credit Risk Flow](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/bdb_data_source.svg)



---

### **Experiment Tracking & Model Lifecycle Setup (AWS MLflow)**


- **MLflow Tracking Server on AWS EC2**

MLflow tracking server hosted on AWS EC2 to log experiments, metrics, and artifacts centrally.

<img width="1852" height="143" alt="EC2_Instance" src="https://github.com/user-attachments/assets/3dd85c69-89f4-490c-962a-a569ce3b2807" />


- **Experiment Run Tracking**

 Multiple model runs tracked with parameters and performance metrics to enable reproducible model comparison.
 

<img width="1917" height="893" alt="Experimental_Tracking_Table_EC2" src="https://github.com/user-attachments/assets/886ff470-703b-4e5a-bb10-3dbdcfce65b1" />

 
<img width="1193" height="512" alt="Comparision_2" src="https://github.com/user-attachments/assets/b72fcd6f-df1a-4e6c-aa33-12c1145eadc3" />



- **Model Registry**


<img width="1911" height="837" alt="Model_Register" src="https://github.com/user-attachments/assets/f246e200-9457-45bd-9828-1f98526183ca" />



---

### **Model Evaluation**

Model evaluation shows strong ranking performance with ROC-AUC ≈ 0.74 and Gini ≈ 0.48, indicating effective separation between good and bad customers.
KS statistic ≈ 34% confirms good risk discrimination across score deciles.
The confusion matrix is based on the model’s default classification output.

<img width="6618" height="1326" alt="Untitled design (5)" src="https://github.com/user-attachments/assets/286464de-6b6d-4993-971e-510cbc1093b9" />



- **KS statistics**

<img width="1392" height="365" alt="image" src="https://github.com/user-attachments/assets/93ad1646-dc63-4bf6-8cf9-d275117640d2" />



-------------

### Monitoring and Data Drift 

**Note :** This project demonstrates model monitoring and drift detection using PSI, CSI. Since real production data was not available, drift analysis was performed using **holdout test predictions** to simulate monitoring conditions.

**PSI (Population Stability Index)**


<img width="915" height="393" alt="image" src="https://github.com/user-attachments/assets/12d7cc8a-04c2-41a3-ba42-001ab0362b1b" />




**Summary :**

Calculated Population Stability Index (PSI) by binning prediction probabilities and comparing train vs test distributions. Total PSI = 0.3996, which indicates significant data drift (> 0.25 threshold). This suggests the score distribution has shifted and the model should be monitored closely and considered for recalibration or retraining.

--------------------

**CSI (Characteristic Stability Index)**

<img width="721" height="660" alt="image" src="https://github.com/user-attachments/assets/0616c1b4-9ec6-4ac5-8c85-9480224ddf2e" />


**Summary:**

Total Characteristic Stability Index ≈ 0.0032, indicating negligible feature drift between train and test datasets. Feature distributions are highly stable and within safe monitoring limits.

-------------------

### Business Impact
 * Helped lower potential credit loss by flagging risky applications before approval.
 * Improved credit approval quality using standardized credit scores and simple risk bands (Low/Medium/High).
 * Detected ~60% of high-risk applicants suggested potential reduction up to 50–60% under early-review strategy
 * Enabled early identification of risky customers using model-based risk scoring.
 * Implemented monitoring and drift checks to detect future behavior changes in model performance and indicating score distribution shift and need for recalibration..

--------
   
## Future Improvements

* Maintain an audit-ready scoring history and performance tracking setup so credit decisions can be reviewed and explained over time.

* Although features are stable, score distribution drift indicates the need for stronger monitoring, drift alerts, recalibration, and periodic retraining to keep risk decisions accurate as data patterns change.

----------

### **Challenges**

* Handling severe **class imbalance**, where bad customers were a small minority — required careful metric selection and resampling strategy testing.
* Avoiding misleading accuracy and shifting evaluation toward **bad-class recall, KS, and Gini**.
* **Feature selection** was challenging due to noisy, correlated, and leakage-prone variables — addressed using WoE/IV and stability checks.
* Balancing recall vs precision when using SMOTE-Tomek — improved detection but increased overfitting in some models.

---
