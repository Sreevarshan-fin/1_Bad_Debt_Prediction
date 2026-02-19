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


![project](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/ml_bdb.svg)



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

![Model Metrics](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/bdb_model_metrics_dashboard.svg)




- **KS statistics**

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

### Monitoring and Data Drift 

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
