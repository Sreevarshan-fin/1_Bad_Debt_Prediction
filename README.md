# Bad Debt Prediction


---

### Business Problem 

Credit-based businesses let customers buy now and pay later through EMI or postpaid models. This helps grow sales, but it also brings repayment risk. Most customers pay on time, but some delay or default, and when dues can’t be recovered, they turn into bad debt that directly affects revenue and cash flow.

Companies already collect customer details, repayment history, and bureau data. But credit approvals are not always fully risk-driven. Because of that, some high-risk customers still get approved, leading to avoidable losses.

This project focuses on predicting whether a customer is likely to be Good or Bad before approval, so decisions can be more risk-aware.

---

### Why This Problem Is Hard 

The biggest challenge is class imbalance — bad customers are a small part of the population. Because of this, accuracy alone can be misleading. A model can show high accuracy while still missing most risky customers.

Another issue is feature stability. Some variables look predictive during training but don’t remain reliable in real portfolios. Also, credit decisions must be explainable and aligned with policy — not just statistically strong.

From a business standpoint, missing a bad customer is more costly than wrongly flagging a good one — so model trade-offs matter.

---

### Solution Approach 

I treated this as a loss-reduction and decision-support problem, not just a modeling exercise.

Instead of optimizing only for accuracy, the focus was on catching risky customers early and ranking customers by risk level.

**Main steps included:**

* Handling class imbalance carefully

* Removing noisy and unstable features

* Selecting stable predictors using WoE and IV

* Prioritizing bad-customer recall over raw accuracy

* Using ROC, KS, and Gini for ranking quality

* Converting predictions into Low / Medium / High risk bands

* Using model scores to support decisions, not replace policy

* The goal was to strengthen credit decisions 

---

### Project Architecture

<img width="702" height="776" alt="image" src="https://github.com/user-attachments/assets/14f873cb-21fa-40ac-882a-51e942b52579" />


### Data Source 

**This project uses real client data under a Non-Disclosure Agreement, so raw dataset details cannot be disclosed.**

The working dataset contains about 92K customers and 99 features, covering demographic, behavioral, credit, and bureau attributes. This master dataset was used for all analysis and modeling.

<img width="1536" height="620" alt="ChatGPT Image Feb 5, 2026, 04_53_42 PM" src="https://github.com/user-attachments/assets/b49eef6a-c0c7-4988-987b-89ae276ab26d" />


---

### Modeling Strategy 

* Treated this as an **imbalanced classification problem**, where the goal was to rank customer risk properly rather than chase overall accuracy, since bad cases were a small share of the data.

* Spent time on **feature stability and explainability**. Used WoE and IV to shortlist useful predictors and removed variables that were noisy, too correlated, or showed signs of leakage during validation.

* Tried multiple model types — Logistic Regression, Random Forest, XGBoost, and CatBoost — to see how both linear and tree-based methods performed on the same cleaned feature set.

* Tested different imbalance treatments like **under-sampling and SMOTE-Tomek**. SMOTE helped improve bad-customer recall, but in some runs it also led to overfitting, especially with boosting models.

* Compared models using **recall for bad customers, ROC-AUC, KS, and Gini**, since these are more meaningful for credit risk than plain accuracy.

* Logged experiments and parameter settings using **MLflow**, which helped track results and compare runs properly. Best models were saved in the registry for version control.

* XGBoost showed slightly better recall at first, but results were not consistent across resamples and folds.

* Random Forest gave **more stable KS and Gini values** and more consistent ranking across validations, so I chose it as the main production model.

* Kept XGBoost as a **challenger model** to compare performance over time instead of discarding it completely.

* Tested multiple probability cutoffs and fixed the working threshold at **0.3** to improve risky-customer detection while keeping false positives manageable.

* Final model outputs are stored in **SQLite as probability risk scores**, which are used for risk banding and review — not for automatic approval decisions.

-----


### Experiment Tracking & Model Lifecycle Setup (AWS MLflow)


- **MLflow Tracking Server on AWS EC2**

MLflow tracking server hosted on AWS EC2 to log experiments, metrics, and artifacts centrally.

<img width="1852" height="143" alt="EC2_Instance" src="https://github.com/user-attachments/assets/850500ce-7d35-4209-92da-e42ac698597f" />

- **Experiment Run Tracking**

 Multiple model runs tracked with parameters and performance metrics to enable reproducible model comparison.
 

 <img width="1917" height="893" alt="Experimental_Tracking_Table_EC2" src="https://github.com/user-attachments/assets/88c83ed9-b15a-425e-ae11-f3ebde624a8d" />
 

 <img width="1193" height="512" alt="Comparision_2" src="https://github.com/user-attachments/assets/76987936-b513-44d7-bd2a-cbb376f124bd" />


- **Model Registry**


<img width="1911" height="837" alt="Model_Register" src="https://github.com/user-attachments/assets/2a71f774-36a8-453c-b51d-4490cc9c0e0b" />



---

### Model Evaluation

Model evaluation shows strong ranking performance with ROC-AUC ≈ 0.74 and Gini ≈ 0.48, indicating effective separation between good and bad customers.
KS statistic ≈ 34% confirms good risk discrimination across score deciles.
The confusion matrix is based on the model’s default classification output.

<img width="6618" height="1326" alt="Untitled design (5)" src="https://github.com/user-attachments/assets/4279e49a-e318-4b29-8ef4-488894037b1c" />

-------------

### Monitoring and Data Drift 

**Note :** This project demonstrates model monitoring and drift detection using PSI, CSI. Since real production data was not available, drift analysis was performed using **holdout test predictions** to simulate monitoring conditions.

**PSI (Population Stability Index)**

<img width="910" height="377" alt="image" src="https://github.com/user-attachments/assets/48ce6200-651f-4cbf-9021-561ddaa2fdfb" />



**Summary :**

Calculated Population Stability Index (PSI) by binning prediction probabilities and comparing train vs test distributions. Total PSI = 0.3996, which indicates significant data drift (> 0.25 threshold). This suggests the score distribution has shifted and the model should be monitored closely and considered for recalibration or retraining.


**CSI (Characteristic Stability Index)**

<img width="703" height="643" alt="image" src="https://github.com/user-attachments/assets/0ed200cb-798e-4e57-b66d-bd9093b62cfe" />

**Summary:**

Total Characteristic Stability Index ≈ 0.0032, indicating negligible feature drift between train and test datasets. Feature distributions are highly stable and within safe monitoring limits.


### Business Impact
 * Helped lower potential credit loss by flagging risky applications before approval.
 * Improved credit approval quality using standardized credit scores and simple risk bands (Low/Medium/High).
 * Simulation suggests potential reduction up to 50–60% under early-review strategy
 * Enabled early identification of risky customers using model-based risk scoring.
 * Implemented monitoring and drift checks to detect future behavior changes in model performance.
   
## Future Improvements

* Maintain an audit-ready scoring history and performance tracking setup so credit decisions can be reviewed and explained over time.

* Although features are stable, score distribution drift indicates the need for stronger monitoring, drift alerts, recalibration, and periodic retraining to keep risk decisions accurate as data patterns change.

---

### **Challenges**

* Handling severe **class imbalance**, where bad customers were a small minority — required careful metric selection and resampling strategy testing.
* Avoiding misleading accuracy and shifting evaluation toward **bad-class recall, KS, and Gini**.
* **Feature selection** was challenging due to noisy, correlated, and leakage-prone variables — addressed using WoE/IV and stability checks.
* Balancing recall vs precision when using SMOTE-Tomek — improved detection but increased overfitting in some models.

---
