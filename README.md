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


![Project Architecture](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/ml_bdb.svg)

-----

### 🔹 **Data Source**

**This project uses real client data under a Non-Disclosure Agreement, so raw dataset details cannot be disclosed.**

The working dataset contains about 100K customers and 99 features, covering demographic, behavioral, credit, and bureau attributes. This master dataset was used for all analysis and modeling.



![Data Architecture](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/bdb_data_source.svg)

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


![Project Architecture](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/ml_bdb.svg)

-----

### 🔹 **Data Source**

**This project uses real client data under a Non-Disclosure Agreement, so raw dataset details cannot be disclosed.**

The working dataset contains about 100K customers and 99 features, covering demographic, behavioral, credit, and bureau attributes. This master dataset was used for all analysis and modeling.



![Data Architecture](https://raw.githubusercontent.com/Sreevarshan-fin/Sreevarshan-fin/main/assets/bdb_data_source.svg)

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


# 📊 Bad Debt Prediction

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/Sreevarshan-fin/Bad_Debt_Prediction?style=for-the-badge&color=yellow)
[![GitHub forks](https://img.shields.io/github/forks/Sreevarshan-fin/Bad_Debt_Prediction?style=for-the-badge)](https://github.com/Sreevarshan-fin/Bad_Debt_Prediction/network)
[![GitHub issues](https://img.shields.io/github/issues/Sreevarshan-fin/Bad_Debt_Prediction?style=for-the-badge)](https://github.com/Sreevarshan-fin/Bad_Debt_Prediction/issues)
[![GitHub license](https://img.shields.io/github/license/Sreevarshan-fin/Bad_Debt_Prediction?style=for-the-badge)](LICENSE)

**An end-to-end credit risk model identifying high-risk borrowers before approval, powered by advanced ML techniques and deployed with Streamlit on AWS.**

[Live Demo](https://demo-link.com) <!-- TODO: Add live demo link if available --> |
[Documentation](https://docs-link.com) <!-- TODO: Add documentation link if available -->

</div>

## 📖 Overview

Bad Debt Prediction is a robust machine learning project designed to revolutionize credit risk assessment. It provides an end-to-end solution for financial institutions to proactively identify potential high-risk borrowers. By leveraging sophisticated feature engineering (Weight of Evidence - WoE and Information Value - IV), advanced data resampling techniques (SMOTE-Tomek), and powerful classification models (Random Forest, XGBoost), the system accurately predicts bad debt likelihood. The project includes a user-friendly Streamlit web application for interactive predictions, model performance tracking with MLflow, and continuous monitoring of model stability using Population Stability Index (PSI) and Characteristic Stability Index (CSI).

This system empowers credit analysts and financial decision-makers with data-driven insights, reducing financial losses and optimizing lending strategies.

## ✨ Features

-   **🎯 Credit Risk Prediction**: Identify high-risk borrowers before loan approval.
-   **⚙️ Advanced Feature Engineering**: Utilizes Weight of Evidence (WoE) and Information Value (IV) for robust feature selection and transformation.
-   **📈 Imbalanced Data Handling**: Employs SMOTE-Tomek resampling to address class imbalance and improve model performance.
-   **🧠 Multiple ML Models**: Implements and evaluates Random Forest and XGBoost classifiers for superior predictive power.
-   **📊 Comprehensive Model Evaluation**: Assesses model performance using key metrics like ROC-AUC, Kolmogorov-Smirnov (KS) statistic, and Gini coefficient.
-   **🖥️ Interactive Web Application**: A user-friendly Streamlit interface for live credit risk predictions.
-   **🧪 MLflow Experiment Tracking**: Tracks model training, parameters, and metrics for improved MLOps and reproducibility.
-   **☁️ AWS Deployment Ready**: Configured for deployment on AWS, ensuring scalability and accessibility.
-   **🔍 Model Monitoring**: Incorporates PSI and CSI for continuous monitoring of model and feature drift, ensuring long-term model reliability.
-   **🐳 Dev Container Support**: Includes `.devcontainer` configuration for a consistent and isolated development environment.

## 🖥️ Screenshots

![Streamlit App Screenshot](docs/streamlit-app.png) <!-- TODO: Add actual screenshot of the Streamlit app -->
*Screenshot of the interactive Streamlit application.*

![Model Evaluation Metrics](docs/model-evaluation.png) <!-- TODO: Add actual screenshot of model evaluation results or MLflow UI -->
*Screenshot depicting key model evaluation metrics or MLflow UI.*

## 🛠️ Tech Stack

**Programming Language:**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Data Science & Machine Learning:**
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-146356?style=for-the-badge&logo=xgboost&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8F8F8F?style=for-the-badge&logo=scipy&logoColor=white)
![Imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-orange?style=for-the-badge)

**Web Application Framework:**
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**MLOps & Deployment:**
![MLflow](https://img.shields.io/badge/MLflow-0099FF?style=for-the-badge&logo=mlflow&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)

**Development Tools:**
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Dev Containers](https://img.shields.io/badge/Dev%20Containers-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)

## 🚀 Quick Start

Follow these steps to set up and run the Bad Debt Prediction project locally.

### Prerequisites

-   **Python 3.8+** (recommended)
-   `pip` (Python package installer)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Sreevarshan-fin/Bad_Debt_Prediction.git
    cd Bad_Debt_Prediction
    ```

2.  **Create a virtual environment** (recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment setup**
    Create a `.env` file in the project root based on the example below. This file will store sensitive information and configuration parameters.
    ```
    # .env example
    MLFLOW_TRACKING_URI=http://localhost:5000  # Or your remote MLflow server URI
    AWS_ACCESS_KEY_ID=your_aws_access_key_id
    AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
    AWS_REGION=your_aws_region
    # Add other necessary environment variables for data paths, model paths etc.
    ```
    **Note**: For local development, you might not need AWS credentials initially, but they are crucial for cloud deployment and remote MLflow tracking.

### Run MLflow Tracking Server (Optional, for full MLOps experience)

If you wish to track experiments using MLflow locally, start the MLflow UI:
```bash
mlflow ui
```
Then navigate to `http://localhost:5000` in your browser. Ensure `MLFLOW_TRACKING_URI` in your `.env` is set accordingly.

### Run the Streamlit Application

After installing dependencies and setting up your environment, you can run the interactive Streamlit application:

```bash
streamlit run app/app.py # Assuming the main app file is app/app.py
```
**Note**: If the Streamlit app's entry point is different, please adjust `app/app.py` to the correct file name (e.g., `app/main.py`).

Once the application starts, it will provide a local URL (e.g., `http://localhost:8501`) which you can open in your web browser.

## 📁 Project Structure

```
Bad_Debt_Prediction/
├── .devcontainer/      # Dev Container configuration for VS Code
│   └── devcontainer.json
├── Notebook/           # Jupyter notebooks for EDA, feature engineering, model training, and evaluation
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training_Evaluation.ipynb
│   └── ...
├── app/                # Streamlit web application
│   └── app.py          # Main Streamlit application script
├── model/              # Directory for storing trained model artifacts
│   ├── bad_debt_predictor.pkl # Example trained model
│   └── preprocessor.pkl     # Example preprocessor
├── .gitattributes      # Git attribute configuration
├── .gitignore          # Specifies intentionally untracked files to ignore
├── requirements.txt    # Python dependencies
└── README.md           # Project README file
```

## ⚙️ Configuration

### Environment Variables
The project uses environment variables for sensitive data and configuration. It is recommended to create a `.env` file in the root directory.

| Variable             | Description                                     | Default          | Required |
|----------------------|-------------------------------------------------|------------------|----------|
| `MLFLOW_TRACKING_URI`| URI for the MLflow tracking server.             | `http://localhost:5000` | No       |
| `AWS_ACCESS_KEY_ID`  | AWS access key ID for cloud services.           | None             | No (for local)|
| `AWS_SECRET_ACCESS_KEY`| AWS secret access key for cloud services.     | None             | No (for local)|
| `AWS_REGION`         | AWS region (e.g., `us-east-1`).                 | None             | No (for local)|
| `MODEL_PATH`         | Path to the trained model artifact.             | `model/bad_debt_predictor.pkl` | No       |
| `PREPROCESSOR_PATH`  | Path to the preprocessor/scaler artifact.       | `model/preprocessor.pkl` | No       |

### Development Container

The `.devcontainer` directory contains configurations to set up a consistent development environment using VS Code Dev Containers. This ensures all team members work with the same dependencies and tools.

## 🔧 Development

### Workflow for Data Scientists / ML Engineers
1.  **Exploratory Data Analysis (EDA):** Use the Jupyter notebooks in `Notebook/` to understand the dataset, identify patterns, and prepare for feature engineering.
2.  **Feature Engineering:** Implement and experiment with various feature engineering techniques, including WoE-IV, within the notebooks.
3.  **Model Training & Evaluation:** Train different machine learning models (Random Forest, XGBoost), tune hyperparameters, and evaluate their performance using metrics like ROC-AUC, KS, and Gini.
4.  **Model Persistence:** Save the best performing models and any necessary preprocessors (e.g., scalers, encoders) to the `model/` directory.
5.  **MLflow Tracking:** Ensure all model training runs are tracked using MLflow to record parameters, metrics, and artifacts.
6.  **Streamlit App Integration:** Update the `app/app.py` script to load the trained model and preprocessor, and integrate the prediction logic into the UI.
7.  **Model Monitoring (PSI/CSI):** Develop and integrate scripts or components to monitor model and feature drift over time, possibly within the `app/` or as separate utilities.

## 🧪 Model Validation & Testing

This project emphasizes rigorous model validation rather than traditional software unit testing.

-   **Model Evaluation Metrics**: The Jupyter notebooks include code to calculate and visualize:
    -   **ROC-AUC Score**: Measures the model's ability to distinguish between classes.
    -   **Kolmogorov-Smirnov (KS) Statistic**: Assesses the separation between positive and negative distributions.
    -   **Gini Coefficient**: Derived from AUC, indicating predictive power.
-   **Cross-Validation**: Models are typically trained and evaluated using k-fold cross-validation to ensure generalization.
-   **Population Stability Index (PSI) & Characteristic Stability Index (CSI)**: These metrics are critical for monitoring the stability of the model and its input features in production. Implementations or conceptual frameworks for calculating these are expected within the project, likely integrated into the Streamlit app or separate monitoring scripts.

## 🚀 Deployment

The project is designed with cloud deployment in mind, specifically on AWS.

### Streamlit App Deployment
The Streamlit application `app/app.py` can be deployed on various platforms:

-   **Streamlit Community Cloud**: Easiest way to deploy Streamlit apps.
-   **AWS EC2/ECS/Fargate**: For more control and integration with AWS services. This involves containerizing the Streamlit app (e.g., using Docker, though a Dockerfile is not explicitly provided, it's a common next step for AWS deployment).

### MLflow Deployment
-   **MLflow Tracking Server**: For production environments, the MLflow tracking server can be deployed on an AWS EC2 instance, managed service, or within a Kubernetes cluster, pointing to an S3 bucket for artifact storage and a relational database for backend store.

## 🤝 Contributing

We welcome contributions to enhance this Bad Debt Prediction project! Please consider the following guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  **Make your changes**, ensuring they adhere to the project's coding style and best practices.
4.  **Commit your changes** with clear and concise messages (`git commit -m 'feat: Add new feature X'`).
5.  **Push your branch** to your forked repository (`git push origin feature/your-feature-name`).
6.  **Open a Pull Request** to the `main` branch of this repository, describing your changes in detail.

### Development Setup for Contributors

For a consistent development experience, we recommend using [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers):

1.  Ensure you have Docker Desktop installed and running.
2.  Open the project in VS Code.
3.  VS Code should prompt you to "Reopen in Container". If not, open the Command Palette (Ctrl+Shift+P) and select "Dev Containers: Reopen in Container".
4.  This will set up a fully configured Python environment with all dependencies pre-installed.

## 📄 License

This project is licensed under the [LICENSE_NAME](LICENSE) - see the LICENSE file for details. <!-- TODO: Add actual license file (e.g., MIT, Apache-2.0) and update this section -->

## 🙏 Acknowledgments

-   Built upon the robust scientific computing ecosystem of **Python** (Pandas, NumPy, Scikit-learn, XGBoost).
-   Utilizes **Streamlit** for rapid application development and interactive dashboards.
-   Integrates **MLflow** for experiment tracking and model lifecycle management.
-   Thanks to the open-source community for invaluable libraries and tools.
-   Authored by [Sreevarshan-fin](https://github.com/Sreevarshan-fin).

## 📞 Support & Contact

-   🐛 Issues: [GitHub Issues](https://github.com/Sreevarshan-fin/Bad_Debt_Prediction/issues)
-   📧 Email: [sreevarshan.fin@example.com] <!-- TODO: Replace with actual contact email if available -->

---

<div align="center">

**⭐ Star this repo if you find this project helpful!**

</div>

