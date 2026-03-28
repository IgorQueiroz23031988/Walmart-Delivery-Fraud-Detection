![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Project-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


# Walmart Delivery Fraud Detection

This project investigates **missing item incidents in Walmart delivery orders** to distinguish between operational errors and potential fraud patterns.

The analysis combines exploratory data analysis, hypothesis testing, and a custom fraud risk scoring framework to identify high-risk delivery scenarios.

An interactive **Fraud Risk Simulator** was developed using Streamlit to operationalize the model.

---

# Project Overview

Missing items in delivery orders generate financial losses and customer dissatisfaction.  
However, identifying whether these incidents are caused by operational issues or fraudulent behavior can be challenging.

This project aims to:

- Analyze patterns in missing items
- Identify operational and behavioral risk factors
- Build a fraud risk scoring framework
- Develop an interactive risk simulator

---

# Dataset

The dataset contains information about:

- Orders
- Drivers
- Customers
- Products
- Missing items

Key features include:

Operational variables
- delivery period
- month
- day of week
- region

Driver variables
- driver ID patterns
- driver age group
- trip volume

Customer variables
- customer age group

Product variables
- macro category
- price bin

---

# Methodology

The analysis followed these steps:

1. Exploratory Data Analysis (EDA)
2. Hypothesis testing across operational, driver, customer, and product dimensions
3. Development of a **Fraud Risk Framework**
4. Creation of a **risk scoring model**
5. Development of an **interactive fraud risk simulator**

---

# Fraud Risk Framework

Risk scores are calculated using normalized metrics including:

- Missing item incidence rate
- Weighted missing item rate
- Average revenue loss
- Weighted revenue loss
- Order volume

Risk levels are classified using quartiles:

Low Risk  
Moderate Risk  
High Risk  
Critical Risk

---

# Key Insights

Major findings include:

- Electronics present the **highest financial loss risk**
- Supermarket products account for **most missing item incidents**
- Drivers with **multiple IDs show elevated fraud risk**
- Certain regions and time periods concentrate higher incident rates
- High-value products significantly increase revenue loss impact

---

# Fraud Risk Simulator

An interactive Streamlit application was developed to simulate delivery fraud risk.

Users can select delivery characteristics and estimate risk levels.

The simulator includes:

- Risk score calculation
- Fraud risk classification
- Risk gauge visualization
- Risk contribution breakdown
- Top risk drivers

App link:

https://delivery-fraud-risk-score-simulator-walmart.streamlit.app/

---

# Tech Stack

Python  
Pandas  
NumPy  
Streamlit  
Plotly  

---

# Project Structure

│
├── data
│   ├── raw
│   └── processed
│
├── notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_risk_framework.ipynb
│
├── dashboard
│   ├── powerbi_file.pbix
│
├── streamlit_app
│   ├── app.py
│   └── model.pkl
│
├── presentation
│   └── fraud_risk_presentation.pptx
│
├── report
│   └── fraud_risk_report.pdf
│
├── images
│   └── charts_for_readme
│
├── requirements.txt
│
└── README.md


---

# Future Improvements

Possible extensions include:

- Machine learning fraud classification models
- real-time monitoring dashboards
- anomaly detection for driver behavior
- integration with operational risk monitoring systems
