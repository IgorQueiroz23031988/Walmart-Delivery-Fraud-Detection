![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Project-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# 🚚 Walmart Delivery Fraud Risk Analysis

## 📌 Project Overview

This project investigates operational risks related to **missing items in delivery orders**, focusing on identifying patterns that may indicate **fraud or operational failures**.

The analysis combines **Exploratory Data Analysis (EDA)**, **Hypothesis Testing**, and a **Fraud Risk Score Framework** to detect high-risk operational scenarios and support decision-making.

### 🔍 Final Solution Includes:
- Full analytical report  
- Executive presentation  
- Interactive dashboard  
- Fraud risk prediction simulator  

---

## 💼 Business Problem

Delivery operations reported an increasing number of **missing items**, leading to operational complaints and **financial losses**.

However, the dataset **does not contain fraud labels**, preventing traditional supervised fraud detection.

### 🎯 Objective:
Identify patterns of operational risk and potential fraud indicators across:
- Orders  
- Products  
- Drivers  
- Customers  

---

## 🧩 Project Structure

The analysis follows a structured analytics workflow:

1. Business Understanding  
2. Data Understanding  
3. Data Preparation  
4. Exploratory Data Analysis  
5. Hypothesis Testing  
6. Fraud Risk Score Framework  
7. Risk Insights  
8. Business Recommendations  

---

## 📊 Data Sources

The dataset consists of the following tables:

| Table          | Description                          |
|----------------|--------------------------------------|
| Orders         | Delivery transaction records         |
| Missing Items  | Reported missing products            |
| Drivers        | Driver information                   |
| Customers      | Customer characteristics             |
| Products       | Product attributes                   |

---

## 🔬 Methodology

### 📈 Exploratory Data Analysis (EDA)

The EDA investigates operational patterns using:
- Univariate analysis  
- Bivariate analysis  
- Hypothesis testing (H1–H12)

### Key metrics analyzed:
- Missing Items Incidence Rate - Frequency of occurrence - Percentage of order with at least one missing item.
- Average Missing Items - Error intensity (severity) - Average number of items missing per order.
- Weighted Missing Items Rate - Real (Operational) Impact - Missing items relative to the total number of delivered items.
- Average Revenue Loss - Financial intensity (severity) - Average revenue loss per order due to missing items.
- Weighted Revenue Loss Rate - Real (Financial) impact - Revenue loss relative to the total order value.

### Suporting Metrics (Proxy Metrics)
- Total Renevue Loss
- Total Missing Items

### Key dimensions analyzed:
- Delivery period  
- Region  
- Product category  
- Customer profile  
- Driver characteristics  

---

## ⚠️ Fraud Risk Score Framework

Since no fraud labels are available, a **Fraud Risk Score** was developed to identify high-risk operational segments.

### 🔢 The score combines:
- Missing item incidence  
- Financial loss  
- Operational anomalies  

### 🚦 Risk Levels:

| Score Range | Risk Level   |
|------------|-------------|
| 0 – 30     | Low Risk     |
| 30 – 60    | Medium Risk  |
| 60 – 100   | High Risk    |

---

## 🔍 Key Insights

- High-value electronics generate the largest financial losses  
- Supermarket items drive the highest frequency of missing items  
- Drivers with multiple IDs show elevated operational risk  
- Driver experience influences delivery performance
- Higher risk in mornings /early week
- Financial Impact Outweighs Incident Frequency Across Regions and Customers

---

## 💡 Business Recommendations

1. Protect High-Value Orders:
- Enforce delivery confirmation (photo + verification)
- Use tamper-proof packaging

2. Strengthen Driver Identity Control
- Flag multiple ID accounts
- Implement identity verification

3.  Improve High-Volume Operations
- Add picking validation
- User automated order checks

4. Develop Driver Performance Strategy
- Train mid-experience drivers
- Incentivize delivery accuracy

5. Implement Risk Monitoring System
- Track high-risk segments in real time
- Monitor products, drivers, and time windows

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

## 📦 Project Deliverables

### 📄 Technical Report  
Full analytical workflow implemented in Python  

### 📊 Executive Presentation  
Business insights and recommendations for stakeholders  

### 📈 Interactive Dashboard  
Operational risk monitoring (Power BI / Looker Studio)  

### 🤖 Fraud Risk Simulator  
Streamlit application to estimate delivery risk based on input attributes  

---

## 🛠️ Tools Used

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  
- Power BI / Looker Studio  
- Streamlit  

---

## 👤 Author

**Igor Queiroz**

---

# Project Structure

│
├── datasets
│   ├── raw
│  
├── documents
│   ├── Project Walmart.pdf
│
├── fraud_ris_project_simulator_app
│   ├── data
│       ├── df_final_risk_summary
│   ├── app.py
│   └── requirements
│
├── images
│   └── delivery_fraud_dimensions
│
├── interacirve_dashboard
│   ├── datasets
│       ├── processed
│   ├── Walmart-Delivery-Fraud-Detection-Dashboarde.pbix
│  
├── notebooks
│   ├── Walmart_Delivery_Fraud_Detection.ipynb
│
├── presentation
│   └── Walmart - Fraud Risk Analysis in Delivery Operations.pptx
│
├── report
│   └── Walmart_Delivery_Fraud_Detection_report.pdf
│
├── scripts
│   └── Walmart_Delivery_Fraud_Detection.py
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
