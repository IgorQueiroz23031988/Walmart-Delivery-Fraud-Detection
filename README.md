![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Project-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# 🚚 Walmart Delivery Fraud Risk Analysis

## 📌 Project Overview

This project investigates operational risks related to missing items in delivery orders, focusing on identifying patterns that may indicate fraud or operational failures.

The analysis combines **Exploratory Data Analysis (EDA)**, **Hypothesis Testing**, and a **Fraud Risk Score Framework** to detect high-risk operational scenarios and support decision-making.

### 🔍 Final Solution Includes:
- Full analytical report  
- Executive presentation  
- Interactive dashboard  
- Fraud risk prediction simulator  

---

## 💼 Business Problem

Delivery operations reported an increasing number of missing items, leading to operational complaints and financial losses.

However, the dataset does not contain fraud labels, preventing traditional supervised fraud detection.

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

---

## 💡 Business Recommendations

- Enhance verification for high-value electronics  
- Monitor drivers with multiple IDs  
- Improve fulfillment accuracy for high-volume products  
- Implement driver performance monitoring and training  
- Develop a Fraud Risk Monitoring Dashboard  

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
