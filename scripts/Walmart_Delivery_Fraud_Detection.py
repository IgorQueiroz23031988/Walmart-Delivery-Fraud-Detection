# %% [markdown]
# 0.0 Imports
# 
# 

# %%
from IPython.display import Image
import pandas as pd
import numpy as np  
import seaborn as sns
from matplotlib import pyplot as plt


# %% [markdown]
# # 1.0 Business Understanding

# %% [markdown]
# ### 1.1. Motivation/Problem (What is the probem? / Who is reporting the issue?)
# 
# In the US, Walmart has a system similar to Uber, where delivery drivers 
# register to deliver orders placed through the Walmart website. 
# These delivery drivers are not Walmart employees, but work independently, 
# accepting delivery orders and receiving these orders from Walmart staff, 
# then delivering them to the consumer's address. Many consumers have 
# reported that certain items in their orders were not delivered, 
# even though the system marked the delivery as complete. This raises some 
# critical questions:
# 
# 1. Delivery Driver Fraud: There is evidence that drivers may be reporting 
# the delivery of items that did not actually reach the customer. 
# They may be omitting or diverting items from the order, 
# while still recording the delivery as complete.
# 
# 2. System or Process Error: It may be that the problem lies in flaws in
# the recording system or delivery process, not limited to intentional fraud.
# 
# 3. Consumer Fraud: In some cases, consumers may claim they did not 
# receive a product that was delivered in order to request a refund.
# 
# ##### Walmart requested the data analyst to identify the potential causes of delivery-related fraud and propose a solution to prevent it.
# 
# ### 1.2. Poblem Root Cause (Why a solution is requested?)
# 
# Customer satisfaction is decreasing due to missing items in completed deliveries.
# 
# Revenue losses are increasing due to customer refunds related to reported missing items.
# 
# ### 1.3. Solution (How to solve the problem?):
# 
# Proceed with a descriptive analysis to identify risk patterns and anomalous behaviors, mainly concerning customers, drivers, and potential system errors, that may indicate fraudulent delivery activities.
# 
# Develop a fraud risk score framework to estimate the likelihood of risk in future deliveries.
# 
# ### 1.4. Deliverable:
# 
# Presentation file including visualization charts and insights about the main causes of missing items in deliveries.
# 
# Interactive dashboard for delivery monitoring.
# 
# Fraud risk score framework to identify high-risk delivery patterns.
# 
# ### 1.5. Tools:
# 
# Python: Exploratory Data Analysis and risk metric development.
# 
# Power BI: Interactive data visualization.
# 
# PowerPoint: Executive presentation of insights.
# 

# %% [markdown]
# # 2.0 Data Understanding
# 
# The dataset is composed of five distinct tables, each operating at a different level of granularity. Understanding these differences is essential to avoid incorrect aggregations and misleading conclusions during the analysis.
# 
# ### 2.1 Orders Table
# 
# The orders table represents delivery orders placed through Walmart’s e-commerce platform. Each row corresponds to a single order and contains information such as order value, delivery region, delivery time, and the number of items delivered and reported as missing.
# 
# It is important to note that the presence of missing items (items_missing > 0) does not confirm fraud, but rather indicates a reported delivery discrepancy.
# 
# ### 2.2 Missing Items Data
# 
# The missing_items_data table contains information only about items reported by customers as not received. This table does not represent the full composition of an order, but only the subset of products that were declared missing.
# 
# As a result, this table should not be interpreted as a complete list of items per order, but rather as a complaint-level dataset.
# 
# ### 2.3 Drivers Data
# 
# The drivers_data table provides demographic and operational information about delivery drivers, including age and total number of trips performed during the year. Each driver may be associated with multiple delivery orders.
# 
# ### 2.4 Customers Data
# 
# The customers_data table contains demographic information about customers who placed orders. Similar to drivers, each customer may be associated with multiple orders over time.
# 
# ### 2.5 Products Data
# 
# The products_data table includes information about individual products, such as category and price. Product-level analysis is only possible when this table is joined with the missing items data.
# 
# ### 2.6 Absence of Fraud Labels
# 
# The dataset does not contain an explicit target variable indicating confirmed fraud. Consequently, the analysis cannot rely on supervised machine learning techniques and instead focuses on exploratory analysis, pattern detection, and risk assessment.
# 

# %% [markdown]
# # 3.0 Solution Strategy
# 
# The solution strategy is diveded into two parts:
# 
# 1. Exploratory Data Analysis - The objective is not to classify orders as fraudulent, but to prioritize investigations and preventive actions. Given the absence of confirmed fraud labels, the proposed approach focuses on identifying risk patterns and anomalous behaviors that may indicate higher likelihood of delivery-related issues.
# 
# 2. Fraud risk score framework - The exploratory findings will serve as the foundation for the development of a Fraud Risk Score framework. This framework will aggregate relevant risk indicators identified during the analysis and assign a relative risk level to future delivery orders, supporting preventive monitoring and investigation prioritization.

# %% [markdown]
# ### 3.1 Exploratory Data Analysis Methodology
# 
# The Fact-Dimension method is used to develop the data descriptive analysis.

# %% [markdown]
# #### 3.1.1 Main (Open) Question
# 
# How can delivery-related fraud risk be identified and reduced?
# 
# Are there observable patterns in delivery data that indicate higher risk of missing items?

# %% [markdown]
# #### 3.1.2 Closed Questions
# 
# Closed questions are structured, measurable questions that can be answered objectively using the available data. They guide the analytical process and support evidence-based conclusions.
# 
# They are divided into two categories:
# 
# ##### 3.1.2.1 Impact-Level Questions
# 
# These questions quantify the magnitude and financial exposure of delivery discrepancies. They measure overall volume and economic impact, but do not identify behavioral concentration or disproportionate risk patterns by themselves.
# 
# ##### 1. How manny items were not delivered? 
# 
# ##### 2.What is the total revenue lost due to these undelivered items?
# 
# ##### 3.1.2.2 Risk-Oriented Questions
# 
# These questions analyze proportional metrics across relevant dimensions in order to identify concentration effects, deviations from baseline behavior, and potential anomalous patterns that may indicate elevated delivery-related risk.
# 
# ##### 3. Do certain drivers present disproportionately higher missing item rates compared to the overall average?
# 
# ##### 4. Are missing item reports concentrated among a small subset of customers?
# 
# ##### 5. Are specific product categories more frequently reported as missing?
# 
# ##### 6. Do certain regions exhibit higher missing item rates?
# 
# ##### 7. Is there a relationship between delivery time (hour) and missing item occurrences?

# %% [markdown]
# #### 3.1.3 Defining the fact table
# 
# order_id
# 
# items_missing
# 
# order_amount
# 
# items_delivered
# 
# revenue_lost (calculated)

# %% [markdown]
# #### 3.1.4 Defining Dimensions

# %%
Image('C:/Users/Igor/Repos/Walmart-Delivery-Fraud-Detection/images/delivery_fraud_dimensions.png')

# %% [markdown]
# ### 3.2 Fraud Risk Score Framework Methodology
# 
# The Fraud Risk Score framework will be developed after identifying statistically and operationally relevant risk indicators during the exploratory analysis.
# 
# The framework will:
# 
# Select key risk variables derived from the EDA
# 
# Normalize and standardize relevant metrics
# 
# Combine weighted indicators into a composite risk score
# 
# Assign relative risk levels (e.g., low, medium, high) to delivery orders
# 
# The score will not represent confirmed fraud probability, but rather a relative risk assessment tool to support preventive decision-making.

# %% [markdown]
# 4.0 Exploratory Data Analysis

# %% [markdown]
# perguntar se eh necessario incluir informacao sbre a Fraud risk score framework no solution strategy

# %% [markdown]
# ### 0.1 Helper Functions

# %%
def missing_values_summary(df, dataset_name=None):
    
    print('='*70)
    if dataset_name:
        print(f'Dataset - {dataset_name}')
    print('='*70)
    
    # =============================
    # Missing Values
    # =============================
    
    missing = pd.DataFrame({
        'attribute': df.columns,
        'missing_values': df.isna().sum().values,
        'missing_%': (df.isna().sum().values / len(df)) * 100})
    
    print('\n Missing Values Summary:\n')

    return missing.sort_values('missing_%', ascending=False)

#--/--

def check_unusual_values(df, 
                         numeric_cols=None, 
                         categorical_cols=None, 
                         id_cols=None,
                         sample=11):
    
    print('='*70)
    print('Checking Unusual / Incoherent Values')
    print('='*70)
    
    
    # ========================
    # Numerical Checks
    # ========================
    
    if numeric_cols:
        print('\n Numerical Columns:\n')
        
        for col in numeric_cols:
            
            values = df[col].dropna().unique()
            sorted_values = np.sort(values)
            
            # Convert numpy types to Python native types
            sorted_values = [v.item() if hasattr(v, 'item') else v 
                             for v in sorted_values]
            
            if len(sorted_values) > sample * 2:
                preview = (
                    sorted_values[:sample] +
                    ['...'] +
                    sorted_values[-sample:]
                )
            else:
                preview = sorted_values
            
            negative_count = (df[col] < 0).sum()
            
            print(f'Column: {col}')
            print(f'Negative / Non-Numeroic values count: {negative_count}')
            print(f'Sample values: {preview}\n')
    
    
    # ========================
    # ID duplicate checks
    # ========================
    
    if id_cols:
        print('\n Duplicate Checks:\n')
        
        for col in id_cols:
            dup_count = df[col].duplicated().sum()
            print(f'Column: {col} → Duplicated values: {dup_count}')
    
    
    # ========================
    # Categorical Checks
    # ========================
    
    if categorical_cols:
        print('\n Categorical Columns:\n')
        
        for col in categorical_cols:
            
            unique_vals = df[col].dropna().sort_values().unique()
            
            # Convert numpy/object types
            unique_vals = [v.item() if hasattr(v, 'item') else v 
                           for v in unique_vals]
            
            if len(unique_vals) > sample:
                preview = unique_vals[:sample] + ['...']
            else:
                preview = unique_vals
            
            print(f'Column: {col}')
            print(f'Unique values sample: {preview}\n')

#--/--

def variables_summary(df, dataset_name=None):
    
    print('='*70)
    if dataset_name:
        print(f'Dataset - {dataset_name}')
    print('='*70)
    
    # =============================
    # Numerical Variables
    # =============================
    
    num = df.select_dtypes(include=['int64', 'float64'])
    
    if not num.empty:
        
        # IQR for outliers
        Q1 = num.quantile(0.25)
        Q3 = num.quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = ((num < (Q1 - 1.5 * IQR)) | 
                    (num > (Q3 + 1.5 * IQR))).sum()
        
        num_summary = num.agg(['min','max','mean','median','std','skew','kurt']).T
        num_summary['range'] = num_summary['max'] - num_summary['min']
        num_summary['outliers_count'] = outliers
        num_summary = num_summary.reset_index().rename(columns={'index':'attribute'})

        # Reorder columns
        num_summary = num_summary[
            ['attribute', 'min', 'max', 'range',
            'mean', 'median', 'std', 'skew', 'kurt',
            'outliers_count']]
        
        print('\n Numerical Summary:\n')
        display(num_summary)

    # =============================
        #  HISTOGRAMS (all together)
        # =============================
        
        cols = num.columns
        n = len(cols)
        
        n_cols = 3
        n_rows = int(np.ceil(n / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = np.array(axes).reshape(-1)
        
        for i, col in enumerate(cols):
            sns.histplot(num[col], kde=True, ax=axes[i])
            axes[i].set_title(f'{col} - Histogram')
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
        
        
        # =============================
        #  BOXPLOTS (all together)
        # =============================
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = np.array(axes).reshape(-1)
        
        for i, col in enumerate(cols):
            sns.boxplot(x=num[col], ax=axes[i])
            axes[i].set_title(f'{col} - Boxplot')
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    
    # =============================
    # Categorical Variables
    # =============================
    
    cat = df.select_dtypes(exclude=['int64','float64','datetime64[ns]'])
    
    if not cat.empty:
        
        cat_summary = pd.DataFrame({
            'attribute': cat.columns,
            'unique_values': cat.nunique().values,
            'most_frequent': cat.mode().iloc[0].values,
            'frequency': [
                cat[col].value_counts().iloc[0]
                for col in cat.columns
            ]
        })
        
        print('\n Categorical Summary:\n')
        display(cat_summary)
    
    return

#--/--

def export_eda_to_excel(file_name, missing=None, num_summary=None, cat_summary=None):
    
    with pd.ExcelWriter(file_name) as writer:
        
        if missing is not None:
            missing.to_excel(writer, sheet_name='Missing', index=False)
            
        if num_summary is not None:
            num_summary.to_excel(writer, sheet_name='Numerical', index=False)
            
        if cat_summary is not None:
            cat_summary.to_excel(writer, sheet_name='Categorical', index=False)
    
    print(f'\n Report exported to {file_name}')

# %% [markdown]
# # 4.0 Loading Data

# %%
df_orders_raw = pd.read_csv ('C:/Users/Igor/Repos/Walmart-Delivery-Fraud-Detection/datasets/orders.csv', low_memory=False)
df_missing_items_raw = pd.read_csv ('C:/Users/Igor/Repos/Walmart-Delivery-Fraud-Detection/datasets/missing_items_data.csv', low_memory=False)
df_products_raw = pd.read_csv ('C:/Users/Igor/Repos/Walmart-Delivery-Fraud-Detection/datasets/products_data.csv', low_memory=False)
df_customers_raw = pd.read_csv ('C:/Users/Igor/Repos/Walmart-Delivery-Fraud-Detection/datasets/customers_data.csv', low_memory=False)
df_drivers_raw = pd.read_csv ('C:/Users/Igor/Repos/Walmart-Delivery-Fraud-Detection/datasets/drivers_data.csv', low_memory=False)


# %%
df_orders_raw.sample()

# %%
df_missing_items_raw.sample()

# %%
df_products_raw.sample()

# %%
df_customers_raw.sample()

# %%
df_drivers_raw.sample()

# %% [markdown]
# # 4.0 Data Description

# %%
df_orders_1 = df_orders_raw.copy()
df_missing_items_1 = df_missing_items_raw.copy()
df_products_1 = df_products_raw.copy()
df_customers_1 = df_customers_raw.copy()
df_drivers_1 = df_drivers_raw.copy()

# %% [markdown]
# ## 4.1 Columns Rename

# %%
df_orders_1.columns

# %%
df_missing_items_1.columns

# %%
df_products_1.columns

# %%
df_customers_1.columns

# %%
df_drivers_1.columns

# %%
# The columns 'produc_id', 'category', 'price', 'age' and 'Trips' from datasets df_products_1 and df_drivers_1 need to be renamed.

df_products_1 = df_products_1.rename(columns={'produc_id': 'product_id', 'category': 'product_category', 'price': 'product_price'})
df_drivers_1 = df_drivers_1.rename(columns={'age': 'driver_age', 'Trips': 'driver_trips'})

# %%
df_products_1.columns

# %%
df_drivers_1.columns

# %% [markdown]
# ## 4.2 Data Dimensions

# %%
# Orders DataFrame

print( 'Number of Rows: {}'.format( df_orders_1.shape[0] ))
print( 'Number of Columns: {}'.format( df_orders_1.shape[1] ))

# %%
# Missing Items DataFrame

print( 'Number of Rows: {}'.format( df_missing_items_1.shape[0] ))
print( 'Number of Columns: {}'.format( df_missing_items_1.shape[1] ))

# %%
# Products DataFrame

print( 'Number of Rows: {}'.format( df_products_1.shape[0] ))
print( 'Number of Columns: {}'.format( df_products_1.shape[1] ))

# %%
# Customers DataFrame

print( 'Number of Rows: {}'.format( df_customers_1.shape[0] ))
print( 'Number of Columns: {}'.format( df_customers_1.shape[1] ))

# %%
# Drivers DataFrame

print( 'Number of Rows: {}'.format( df_drivers_1.shape[0] ))
print( 'Number of Columns: {}'.format( df_drivers_1.shape[1] ))

# %% [markdown]
# ## 4.3 Data Types

# %%
 # Orders DataFrame

# Before to update the column order_amount from object to float, it is necessary to remove the dollar sign $.
df_orders_1['order_amount'] = (df_orders_1['order_amount'].str.replace('$', '', regex=False).str.replace(',', '', regex=False))

# Updating the colums type.
df_orders_1['date'] = pd.to_datetime( df_orders_1['date'].str.replace( '-' , '/' ), dayfirst=True, errors = 'coerce' )
df_orders_1['delivery_hour'] = pd.to_datetime( df_orders_1['delivery_hour'],format='%H:%M:%S', errors = 'coerce' )
df_orders_1['order_amount'] = df_orders_1['order_amount'].astype('float64')
df_orders_1['items_delivered'] = df_orders_1['items_delivered'].astype('int64')
df_orders_1['items_missing'] = df_orders_1['items_missing'].astype('int64')

# %%
df_orders_1.dtypes

# %%
# Missing Items DataFrame

df_missing_items_1.dtypes

# Not necessary to update the columns types.

# %%
# Products DataFrame

# Before to update the column product_price from object to float, it is necessary to remove the dollar sign $.
df_products_1['product_price'] = (df_products_1['product_price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False))

# Updating the colums type.
df_products_1['product_price'] = df_products_1['product_price'].astype('float64')

# %%
df_products_1.dtypes

# %%
# Customers DataFrame

df_customers_1.dtypes

# Not necessary to update the columns types.

# %%
# Drivers DataFrame

df_drivers_1.dtypes

# Not necessary to update the columns types.

# %% [markdown]
# ### 4.4 Check NA

# %%
# Orders Dataset
 
missing_values_summary(df_orders_1, 'Orders')

# %% [markdown]
# No missing values.

# %%
# Missing Items Dataset

missing_values_summary(df_missing_items_1, 'Missing Items')

# %% [markdown]
# Missing values are expected because this dataframe lists the quantity of products not delivered, which variate between 1 to 3.
# 
# The columns product_id_1 2 and 3 lists the product id that was no delivered, if column product_id_2 and product_id_3 are set as NA, means that only one product is missing.
# 
# If only product_id_3 is set as NA, means that 2 products are missing.
# 
# If all products Ids columns have the Ids listed, means that 3 products are missing.

# %%
# Products Dataset

missing_values_summary(df_products_1, 'Products')

# %% [markdown]
#  No Missing values

# %%
# Customers DataFrame

missing_values_summary(df_customers_1, 'Customers')

# %% [markdown]
#  No Missing values

# %%
# Drivers Dataset

missing_values_summary(df_drivers_1, 'Drivers')

# %% [markdown]
#  No Missing values

# %% [markdown]
# ### 4.5 Check Unusual/Incoherent Values

# %%
# Orders Dataset

check_unusual_values( df_orders_1, numeric_cols=['order_amount', 'items_delivered', 'items_missing'], categorical_cols=['region'], id_cols=['order_id'])

# %%
# Missing Items Dataset

check_unusual_values( df_missing_items_1, id_cols=['order_id', 'product_id_1', 'product_id_2', 'product_id_3'])

# %% [markdown]
# Duplicates on columns 'product_id_1', 'product_id_2', 'product_id_3' are expected.

# %%
# Products Dataset

check_unusual_values( df_products_1, numeric_cols=['product_price'], categorical_cols=['product_category'], id_cols=['product_id', 'product_name'])

# %%
# Customers Dataset

check_unusual_values( df_customers_1, numeric_cols=['customer_age'], id_cols=['customer_id', 'customer_name'])

# %%
# Customers with duplicated names and ages.

# df_customers_1[df_customers_1.duplicated(subset=['customer_name', 'customer_age'], keep=False)].sort_values(['customer_name', 'customer_age'])
# df_customers_1['customer_name'].value_counts()[lambda x: x > 1]

df_customers_1.groupby(['customer_name', 'customer_age']).size().reset_index(name='count')
df_customers_1.groupby(['customer_name', 'customer_age']).size() \
    .reset_index(name='count') \
    .query('count > 1')

# %% [markdown]
# Despite there are duplicated customer names, they are not duplicated customer_id or customer_age, which means that they are different customers with the same name. Therefore, there is no inconsistency in the dataset.

# %%
# Drivers Dataset

check_unusual_values( df_drivers_1, numeric_cols=['driver_age', 'driver_trips'], id_cols=['driver_id', 'driver_name'])

# %%
# Drivers with duplicated names and ages.

# df_drivers_1[df_drivers_1.duplicated(subset=['driver_name', 'driver_age'], keep=False)].sort_values(['driver_name', 'driver_age'])

df_drivers_1.groupby(['driver_name', 'driver_age']).size().reset_index(name='count')
df_drivers_1.groupby(['driver_name', 'driver_age']).size() \
    .reset_index(name='count') \
    .query('count > 1')

# %% [markdown]
# These duplicated driver names and ages could be the same person, but with different driver_id, which could be a indication of fraudulent activity, 
# such as a driver using multiple identities to steal deliveries and earn more money.
# 
# Necessary to investigate further these duplicated driver names and ages, by checking if they have missing items on the orders they were responsible for.

# %% [markdown]
# # 5.0 Descriptive Statistical

# %% [markdown]
# ## 5.1 Orders Dataset

# %%
df_orders_2 = df_orders_1.copy()
df_missing_items_2 = df_missing_items_1.copy()
df_products_2 = df_products_1.copy()
df_customers_2 = df_customers_1.copy()
df_drivers_2 = df_drivers_1.copy()

# %%
# Central Tendecy
# Definition: resume os dados em um unico numero.
# Metrics: mean , median

# Dispersion
# Definition: Dizem se os dados estao muito concentrados perto da media/mediana ou se os dados estao muito dispersos da media/mediana
# Metrics: std, min, max, range, skew (como eh a deformacao dos dados em relacao a distribuicao normal (forma de sino). Para e esquerda a skem eh positiva, para direite negativa), 
# kurtosis (fala sobre a concentracao dos dados. Quanto maior a kurtose positiva, mais dados concentrados com pico muito alto. Quanto menor a kurtose negativa,
# tem-se pico menor e valores mais dispersos).

# %%
variables_summary(df_orders_2, dataset_name='Orders')

# %% [markdown]
# ### Numerical Attributes:
# 
# #### Order_amount
# 
# Mean and Median: Values are very close, indicating a centered distribution. The distribution is right-skewed (positive skew), meaning a few high-value orders stretch the tail toward larger amounts.
# 
# Skew: Positive, the tail extends toward higher values, while most orders are concentrated between $0 and $500.
# 
# Kurtosis: High, leptokurtic distribution, with a sharp central peak and data tightly concentrated around the mean.
# 
# #### Items_delivered
# 
# Mean and Median: Similar values, indicating a uniform distribution. The frequencies of different quantities delivered are approximately equal.
# 
# Skew: Close to 0, symmetric distribution.
# 
# Kurtosis: Negative, platykurtic distribution, with more dispersed values and no strong concentration around the mean.
# 
# #### Items_missing
# 
# Mean and Median: Multimodal distribution, with approximately four decreasing peaks as the number of missing items increases.
# 
# Skew: Highly positive, most data are near 0, with few higher values.
# 
# Kurtosis: High, leptokurtic distribution, concentrating many observations at 0.
# 
# Outliers: Values 1, 2, and 3 could be considered statistical outliers, but they will not be treated as such since this attribute is the fact table used for identifying fraud risk.

# %% [markdown]
# ## 5.2 Missing Items Dataset

# %%
variables_summary(df_missing_items_2, dataset_name='Missing Items')

# %% [markdown]
# ## 5.3 Products Dataset

# %%
# spliting numerical columns from categorical columns

variables_summary(df_products_2, dataset_name='Products')

# %% [markdown]
# ### Numerical Attributes:
# 
# #### Product_price
# 
# Mean and Median: The mean and median are very close, but the distribution is slightly right-skewed, meaning most values are concentrated toward the lower end.
# 
# Skew: Positive, indicating a tail extending toward higher prices. Most values are clustered between $0 and $200.
# 
# Kurtosis: High, suggesting a leptokurtic distribution with a sharp peak, meaning the data are tightly concentrated. According to the histogram and skew, most prices are between $0 and $200.

# %% [markdown]
# ## 5.4 Customers Dataset

# %%
variables_summary(df_customers_2, dataset_name='Customers')

# %% [markdown]
# ### Numerical Attributes:
# 
# #### Customer_age
# 
# Mean and Median: The mean and median are very close, and the frequency of values is fairly uniform. The histogram bars have approximately equal heights, indicating a uniform distribution.
# 
# Skew: Close to zero, showing that the distribution is symmetric, with data roughly equally distributed on both sides of the mean.
# 
# Kurtosis: Negative, indicating a platykurtic distribution, meaning the data are more dispersed and not concentrated around a single value.

# %% [markdown]
# ## 5.5 Drivers Dataset

# %%
variables_summary(df_drivers_2, dataset_name='Drivers')

# %% [markdown]
# ### Numerical Attributes:
# 
# #### Driver_age
# 
# Mean and Median: Values are fairly uniformly distributed, with histogram bars of similar height, although there is a concentration of data between 20 and 23 years.
# 
# Skew: Close to zero, showing a symmetric distribution, despite the accumulation of drivers aged 20–23.
# 
# Kurtosis: Negative, suggesting a platykurtic distribution, with values well dispersed and no extreme concentration, although the peak around 20–23 years is noticeable.
# 
# #### Driver_trips
# 
# Mean and Median: Values are fairly uniform, with histogram bars of roughly equal height, indicating a uniform distribution.
# 
# Skew: Close to zero, showing a symmetric distribution, with roughly equal values on both sides of the mean.
# 
# Kurtosis: Negative, indicating a platykurtic distribution, with data spread evenly and no strong concentration around a specific value.
# 
# 

# %% [markdown]
# # 6.0 Data Featuring

# %%
print(df_products_1['product_category'].value_counts(), end='\n\n')

# %% [markdown]
# Aside the Eletronics, all other categories are sub-categories from the macro category Supermaket. 
# Therefore, a category called 'macro_category' will be createed to separate eletronics from groceries.

# %%
# Creating macro category column

df_products_1['macro_category'] = df_products_1['product_category'].apply(
    lambda x: 'Electronics' if x == 'Electronics' else 'Supermarket' )

# Moving the column 'macro_category' to the penultimate position.

#  Remove the column from the end
col = df_products_1.pop('macro_category')

# 2. Insert the column in the desired position (penultimate).
df_products_1.insert(len(df_products_1.columns)-1, 'macro_category', col)

# %% [markdown]
# In order to know which sub-categories are significant for the analysis, it is necessary to verify the missing items per sub-category.

# %%
# First, it is necessary to merge the orders dataframe with the missing items dataframe, because the missing items dataframe has the product_id of the products that were not delivered,
#  and this information is necessary to merge with the products dataframe to get the product category and price information.

# Merging df_orders_2 with df_missing_items_2 to get the product_id of the products that were not delivered.
df = pd.merge(df_orders_2, df_missing_items_2, on='order_id', how='left')

# Second, it is necessary to merge the resulting dataframe with the products dataframe to get the product category and price information.

# 🔹 Merging all columns from df_products_1
for i in [1, 2, 3]:
    df = df.merge(
        df_products_1.add_suffix(f'_{i}'),
        left_on=f'product_id_{i}',
        right_on=f'product_id_{i}',
        how='left'
    )

# 🔹 Filling NANs.
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna('None')

# Reorganizing colmuns.

def move_columns_after(df, move_map):
    """
    move_map: dicionário {coluna_a_mover: coluna_destino}
    Move cada coluna_a_mover para ficar logo após coluna_destino
    """
    cols = list(df.columns)
    
    for col_to_move, col_after in move_map.items():
        if col_to_move in cols and col_after in cols:
            cols.remove(col_to_move)
            idx = cols.index(col_after) + 1
            cols.insert(idx, col_to_move)
    
    return df[cols]

# Defining which columns to move and their destination columns.
move_map = {
    'product_id_2': 'macro_category_1',
    'product_id_3': 'macro_category_2'
}

df = move_columns_after(df, move_map)

# %%
df.columns

# %%
# Now it is possible to vefify the amount of missing products per sub-category, 

# df_products_1[df_products_1['product_name'].str.contains('yogurt', case=False)]
print(df.loc[df['items_missing'] != 0, 'product_category_1'].value_counts(), end='\n\n')
print(df.loc[df['items_missing'] != 0, 'product_category_2'].value_counts(), end='\n\n')
print(df.loc[df['items_missing'] != 0, 'product_category_3'].value_counts(), end='\n\n')

# %% [markdown]
# All the products that were not delivered are from the Supermarket category or Electronics category, therefore, it is not relelevant to run an analysis by sub-category level, to verify if there is any sub-category that has a higher amount of missing products.
# 
# Additonally, there is no need to verify if the products are correctly categorized on the sub-category level.


