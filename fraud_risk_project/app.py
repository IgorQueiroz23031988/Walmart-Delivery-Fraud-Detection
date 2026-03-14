import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Delivery Fraud Risk Simulator",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
fraud_risk_project
df_risk = pd.read_csv("../data/df_final_risk_summary.csv")

# quartile thresholds (same logic as fraud framework)
q1 = df_risk["risk_score_0_100"].quantile(0.25)
q2 = df_risk["risk_score_0_100"].quantile(0.50)
q3 = df_risk["risk_score_0_100"].quantile(0.75)

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------

def risk_level(score):

    if score <= q1:
        return "Low Risk"

    elif score <= q2:
        return "Moderate Risk"

    elif score <= q3:
        return "High Risk"

    else:
        return "Critical Risk"


def get_segment_score(attribute, segment):

    row = df_risk[
        (df_risk["attribute"] == attribute) &
        (df_risk["segment"] == segment)
    ]

    if len(row) > 0:
        return row["risk_score_0_100"].values[0]

    return None


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🚚 Delivery Fraud Risk Simulator")

st.write(
"""
This simulator estimates delivery fraud risk using the **Fraud Risk Framework**.
Select operational, customer, driver, and product characteristics to evaluate the risk level of a delivery.
"""
)

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    period = st.selectbox(
        "Delivery Period",
        df_risk[df_risk["attribute"]=="period"]["segment"].unique()
    )

    month = st.selectbox(
        "Month",
        df_risk[df_risk["attribute"]=="month"]["segment"].unique()
    )

    day_of_week = st.selectbox(
        "Day of Week",
        df_risk[df_risk["attribute"]=="day_of_week"]["segment"].unique()
    )

with col2:

    region = st.selectbox(
        "Region",
        df_risk[df_risk["attribute"]=="region"]["segment"].unique()
    )

    customer_age = st.selectbox(
        "Customer Age Group",
        df_risk[df_risk["attribute"]=="customer_age_group"]["segment"].unique()
    )

    driver_id = st.selectbox(
        "Driver ID Type",
        df_risk[df_risk["attribute"]=="driver_id_type"]["segment"].unique()
    )

with col3:

    driver_age = st.selectbox(
        "Driver Age Group",
        df_risk[df_risk["attribute"]=="driver_age_group"]["segment"].unique()
    )

    trip_bin = st.selectbox(
        "Driver Trip Bin",
        df_risk[df_risk["attribute"]=="trip_bin"]["segment"].unique()
    )

    price_bin = st.selectbox(
        "Product Price Bin",
        df_risk[df_risk["attribute"]=="price_bin"]["segment"].unique()
    )

macro_category = st.selectbox(
    "Product Category",
    df_risk[df_risk["attribute"]=="macro_category"]["segment"].unique()
)

# ---------------------------------------------------
# CALCULATE RISK
# ---------------------------------------------------

if st.button("Calculate Fraud Risk"):

    attributes = {

        "period":period,
        "month":month,
        "day_of_week":day_of_week,
        "region":region,

        "customer_age_group":customer_age,

        "driver_id_type":driver_id,
        "driver_age_group":driver_age,
        "trip_bin":trip_bin,

        "price_bin":price_bin,
        "macro_category":macro_category
    }

    results = []

    for attr, seg in attributes.items():

        score = get_segment_score(attr, seg)

        if score is not None:

            results.append({
                "attribute":attr,
                "segment":seg,
                "risk_score":score
            })

    df_results = pd.DataFrame(results)

    final_score = df_results["risk_score"].mean()

    level = risk_level(final_score)

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------

    st.subheader("Fraud Risk Score")

    col1, col2 = st.columns(2)

    with col1:

        st.metric("Risk Score", round(final_score,1))

        if level == "Critical Risk":
            st.error(level)

        elif level == "High Risk":
            st.warning(level)

        elif level == "Moderate Risk":
            st.info(level)

        else:
            st.success(level)

# ---------------------------------------------------
# GAUGE CHART
# ---------------------------------------------------

    with col2:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_score,
            title={'text':"Fraud Risk Score"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"darkred"},
                'steps':[
                    {'range':[0,25],'color':"green"},
                    {'range':[25,50],'color':"yellow"},
                    {'range':[50,75],'color':"orange"},
                    {'range':[75,100],'color':"red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# RISK BREAKDOWN
# ---------------------------------------------------

    st.subheader("Risk Contribution by Factor")

    st.dataframe(df_results)

# ---------------------------------------------------
# TOP RISK DRIVERS
# ---------------------------------------------------

    st.subheader("Top Risk Drivers")

    top_risk = df_results.sort_values(
        "risk_score",
        ascending=False
    ).head(5)

    st.dataframe(top_risk)