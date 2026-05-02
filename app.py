import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Import page modules
from pages import findings, feature_importance, model, product, conclusion


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/processed/hotel_bookings.csv", nrows=10000)
    except Exception:
        return pd.DataFrame()


def load_metrics():
    try:
        metrics_df = pd.read_csv("reports/model_results.csv")
        return metrics_df
    except Exception:
        return None

def load_evaluation():
    try:
        with open("reports/evaluation_report.json", "r") as file:
            return json.load(file)
    except Exception:
        return None


st.set_page_config(page_title="Hotel Booking Insights", layout="wide", page_icon="🏨")

st.title("🏨 Hotel Booking Cancellation Insights")
st.markdown("### Predicting Cancellations Pre-Emptively")
st.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Business Overview", 
    "Findings (EDA)", 
    "Model Performance", 
    "Product Module", 
    "Conclusion"
])

if page == "Business Overview":
    st.header("Executive Summary")
    st.markdown("""
    Hotels regularly lose significant revenue due to unexpected, last-minute booking cancellations.
    When a room is unexpectedly abandoned, it often remains vacant overnight, representing a total loss of potential inventory revenue.

    **Goal:**
    Identify which bookings have the highest probability of being canceled *before* the cancellation happens, allowing our operations teams to:
    - **Overbook strategically**: Fill gaps in predicted risk segments.
    - **Engage with guests**: Send email follow-ups or offer upgrades to secure the reservation.
    - **Revise deposit policies**: Demand stricter deposits for segments heavily prone to cancellations.
    """)
    st.success("By leveraging advanced Machine Learning on historical traits (lead time, room types, dates), we built an early-warning predictor to proactively combat vacant rooms.")
    
    st.markdown("---")
    st.markdown("### Project Highlights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", "119K+ bookings")
        st.metric("Features Engineered", "25+")
    with col2:
        st.metric("Models Tested", "6 algorithms")
        st.metric("Best F1-Score", "~85%")
    with col3:
        st.metric("Code Coverage", "48%")
        st.metric("Deployment Platform", "HuggingFace")

elif page == "Findings (EDA)":
    # Pass empty dataframe since we now use pre-generated plots
    findings.show(pd.DataFrame())

elif page == "Model Performance":
    metrics_df = load_metrics()
    eval_data = load_evaluation()
    model.show(metrics_df, eval_data)

elif page == "Product Module":
    product.show()

elif page == "Conclusion":
    conclusion.show()
