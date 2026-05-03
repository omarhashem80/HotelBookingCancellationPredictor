import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Findings", layout="wide")


def load_image_plot(plot_path, caption=None):
    """Load and display an image plot file with optional caption."""
    try:
        if plot_path.exists():
            img = Image.open(plot_path)
            st.image(img, use_container_width=True, caption=caption)
        else:
            st.warning(
                f"Plot not found: {plot_path.name}. Run `poetry run python scripts/generate_eda_reports.py` first."
            )
    except FileNotFoundError:
        st.warning(
            f"Plot not found: {plot_path.name}. Run `poetry run python scripts/generate_eda_reports.py` first."
        )
    except Exception as e:
        st.error(f"Error loading plot: {e}")


st.header("Exploratory Data Analysis (EDA)")
st.markdown(
    "Comprehensive analysis of hotel booking patterns and cancellation drivers."
)

# Define plot directories
univariate_plots = Path("reports/figures/univariate_plots")
bivariate_plots = Path("reports/figures/bivariate_plots")

# Check if plots exist
if not univariate_plots.exists() or not bivariate_plots.exists():
    st.warning(
        "⚠️ EDA plots not found. Please run: `poetry run python scripts/generate_eda_reports.py`"
    )
    return

# Create tabs for different analysis sections
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Univariate Analysis",
        "🔗 Bivariate Analysis",
        "🌍 Geographic",
        "📅 Time Series",
    ]
)

with tab1:
    st.markdown("### Distribution of Key Variables")
    st.markdown(
        "Understanding individual feature distributions helps identify patterns and outliers."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📅 Lead Time Distribution")
        st.info(
            "💡 **Insight:** The median lead time is 49 days, with 75% of bookings made within 125 days of arrival. Most bookings cluster between 0-150 days. A long tail extends to 700+ days for extreme advance planners."
        )
        load_image_plot(
            univariate_plots / "lead_time_dist_histogram_box_plot.png"
        )

        st.markdown("#### 👥 Adults per Booking")
        st.caption(
            "**Analysis:** The majority of bookings are for 2 adults (typical couples). Single-adult bookings are common for business travel. Very few bookings exceed 3 adults, indicating room capacity constraints."
        )
        load_image_plot(
            univariate_plots / "adults_dist_histogram_box_plot.png"
        )

        st.markdown("#### 💰 Average Daily Rate (ADR)")
        st.caption(
            "**Analysis:** ADR shows a right-skewed distribution with most rates between $50-$150. Outliers above $400 represent luxury suites or special packages. Lower rates may indicate off-season or promotional discounts."
        )
        load_image_plot(univariate_plots / "adr_dist_histogram_box_plot.png")

        st.markdown("#### 🌙 Weekend Nights Stay")
        st.caption(
            "**Analysis:** Most guests stay 0-2 weekend nights. Pure weekend getaways (2 nights) are popular. Extended weekend stays are less common, suggesting most bookings are business-oriented or short vacations."
        )
        load_image_plot(
            univariate_plots
            / "stays_in_weekend_nights_dist_histogram_box_plot.png"
        )

    with col2:
        st.markdown("#### ❌ Cancellation Status")
        st.warning(
            "**Key Metric:** 27.5% of all bookings are canceled, representing significant revenue risk. More than 1 in 4 bookings result in cancellations, highlighting the need for predictive intervention."
        )
        load_image_plot(univariate_plots / "is_canceled_dist_pie_chart.png")

        st.markdown("#### 🔒 Deposit Type Distribution")
        st.info(
            "**Policy Insight:** 98.7% of bookings have no deposit required. Only 1.2% have non-refundable deposits. The overwhelming majority of bookings operate on a free cancellation model."
        )
        load_image_plot(univariate_plots / "deposit_type_dist_pie_chart.png")

        st.markdown("#### 🏷️ Customer Type")
        st.caption(
            "**Segmentation:** Transient customers represent 82% of all bookings. Transient-Party accounts for 13%, while Contract (4%) and Group (0.6%) bookings are minority segments. Each type shows distinct cancellation patterns."
        )
        load_image_plot(univariate_plots / "customer_type_dist_pie_chart.png")

        st.markdown("#### 🍽️ Meal Package")
        st.caption(
            "**Analysis:** Bed & Breakfast (BB) is the most popular meal option. Half-board and full-board packages are less common. Meal preferences may correlate with booking duration and cancellation likelihood."
        )
        load_image_plot(univariate_plots / "meal_dist_pie_chart.png")

with tab2:
    st.markdown("### Cancellation Patterns by Feature")
    st.markdown(
        "Analyzing how different features correlate with cancellation rates."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📅 Lead Time vs. Cancellation")
        st.info(
            "💡 **Critical Finding:** Canceled bookings have a mean lead time of 109 days vs. 67 days for non-canceled bookings. Longer booking windows create more opportunities for plans to change, increasing cancellation risk."
        )
        load_image_plot(bivariate_plots / "lead_time_cancellation_bar.png")

        st.markdown("#### 🔒 Deposit Type Impact")
        st.error(
            "**Critical Finding:** Non-refundable deposits have a 94.7% cancellation rate! This counterintuitive result suggests these deposits may be required for already-risky bookings. No-deposit bookings have 26.7% cancellation, while refundable deposits show 24.5%."
        )
        load_image_plot(bivariate_plots / "deposit_type_cancellation_bar.png")

        st.markdown("#### 🏷️ Customer Type Impact")
        st.caption(
            "**Segmentation Insight:** Transient customers have the highest cancellation rate at 30.1%. Contract customers are most reliable at 16.3%. Group bookings are exceptionally stable at 9.8%, while Transient-Party shows 15.3% cancellations."
        )
        load_image_plot(bivariate_plots / "customer_type_cancellation_bar.png")

        st.markdown("#### 🌙 Weekend vs. Week Nights")
        st.caption(
            "**Pattern:** Bookings with more weekend nights tend to have slightly higher cancellation rates, possibly because leisure travelers have more flexibility than business travelers (weekday-focused)."
        )
        load_image_plot(
            bivariate_plots / "stays_in_weekend_nights_cancellation_bar.png"
        )

    with col2:
        st.markdown("#### 🔁 Previous Cancellations")
        st.warning(
            "🚨 **Behavioral Predictor:** Guests with 0 previous cancellations have 26.7% cancellation rate. Those with 1 previous cancellation jump to 76.2% likelihood of canceling again! Past behavior is a powerful predictor. Flag repeat offenders for stricter policies."
        )
        load_image_plot(
            bivariate_plots / "previous_cancellations_cancellation_bar.png"
        )

        st.markdown("#### 🎯 Market Segment Impact")
        st.caption(
            "**Channel Analysis:** Online travel agents show higher cancellation rates than direct bookings. Corporate segments are more reliable. Groups have moderate cancellation rates despite coordination complexity."
        )
        load_image_plot(
            bivariate_plots / "market_segment_cancellation_bar.png"
        )

        st.markdown("#### 📡 Distribution Channel")
        st.caption(
            "**Sales Channel:** TA/TO (travel agents/tour operators) channels show varying cancellation patterns. Direct bookings tend to be more stable. Corporate channels have the lowest cancellation rates."
        )
        load_image_plot(
            bivariate_plots / "distribution_channel_cancellation_bar.png"
        )

        st.markdown("#### 💰 ADR vs. Cancellation")
        st.caption(
            "**Pricing Insight:** Higher-priced bookings don't necessarily have lower cancellation rates. Mid-range prices show optimal commitment. Very low rates may attract less committed customers."
        )
        load_image_plot(bivariate_plots / "adr_cancellation_bar.png")

    st.markdown("### 🔗 Correlation Matrix")
    st.markdown(
        "**Multivariate Relationships:** This Spearman correlation matrix reveals relationships between numerical features. Stay duration components (weekend/week nights) correlate with total guests. Lead time and booking changes show distinct patterns that help predict cancellations."
    )
    load_image_plot(
        bivariate_plots / "correlation_matrix_for_numerical_features.png"
    )

with tab3:
    st.markdown("### 🌍 Geographic Insights")
    st.markdown(
        "Guest origin analysis and country-specific patterns reveal international travel dynamics."
    )

    st.markdown("#### 🗺️ Guest Home Countries (Heatmap)")
    st.info(
        "**Global Reach:** Portugal (PRT) dominates as the primary source market, followed by UK (GBR), France (FRA), Spain (ESP), and Germany (DEU). This suggests the hotels are likely located in Portugal with strong European appeal."
    )
    load_image_plot(univariate_plots / "Home_Country_Of_Guests.png")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌐 Top Countries Distribution")
        st.caption(
            "**Market Concentration:** The top 5 countries account for the majority of bookings. Other international markets are consolidated into 'Other' category. This concentration suggests focused marketing opportunities."
        )
        load_image_plot(univariate_plots / "country_dist_pie_chart.png")
    with col2:
        st.markdown("#### 🌍 Country vs. Cancellation Rate")
        st.caption(
            "**Geographic Risk:** Different countries show varying cancellation behaviors. Long-distance international travelers may have higher cancellation rates due to logistical complexity. Domestic travelers (PRT) show different patterns than international guests."
        )
        load_image_plot(bivariate_plots / "country_cancellation_bar.png")

with tab4:
    st.markdown("### 📅 Temporal Patterns")
    st.markdown(
        "Booking and cancellation trends over time reveal seasonal patterns and year-over-year growth."
    )

    st.markdown("#### 📈 Yearly Comparison - Booking Trends")
    st.info(
        "**Growth Pattern:** The overlay of different years shows booking volume trends. Summer peaks are consistent across years. Year-over-year comparison helps identify growth patterns and seasonal consistency."
    )
    load_image_plot(univariate_plots / "yearly_comparsion_arrival_date.png")

    st.markdown("#### 📊 Monthly Booking Patterns")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📅 Arrival Month Distribution")
        st.caption(
            "**Seasonality:** Summer months (July-August) show peak arrivals. Spring (April-May) and fall (September-October) are shoulder seasons. Winter months have lowest occupancy, suggesting opportunities for off-season promotions."
        )
        load_image_plot(
            univariate_plots / "arrival_date_month_dist_pie_chart.png"
        )
    with col2:
        st.markdown("##### ❌ Month vs. Cancellation Rate")
        st.caption(
            "**Seasonal Risk:** Cancellation rates vary by month. Winter months may show higher cancellation rates. Summer peak season might have lower cancellations due to high demand and limited alternatives."
        )
        load_image_plot(
            bivariate_plots / "arrival_date_month_cancellation_bar.png"
        )

    st.markdown("#### 📉 Detailed Time Series by Year")
    st.markdown(
        "**Daily Granularity:** These plots show day-by-day booking patterns throughout each year, revealing weekly cycles, holiday impacts, and special events."
    )

    year_cols = st.columns(3)
    with year_cols[0]:
        st.markdown("##### 2015 Daily Arrivals")
        st.caption(
            "Baseline year showing established seasonal patterns and weekly fluctuations."
        )
        load_image_plot(
            univariate_plots / "arrival_date_time_series_plot_at_2015.png"
        )
    with year_cols[1]:
        st.markdown("##### 2016 Daily Arrivals")
        st.caption(
            "Year-over-year growth comparison. Consistent seasonal patterns with possible volume increases."
        )
        load_image_plot(
            univariate_plots / "arrival_date_time_series_plot_at_2016.png"
        )
    with year_cols[2]:
        st.markdown("##### 2017 Daily Arrivals")
        st.caption(
            "Most recent year data. Compare peak heights and trough depths to identify trends."
        )
        load_image_plot(
            univariate_plots / "arrival_date_time_series_plot_at_2017.png"
        )
