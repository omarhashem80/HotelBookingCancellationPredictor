import streamlit as st


def show():
    st.header("Conclusion & Business Recommendations")
    st.markdown(
        "By implementing this Machine Learning pipeline, the hotel franchise transitions from a *reactive* state to a *proactive* revenue engine."
    )

    st.markdown("### Strategic Recommendations:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Revenue Protection")
        st.markdown("""
        - **Overbook with Confidence:** Using the predictions from the '
                    Product Module', the front desk can safely overbook
                    specific room types on dates where high-risk cancellations
                    are clustered.
        - **Dynamic Deposit Enforcement:** Given the overwhelming evidence
                    that 'Non Refundable' deposits stop cancellations, the
                    booking engine should automatically require non-refundable
                    deposits for reservations made >90 days in advance.
        """)

    with col2:
        st.markdown("#### Customer Engagement")
        st.markdown("""
        - **Targeted Marketing Engagement:** Guests flagged as 'High Risk' who
                    have not yet canceled should receive targeted engagements.
                    For example, offering a discounted dinner reservation
                    incentivizes them to lock in their plans.
        - **Resource Allocation:** Housekeeping and culinary teams can adjust
                    their daily prep volumes based on the *predicted* arrivals
                    rather than the *booked* arrivals, saving significant
                    overhead waste.
        """)

    st.markdown("---")
    st.markdown("### Implementation Roadmap")

    st.markdown("""
    **Phase 1: Pilot (Month 1-2)**
    - Deploy model predictions to operations team dashboard
    - A/B test overbooking strategies on low-occupancy weekdays
    - Monitor false positive rates and adjust threshold

    **Phase 2: Automation (Month 3-4)**
    - Integrate prediction API with booking engine
    - Automate deposit requirement triggers
    - Launch targeted email campaigns for high-risk bookings

    **Phase 3: Scale (Month 5+)**
    - Roll out to all properties in the franchise
    - Continuous model retraining with fresh data
    - Expand to predict no-shows and early checkouts
    """)

    st.success(
        "✨ The application is fully containerized and production-ready for "
        "HuggingFace Spaces deployment."
    )

    st.markdown("---")
    st.markdown("### Technical Stack")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Data & ML**")
        st.markdown("- Python 3.10+")
        st.markdown("- Scikit-Learn")
        st.markdown("- XGBoost/CatBoost")
        st.markdown("- Pandas/NumPy")
    with col_b:
        st.markdown("**Deployment**")
        st.markdown("- Streamlit")
        st.markdown("- Poetry")
        st.markdown("- Docker")
        st.markdown("- GitHub Actions")
    with col_c:
        st.markdown("**Quality Assurance**")
        st.markdown("- Pytest (48% coverage)")
        st.markdown("- Black formatter")
        st.markdown("- Flake8 linting")
        st.markdown("- CI/CD pipelines")
