import streamlit as st


def show():
    st.header("Live Cancellation Predictor (Product Simulation)")
    st.markdown("Use this interactive tool to simulate a new booking and predict whether the guest will cancel their stay.")
    
    st.markdown("### Enter Booking Details:")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            lead_time = st.number_input("Lead Time (Days)", min_value=0, max_value=800, value=30)
            adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
        with col2:
            deposit_type = st.selectbox("Deposit Type", options=["No Deposit", "Non Refund", "Refundable"])
            customer_type = st.selectbox("Customer Type", options=["Transient", "Contract", "Transient-Party", "Group"])
            special_requests = st.number_input("Total Special Requests", min_value=0, max_value=5, value=0)
        with col3:
            prev_cancel = st.number_input("Previous Cancellations", min_value=0, max_value=30, value=0)
            parking_spaces = st.number_input("Required Parking Spaces", min_value=0, max_value=5, value=0)
            adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=1000.0, value=100.0)
            
        submit = st.form_submit_button("🔮 Predict Cancellation Risk")
        
    if submit:
        # Mock prediction logic to demonstrate the product frontend
        # In a real deployed app, you would load the .pkl file and do model.predict_proba()
        st.markdown("---")
        st.subheader("Prediction Results:")
        
        # Super simple rule-based mock for the sake of the UX demo
        risk_score = 0.20
        if lead_time > 60: 
            risk_score += 0.30
        if deposit_type == "Non Refund": 
            risk_score -= 0.45
        if prev_cancel > 0: 
            risk_score += 0.40
        if special_requests > 0: 
            risk_score -= 0.15
        
        risk_score = max(0.01, min(0.99, risk_score))  # clamp between 1% and 99%
        
        # Display prediction
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Cancellation Probability", f"{risk_score*100:.1f}%")
        
        with col_b:
            if risk_score > 0.50:
                st.error(f"🚨 **HIGH RISK** | This booking has elevated cancellation probability.")
                st.markdown("**Suggested Action:** Trigger an automated follow-up email 7 days before arrival, or consider overbooking this room class.")
            else:
                st.success(f"✅ **LOW RISK** | High confidence the guest will arrive successfully.")
                st.markdown("**Suggested Action:** Proceed as normal. Great candidate for upselling spa packages or room upgrades upon arrival.")
        
        st.markdown("---")
        st.markdown("### Input Summary")
        summary_data = {
            "Attribute": ["Lead Time", "Adults", "Children", "Deposit Type", "Previous Cancellations", "Special Requests", "ADR"],
            "Value": [f"{lead_time} days", adults, children, deposit_type, prev_cancel, special_requests, f"${adr:.2f}"]
        }
        st.table(summary_data)
