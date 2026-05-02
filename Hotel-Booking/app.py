import gradio as gr
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# Load the trained model using joblib for better compatibility
model = joblib.load("best_model.pkl")

def predict_cancellation(
    hotel, lead_time, arrival_year, arrival_month, arrival_day,
    weekend_nights, week_nights, adults, children, babies,
    meal, country, market_segment, distribution_channel,
    is_repeated_guest, previous_cancellations, previous_bookings,
    reserved_room, assigned_room, booking_changes,
    deposit_type, days_waiting, customer_type,
    adr, parking_spaces, special_requests, 
    agent, company, is_holiday, reservation_status_date,
    days_to_holiday, days_from_holiday
):
    """Predict if a hotel booking will be canceled."""
    
    # Create arrival_date from components
    arrival_date = datetime(arrival_year, arrival_month, arrival_day)
    
    # Parse reservation status date
    res_date = pd.to_datetime(reservation_status_date)
    
    # Create input DataFrame with ALL features expected by the model
    input_data = pd.DataFrame({
        'hotel': [hotel],
        'lead_time': [lead_time],
        'arrival_date_year': [arrival_year],
        'arrival_date_month': [arrival_month],
        'arrival_date_week_number': [arrival_date.isocalendar()[1]],
        'arrival_date_day_of_month': [arrival_day],
        'stays_in_weekend_nights': [weekend_nights],
        'stays_in_week_nights': [week_nights],
        'adults': [adults],
        'children': [children],
        'babies': [babies],
        'meal': [meal],
        'country': [country],
        'market_segment': [market_segment],
        'distribution_channel': [distribution_channel],
        'is_repeated_guest': [1 if is_repeated_guest == "Yes" else 0],
        'previous_cancellations': [previous_cancellations],
        'previous_bookings_not_canceled': [previous_bookings],
        'reserved_room_type': [reserved_room],
        'assigned_room_type': [assigned_room],
        'booking_changes': [booking_changes],
        'deposit_type': [deposit_type],
        'agent': [str(agent) if agent and str(agent).strip() else "Other"],
        'company': [float(company) if company and str(company).strip() else np.nan],
        'days_in_waiting_list': [days_waiting],
        'customer_type': [customer_type],
        'adr': [adr],
        'required_car_parking_spaces': [parking_spaces],
        'total_of_special_requests': [special_requests],
        'reservation_status_date': [res_date],
        'arrival_date': [arrival_date],
        'is_holiday': [1 if is_holiday == "Yes" else 0],
        'days_to_next_holiday': [days_to_holiday],
        'days_from_last_holiday': [days_from_holiday],
        'reservation_status_year': [res_date.year],
        'reservation_status_month': [res_date.month],
        'reservation_status_day': [res_date.day]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Create result message
    if prediction == 1:
        result = f"⚠️ **HIGH RISK OF CANCELLATION**\n\nProbability: {probability[1]*100:.1f}%"
        recommendation = """
**Recommended Actions:**
- Contact guest to confirm booking
- Offer incentives to maintain reservation
- Consider overbooking buffer
- Flag for follow-up
        """
    else:
        result = f"✅ **LOW RISK OF CANCELLATION**\n\nProbability of cancellation: {probability[1]*100:.1f}%"
        recommendation = """
**Status:**
- Booking appears stable
- Standard confirmation process
- No special intervention needed
        """
    
    return result + "\n\n" + recommendation

# Create Gradio interface
with gr.Blocks(title="Hotel Booking Cancellation Predictor") as demo:
    gr.Markdown("# 🏨 Hotel Booking Cancellation Predictor")
    gr.Markdown("Predict the likelihood of hotel booking cancellations using advanced ML")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Booking Details")
            hotel = gr.Dropdown(["City Hotel", "Resort Hotel"], label="Hotel Type", value="City Hotel")
            lead_time = gr.Number(label="Lead Time (days)", value=30)
            arrival_year = gr.Number(label="Arrival Year", value=2024)
            arrival_month = gr.Slider(1, 12, label="Arrival Month", value=7, step=1)
            arrival_day = gr.Slider(1, 31, label="Arrival Day", value=15, step=1)
            
            gr.Markdown("### Stay Details")
            weekend_nights = gr.Number(label="Weekend Nights", value=1)
            week_nights = gr.Number(label="Week Nights", value=2)
            adults = gr.Number(label="Adults", value=2)
            children = gr.Number(label="Children", value=0)
            babies = gr.Number(label="Babies", value=0)
            
        with gr.Column():
            gr.Markdown("### Guest Information")
            meal = gr.Dropdown(["BB", "HB", "FB", "SC"], label="Meal Plan", value="BB")
            country = gr.Textbox(label="Country Code", value="PRT")
            market_segment = gr.Dropdown(["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"], 
                                        label="Market Segment", value="Online TA")
            distribution_channel = gr.Dropdown(["TA/TO", "Direct", "Corporate", "GDS"], 
                                              label="Distribution Channel", value="TA/TO")
            is_repeated_guest = gr.Radio(["Yes", "No"], label="Repeated Guest", value="No")
            previous_cancellations = gr.Number(label="Previous Cancellations", value=0)
            previous_bookings = gr.Number(label="Previous Bookings (Not Canceled)", value=0)
            
            gr.Markdown("### Room & Payment")
            reserved_room = gr.Textbox(label="Reserved Room Type", value="A")
            assigned_room = gr.Textbox(label="Assigned Room Type", value="A")
            booking_changes = gr.Number(label="Booking Changes", value=0)
            deposit_type = gr.Dropdown(["No Deposit", "Refundable", "Non Refund"], 
                                       label="Deposit Type", value="No Deposit")
            
    with gr.Row():
        with gr.Column():
            days_waiting = gr.Number(label="Days in Waiting List", value=0)
            customer_type = gr.Dropdown(["Transient", "Contract", "Transient-Party", "Group"], 
                                       label="Customer Type", value="Transient")
            adr = gr.Number(label="Average Daily Rate", value=100)
            parking_spaces = gr.Number(label="Parking Spaces Required", value=0)
            special_requests = gr.Number(label="Special Requests", value=0)
        
        with gr.Column():
            agent = gr.Textbox(label="Agent ID", value="Other", placeholder="Travel agent ID or 'Other' for direct booking")
            company = gr.Number(label="Company ID", value=None, placeholder="Company ID (leave empty if none)")
            is_holiday = gr.Radio(["Yes", "No"], label="Is Holiday Period", value="No")
            reservation_status_date = gr.Textbox(label="Reservation Date", value="2024-07-01", placeholder="YYYY-MM-DD")
            days_to_holiday = gr.Number(label="Days to Next Holiday", value=30)
            days_from_holiday = gr.Number(label="Days from Last Holiday", value=30)
    
    predict_btn = gr.Button("🔮 Predict Cancellation Risk", variant="primary")
    output = gr.Markdown()
    
    predict_btn.click(
        fn=predict_cancellation,
        inputs=[hotel, lead_time, arrival_year, arrival_month, arrival_day,
               weekend_nights, week_nights, adults, children, babies,
               meal, country, market_segment, distribution_channel,
               is_repeated_guest, previous_cancellations, previous_bookings,
               reserved_room, assigned_room, booking_changes,
               deposit_type, days_waiting, customer_type,
               adr, parking_spaces, special_requests,
               agent, company, is_holiday, reservation_status_date,
               days_to_holiday, days_from_holiday],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
