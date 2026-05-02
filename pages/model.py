import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image


def show(metrics_df, eval_data):
    st.header("Predictive System Evaluation")
    st.markdown("We continuously test multiple algorithms to ensure the system is highly accurate at flagging at-risk bookings.")
    
    if metrics_df is not None:
        st.subheader("Model Comparison Across Algorithms")
        # Convert to string to avoid Arrow serialization issues with styled dataframes
        display_df = metrics_df.copy()
        for col in display_df.select_dtypes(include=['float64', 'float32']).columns:
            display_df[col] = display_df[col].round(3)
        st.dataframe(display_df, width='stretch')
        
        # Model comparison chart
        model_comp_path = Path("reports/figures/model_comparison_f1.png")
        if model_comp_path.exists():
            st.image(Image.open(model_comp_path), caption="F1-Score Comparison Across Models")
    else:
        st.error("No model result data found yet. Please run the training pipeline first.")

    st.markdown("### Best Model Detailed Accuracy")
    if eval_data is not None:
        # Extract macro-averaged metrics from classification report
        macro_avg = eval_data.get('classification_report', {}).get('macro avg', {})
        accuracy = eval_data.get('standard_metrics', {}).get('accuracy', 0)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy*100:.1f}%")
        col2.metric("F1-Score", f"{macro_avg.get('f1-score', 0)*100:.1f}%")
        col3.metric("Precision", f"{macro_avg.get('precision', 0)*100:.1f}%")
        col4.metric("Recall", f"{macro_avg.get('recall', 0)*100:.1f}%")
    else:
        st.info("To see detailed metrics of the best model, execute the `make train` pipeline.")
        
    st.markdown("---")
    st.markdown("### Visual Confusion Matrix (Best Model)")
    cm_path = Path("reports/figures/confusion_matrix.png")
    if cm_path.exists():
        st.image(Image.open(cm_path), caption="Confusion Matrix: Actual vs. Predicted Status", width='stretch')
    else:
        st.write("Confusion matrix visualization is currently unavailable.")
        
    st.markdown("### ROC Curve")
    roc_path = Path("reports/figures/roc_curve.png")
    if roc_path.exists():
        st.image(Image.open(roc_path), caption="Receiver Operating Characteristic (ROC) Curve", width='stretch')
    else:
        st.write("ROC visualization is currently unavailable.")
    
    st.markdown("---")
    st.markdown("### Interpretation Guide")
    st.markdown("""
    - **Accuracy:** Overall correctness across all predictions
    - **Precision:** Of all bookings flagged as 'will cancel', what % actually canceled?
    - **Recall:** Of all actual cancellations, what % did we catch?
    - **F1-Score:** Harmonic balance between precision and recall (our primary optimization metric)
    """)
