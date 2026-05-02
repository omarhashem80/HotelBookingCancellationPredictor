---
title: Hotel Booking Cancellation Predictor
emoji: 🏨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# 🏨 Hotel Booking Cancellation Predictor

An AI-powered system that predicts hotel booking cancellations with 99.6% accuracy using XGBoost machine learning.

## 🎯 Features

- **High Accuracy Predictions**: 99.8% accuracy, 99.6% F1-score
- **Risk Assessment**: Detailed probability scores for each booking
- **Business Recommendations**: Actionable insights to reduce cancellations
- **Interactive Interface**: User-friendly Gradio web interface
- **Real-time Analysis**: Instant predictions with risk factor breakdown

## 🚀 How It Works

The model analyzes 35+ booking features including:

- Lead time and booking patterns
- Guest demographics and history
- Room preferences and changes
- Deposit and payment details
- Temporal patterns (holidays, seasons)

## 📊 Model Performance

- **Algorithm**: XGBoost with hyperparameter optimization
- **Training Data**: 87,000+ hotel bookings
- **Metrics**:
  - Accuracy: 99.8%
  - Precision: 99.8%
  - Recall: 99.6%
  - F1-Score: 99.6%
  - ROC-AUC: 99.99%

## 💼 Business Impact

Hotels can use this system to:

- Proactively identify at-risk bookings
- Implement targeted retention strategies
- Optimize overbooking policies
- Reduce revenue loss from last-minute cancellations

## 🔬 Technical Details

Built with:

- scikit-learn 1.8.0
- XGBoost 2.1.4
- Gradio 5.9.1
- Python 3.11

Model trained on real hotel booking data with comprehensive feature engineering and hyperparameter tuning.

---

**Auto-deployed from GitHub Actions** | Last updated: {{ date }}
