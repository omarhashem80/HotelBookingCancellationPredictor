"""Export model with joblib for maximum cross-platform compatibility."""

import joblib
from pathlib import Path

print("Loading model from reports/best_model.pkl...")
model_path = Path("reports/best_model.pkl")

# Load with joblib (more robust than pickle)
model = joblib.load(model_path)
print(f"Loaded model: {type(model)}")
print(f"   Pipeline steps: {[step[0] for step in model.steps]}")

deployment_path = Path("Hotel-Booking/best_model.pkl")
deployment_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, deployment_path, compress=3, protocol=4)
print(f"\nExported to {deployment_path}")
print(f"   Size: {deployment_path.stat().st_size / 1024 / 1024:.2f} MB")
print("   Protocol: 4 (Python 3.4+ compatible)")

# Verify it can be reloaded
test_model = joblib.load(deployment_path)
print("\nVerified: Model can be reloaded successfully")
