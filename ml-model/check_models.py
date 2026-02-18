import joblib
import os

# Define the path to your saved XGBoost model
model_path = "models/xgboost.pkl"

if os.path.exists(model_path):
    # This 'loads' the binary file back into a Python object you can use
    model = joblib.load(model_path)
    print("✅ Successfully loaded the XGBoost model!")
    print(f"Model Type: {type(model)}")
    
    # Check if the scaler exists too
    if os.path.exists("models/scaler.pkl"):
        scaler = joblib.load("models/scaler.pkl")
        print("✅ Successfully loaded the Scaler!")
    else:
        print("⚠️ Scaler file missing in models/ folder.")
else:
    print("❌ Model file not found. Make sure you ran train_models.py first.")