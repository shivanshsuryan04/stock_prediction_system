import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. LOAD DATA
DATA_DIR = "data/processed"
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_processed.csv")]
df_list = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in all_files]
data = pd.concat(df_list, ignore_index=True)

# 2. FEATURE SELECTION (Matching your XGBoost script)
features = ["Volume", "Return", "MA_10", "MA_20", "MA_50", "Volatility", 
            "RSI", "MACD", "Signal_Line", "Dist_MA_50", "Lag_1", "Lag_2"]

# 3. SCALING & SEQUENCE CREATION
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])
y = data["Trend"].map({-1: 0, 1: 1}).values 

def create_sequences(X, y, time_steps=10):
    X_s, y_s = [], []
    for i in range(len(X) - time_steps):
        X_s.append(X[i : i + time_steps])
        y_s.append(y[i + time_steps])
    return np.array(X_s), np.array(y_s)

X_final, y_final = create_sequences(X_scaled, y, time_steps=10)

# 4. LSTM ARCHITECTURE
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(10, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # Binary probability
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. TRAIN & SAVE
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_final, y_final, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop])

os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/lstm_scaler.pkl") # Important to save separate scaler if needed
print("✅ LSTM training complete and saved.")