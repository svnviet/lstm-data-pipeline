import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt

df = pd.read_csv("/content/xauusd_M1_exness_2025-08-25.csv")
empty_rows = df.isnull().all(axis=1)

# Print the indices of empty rows
if empty_rows.any():
    print("Empty rows found at indices:")
    print(empty_rows[empty_rows].index)
else:
    print("No empty rows found in the dataset.")

df.head(1)

df['Time'] = pd.to_datetime(df['Time'], utc=True)
df.sort_values(by='Time', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.drop(columns=['Spread', 'RealVolume'], errors='ignore')

NumCols = df.columns.drop(['Time'])
df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
df[NumCols] = df[NumCols].astype('float64')


def split_sequence_dataset(df: pd.DataFrame,
                           feature_cols,
                           target_cols,
                           window_size=60,
                           ratios=(0.7, 0.2, 0.1)):
    """
    Create (train/test/val) splits for a multivariate time series.
    X: sliding windows over `feature_cols` (scaled)
    y: next-step values of `target_cols` (scaled)
    z: next-step values of `target_cols` (UNscaled, for inspection/plots)
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1.0"

    # Ensure columns exist
    for c in feature_cols + target_cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not in DataFrame. Available: {list(df.columns)}")

    # Raw arrays
    raw_X = df[feature_cols].to_numpy(dtype=float)
    raw_y = df[target_cols].to_numpy(dtype=float)

    # Scale separately for X and y
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    scaled_X = x_scaler.fit_transform(raw_X)
    scaled_y = y_scaler.fit_transform(raw_y)

    N = len(df)
    if N <= window_size:
        raise ValueError(f"Need N > window_size. Got N={N}, window_size={window_size}")

    M = N - window_size  # usable target points
    n_train = int(M * ratios[0])
    n_test = int(M * ratios[1])
    n_val = M - n_train - n_test

    train_targets = range(window_size, window_size + n_train)
    test_targets = range(window_size + n_train, window_size + n_train + n_test)
    val_targets = range(window_size + n_train + n_test, window_size + M)

    def build_split(target_indices):
        X, y, z, idx = [], [], [], []
        for i in target_indices:
            X.append(scaled_X[i - window_size:i, :])  # (window_size, n_features)
            y.append(scaled_y[i, :])  # (n_targets,)
            z.append(raw_y[i, :])  # unscaled target for reference
            idx.append(i)
        return np.array(X), np.array(y), np.array(z), np.array(idx, dtype=int)

    X_train, y_train, z_train, idx_train = build_split(train_targets)
    X_test, y_test, z_test, idx_test = build_split(test_targets)
    X_val, y_val, z_val, idx_val = build_split(val_targets)

    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train": (X_train, y_train, z_train, idx_train),
        "test": (X_test, y_test, z_test, idx_test),
        "val": (X_val, y_val, z_val, idx_val),
        "meta": {
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "window_size": window_size
        }
    }


def define_model(window_size: int, n_features: int, n_targets: int) -> Model:
    inp = Input(shape=(window_size, n_features))
    x = LSTM(128, return_sequences=True)(inp)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(n_targets)(x)  # predict targets only
    model = Model(inp, out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# ===== 1) Prepare splits (TickVolume as extra input feature) =====
feature_cols = ['Open', 'High', 'Low', 'Close', 'TickVolume']  # inputs
target_cols = ['Open', 'Close', 'High', 'Low']  # outputs (unchanged)

splits = split_sequence_dataset(df, feature_cols, target_cols, window_size=60, ratios=(0.7, 0.2, 0.1))
x_scaler = splits["x_scaler"]
y_scaler = splits["y_scaler"]

X_train, y_train, z_train, idx_train = splits["train"]
X_test, y_test, z_test, idx_test = splits["test"]
X_val, y_val, z_val, idx_val = splits["val"]

# ===== 2) Model =====
model = define_model(window_size=60, n_features=len(feature_cols), n_targets=len(target_cols))
model.summary()

# ===== 3) Train (use explicit validation set) =====
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# ===== 4) Evaluate & predict =====
test_loss = model.evaluate(X_test, y_test, verbose=0)
y_pred_test_s = model.predict(X_test, verbose=0)  # scaled
y_test_inv = y_scaler.inverse_transform(y_test)  # back to real prices
y_pred_test_inv = y_scaler.inverse_transform(y_pred_test_s)

# Metrics in REAL units (per target)
mape_per_target = mean_absolute_percentage_error(y_test_inv, y_pred_test_inv, multioutput='raw_values')
acc_per_target = 1 - mape_per_target

print("Test Loss (scaled MSE):", test_loss)
print("Test MAPE per target", dict(zip(target_cols, mape_per_target)))
print("Test Accuracy per target", dict(zip(target_cols, acc_per_target)))

# ===== 5) Example comparisons =====
print("\nExample Predictions vs Actual (first 5 rows, real units):")
for i in range(min(5, len(y_test_inv))):
    print("Pred:", y_pred_test_inv[i], "Actual:", y_test_inv[i])

# ===== 6) Plot a single target (e.g., Close) =====
tidx = target_cols.index('Close')  # choose which target to visualize
n = min(365, len(y_test_inv))
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv[:n, tidx], label="Actual Close", marker="o")
plt.plot(y_pred_test_inv[:n, tidx], label="Predicted Close", marker="x")
plt.title(f"Predictions vs Actual ({target_cols[tidx]}) — first {n} samples")
plt.xlabel("Sample index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

import joblib
import json

model.save("artifacts/model.h5")
joblib.dump(x_scaler, "artifacts/x_scaler.joblib")
joblib.dump(y_scaler, "artifacts/y_scaler.joblib")
with open("artifacts/meta.json", "w") as f:
    json.dump({"feature_cols": feature_cols, "target_cols": target_cols, "window_size": 60}, f)
