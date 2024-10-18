import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt  # Updated import for Keras Tuner
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------- Configuration --------------------
# # Groovy SantaCruz model, updated neurons to 550, changed lookback to step 5 up to 90 minutes prior (big orders tend to have this effect)
#1. SantaCruz includes volume data in row 80
#2. Row 41 is to turn retuning on and off
#3. Row 133-143 is for tuning dropout rate, lookback, and number of neurons

# Paths
DATA_FILE_PATH = "C:/Users/Chopp/Desktop/3.csv"  # Replace with your actual file path
BEST_HYPERPARAMS_FILE = 'configs/SC_hyperparameters.json'
TRAINED_MODEL_FILE = 'models/SantaCruz.keras'  # Updated file extension to .keras 
PLOT_OUTPUT_FILE = 'plots/SC_model_performance.png'

# Hyperparameter Tuning Configuration
MAX_TRIALS = 200
EXECUTIONS_PER_TRIAL = 1
TUNER_PROJECT_BASE = 'lstm_agg_forecast'
TUNER_DIR_BASE = 'kt_dir'

# Model Training Configuration
EPOCHS_TUNER = 50
EPOCHS_FINAL = 20
BATCH_SIZE = 36
PATIENCE = 5  # For EarlyStopping

# Forecast Configuration
FORECAST_HORIZON = 3  # Number of future steps to predict

# Control Retuning Behavior
RETUNE = False  # Set to False if you don't want to retune 

# ------------------- ▄︻デ══━一 -------------------

# 1. Ensure Required Directories Exist
required_directories = [
    os.path.dirname(BEST_HYPERPARAMS_FILE),
    os.path.dirname(TRAINED_MODEL_FILE),
    os.path.dirname(PLOT_OUTPUT_FILE),
]

for directory in required_directories:
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# 2. Load the dataset
if not os.path.exists(DATA_FILE_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_FILE_PATH}")

df = pd.read_csv(DATA_FILE_PATH)
print(f"Successfully loaded data from {DATA_FILE_PATH} with shape {df.shape}")

# 3. Engineer Aggressive Buy/Sell Ratio
if 'buyers' in df.columns and 'sellers' in df.columns:
    # Use np.where to handle division by zero
    df['agg_buy_sell_ratio'] = np.where(
        df['sellers'] != 0,
        df['buyers'] / df['sellers'],
        0  # Set ratio to 0 when sellers is 0
    )
    print("Engineered 'agg_buy_sell_ratio' with division by zero handled.")
else:
    raise KeyError("Missing required columns 'buyers' and/or 'sellers' for ratio engineering.")

# 4. Independent Variables (include the engineered ratio)### #####  FEATURES
required_features = ['High', 'Low', 'Close', 'agg_buy_sell_ratio','Volume'] ####
missing_features = [feature for feature in required_features if feature not in df.columns]
if missing_features:
    raise KeyError(f"Missing required features: {missing_features}")
features = df[required_features].values
print(f"Selected features: {required_features}")

# Additional check for invalid values
print("Checking for invalid values in features...")
invalid_values = False

if np.any(np.isnan(features)):
    print("Error: Found NaN values in features.")
    invalid_values = True
if np.any(np.isinf(features)):
    print("Error: Found infinite values in features.")
    invalid_values = True

# Optionally, check for extremely large values
extreme_value_threshold = 1e10  # Adjust the threshold as needed
if np.any(np.abs(features) > extreme_value_threshold):
    print(f"Warning: Found values exceeding {extreme_value_threshold} in features.")
    invalid_values = True

if invalid_values:
    raise ValueError("Invalid values detected in features. Please check your data preprocessing steps.")

print("No invalid values found in features.")

# 5. Dependent Variable (target)
if 'Open' in df.columns:
    target = df['Open'].values.reshape(-1, 1)
    print("Selected target variable: 'Open'")
else:
    raise KeyError("Missing required column 'Open' for target variable.")

# 6. Scale the features and target using StandardScaler
feature_scaler = StandardScaler()
scaled_features = feature_scaler.fit_transform(features)

target_scaler = StandardScaler()
scaled_target = target_scaler.fit_transform(target)

# 7. Function to generate lookback sequences (for LSTM input)
def create_sequences(features, target, lookback, forecast_horizon):
    X, y = [], []
    for i in range(lookback, len(features) - forecast_horizon + 1):
        X.append(features[i - lookback:i])  # Use previous 'lookback' data points
        y.append(target[i:i + forecast_horizon].flatten())  # Collect the next 'forecast_horizon' target values
    return np.array(X), np.array(y)

# 8. LSTM model structure with tunable lookback window <--------------------------------------------######
def build_lstm_model(hp, forecast_horizon, num_features):
    # Define 'lookback' hyperparameter here to register it
    lookback = hp.Int('lookback', min_value=5, max_value=90, step=5) #
    # Adjust input shape according to lookback and number of features
    model = Sequential()
    model.add(Input(shape=(lookback, num_features)))

    # LSTM layer with tunable units
    lstm_units = hp.Int('units', min_value=50, max_value=550, step=50) # -----------------------------------------------
    #                                                                      
    model.add(LSTM(units=lstm_units, activation='tanh', return_sequences=False))

    # Dropout layer for regularization with tunable dropout rate## #
    dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.7, step=0.05)
    model.add(Dropout(rate=dropout_rate))

    # Output layer: adjust units to forecast_horizon
    model.add(Dense(forecast_horizon))  # Output layer now has 'forecast_horizon' units

    # Compile the model
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

    return model

# 9. Custom Tuner to handle variable lookback window
class MyTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters

        # Retrieve 'lookback' hyperparameter
        lookback = hp.get('lookback')

        # Recreate sequences based on the tuned lookback window
        X, y = create_sequences(scaled_features, scaled_target, lookback, FORECAST_HORIZON)

        # Split into training and validation sets (time series split)
        split_index = int(len(X) * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        print(f"Trial {trial.trial_id}: Using lookback window of {lookback}")

        # Update the kwargs to pass the new data
        kwargs['x'] = X_train
        kwargs['y'] = y_train
        kwargs['validation_data'] = (X_val, y_val)

        # Build the model using the hyperparameters
        model = build_lstm_model(hp, forecast_horizon=FORECAST_HORIZON, num_features=scaled_features.shape[1])

        # Fit the model
        history = model.fit(*args, **kwargs)

        # Return the result of the superclass's run_trial method
        return history

# 10. Function to perform hyperparameter tuning
def perform_hyperparameter_tuning():
    # Initialize the tuner
    tuner = MyTuner(
        hypermodel=lambda hp: build_lstm_model(hp, forecast_horizon=FORECAST_HORIZON, num_features=scaled_features.shape[1]),
        objective='val_loss',
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory=TUNER_DIR_BASE,
        project_name=TUNER_PROJECT_BASE
    )

    # Early stopping callback to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    print("Starting hyperparameter tuning...")
    tuner.search(
        epochs=EPOCHS_TUNER,
        callbacks=[early_stopping],
        verbose=1
    )
    print("Hyperparameter tuning completed.")

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Save the best hyperparameters to a JSON file using the fixed filename
    print(f"Attempting to save best hyperparameters to {BEST_HYPERPARAMS_FILE}")
    with open(BEST_HYPERPARAMS_FILE, 'w') as f:
        json.dump(best_hps.values, f)
    print(f"Saved best hyperparameters to {BEST_HYPERPARAMS_FILE}")

    return best_hps

# 11. Function to load best hyperparameters from file
def load_best_hyperparameters():
    print(f"Attempting to load best hyperparameters from {BEST_HYPERPARAMS_FILE}")
    if not os.path.exists(BEST_HYPERPARAMS_FILE):
        raise FileNotFoundError(f"Best hyperparameters file not found at {BEST_HYPERPARAMS_FILE}")
    
    with open(BEST_HYPERPARAMS_FILE, 'r') as f:
        best_hps_values = json.load(f)
    best_hps = kt.HyperParameters()
    best_hps.values = best_hps_values
    print(f"Loaded best hyperparameters from {BEST_HYPERPARAMS_FILE}")
    return best_hps

# 12. Main Execution Flow
def main():
    best_hps = None
    num_features = scaled_features.shape[1]  # Number of input features

    # Check if both trained model and hyperparameters files exist
    if not RETUNE and os.path.exists(TRAINED_MODEL_FILE) and os.path.exists(BEST_HYPERPARAMS_FILE):
        print(f"Loading the saved model from {TRAINED_MODEL_FILE}")
        model = load_model(TRAINED_MODEL_FILE)

        print("Loading saved hyperparameters.")
        best_hps = load_best_hyperparameters()
        best_lookback = best_hps.get('lookback')
        print(f"Using lookback window: {best_lookback}")

        # Recreate sequences with the best lookback
        X, y = create_sequences(scaled_features, scaled_target, best_lookback, FORECAST_HORIZON)
        print(f"Created sequences with shape X: {X.shape}, y: {y.shape}")

        # Split into training and testing sets (time series split)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        print(f"Split data into training and testing sets at index: {split_index}")

    else:
        if RETUNE:
            print("Starting hyperparameter tuning process.")
            best_hps = perform_hyperparameter_tuning()
            best_lookback = best_hps.get('lookback')
            print(f"Using lookback window: {best_lookback}")
        elif os.path.exists(BEST_HYPERPARAMS_FILE):
            print("Loading saved hyperparameters.")
            best_hps = load_best_hyperparameters()
            best_lookback = best_hps.get('lookback')
            print(f"Using lookback window: {best_lookback}")
        else:
            raise FileNotFoundError("Best hyperparameters file not found. Cannot proceed without hyperparameters.")

        # Build the model with the best hyperparameters
        model = build_lstm_model(best_hps, forecast_horizon=FORECAST_HORIZON, num_features=num_features)
        print("Built the LSTM model with the best hyperparameters.")

        # Early stopping callback for final training
        early_stopping_final = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

        # Recreate sequences with the best lookback
        X, y = create_sequences(scaled_features, scaled_target, best_lookback, FORECAST_HORIZON)
        print(f"Created sequences with shape X: {X.shape}, y: {y.shape}")

        # Split into training and testing sets (time series split)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        print(f"Split data into training and testing sets at index: {split_index}")

        # Train the model
        print("Starting model training...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS_FINAL,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping_final],
            verbose=1
        )
        print("Model training completed.")

        # Save the trained model
        print(f"Attempting to save trained model to {TRAINED_MODEL_FILE}")
        model.save(TRAINED_MODEL_FILE)
        print(f"Saved trained model to {TRAINED_MODEL_FILE}")

    # 13. Evaluate the model
    print("Evaluating the model on the test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # 14. Make predictions on the test set
    y_pred = model.predict(X_test)
    print("Made predictions on the test set.")

    # 15. Rescale predictions back to original scale
    # Since y_pred and y_test are now multi-dimensional, we need to reshape them
    try:
        y_pred_rescaled = target_scaler.inverse_transform(y_pred)
        y_test_rescaled = target_scaler.inverse_transform(y_test)
        print("Rescaled predictions back to original scale.")
    except Exception as e:
        print(f"Error during rescaling: {e}")
        return

    # 16. Calculate R-squared value for each horizon step
    r2_scores = []
    for i in range(FORECAST_HORIZON):
        try:
            r2 = r2_score(y_test_rescaled[:, i], y_pred_rescaled[:, i])
            r2_scores.append(r2)
            print(f"R-squared for horizon t+{i+1}: {r2}")
        except IndexError:
            print(f"IndexError: Not enough data to compute R-squared for horizon t+{i+1}")
            r2_scores.append(None)

    # 17. Plot the results and save the plot
    plt.figure(figsize=(12, 6))
    for i in range(FORECAST_HORIZON):
        plt.plot(y_test_rescaled[:, i], label=f"True Values (t+{i+1})")
        plt.plot(y_pred_rescaled[:, i], label=f"Predicted Values (t+{i+1})")
    plt.legend()
    plt.title(f"True vs Predicted 'Open' Values for Next {FORECAST_HORIZON} Minutes")
    plt.savefig(PLOT_OUTPUT_FILE)
    plt.close()
    print(f"Saved performance plot to {PLOT_OUTPUT_FILE}")

    # 18. One-shot prediction: Predict the next 5 'Open' values
    # Prepare the last 'best_lookback' sequence from the scaled features
    last_sequence = scaled_features[-best_lookback:]  # Shape: (lookback, num_features)
    print(f"Prepared last sequence with shape: {last_sequence.shape}")

    # Reshape to match the model's input shape: (1, lookback, num_features)
    last_sequence = np.expand_dims(last_sequence, axis=0)
    print(f"Reshaped last sequence to: {last_sequence.shape}")

    # Predict the next values
    next_preds_scaled = model.predict(last_sequence)
    print("Predicted the next 'Open' values (scaled).")

    # Rescale the predictions back to the original scale
    try:
        next_preds = target_scaler.inverse_transform(next_preds_scaled)
        print("Rescaled the predicted values back to original scale.")
    except Exception as e:
        print(f"Error during rescaling predictions: {e}")
        return

    # Print the predicted next 'Open' values
    for i, pred in enumerate(next_preds[0], 1):
        print(f"Predicted 'Open' value at t+{i}: {pred}")

# -------------------- Execute Script --------------------
if __name__ == "__main__":
    main()
# ---------------------------------------------------------
