# Author: Hedman
# Project: LSTM-Based E-Commerce Revenue Forecasting
# Date: 2025-01-18
# --------------------------------------------------------------------------------------------------------------
# DESCRIPTION:
## This model predicts future e-commerce revenue based on historical sales data using a Long Short-Term Memory (LSTM) network.
## It leverages synthetic revenue data that includes a combination of trend, seasonality, and random noise to mimic real-world fluctuations.
## The LSTM model is designed to learn time-series dependencies, helping businesses forecast revenue more accurately.
## Techniques such as sequence modeling, batch processing, dropout regularization, learning rate scheduling, and early stopping 
## are applied to optimize and assess model performance.

# OBJECTIVE:
## Develop and evaluate a predictive model that accurately forecasts e-commerce revenue based on past revenue patterns.
## Identify key trends in sales behavior and capture seasonality effects in the data.
## Utilize deep learning techniques (LSTM) to enhance forecasting accuracy over traditional statistical models.

# Hypothesis:
## H0: Past revenue patterns do not significantly influence future revenue.
## H1: Historical revenue data, including trend and seasonality, significantly influences future revenue predictions.
# --------------------------------------------------------------------------------------------------------------
# Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
# --------------------------------------------------------------------------------------------------------------
# 1. Generate synthetic e-commerce data
# --------------------------------------------------------------------------------------------------------------
def generate_synthetic_ecommerce_data(num_days=365):
    """
    Generates synthetic daily e-commerce revenue data with seasonality, trend, and noise.
    
    Parameters:
        - num_days (int): Number of days to simulate revenue data.
    
    Returns:
        - df (DataFrame): Pandas DataFrame with date and synthetic revenue values.
    """
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start="2025-01-01", periods=num_days, freq="D")

    # Define revenue components
    trend_component = np.linspace(50, 200, num_days)  # Linear increase in revenue
    seasonal_component = 20 * np.sin(np.linspace(0, 4 * np.pi, num_days))  # Weekly seasonality
    noise_component = np.random.normal(0, 10, num_days)  # Random fluctuations
    
    # Generate final revenue series
    revenue_series = trend_component + seasonal_component + noise_component
    
    # Return as DataFrame
    df = pd.DataFrame({"date": dates, "revenue": revenue_series})
    
    # Debugging Insert: Check first few rows
    print("Generated synthetic e-commerce data sample:")
    print(df.head())

    return df

# Generate and visualize synthetic e-commerce data
df = generate_synthetic_ecommerce_data(num_days=365)

plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["revenue"], label="Synthetic Revenue Data")
plt.xlabel("Date")
plt.ylabel("Revenue ($)")
plt.title("Synthetic E-Commerce Revenue Over Time")
plt.legend()
plt.show()
# --------------------------------------------------------------------------------------------------------------
# 2. Data pre-processing
# --------------------------------------------------------------------------------------------------------------
# Extract revenue values and reshape for scaling
revenue_values = df["revenue"].values.reshape(-1, 1)

# Normalize revenue data using StandardScaler
scaler = StandardScaler()
scaled_revenue = scaler.fit_transform(revenue_values)

def create_lstm_sequences(data, sequence_length):
    """
    Converts time-series data into sequences suitable for LSTM input.

    Parameters:
        - data (numpy array): Normalized time-series revenue data.
        - sequence_length (int): Number of past time steps to use for forecasting.

    Returns:
        - X_sequences (numpy array): LSTM input sequences.
        - y_targets (numpy array): Target revenue values.
    """
    X_sequences, y_targets = [], []
    for i in range(len(data) - sequence_length):
        X_sequences.append(data[i:i + sequence_length])
        y_targets.append(data[i + sequence_length])
    
    return np.array(X_sequences), np.array(y_targets)

# Define sequence length (how many past days to use for prediction)
SEQUENCE_LENGTH = 60  # Increased to capture longer-term trends

# Prepare LSTM sequences
X_data, y_data = create_lstm_sequences(scaled_revenue, SEQUENCE_LENGTH)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32).squeeze()  # Fix shape warning

# Split dataset into training (80%) and test (20%)
train_size = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Create DataLoader for batching
BATCH_SIZE = 32  
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# Debugging Insert: Print dataset shapes
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
# --------------------------------------------------------------------------------------------------------------
# 3. Define LSTM forecasting model
# --------------------------------------------------------------------------------------------------------------
class ECommerceLSTM(nn.Module):
    def __init__(self, input_features=1, hidden_units=128, num_layers=3, dropout_rate=0.2):
        """
        Improved LSTM model with dropout to prevent overfitting.

        Parameters:
            - input_features (int): Number of input features (1 for univariate time series).
            - hidden_units (int): Number of hidden units per LSTM layer.
            - num_layers (int): Number of stacked LSTM layers.
            - dropout_rate (float): Dropout rate to prevent overfitting.
        """
        super(ECommerceLSTM, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        # LSTM layers with dropout
        self.lstm = nn.LSTM(input_features, hidden_units, num_layers, batch_first=True, dropout=dropout_rate)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x):
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Fully connected output (predict next revenue value)
        return self.fc(lstm_out[:, -1, :]).squeeze()

# Initialize the model
model = ECommerceLSTM().to(torch.device("cpu"))

# Define loss function and optimizer
criterion = nn.L1Loss()  # MAE (L1) Loss is more robust for outliers
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Reduce LR over time
# --------------------------------------------------------------------------------------------------------------
# 4. Train the LSTM model stopping mechanism included
# --------------------------------------------------------------------------------------------------------------
EPOCHS = 200
best_loss = float("inf")
early_stopping_patience = 10  
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()  

    avg_epoch_loss = epoch_loss / len(train_loader)

    # Logging progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_epoch_loss:.6f}")

    # Early stopping mechanism
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered! Training stopped.")
            break
# --------------------------------------------------------------------------------------------------------------
# 5. Evaluate the LSTM model and make predictions
# --------------------------------------------------------------------------------------------------------------
model.eval()
predictions, actual_values = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        batch_predictions = model(X_batch)
        predictions.extend(batch_predictions.numpy())
        actual_values.extend(y_batch.numpy())

# Convert predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
actual_values = scaler.inverse_transform(np.array(actual_values).reshape(-1, 1))

# Plot actual vs. predicted revenue
plt.figure(figsize=(12, 5))
plt.plot(actual_values, label="Actual Revenue", color="blue")
plt.plot(predictions, label="Predicted Revenue", color="red", linestyle="dashed")
plt.xlabel("Days")
plt.ylabel("Revenue ($)")
plt.title("Improved E-Commerce Revenue Forecasting with LSTM")
plt.legend()
plt.show()
# --------------------------------------------------------------------------------------------------------------
