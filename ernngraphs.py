import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, SimpleRNN, Dropout, LayerNormalization)
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# Enable mixed precision for potential performance benefits
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.optimizer.set_jit(True)

# Preprocessing with detailed scaling
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# RNN input preparation
def prepare_rnn_input(features, target, timesteps=7):
    X, y = [], []
    for i in range(timesteps, len(features)):
        X.append(features[i - timesteps : i])
        y.append(target[i])
    return np.array(X), np.array(y)

# Load GOOG stock data
stock_data = pd.read_csv('./price/raw/GOOG.csv')

# Ensure the Date column is in datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Sort the data by date
stock_data = stock_data.sort_values('Date')

# Calculate technical indicators
def add_technical_indicators(df):
    df = df.copy()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['RSI_14'] = calculate_rsi(df['Close'], window=14)
    df['MACD'] = (
        df['Close'].ewm(span=12, adjust=False).mean()
        - df['Close'].ewm(span=26, adjust=False).mean()
    )
    df = df.dropna()
    return df

# RSI Calculation Helper
def calculate_rsi(series, window=14):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    gain_avg = gain.rolling(window=window).mean()
    loss_avg = loss.rolling(window=window).mean()
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Apply technical indicators
stock_data = add_technical_indicators(stock_data)

# Introduce a threshold for 'No Change' classification
stock_data['Price_Change'] = (stock_data['Close'].shift(-1) - stock_data['Close']) / stock_data['Close']
stock_data['Target'] = (stock_data['Close'] > stock_data['Close'].shift(1)).astype(int)

# Drop any rows with NaN values (due to shifting and indicators)
stock_data = stock_data.dropna()

# Features (exclude 'Date' and 'Target')
features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI_14', 'MACD'
]

# Prepare feature data
feature_data = stock_data[features].values

# Preprocess the data (scale)
scaled_features, scaler = preprocess_data(feature_data)

# Prepare RNN inputs
timesteps = 30  # You can adjust the number of timesteps
X, y = prepare_rnn_input(scaled_features, stock_data['Target'].values, timesteps=timesteps)

# Encode labels to integers and get the mapping
label_encoder = LabelEncoder()
integer_encoded_y = label_encoder.fit_transform(y)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Convert integers to one-hot encoding
y_categorical = to_categorical(integer_encoded_y)

# Analyze class distribution
unique, counts = np.unique(integer_encoded_y, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class Distribution:", class_distribution)

# Calculate class weights using integer-encoded labels
class_weights_values = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(integer_encoded_y),
    y=integer_encoded_y
)
class_weights_dict = {i: class_weights_values[i] for i in range(len(class_weights_values))}
print("Class Weights Dictionary:", class_weights_dict)

# Adjust dates and prices for plotting
dates = stock_data['Date'].values[timesteps:]
actual_prices = stock_data['Close'].values[timesteps:]

# Train-test split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_categorical[:split_idx], y_categorical[split_idx:]
test_dates = dates[split_idx:]
test_actual_prices = actual_prices[split_idx:]
y_test_labels = integer_encoded_y[split_idx:]

# Build the improved model for multiclass classification using SimpleRNN
def build_improved_simple_rnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=input_shape),
        LayerNormalization(),
        Dropout(0.2),
        
        SimpleRNN(128),
        LayerNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='linear')
    ])
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

# Ensure model is None to force retraining
model = None
try:
    model = tf.keras.models.load_model('stock_price_prediction_multiclass_simple_rnn4.keras')
    print("Model loaded successfully.")
except:
    print("No existing model found. Training a new model.")

if model is None:
    model = build_improved_simple_rnn((X_train.shape[1], X_train.shape[2]), y_categorical.shape[1])

    # Train the model with class weights
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        class_weight=class_weights_dict,
        verbose=1
    )

# Predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

# Map integer labels back to original labels
predicted_labels = label_encoder.inverse_transform(predicted_classes)
actual_labels = label_encoder.inverse_transform(actual_classes)

# Calculate model performance metrics
accuracy = accuracy_score(actual_labels, predicted_labels) * 100
report = classification_report(actual_labels, predicted_labels, target_names=label_encoder.classes_.astype(str))
conf_matrix = confusion_matrix(actual_labels, predicted_labels)

print(f"Model Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}%")
print("Classification Report:")
print(report)

# Multiclass Confusion Matrix
print("Confusion Matrix (Up vs Down):")
print(conf_matrix)

# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Save model
model.save('stock_price_prediction_multiclass_simple_rnn4.keras')
print("Model saved successfully.")

# Prepare predicted prices for plotting
predicted_prices = test_actual_prices.copy()
price_change = 0.01  # Assume a 1% change for visualization

for i in range(len(predicted_labels)):
    if predicted_labels[i] == 1:  # Predicted Up
        predicted_prices[i] = test_actual_prices[i] * (1 + price_change)
    elif predicted_labels[i] == 0:  # Predicted Down
        predicted_prices[i] = test_actual_prices[i] * (1 - price_change)
    else:
        predicted_prices[i] = test_actual_prices[i]  # No Change

# Simplified Trading Simulation
initial_investment = 1000  # Initial investment in NOK
current_balance = initial_investment  # Available cash
shares_held = 0  # Number of shares held
buy_actions = []  # List of (index, price)
sell_actions = []  # List of (index, price)
profits = []  # List of profits from each trade
buy_price = 0

for i in range(len(predicted_labels)):
    # Today's actual price
    current_price = test_actual_prices[i]
    # tomorrows's prediction
    prediction = predicted_labels[i+1] if i < len(predicted_labels) - 1 else 0
    
    # Buy logic
    if prediction == 1 and current_balance > 0:
        # Buy as many shares as possible
        shares_held = current_balance / current_price
        current_balance = 0
        buy_price = current_price
        buy_actions.append((i, current_price))
    # Sell logic
    elif prediction == 0 and shares_held > 0:
        # Sell all shares
        proceeds = shares_held * current_price
        profit = proceeds - (shares_held * buy_price)
        current_balance += proceeds
        profits.append(profit)
        shares_held = 0
        sell_actions.append((i, current_price))
        buy_price = 0
    # Hold logic: do nothing

# If shares are still held at the end, sell them at the last available price
if shares_held > 0:
    current_price = test_actual_prices[-1]
    proceeds = shares_held * current_price
    profit = proceeds - (shares_held * buy_price)
    current_balance += proceeds
    profits.append(profit)
    sell_actions.append((len(test_actual_prices) - 1, current_price))
    shares_held = 0

# Metrics for Trading Simulation
total_trades = len(profits)
profitable_trades = sum(1 for profit in profits if profit > 0)
profitability_accuracy = (
    (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
)
total_profit = current_balance - initial_investment
profit_percentage = (total_profit / initial_investment) * 100
avg_trade_profit = np.mean(profits) if profits else 0

# Display Trading Simulation Results
print(f"\nTrading Simulation Results:")
print(f"Profitability Accuracy: {profitability_accuracy:.2f}%")
print(f"Total Profit: {total_profit:.2f} NOK ({profit_percentage:.2f}%)")
print(f"Final Balance: {current_balance:.2f} NOK")
print(f"Total Trades: {total_trades}")
print(f"Average Profit per Trade: {avg_trade_profit:.2f} NOK")

# Corrected Plotting
plt.figure(figsize=(14, 7))

# Plot historical data for the previous 30 days
# plt.plot(
#     dates[split_idx:],
#     actual_prices[split_idx:],
#     label="Historical Prices",
#     linestyle='-',
#     color='gray'
# )

# Plot actual prices in test set
plt.plot(
    test_dates,
    test_actual_prices,
    label="Actual Prices",
    linestyle='-',
    color='blue'
)
# Plot predicted prices
plt.plot(
    test_dates,
    predicted_prices,
    label="Predicted Prices",
    linestyle='--',
    color='orange'
)

# Plot buy actions on predicted price line
if buy_actions:
    buy_indices, _ = zip(*buy_actions)
    buy_dates = test_dates[np.array(buy_indices)]
    buy_prices = predicted_prices[np.array(buy_indices)]
    plt.scatter(
        buy_dates,
        buy_prices,
        color='green',
        label='Buy Action',
        marker='^',
        s=100
    )

# Plot sell actions on predicted price line
if sell_actions:
    sell_indices, _ = zip(*sell_actions)
    sell_dates = test_dates[np.array(sell_indices)]
    sell_prices = predicted_prices[np.array(sell_indices)]
    plt.scatter(
        sell_dates,
        sell_prices,
        color='red',
        label='Sell Action',
        marker='v',
        s=100
    )

plt.title("Trade Simulation")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

corr_matrix = stock_data[features + ['Target']].corr()
corr_with_target = corr_matrix['Target'].drop('Target').sort_values(ascending=False)
print("Correlation of features with Target:")
print(corr_with_target)

# Prepare the data
dates = np.array([pd.to_datetime(date) for date in stock_data['Date'].values[timesteps + split_idx:]])
actual_prices = stock_data['Close'].values[timesteps + split_idx:]
predicted_labels = np.array(predicted_classes)  # Assuming predicted_classes contains your model predictions after decoding

# Define markers for 'Up' and 'Down' predictions
up_dates = dates[predicted_labels == 1]
down_dates = dates[predicted_labels == 0]
up_prices = actual_prices[predicted_labels == 1]
down_prices = actual_prices[predicted_labels == 0]

# Create the plot
plt.figure(figsize=(14, 7))
plt.plot(dates, actual_prices, label="Actual Prices", linestyle='-', color='blue')

# Add markers for buy and sell actions based on predictions
plt.scatter(up_dates, up_prices, color='green', label='Predicted Up', marker='^', s=100)
plt.scatter(down_dates, down_prices, color='red', label='Predicted Down', marker='v', s=100)

plt.title("Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
