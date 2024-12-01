import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, SimpleRNN, Dropout, LayerNormalization, Bidirectional)
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, precision_score)
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Enable mixed precision for potential performance benefits

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Stocks, epochs, sequence lengths, and bidirectional settings
stocks = ['AAPL', 'GOOG', 'AMZN', 'XOM', 'NEE']
epochs_list = [50, 100, 200]
seq_lengths = [7, 30]
bidirectional_options = [False, True]

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

# Build the improved model for multiclass classification using SimpleRNN
def build_improved_simple_rnn(input_shape, num_classes, bidirectional=False):
    layers = []
    if bidirectional:
        layers.append(Bidirectional(SimpleRNN(64, return_sequences=True, input_shape=input_shape)))
    else:
        layers.append(SimpleRNN(64, return_sequences=True, input_shape=input_shape))
    layers.append(LayerNormalization())
    layers.append(Dropout(0.2))
    if bidirectional:
        layers.append(Bidirectional(SimpleRNN(128)))
    else:
        layers.append(SimpleRNN(128))
    layers.append(LayerNormalization())
    layers.append(Dropout(0.2))
    layers.append(Dense(num_classes, activation='linear'))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model


# Results list to store precision scores
results = []

# Loop over configurations
for stock in stocks:
    for epochs in epochs_list:
        for seq_length in seq_lengths:
            for bidirectional in bidirectional_options:
                print(f"Running for Stock: {stock}, Epochs: {epochs}, Seq Length: {seq_length}, Bidirectional: {bidirectional}")
                # Load stock data
                stock_data = pd.read_csv(f'./price/raw/{stock}.csv')

                # Ensure the Date column is in datetime format
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])

                # Sort the data by date
                stock_data = stock_data.sort_values('Date')

                # Apply technical indicators
                stock_data = add_technical_indicators(stock_data)

                # Introduce a threshold for 'No Change' classification
                threshold = 0.000  # 0.1%
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
                timesteps = seq_length  # Adjust the number of timesteps
                X, y = prepare_rnn_input(scaled_features, stock_data['Target'].values, timesteps=timesteps)

                # Encode labels to integers and get the mapping
                label_encoder = LabelEncoder()
                integer_encoded_y = label_encoder.fit_transform(y)
                label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

                # Convert integers to one-hot encoding
                y_categorical = to_categorical(integer_encoded_y)

                # Analyze class distribution
                unique, counts = np.unique(integer_encoded_y, return_counts=True)
                class_distribution = dict(zip(unique, counts))

                # Calculate class weights using integer-encoded labels
                class_weights_values = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(integer_encoded_y),
                    y=integer_encoded_y
                )
                class_weights_dict = {i: class_weights_values[i] for i in range(len(class_weights_values))}

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

                
                # Build the model
                model = build_improved_simple_rnn((X_train.shape[1], X_train.shape[2]), y_categorical.shape[1], bidirectional=bidirectional)

                with tf.device('/GPU:0'):
                    # Train the model with class weights
                    history = model.fit(
                        X_train,
                        y_train,
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=512,
                        class_weight=class_weights_dict,
                        verbose=0  # Set to 1 to see training progress
                    )

                # Predictions
                predictions = model.predict(X_test)
                predicted_classes = np.argmax(predictions, axis=1)
                actual_classes = np.argmax(y_test, axis=1)

                # Map integer labels back to original labels
                predicted_labels = label_encoder.inverse_transform(predicted_classes)
                actual_labels = label_encoder.inverse_transform(actual_classes)

                # Calculate model performance metrics
                report = classification_report(actual_labels, predicted_labels, target_names=label_encoder.classes_.astype(str),zero_division=0, output_dict=True)
                
                precision = report['1']['precision'] * 100  # Precision for the 'Up' class

                print(precision)
                # Append results
                results.append({
                    'Stock': stock,
                    'Epochs': epochs,
                    'Bidirectional': bidirectional,
                    'Seq_Length': seq_length,
                    'Precision': precision
                })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Pivot the DataFrame to match the desired table format
pivot_df = results_df.pivot_table(
    index=['Stock', 'Epochs'],
    columns=['Bidirectional', 'Seq_Length'],
    values='Precision'
).reset_index()

# Rename columns for clarity
pivot_df.columns = ['Stock', 'Epochs', 'Bidirectional=False, Seq Length 7', 'Bidirectional=False, Seq Length 30',
                    'Bidirectional=True, Seq Length 7', 'Bidirectional=True, Seq Length 30']

# Display the DataFrame
print(pivot_df)

# Highlight the highest precision for each stock
def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

pivot_df_style = pivot_df.style.apply(highlight_max, subset=[
    'Bidirectional=False, Seq Length 7', 'Bidirectional=False, Seq Length 30',
    'Bidirectional=True, Seq Length 7', 'Bidirectional=True, Seq Length 30'
], axis=1)

# Display styled DataFrame
pivot_df_style

# Optionally, save the DataFrame to LaTeX format
pivot_df.to_latex('results_table.tex', index=False)
