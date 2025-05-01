import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the dataset (replace 'your_stock_data.csv' with the path to your CSV file)
df = pd.read_csv('your_stock_data.csv')  # Ensure the file is in the correct directory

# Display first few rows to understand the dataset
print(df.head())

# Assuming the dataset has columns 'Date' and 'Close' for stock prices, adjust as necessary
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Use only 'Close' price for prediction
data = df[['Close']]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the data for LSTM (timesteps)
train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]

# Create the training data
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])  # 60 previous days' data
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape X_train to be compatible with LSTM input shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))  # Prediction of next day's price

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Prepare the test data in the same way
X_test, y_test = [], []
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape X_test to be compatible with LSTM input shape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict the stock prices
predicted_prices = model.predict(X_test)

# Inverse the scaling to get the original prices
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(df.index[int(len(scaled_data)*0.8)+60:], y_test, color='blue', label='Actual Prices')
plt.plot(df.index[int(len(scaled_data)*0.8)+60:], predicted_prices, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()