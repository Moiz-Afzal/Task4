from flask import Flask, request, jsonify
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)

# Load the pre-trained model


def benchmark():
    # Benchmark
    from sklearn.metrics import r2_score
    import keras.backend as K
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from keras.callbacks import LearningRateScheduler
    from keras.callbacks import ModelCheckpoint
    from keras.models import Sequential
    from keras.layers import Bidirectional, LSTM, Dropout, Dense
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    # from keras.layers import *

    df = pd.read_csv('/C:/Users/moiz2/OneDrive/Desktop/EURUSD_M15.csv')
    print(df.count())

    # Rename bid OHLC columns
    df.rename(columns={'Time': 'timestamp','Open': 'open', 'Close': 'close',
                      'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)

    # Add additional features
    df['avg_price'] = (df['low'] + df['high']) / 2
    # df['range'] = df['high'] - df['low']
    df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close']) / 4
    df['oc_diff'] = df['open'] - df['close']

    df.drop(columns=['volume'], inplace=True)
    print(df.head())

    def create_dataset(dataset, look_back=20):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)


    # Scale and create datasets
    target_index = df.columns.tolist().index('close')
    high_index = df.columns.tolist().index('high')
    low_index = df.columns.tolist().index('low')
    dataset = df.values.astype('float32')

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Create y_scaler to inverse it later
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    t_y = df['close'].values.astype('float32')
    t_y = np.reshape(t_y, (-1, 1))
    y_scaler = y_scaler.fit(t_y)

    X, y = create_dataset(dataset, look_back=30)
    y = y[:, target_index]

    train_size = int(len(X) * 0.90)
    trainX = X[:train_size]
    trainY = y[:train_size]
    testX = X[train_size:]
    testY = y[train_size:]

    # create a small LSTM network
    model = Sequential()
    model.add(
        Bidirectional(LSTM(30, input_shape=(X.shape[1], X.shape[2]),
                          return_sequences=True),
                      merge_mode='sum',
                      weights=None,
                      input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(10, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(4, return_sequences=False))
    model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    print(model.summary())

    # Load the model weights
    model.load_weights("/C:/Users/moiz2/OneDrive/Desktop/weights.best.hdf5")

    # Predictions
    pred = model.predict(testX)
    pred = y_scaler.inverse_transform(pred)
    close = y_scaler.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))

    # Create a DataFrame for predictions
    predictions = pd.DataFrame()
    predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
    predictions['close'] = pd.Series(np.reshape(close, (close.shape[0])))

    # Ensure the indices of predictions and df match
    predictions.index = df.index[-len(pred):]

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(predictions.index, predictions['close'], label='Actual Close Price')
    plt.plot(predictions.index, predictions['predicted'], label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Close Price')
    plt.legend()
    plt.show()

    # Calculate the difference between predicted and actual close prices
    predictions['diff'] = predictions['predicted'] - predictions['close']

    # Plot distribution of differences
    plt.figure(figsize=(10, 10))
    sns.distplot(predictions['diff']);
    plt.title('Distribution of differences between actual and prediction ')
    plt.show()

    # Plot jointplot
    g = sns.jointplot(x="diff", y="predicted", data=predictions, kind="kde", space=0)
    plt.title('Distribution of error and price')
    plt.show()

    # Calculate MSE, MAE, and R^2
    mse = mean_squared_error(predictions['predicted'].values, predictions['close'].values)
    mae = mean_absolute_error(predictions['predicted'].values, predictions['close'].values)
    r2 = r2_score(predictions['predicted'].values, predictions['close'].values)

    print("MSE : ", mse)
    print("MAE : ", mae)
    print("R^2 : ", r2)
    predictions['diff'].describe()

@app.route('/predict', methods=['POST'])
def predict():
    predictions = benchmark()
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
