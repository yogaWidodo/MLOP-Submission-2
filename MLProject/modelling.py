import argparse
import argparse
import mlflow
import mlflow.keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM

def define_model(units, dropout_rate, window_size):
    input1 = Input(shape=(window_size,1))
    x = LSTM(units = units, return_sequences=True)(input1)  
    x = Dropout(dropout_rate)(x)
    x = LSTM(units = units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units = units)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='softmax')(x)
    dnn_output = Dense(1)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    model.summary()
    return model

def main(args):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("LSTM Model for Gold Price Prediction")

    df = pd.read_csv('MLProject/processed_gold_price.csv', parse_dates=['Date'])

    test_size = df[df.Date.dt.year==2022].shape[0]
    scaler = MinMaxScaler()
    scaler.fit(df.Price.values.reshape(-1,1))

    window_size = args.window_size

    train_data = df.Price[:-test_size]
    train_data = scaler.transform(train_data.values.reshape(-1,1))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])

    test_data = df.Price[-test_size-window_size:]
    test_data = scaler.transform(test_data.values.reshape(-1,1))

    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-window_size:i, 0])
        y_test.append(test_data[i, 0])

    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1,1))
    y_test  = np.reshape(y_test, (-1,1))

    with mlflow.start_run():
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("units", args.units)
        mlflow.log_param("dropout_rate", args.dropout_rate)

        model = define_model(args.units, args.dropout_rate, window_size)
        mlflow.autolog()
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name="LSTM_Gold_Price_Prediction_Model",
        )
        history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1, verbose='1')
        y_pred = model.predict(X_test)
        MAPE = mean_absolute_percentage_error(y_test, y_pred)
        Accuracy = 1 - MAPE
        mlflow.log_metric("MAPE", float(MAPE))
        mlflow.log_metric("Accuracy", float(Accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model for gold price prediction')
    parser.add_argument('--units', type=int, default=64, help='Number of LSTM units')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--window_size', type=int, default=60, help='Window size for time series')
    args = parser.parse_args()
    main(args)
