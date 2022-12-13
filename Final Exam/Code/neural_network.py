import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from numpy.random import seed
import tensorflow
import matplotlib.pyplot as plt


def get_test_data(path_to_train_temp_data, path_to_test_data):
    tensorflow.random.set_seed(0)
    seed(0)

    dataset = pd.read_csv(path_to_train_temp_data)
    n = int(len(dataset) - (len(dataset) / 5))
    X_Train = dataset.iloc[:n, 0:19].values
    Y_Train = dataset.iloc[:n, 19:21].values
    X_Val = dataset.iloc[n:, 0:19].values
    Y_Val = dataset.iloc[n:, 19:21].values
    dataset2 = pd.read_csv(path_to_test_data)
    X_Test = dataset2.iloc[:, 0:19].values
    Y_Test = dataset2.iloc[:, 19:21].values

    return X_Train, Y_Train, X_Test, Y_Test, X_Val, Y_Val


def neural_network(path_to_train_temp_data, path_to_test_data):
    X_Train, Y_Train, X_Test, Y_Test, X_Val, Y_Val = get_test_data(path_to_train_temp_data, path_to_test_data)

    model = Sequential()
    model.add(Dense(100, input_dim=19, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="linear"))
    model.add(Dense(100, activation="linear"))
    model.add(Dense(2, activation="linear"))
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["MAE"])
    history = model.fit(X_Train, Y_Train, epochs=25, batch_size=100000)
    accuracy = model.evaluate(X_Test, Y_Test)
    print(f"MSE & MAE for Test = {accuracy}")
    plt.plot(history.history["loss"])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Test"], loc="upper right")
    plt.show()
