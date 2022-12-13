import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from numpy.random import seed
import tensorflow
from keras.regularizers import l1
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


def plot_graphs(history_train, history_test):
    plt.plot(history_train.history["loss"])
    plt.plot(history_test.history["loss"])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"], loc="upper right")
    plt.show()


def testing_neural_network(path_to_train_temp_data, path_to_test_data):
    X_Train, Y_Train, X_Test, Y_Test, X_Val, Y_Val = get_test_data(path_to_train_temp_data, path_to_test_data)

    x = [X_Train, X_Val]
    y = [Y_Train, Y_Val]
    histories = []
    accuracies = []
    for i in range(2):
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
        history = model.fit(x[i], y[i], epochs=25, batch_size=100000)
        accuracy = model.evaluate(x[i], y[i])

        histories.append(history)
        accuracies.append(accuracy)

    print(f"MSE & MAE for Train = {accuracies[0]}\nMSE & MAE for Validation = {accuracies[1]}")
    plot_graphs(histories[0], histories[1])


if __name__ == "__main__":
    import os

    path_to_test_csv = os.path.join(os.getcwd(), "Final Data", "Test.csv")
    path_to_train_temp_csv = os.path.join(os.getcwd(), "Final Data", "Train_Temp.csv")
    testing_neural_network(path_to_train_temp_csv, path_to_test_csv)
