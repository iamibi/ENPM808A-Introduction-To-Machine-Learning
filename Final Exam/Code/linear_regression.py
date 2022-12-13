from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd


def linear_regression(path_to_train_temp_csv, path_to_test_csv):
    training_dataset = pd.read_csv(path_to_train_temp_csv)
    n = int(len(training_dataset) - (len(training_dataset) / 5))
    print(f"N = {n}")

    X_Train = training_dataset.iloc[:n, 0:19].values
    Y_Train = training_dataset.iloc[:n, 19:21].values
    X_Val = training_dataset.iloc[n:, 0:19].values
    Y_Val = training_dataset.iloc[n:, 19:21].values

    test_dataset = pd.read_csv(path_to_test_csv)
    X_Test = test_dataset.iloc[:, 0:19].values
    Y_Test = test_dataset.iloc[:, 19:21].values

    training_dataset = pd.read_csv(path_to_train_temp_csv)
    X_Train_final = training_dataset.iloc[:, 0:19].values
    Y_Train_final = training_dataset.iloc[:, 19:21].values

    regressor = LinearRegression()
    regressor.fit(X_Train, Y_Train)

    # Predicting the validation set results
    yPred_val = regressor.predict(X_Val)
    yPred_train = regressor.predict(X_Train)
    print("MSE value for validation data", mean_squared_error(Y_Val, yPred_val))

    # For test data
    regressor1 = LinearRegression()
    regressor1.fit(X_Train_final, Y_Train_final)

    # Predicting the Test set results
    yPred_test = regressor1.predict(X_Test)
    print("MSE value for test data", mean_squared_error(Y_Test, yPred_test))
    print("MSE value for Train data", mean_squared_error(Y_Train, yPred_train))
