from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def main():
    data = pd.read_excel("training_data.xlsx", header=None)
    x1 = pd.DataFrame(data, columns=[0, 1])
    x2 = pd.DataFrame(data, columns=[2])

    data_1 = pd.read_excel("test_data.xlsx", header=None)
    tx1 = pd.DataFrame(data_1, columns=[0, 1])
    tx2 = pd.DataFrame(data_1, columns=[2])

    training_data = x1.to_numpy()
    y_data = x2.to_numpy()
    transformed_x = transform_data(training_data)

    trained_weights, model = linear_regression(transformed_x, y_data)
    output = predict(model, transformed_x)
    output = np.array(output)

    scaled_output = output
    for i in range(len(output)):
        if output[i] > 0:
            scaled_output[i] = 1
        elif output[i] < 0:
            scaled_output[i] = -1

    error_in = np.mean(scaled_output != y_data)
    print(error_in)

    test_x_data = tx1.to_numpy()
    test_y_data = tx2.to_numpy()

    transformed_test_data = transform_data(test_x_data)
    output_test_data = predict(model, transformed_test_data)
    output_test_data = np.array(output_test_data)

    scaled_output_1 = output_test_data
    for i in range(len(output_test_data)):
        if output_test_data[i] > 0:
            scaled_output_1[i] = 1
        elif output_test_data[i] < 0:
            scaled_output_1[i] = -1

    error_test = np.mean(scaled_output_1 != test_y_data)
    print(error_test)

def transform_data(input_data):
    transform = lambda x: [1, x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1], np.abs(x[0] - x[1]), np.abs((x[0] + x[1]))]

    transformed_data=[transform(x) for x in input_data]
    transformed_data=np.array(transformed_data)
       
    return transformed_data


def linear_regression(x_data, y_data):

    model = LinearRegression()
    model.fit(x_data,y_data)
    print(model.coef_)
    return model.coef_,model

def predict(model,predict_data):
    predicted_output=model.predict(predict_data)
    return predicted_output


if __name__ == "__main__":
    main()
