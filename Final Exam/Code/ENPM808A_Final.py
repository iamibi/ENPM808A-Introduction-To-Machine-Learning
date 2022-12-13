from pathlib import Path
from linear_regression import linear_regression
from neural_network import neural_network
import numpy as np
import pandas as pd
import os


class FinalExam:
    @classmethod
    def preprocess_dataset(cls, dataset):
        dataset = np.array(dataset)
        sample = np.empty([len(dataset), 21], dtype=object)
        print(f"Dataset Length: {len(dataset)}")
        ranges = [
            (0, 108),
            (108, 216),
            (216, 324),
            (324, 432),
            (432, 540),
            (540, 648),
            (648, 756),
            (756, 864),
            (864, 972),
            (972, 1080),
        ]
        for i in range(len(dataset)):
            for j in range(10):
                sum_c = 0
                avg_c = 0
                range_c = ranges[j]
                for k in range(range_c[0], range_c[1]):
                    sum_c += dataset[i][k]
                    avg_c = sum_c / 108
                sample[i][j] = avg_c
            sample[i][10] = dataset[i][1080]
            sample[i][11] = dataset[i][1081]
            sample[i][12] = dataset[i][1082] + dataset[i][1083]
            sample[i][13] = dataset[i][1084]
            sample[i][14] = dataset[i][1085]
            sample[i][15] = dataset[i][1086] + dataset[i][1087]
            sample[i][16] = dataset[i][1088]
            sample[i][17] = dataset[i][1089]
            sample[i][18] = dataset[i][1090] + dataset[i][1091]
            sample[i][19] = dataset[i][1092]
            sample[i][20] = dataset[i][1093]
        sample = pd.DataFrame(sample)
        return sample

    @classmethod
    def generate_test_data(cls):
        path = os.path.join(os.getcwd(), "Final Data", "Test Data")
        csv_files = os.listdir(path)
        print(f"CSV Files: {csv_files}")

        output_file = os.path.join(os.getcwd(), "Final Data", "Test.csv")
        if os.path.exists(output_file) is False:
            Path(output_file).touch()

        cls.__generate_csv(path, csv_files, output_file)

    @classmethod
    def generate_samples(cls):
        cls.generate_test_data()
        cls.generate_train_data()

    @classmethod
    def generate_train_data(cls):
        # The corridor_CSV folder is expected to be located at `Final Data/Training Data/corridor_CSV/`
        path = os.path.join(os.getcwd(), "Final Data", "Training Data", "corridor_CSV")
        csv_files = os.listdir(path)
        print(f"CSV Files: {csv_files}")

        output_file = os.path.join(os.getcwd(), "Final Data", "Train_Temp.csv")
        if os.path.exists(output_file) is False:
            Path(output_file).touch()

        cls.__generate_csv(path, csv_files, output_file)

    @classmethod
    def linear_regression_predictor(cls):
        path_to_train_temp_csv, path_to_test_csv = cls.__get_train_and_test_paths()

        # Run the linear regression predictor
        linear_regression(path_to_train_temp_csv, path_to_test_csv)

    @classmethod
    def neural_network_predictor(cls):
        path_to_train_temp_csv, path_to_test_csv = cls.__get_train_and_test_paths()

        # Run the neural network predictor
        neural_network(path_to_train_temp_csv, path_to_test_csv)

    @classmethod
    def __generate_csv(cls, base_path, csv_files, output_file):
        temp = 0
        for file in csv_files:
            if Path(file).suffix == ".csv":
                csv_data = pd.read_csv(os.path.join(base_path, file))
                rows_csv_data, columns_csv_data = csv_data.shape
                temp += rows_csv_data
                ex_data = cls.preprocess_dataset(csv_data)
                ex_data.to_csv(output_file, mode="a", header=None, index=False)

    @classmethod
    def __get_train_and_test_paths(cls):
        path_to_test_csv = os.path.join(os.getcwd(), "Final Data", "Test.csv")
        if os.path.exists(path_to_test_csv) is False:
            raise f"File Test.csv not found in {path_to_test_csv}"

        path_to_train_temp_csv = os.path.join(os.getcwd(), "Final Data", "Train_Temp.csv")
        if os.path.exists(path_to_train_temp_csv) is False:
            raise f"File Train_Temp.csv not found in {path_to_train_temp_csv}"

        return path_to_train_temp_csv, path_to_test_csv


if __name__ == "__main__":
    # Generate the training data and test data combined
    # FinalExam.generate_samples()

    # Run Linear Regression Predictor
    FinalExam.linear_regression_predictor()

    # Run Neural Network Predictor
    FinalExam.neural_network_predictor()
