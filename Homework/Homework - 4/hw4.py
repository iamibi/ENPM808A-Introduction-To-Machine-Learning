from util import Util
import numpy as np
import pandas as pd
import linear_models as lm
import matplotlib.pyplot as plt


class Homework4:
    @classmethod
    def question_2(cls):
        N = 20
        N_t = 100
        df = Util.generate_data(N)
        print(f"Number of positive points = {df.loc[df['y'] > 0]['y'].count()}")
        X = df[["x1", "x2"]].values
        y = df["y"].values
        my_svm = lm.SVM()
        my_svm.fit(X, y)
        print(f"In-Sample Error = {my_svm.calc_error(X, y)}")
        print(f"SVM Margin = {my_svm.margin()}")
        df_t = Util.generate_data(N_t)
        X_t = df_t[["x1", "x2"]].values
        y_t = df_t["y"].values
        print(f"Out-of-Sample Error = {my_svm.calc_error(X_t, y_t)}")

        xsp1 = df.loc[df["y"] == 1]["x1"].values
        ysp1 = df.loc[df["y"] == 1]["x2"].values
        xsm1 = df.loc[df["y"] == -1]["x1"].values
        ysm1 = df.loc[df["y"] == -1]["x2"].values
        message = my_svm
        x1_min, x1_max = 0, 1
        x2_min, x2_max = -1, 1
        xx1, xx2 = Util.get_grid(x1_min, x1_max, x2_min, x2_max, step=0.02)
        Util.plot_decision_boundaries(xx1, xx2, 2, message)

        Util.plt_plot(
            [xsp1, xsm1],
            [ysp1, ysm1],
            "scatter",
            colors=["g", "b"],
            markers=["o", "+"],
            labels=["+1", "-1"],
            title="SVM",
            yscale=None,
            ylb=-1,
            yub=1,
            xlb=0,
            xub=1,
            xlabel=None,
            ylabel=None,
            legends=["+1", "-1"],
            legendx=None,
            legendy=None,
            marker_sizes=[5, 5],
        )

        df2 = pd.DataFrame({"x0": np.ones(N)})
        df1 = df.copy()
        df1.insert(0, "x0", df2["x0"])

        indexes = np.arange(N)
        indexes[18], indexes[0], indexes[19], indexes[17] = 0, 18, 17, 19
        df1 = df1.reindex(indexes)

        eta = 1
        use_adaline = False
        maxit = 1000
        dim = 2

        positives = df1.loc[df1["y"] == 1]
        negatives = df1.loc[df1["y"] == -1]

        figsize = plt.figaspect(1)
        f, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter(positives[["x1"]].values, positives[["x2"]].values, marker="+", c="g", label="+1 labels")
        ax.scatter(negatives[["x1"]].values, negatives[["x2"]].values, marker=r"$-$", c="b", label="-1 labels")
        print(f"Number of positive points = {len(positives)}\nNumber of negatives points = {len(negatives)}")

        norm_g, num_its, _ = lm.perceptron(df1.values, dim, maxit, use_adaline, eta, randomize=False, print_out=True)

        x1 = np.arange(0, 1, 0.02)
        norm_g = norm_g / norm_g[-1]

        c = np.ones((N_t, 1))
        XX_t = np.hstack((c, X_t))
        print(f"Out-of-Sample Error = {Util.calc_error(norm_g, XX_t, y_t)}")

        ax.plot(x1, -(norm_g[0] + norm_g[1] * x1), c="g", label="PLA")
        ax.plot(x1, -(my_svm.w[0] * x1 + my_svm.b) / my_svm.w[1], c="r", label="SVM")

        ax.set_ylabel(r"$x_2$", fontsize=11), ax.set_xlabel(r"$x_1$", fontsize=11)
        ax.set_title(f"Data set size = {N}", fontsize=9)
        ax.axis("tight")
        legend_x, legend_y = 2.0, 0.5
        ax.legend(
            [
                "PLA",
                "SVM",
                "+1 labels",
                "-1 labels",
            ],
            loc="center right",
            bbox_to_anchor=(legend_x, legend_y),
        )
        plt.show()

    @classmethod
    def question_4(cls):
        df = cls.__gen_data_q4()

        X_train = df[["x1", "x2"]].values
        y_train = df["y"].values

        for lamda_t in range(2):
            for Q in range(2, 4):
                linear_reg = lm.LinearRegression(reg_param=lamda_t, poly_degree=Q, to_classify=True)
                linear_reg.fit(X_train, y_train)
                cls.__plot_q4(df, linear_reg, f"Question 4 Q = {Q}, Lambda = {lamda_t}")

    @classmethod
    def __plot_q4(cls, df, message, tit):
        xsp1 = df.loc[df["y"] == 1]["x1"].values
        ysp1 = df.loc[df["y"] == 1]["x2"].values
        xsm1 = df.loc[df["y"] == -1]["x1"].values
        ysm1 = df.loc[df["y"] == -1]["x2"].values

        x1_min, x1_max = -1.2, 1.2
        x2_min, x2_max = -1.2, 1.2
        xx1, xx2 = Util.get_grid(x1_min, x1_max, x2_min, x2_max, step=0.02)
        Util.plot_decision_boundaries(xx1, xx2, 2, message, alpha=0.5)

        Util.plt_plot(
            [xsp1, xsm1],
            [ysp1, ysm1],
            "scatter",
            colors=["g", "b"],
            markers=["o", "x"],
            labels=["+1", "-1"],
            title=tit,
            yscale=None,
            ylb=-1,
            yub=1,
            xlb=0,
            xub=1,
            xlabel=None,
            ylabel=None,
            legends=["+1", "-1"],
            legendx=None,
            legendy=None,
            marker_sizes=[25, 25],
        )

    @classmethod
    def __gen_data_q4(cls):
        datap = np.array(
            [
                [-0.494, 0.363],
                [-0.311, -0.101],
                [-0.0064, 0.374],
                [-0.0089, -0.173],
                [0.0014, 0.138],
                [-0.189, 0.718],
                [0.085, 0.32208],
                [0.171, -0.302],
                [0.142, 0.568],
            ]
        )
        datan = np.array(
            [
                [0.491, 0.920],
                [-0.892, -0.946],
                [-0.721, -0.710],
                [0.519, -0.715],
                [-0.775, 0.551],
                [-0.646, 0.773],
                [-0.803, 0.878],
                [0.944, 0.801],
                [0.724, -0.795],
                [-0.748, -0.853],
                [-0.635, -0.905],
            ]
        )
        p_df = pd.DataFrame(datap, columns=["x1", "x2"])
        p_df["y"] = 1
        n_df = pd.DataFrame(datan, columns=["x1", "x2"])
        n_df["y"] = -1
        df = pd.concat([p_df, n_df])
        return df


# Execution starts here
if __name__ == "__main__":
    question_to_run = input("Enter the question number to run the program: ").strip()
    if question_to_run == "2":
        Homework4.question_2()
    elif question_to_run == "4":
        Homework4.question_4()
    else:
        raise ValueError("Invalid choice")
