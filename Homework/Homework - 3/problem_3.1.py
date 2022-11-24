import matplotlib.pyplot as plt
from util import Util
import numpy as np
import pandas as pd
import linear_models as lm


def main():
    N = 2000
    max_v = 10000
    rad, thk = 10, 5
    sep = 5
    eta = 1
    use_adaline = False
    maxit = 1000
    dim = 2

    radiuses, radians = Util.generate_random_ring(N, rad, rad + thk, max_v)
    xs, ys, signs = Util.move_bottom_ring_and_assign(
        radiuses, radians, rad + thk / 2.0, -sep
    )
    df = pd.DataFrame({"x1": xs.flatten(), "x2": ys.flatten(), "y": signs.flatten()})
    df["x0"] = 1
    df = df[["x0", "x1", "x2", "y"]]
    positives = df.loc[df["y"] == 1]
    negatives = df.loc[df["y"] == -1]

    figsize = plt.figaspect(1)
    f, ax = plt.subplots(1, 1, figsize=figsize)

    ps = ax.scatter(
        positives[["x1"]].values,
        positives[["x2"]].values,
        marker="+",
        c="g",
        label="+1 labels",
    )
    ns = ax.scatter(
        negatives[["x1"]].values,
        negatives[["x2"]].values,
        marker=r"$-$",
        c="b",
        label="-1 labels",
    )
    print("Number of positive points: ", len(positives))
    print("Number of negatives points: ", len(negatives))

    norm_g, num_its, _ = lm.perceptron(
        df.values, dim, maxit, use_adaline, eta, randomize=False, print_out=True
    )
    x1 = np.arange(-(rad + thk), (rad + thk) + rad + thk / 2)
    norm_g = norm_g / norm_g[-1]
    hypothesis = ax.plot(
        x1, -(norm_g[0] + norm_g[1] * x1), c="g", label="Final Hypothesis"
    )

    w_lin = lm.linear_regression(df[["x0", "x1", "x2"]].values, df["y"].values)
    print("Liner regression coefficients: ", w_lin)
    linear = ax.plot(x1, -(w_lin[0] + w_lin[1] * x1), c="r", label="Linear Regression")

    ax.set_ylabel(r"$x_2$", fontsize=11)
    ax.set_xlabel(r"$x_1$", fontsize=11)
    ax.set_title("Data set size = %s" % N, fontsize=9)
    ax.axis("tight")
    legend_x = 2.0
    legend_y = 0.5
    ax.legend(
        [
            "PLA",
            "Linear Regression",
            "+1 labels",
            "-1 labels",
        ],
        loc="center right",
        bbox_to_anchor=(legend_x, legend_y),
    )
    # ax.set_ylim(bottom=lb, top=ub)
    plt.show()


# Execution starts here
if __name__ == "__main__":
    main()
