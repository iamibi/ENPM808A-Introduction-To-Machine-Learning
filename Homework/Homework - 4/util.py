from matplotlib.colors import ListedColormap
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


class Util:
    @classmethod
    def move_bottom_ring_and_assign(cls, radiuses, radians, diffx, diffy):
        xs = radiuses * np.cos(radians)
        ys = radiuses * np.sin(radians)
        signs = np.ones(len(xs))

        for idx, r in enumerate(radiuses):
            rad = radians[idx]
            xi, yi = xs[idx], ys[idx]
            if math.pi < rad < 2 * math.pi:
                xs[idx] = xi + diffx
                ys[idx] = yi + diffy
                signs[idx] = -1
        return xs, ys, signs

    @classmethod
    def generate_random_numbers01(cls, N, dim, max_v=10000):
        random_ints = np.random.randint(max_v, size=(N, dim))
        init_lb = 0
        return (random_ints - init_lb) / (max_v - 1 - init_lb)

    @classmethod
    def generate_random_numbers(cls, N, dim, max_v, lb, ub):
        zero_to_one_points = cls.generate_random_numbers01(N, dim, max_v)
        res = lb + (ub - lb) * zero_to_one_points
        return res

    @classmethod
    def generate_random_ring(cls, N, r1, r2, max_v):
        radiuses = cls.generate_random_numbers(N, 1, max_v, r1, r2)
        radians = cls.generate_random_numbers(N, 1, max_v, 0, 2.0 * math.pi)
        return radiuses, radians

    @classmethod
    def get_grid(cls, x1_min, x1_max, x2_min, x2_max, step=0.02):
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
        return xx1, xx2

    @classmethod
    def plot_decision_boundaries(cls, xx1, xx2, num_cats, classifier, transformer=None, alpha=0.4):
        colors = ("blue", "red", "green", "yellow")
        cmap = ListedColormap(colors[:num_cats])

        Xgrid = np.array([xx1.ravel(), xx2.ravel()]).T
        if transformer:
            Xgrid = transformer(Xgrid)
        y = classifier.predict(Xgrid)
        y = y.reshape(xx1.shape)
        plt.contourf(xx1, xx2, y, alpha=alpha, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

    @classmethod
    def plt_plot(
        cls,
        xs,
        ys,
        plot_func,
        colors,
        markers,
        labels,
        title=None,
        yscale=None,
        ylb=None,
        yub=None,
        xlb=None,
        xub=None,
        xlabel=None,
        ylabel=None,
        legends=None,
        legendx=None,
        legendy=None,
        marker_sizes=np.ones(5),
    ):

        # Plot a subplot graph
        for x, y, c, m, label, s in zip(xs, ys, colors, markers, labels, marker_sizes):
            if plot_func == "plot":
                plt.plot(x, y, color=c, marker=m, label=label)
            elif plot_func == "scatter":
                plt.scatter(x, y, color=c, marker=m, s=s, label=label)

        if yscale:
            plt.yscale(yscale)
        if title:
            plt.title(title)
        if ylb or yub:
            plt.ylim(bottom=ylb, top=yub)
        if xlb or xub:
            plt.xlim(left=xlb, right=xub)

        if ylabel:
            plt.ylabel(ylabel, fontsize=11)
        if xlabel:
            plt.xlabel(xlabel, fontsize=11)

        if legends:
            if legendx and legendy:
                legend_x = legendx
                legend_y = legendy
                plt.legend(legends, loc="center right", bbox_to_anchor=(legend_x, legend_y))
            else:
                plt.legend()

        plt.axis("tight")
        plt.show()

    @classmethod
    def calc_error(cls, w, xs, ys):
        c = 0
        for x, y in zip(xs, ys):
            prod = np.dot(w.T, x) * y
            if prod < 0:
                c += 1
        return c / len(ys)

    @classmethod
    def polynomial_transform(cls, q, X):
        poly = PolynomialFeatures(q)
        return poly.fit_transform(X)

    @classmethod
    def pseudo_inv(cls, X, y, reg):
        Xa = X
        t = np.matmul(Xa.transpose(), Xa)
        pseudo_inv = np.matmul(np.linalg.inv(t + reg), Xa.transpose())
        w = np.matmul(pseudo_inv, y.reshape((-1, 1)))
        return w.reshape(-1, 1)

    @classmethod
    def generate_data(cls, N, x1_lb=0, x1_ub=1, x2_lb=-1, x2_ub=1):
        dim = 1
        max_v = 10000
        x1 = cls.generate_random_numbers(N, dim, max_v, x1_lb, x1_ub)
        x2 = cls.generate_random_numbers(N, dim, max_v, x2_lb, x2_ub)
        df = pd.DataFrame({"x1": x1.flatten(), "x2": x2.flatten()})
        df["y"] = np.sign(df["x2"])
        return df
