from util import Util
import numpy as np
import math
import cvxopt


def perceptron(points, dim, max_it=100, use_adaline=False, eta=1, randomize=False, print_out=True):
    w = np.zeros(dim + 1)
    xs, ys = points[:, : dim + 1], points[:, dim + 1]
    num_points = points.shape[0]
    for it in range(max_it):
        correctly_predicted_ids = set()
        idxs = np.arange(num_points)
        if randomize:
            idxs = np.random.choice(np.arange(num_points), num_points, replace=False)
        for idx in idxs:
            x, y = xs[idx], ys[idx]
            st = np.dot(w.T, x)
            prod = st * y  # np.dot(w.T, x)*y
            if prod < -100:  # avoid out of bound error
                st = -100
            threshold = 1 if use_adaline else 0
            st = st if use_adaline else 0
            if prod <= threshold:
                w = w + eta * (y - st) * x
                break  # PLA picks one example at each iteration
            else:
                correctly_predicted_ids.add(idx)
        if len(correctly_predicted_ids) == num_points:
            break

    rou = math.inf
    R = 0
    c = 0
    for x, y in zip(xs, ys):
        prod = np.dot(w.T, x) * y
        if prod > 0:
            c += 1
        if prod < rou:
            rou = prod
        abs_x = np.linalg.norm(x)
        if abs_x > R:
            R = abs_x
    theoretical_t = (R**2) * (np.linalg.norm(w) ** 2) / rou / rou

    if print_out is True:
        print(f"Final correctness = {c}. Total iteration = {it}\nFinal w = {w}")
    return w, it, theoretical_t


def linear_regression(X, y):
    XT = np.transpose(X)
    x_pseudo_inv = np.matmul(np.linalg.inv(np.matmul(XT, X)), XT)
    w = np.matmul(x_pseudo_inv, y)
    return w


class SVM:
    def __init__(self, is_soft=False):
        self.is_soft = is_soft  # hard margin or soft margin
        self.w = None
        self.b = None

    def margin(self):
        w_norm = np.linalg.norm(self.w)
        return 1.0 / w_norm

    def fit(self, X, y, solver=None):
        # This algo is implented according to the "Linear Hard-Margin SVM with QP"
        # on page 8-10 in Chapter 8 from "Learning From Data: A Short Course"

        # Apply quadratic programming to solve this
        # solver is None or 'mosek'

        # N.B. X doesn't have the 1 column here
        N, d = X.shape
        p = np.zeros((d + 1, 1))
        c = np.ones((N, 1))
        dzeros = np.zeros((d, 1))
        Qup = np.hstack((np.zeros((1, 1)), dzeros.transpose()))
        Qdown = np.hstack((dzeros, np.identity(d)))
        Q = np.vstack((Qup, Qdown))
        A = np.hstack((c, X))
        A = np.multiply(A, y.reshape(-1, 1))

        P = cvxopt.matrix(Q)
        q = cvxopt.matrix(p)
        G = cvxopt.matrix(-A)
        h = cvxopt.matrix(-c)
        res = cvxopt.solvers.qp(P, q, G, h, solver=solver, options={"show_progress": False})
        if res["status"] != "optimal":
            print(f"Couldn't find optimal solution!")
            print(f"Final status: {res['status']}")

        u = np.array(res["x"])
        self.b = u[0, :]
        self.w = u[1:, :]

    def predict(self, X):
        N, _ = X.shape
        c = np.ones((N, 1))
        A = np.hstack((c, X))
        u = np.vstack((self.b, self.w))
        pred = np.sign(np.matmul(A, u))
        return pred

    def calc_error(self, X, y):
        N, _ = X.shape
        c = np.ones((N, 1))
        XX = np.hstack((c, X))
        u = np.vstack((self.b, self.w))
        error = Util.calc_error(u, XX, y)
        return error


class LinearRegression:
    def __init__(self, reg_param, poly_degree=None, solver=None, to_classify=False):
        self.w = None
        self.reg_param = reg_param
        self.poly_degree = poly_degree  # If apply polynomial transformation first
        self.solver = solver
        self.to_classify = to_classify

    def fit(self, X, y):
        Z = X
        if self.poly_degree:
            Z = Util.polynomial_transform(self.poly_degree, X)

        self.w = Util.pseudo_inv(Z, y, self.reg_param)

    def predict(self, X):
        Z = X
        if self.poly_degree:
            Z = Util.polynomial_transform(self.poly_degree, X)
        y_pred = np.matmul(Z, self.w)
        if self.to_classify:
            y_pred = np.sign(y_pred)
        return y_pred

    def calc_error(self, X, y):
        y_pred = self.predict(X)
        err = y_pred - y
        error = np.matmul(err.transpose(), err).flatten() / y.shape[0]
        return error
