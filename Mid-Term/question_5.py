import matplotlib.pyplot as plt
import numpy as np


def f(x1, x2):
    return np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)


def gradient_function(x1, x2):
    expression_1 = np.exp(x1 + 3 * x2 - 0.1)
    expression_2 = np.exp(x1 - 3 * x2 - 0.1)
    expression_3 = np.exp(-x1 - 0.1)

    dx1 = expression_1 + expression_2 - expression_3
    dx2 = 3 * np.exp(x1 + 3 * x2 - 0.1) - np.exp(x1 - 3 * x2 - 0.1)

    return dx1, dx2


def gradient_descent(x1, x2, max_iter, learn_rate):
    x_steps, y_steps = [x1], [x2]
    for i in range(max_iter):
        gradient = gradient_function(x1, x2)
        x1 -= learn_rate * gradient[0]
        gradient = gradient_function(x1, x2)
        x2 -= learn_rate * gradient[1]

        x_steps.append(x1)
        y_steps.append(x2)
    return x_steps, y_steps


def main():
    x1 = np.linspace(-6, 6, 100)
    x2 = np.linspace(-6, 6, 100)
    X1, X2 = np.meshgrid(x1, x2)
    func = np.exp(X1 + 3 * X2 - 0.1) + np.exp(X1 - 3 * X2 - 0.1) + np.exp(-X1 - 0.1)
    k = np.arange(15)

    plt.contour(X1, X2, func, np.arange(20))
    plt.show()

    plt.contour(X1, X2, func, k)
    plt.show()

    x1 = np.linspace(-6, 6, 100)
    x2 = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x1, x2)
    gradient_val = gradient_function(X, Y)
    plt.quiver(X, Y, gradient_val[0], gradient_val[1], gradient_val[1])
    plt.xlabel("x1, x2")
    plt.ylabel("f(x1, x2")
    plt.title("Gradient function")
    plt.show()

    x1, x2 = np.arange(-6, 6, 0.2), np.arange(-6, 6, 0.2)
    X, Y = np.meshgrid(x1, x2)

    x_dir, y_dir = gradient_function(X, Y)[0], gradient_function(X, Y)[1]
    plt.quiver(X, Y, x_dir, y_dir, color="k")
    plt.xlabel("x1"), plt.ylabel("x2")
    plt.title("Gradient descent to optimize f(x1, x2)")

    for i in k:
        plt.contour(X, Y, f(X, Y), [i], colors="b")

    x_steps, y_steps = gradient_descent(-1, -1, 500, 0.15)
    print(f"f(x1, x2) is minimum at: ({x_steps[-1]}, {y_steps[-1]})")
    plt.plot(x_steps, y_steps, "g", linewidth=3)
    plt.show()


if __name__ == "__main__":
    main()
