import numpy as np
from scipy.optimize import minimize
from functools import partial
import math


def find_newton_direction(u,v,norm):
    multipler = norm**2 / (u**2 + v**2)
    return u * np.sqrt(multipler), v*np.sqrt(multipler)


def calc_E(x):
    u, v = x[0], x[1]
    return np.exp(u) + np.exp(2.0*v) + np.exp(u*v) + u**2 - 3*u*v + 4*v**2 - 3*u - 5*v


def calc_E_with_length(theta, norm):
    u, v = norm*math.sin(theta), norm*math.cos(theta)
    return np.exp(u) + np.exp(2.0*v) + np.exp(u*v) + u**2 - 3*u*v + 4*v**2 - 3*u - 5*v


def main():
    # Note, we only compare values at point (u=0, v=0)
    print('E(u+du,v+dv) with (du,dv) from E_1 approximation: ', calc_E([1 / np.sqrt(13), 3 / np.sqrt(13) / 2]))
    new_u, new_v = find_newton_direction(30 / 32, 13 / 32, 0.5)
    print('Newton direction: (', new_u, ',', new_v, ')')
    new_E = calc_E([new_u, new_v])
    print('E(u+du,v+dv) = ', new_E)

    calc_E_half = partial(calc_E_with_length, norm=0.5)
    x0 = 0
    res = minimize(calc_E_half, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    print('Optimal direction by minimizing E(u+du,v+dv): ', 0.5 * math.sin(res.x), 0.5 * math.cos(res.x))
    print('Minimal E(u+du, v+dv): ', res.fun)


if __name__ == "__main__":
    main()
