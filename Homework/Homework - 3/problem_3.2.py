from util import Util
import numpy as np
import linear_models as lm
import matplotlib.pyplot as plt
import pandas as pd


def main():
    N = 2000
    max_v = 10000
    rad, thk = 10, 5
    eta = 1
    use_adaline=False
    maxit = 1000
    dim = 2
    seps = np.arange(0.2, 5.2, 0.2)

    radiuses, radians = Util.generate_random_ring(N, rad, rad+thk, max_v)
    its, threoticals_ts = [], []
    for sep in seps:
        xs, ys, signs = Util.move_bottom_ring_and_assign(radiuses, radians, rad + thk/2.0, -sep)
        df = pd.DataFrame({'x1':xs.flatten(), 'x2':ys.flatten(), 'y':signs.flatten()})
        df['x0'] = 1
        df = df[['x0','x1','x2','y']]
        positives = df.loc[df['y']==1]
        negatives = df.loc[df['y']==-1]
        norm_g, num_its, theoretical_t = lm.perceptron(df.values, dim, maxit, use_adaline, eta,
                                                    randomize=False, print_out = True)
        its.append(num_its)
        threoticals_ts.append(theoretical_t)

    plt.plot(seps, its, marker='.')
    plt.show()


if __name__ == "__main__":
    main()
