import numpy as np

from simulators.base import BaseSimulator

EPS = 1e-8


class Branin2d(BaseSimulator):
    """
    Branin function [https://www.sfu.ca/~ssurjano/branin.html]
    With the modification by Forrester et al. (2008).
    The std. dev. is 57
    """

    def __init__(self, n_points=100, noise_sigma=57, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        a = 1
        b = 5.1 / (4 * np.power(np.pi, 2))
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        mean = a * np.power(x[:, 1] - b * np.power(x[:, 0], 2) + c * x[:, 0] - r, 2) + \
            s * (1 - t) * np.cos(x[:, 0]) + s + 5 * x[:, 0]
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(-5, 10, self.n_points)
        x2 = np.linspace(0, 15, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class ChengAndSandu(BaseSimulator):
    """
    Cheng & Sandu (2010) [https://www.sfu.ca/~ssurjano/chsan10.html]
    The std. dev. is 0.37
    """

    def __init__(self, n_points=100, noise_sigma=0.37, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        mean = np.cos(x[:, 0] + x[:, 1]) * np.exp(x[:, 0] * x[:, 1])
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class CurrinExp(BaseSimulator):
    """
    Currin et al. (1988) [https://www.sfu.ca/~ssurjano/curretal88exp.html]
    The std. dev. is 2.6
    """

    def __init__(self, n_points=100, noise_sigma=2.6, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        term1 = (1 - np.exp(-1 / (2 * x[:, 1] + EPS)))
        term2 = 2300 * np.power(x[:, 0], 3) + 1900 * np.power(x[:, 0], 2) + 2092 * x[:, 0] + 60
        term3 = 100 * np.power(x[:, 0], 3) + 500 * np.power(x[:, 0], 2) + 4 * x[:, 0] + 20
        mean = term1 * term2 / term3
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class CurrinPoly(BaseSimulator):
    """
    Currin et al. (1988) [https://www.sfu.ca/~ssurjano/curretal91.html]
    The std. dev. is 2.3
    """

    def __init__(self, n_points=100, noise_sigma=2.3, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        mean = 4.9 + 21.15 * x[:, 0] - 2.17 * x[:, 1] - 15.88 * np.power(x[:, 0], 2) \
                    - 1.38 * np.power(x[:, 1], 2) - 5.26 * x[:, 0] * x[:, 1]
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class CurrinSinus(BaseSimulator):
    """
    Function: https://www.sfu.ca/~ssurjano/curretal88sin.html
    The std. dev. is 0.70
    """

    def __init__(self, n_points=101, noise_sigma=0.70, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)
        self.n_inputs = 1
        self.n_outputs = 1

    def mean(self, x): return np.sin(2 * np.pi * (x - 0.1))

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self): return np.linspace(0, 1, self.n_points).reshape(-1, 1)


class CurrinSurvival(BaseSimulator):
    """
    Currin et al. (1988) Survival function [https://www.sfu.ca/~ssurjano/curretal88sur.html]
    The std. dev. is 0.2047772
    """

    def __init__(self, n_points=101, noise_sigma=0.20, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)
        self.n_inputs = 1
        self.n_outputs = 1

    def mean(self, x): return 1 - np.exp(-1 / (2 * x + EPS))

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self): return np.linspace(0, 1, self.n_points).reshape(-1, 1)


class Dette8d(BaseSimulator):
    """
    DETTE & PEPELYSHEV (2010) 8-DIMENSIONAL FUNCTION [https://www.sfu.ca/~ssurjano/detpep108d.html]
    The std. dev. is 9.3.
    """

    def __init__(self, n_points=100, noise_sigma=9.3, n_inputs=8, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        term1 = 4 * np.power(x[:, 0] - 2 + 8 * x[:, 1] - 8 * np.power(x[:, 1], 2), 2)
        term2 = np.power(3 - 4 * x[:, 1], 2)
        term3 = 16 * np.sqrt(x[:, 2] + 1) * np.power(2 * x[:, 2] - 1, 2)

        inner = x[:, 2:8]
        inner = np.repeat(inner, repeats=5, axis=0).reshape(-1, 5, 6)
        inner = np.tril(inner, k=1)
        inner = np.sum(inner, axis=2)
        outer = np.sum(np.arange(4, 9) * np.log(1 + inner), axis=1)

        mean = term1 + term2 + term3 + outer
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        search_space = np.random.rand(1000000, 8)
        return search_space


class DetteExp(BaseSimulator):
    """
    DETTE & PEPELYSHEV (2010) EXPONENTIAL FUNCTION [https://www.sfu.ca/~ssurjano/detpep10exp.html]
    The std. dev. is 6.8
    """

    def __init__(self, n_points=100, noise_sigma=6.8, n_inputs=3, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        term1 = np.exp(-2 / np.power(x[:, 0] + EPS, 1.75))
        term2 = np.exp(-2 / np.power(x[:, 1] + EPS, 1.5))
        term3 = np.exp(-2 / np.power(x[:, 2] + EPS, 1.25))
        mean = 100 * (term1 + term2 + term3)
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        x3 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, x3, indexing='ij')).T.reshape(-1, 3)
        return search_space


class DettePoly(BaseSimulator):
    """
    DETTE & PEPELYSHEV (2010) CURVED FUNCTION [https://www.sfu.ca/~ssurjano/detpep10curv.html]
    The std. dev. is 7.2
    """

    def __init__(self, n_points=100, noise_sigma=7.2, n_inputs=3, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        term1 = 4 * np.power(x[:, 0] - 2 + 8 * x[:, 1] - 8 * np.power(x[:, 1], 2), 2)
        term2 = np.power(3 - 4 * x[:, 1], 2)
        term3 = 16 * np.sqrt(x[:, 2] + 1) * np.power(2 * x[:, 2] - 1, 2)
        mean = term1 + term2 + term3
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        x3 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, x3, indexing='ij')).T.reshape(-1, 3)
        return search_space


class Forrester1d(BaseSimulator):
    """
    Forrester et al. (2008) [https://www.sfu.ca/~ssurjano/forretal08.html]
    The std. dev. of the data is 4.5
    """
    def __init__(self, n_points=101, noise_sigma=4.5, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x): return np.power(6 * x - 2, 2) * np.sin(12 * x - 4)

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self): return np.linspace(0, 1, self.n_points).reshape(-1, 1)


class Franke(BaseSimulator):
    """
    Franke et al. (1979) [https://www.sfu.ca/~ssurjano/franke2d.html]
    The std. dev. is 0.29
    """

    def __init__(self, n_points=100, noise_sigma=0.29, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        term1 = 0.75 * np.exp(-np.power(9 * x[:, 0] - 2, 2) / 4 - np.power(9 * x[:, 1] - 2, 2) / 4)
        term2 = 0.75 * np.exp(-np.power(9 * x[:, 0] + 1, 2) / 49 - (9 * x[:, 1] + 1) / 10)
        term3 = 0.50 * np.exp(-np.power(9 * x[:, 0] - 7, 2) / 4 - np.power(9 * x[:, 1] - 3, 2) / 4)
        term4 = -0.2 * np.exp(-np.power(9 * x[:, 0] - 4, 2) - np.power(9 * x[:, 1] - 7, 2) )
        mean = term1 + term2 + term3 + term4
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class Friedman(BaseSimulator):
    """
    Friedman [https://www.sfu.ca/~ssurjano/fried.html]
    The std. dev. is 4.9
    """

    def __init__(self, n_points=100, noise_sigma=4.9, n_inputs=5, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        mean = 10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * np.power(x[:, 2] - 0.5, 2) + 10 * x[:, 3] + 5 * x[:, 4]
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        """
        This is too big!!
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        x3 = np.linspace(0, 1, self.n_points)
        x4 = np.linspace(0, 1, self.n_points)
        x5 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, x3, x4, x5, indexing='ij')).T.reshape(-1, 5)
        Instead, we cut the size of the space at 10^6 data points
        """
        search_space = np.random.rand(1000000, 5)
        return search_space


class GramacyAndLee1d(BaseSimulator):
    """
    Gramacy and Lee (2012) [https://www.sfu.ca/~ssurjano/grlee12.html]
    The std. dev is 1.31.
    """

    def __init__(self, n_points=101, noise_sigma=1.31, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x): return np.sin(10 * np.pi * x) / (2 * x) + np.power(x - 1, 4)

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self): return np.linspace(0.5, 2.5, self.n_points).reshape(-1, 1)


class GramacyAndLee2d(BaseSimulator):
    """
    Gramacy and Lee (2008) [https://www.sfu.ca/~ssurjano/grlee08.html]
    The std. dev. is 0.078
    """

    def __init__(self, n_points=100, noise_sigma=0.078, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        mean = x[:, 0] * np.exp(-np.power(x[:, 0], 2) - np.power(x[:, 1], 2))
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(-2, 6, self.n_points)
        x2 = np.linspace(-2, 6, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class GramacyAndLee6d(BaseSimulator):
    """
    Gramacy & Lee (2009) [https://www.sfu.ca/~ssurjano/grlee09.html]
    The std. dev. is 0.71
    """

    def __init__(self, n_points=100, noise_sigma=0.71, n_inputs=6, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        # the last two inputs are not active
        mean = np.exp(np.sin(np.power(0.9 * (x[:, 0] + 0.48), 10))) + x[:, 1] * x[:, 2] + x[:, 3]
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        search_space = np.random.rand(1000000, 6)
        return search_space


class Hartmann6d(BaseSimulator):
    """
    The original function has no noise. Following Picheny, we add 5% of the standard deviation of the function
    as noise, e.g., 0.019. The std. dev. is 0.38.
    All references can be found here: https://www.sfu.ca/~ssurjano/hart6.html
    """

    def __init__(self, n_points=100, noise_sigma=0.0192, n_inputs=6, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        alpha = np.array([1, 1.2, 3, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [.05, 10, 17, .1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, .05, 10, .1, 14]
                      ])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])

        # change x to have to size [batch_size, 1, features]
        x = x.reshape(x.shape[0], 6, 1).transpose(0, 2, 1)
        # add batch_dim to P
        P = P.reshape(4, 6, 1).transpose(2, 0, 1)
        inner = np.sum(A * np.power(x - P, 2), axis=2)
        # inner = np.sum(np.power(x - P, 2), axis=2)
        mean = - np.sum(alpha * np.exp(- inner), axis=1)
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        """
        This is too big!!
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        x3 = np.linspace(0, 1, self.n_points)
        x4 = np.linspace(0, 1, self.n_points)
        x5 = np.linspace(0, 1, self.n_points)
        x6 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, x3, x4, x5, x6, indexing='ij')).T.reshape(-1, 6)
        Instead, we cut the size of the space at 10^6 data points
        """
        search_space = np.random.rand(1000000, 6)
        return search_space


class Higdon1d(BaseSimulator):
    """
    Higdon (2002), Gramacy and Lee (2008) [https://www.sfu.ca/~ssurjano/hig02grlee08.html]
    The std. dev. is 0.60.
    """

    def __init__(self, n_points=101, noise_sigma=0.60, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)
        self.n_inputs = 1
        self.n_outputs = 1

    def mean(self, x):
        mean = np.piecewise(x,
                            [x < 10, x >= 10],
                            [lambda xx: np.sin(np.pi * xx / 5) + 0.2 * np.cos(4 * np.pi * xx / 5),
                             lambda xx: xx / 10 - 1])
        return mean

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self): return np.linspace(0, 20, self.n_points).reshape(-1, 1)


class Higdon1dSimple(BaseSimulator):
    """
    Higdon (2002) function: https://www.sfu.ca/~ssurjano/hig02.html
    The std. dev. is 0.72
    """

    def __init__(self, n_points=101, noise_sigma=0.72, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)
        self.n_inputs = 1
        self.n_outputs = 1

    def mean(self, x):
        return np.sin(np.pi * x / 5) + 0.2 * np.cos(4 * np.pi * x / 5)

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self): return np.linspace(0, 10, self.n_points).reshape(-1, 1)


class HolsclawSinus(BaseSimulator):
    """
    Holsclaw et al. (2013) [https://www.sfu.ca/~ssurjano/holsetal13sin.html]
    The std. dev. of the data is 0.37
    """
    def __init__(self, n_points=101, noise_sigma=0.37, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x): return x * np.sin(x) / 10

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self): return np.linspace(0, 10, self.n_points).reshape(-1, 1)


class HolsclawLog(BaseSimulator):
    """
    Holsclaw et al. (2013) [https://www.sfu.ca/~ssurjano/holsetal13log.html]
    The std. dev. of the data is 0.48
    """
    def __init__(self, n_points=101, noise_sigma=0.48, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x): return np.log(1 + x)

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self): return np.linspace(0, 5, self.n_points).reshape(-1, 1)


class Ishigami3d(BaseSimulator):
    """
    Ishigami function
    The original function has no noise. Following Picheny, we add 5% of the standard deviation of the function
    as noise, e.g., 0.187. The std. dev. is 3.76.
    All references can be found here: https://www.sfu.ca/~ssurjano/ishigami.html
    """

    def __init__(self, n_points=100, noise_sigma=0.187, n_inputs=3, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        a = 7
        b = 0.1
        mean = np.sin(x[:, 0]) + a * np.power(np.sin(x[:, 1]), 2) + b * np.power(x[:, 2], 4) * np.sin(x[:, 0])
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(-np.pi, np.pi, self.n_points)
        x2 = np.linspace(-np.pi, np.pi, self.n_points)
        x3 = np.linspace(-np.pi, np.pi, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, x3, indexing='ij')).T.reshape(-1, 3)
        return search_space


class Ishigami3d_b0(BaseSimulator):
    def __init__(self, n_points=100, noise_sigma=0.187, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        a = 7
        mean = np.sin(x[:, 0]) + a * np.power(np.sin(x[:, 1]), 2)
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(-np.pi, np.pi, self.n_points)
        x2 = np.linspace(-np.pi, np.pi, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class Ishigami3d_a0(BaseSimulator):
    def __init__(self, n_points=100, noise_sigma=0.187, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        b = 0.1
        mean = np.sin(x[:, 0]) + b * np.power(x[:, 1], 4) * np.sin(x[:, 0])
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(-np.pi, np.pi, self.n_points)
        x2 = np.linspace(-np.pi, np.pi, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class Ishigami3d_ab0(Ishigami3d):
    def __init__(self, n_points=100, noise_sigma=0.187, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x): return np.sin(x)

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        return np.linspace(-np.pi, np.pi, self.n_points).reshape(-1, 1)


class LimNonPoly(BaseSimulator):
    """
    Lim et al. (2002) [https://www.sfu.ca/~ssurjano/limetal02non.html]
    The std. dev. is 2.0
    """

    def __init__(self, n_points=100, noise_sigma=2.0, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        mean = (1 / 6) * ((30 + 5* x[:, 0] * np.sin(5 * x[:, 0])) * (4 + np.exp(-5 * x[:, 1])) - 100)
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class LimPoly(BaseSimulator):
    """
    Lim et al. (2002) [https://www.sfu.ca/~ssurjano/limetal02pol.html]
    The std. dev. is 1.7
    """

    def __init__(self, n_points=100, noise_sigma=1.7, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        mean = 9 + (5 / 2) * x[:, 0] \
               - (35 / 2) * x[:, 1] \
               + (5 / 2) * x[:, 0] * x[:, 1] \
               + 19 * np.power(x[:, 1], 2) \
               - (15 /2) * np.power(x[:, 0], 3)\
               - (5 / 2) * x[:, 0] * np.power(x[:, 1], 2) \
               - (11 / 2) * np.power(x[:, 1], 4) \
               + np.power(x[:, 0], 3) * np.power(x[:, 1], 2)
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class Motorcycle(BaseSimulator):
    """
    Mean and stddev list are fitted with an heteroscedastic GP (inducing points)
    """

    def __init__(self, n_points=None, noise_sigma=None, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)
        self.n_inputs = 1
        self.n_outputs = 1

    def convert_x_to_list(self, x):
        x = [np.where(self.search_space() == int(tmp_x))[0][0] for tmp_x in x]
        # x = [x] if x.shape == () else x
        return [int(x) for x in x]

    def mean(self, x):
        gp_mean = [0.5107, 0.5032, 0.4939, 0.4843, 0.4765, 0.4718, 0.4713, 0.4748, 0.4805, 0.4856,
                   0.4870, 0.4824, 0.4724, 0.4604, 0.4523, 0.4543, 0.4695, 0.4953, 0.5216, 0.5319,
                   0.5067, 0.4284, 0.2868, 0.0825, -0.1722, -0.4559, -0.7433, -1.0112, -1.2435, -1.4339,
                   -1.5851, -1.7049, -1.8015, -1.8788, -1.9333, -1.9549, -1.9298, -1.8453, -1.6947, -1.4804,
                   -1.2140, -0.9140, -0.6013, -0.2950, -0.0080, 0.2536, 0.4900, 0.7046, 0.9006, 1.0778,
                   1.2317, 1.3542, 1.4362, 1.4710, 1.4573, 1.4012, 1.3147, 1.2139, 1.1143, 1.0277,
                   0.9589, 0.9056, 0.8605, 0.8146, 0.7605, 0.6959, 0.6241, 0.5533, 0.4935, 0.4538,
                   0.4395, 0.4501, 0.4795, 0.5173, 0.5507, 0.5678, 0.5600, 0.5241, 0.4632, 0.3869,
                   0.3092, 0.2456, 0.2098, 0.2103, 0.2482, 0.3164, 0.4015, 0.4868, 0.5568, 0.6006,
                   0.6147, 0.6033, 0.5769, 0.5487, 0.5311, 0.5324, 0.5549, 0.5950, 0.6441, 0.6919,
                   0.7285]
        return np.array([gp_mean[x] for x in self.convert_x_to_list(x)]).reshape(-1, 1)

    def stddev(self, x):
        gp_stddev = [0.0477, 0.0400, 0.0463, 0.0514, 0.0529, 0.0502, 0.0438, 0.0366, 0.0330, 0.0328, 0.0327,
                     0.0318, 0.0315, 0.0331, 0.0363, 0.0433, 0.0560, 0.0702, 0.0780, 0.0733, 0.0615, 0.0799,
                     0.1468, 0.2393, 0.3408, 0.4382, 0.5196, 0.5747, 0.5959, 0.5800, 0.5295, 0.4540, 0.3712,
                     0.3067, 0.2822, 0.2915, 0.3074, 0.3135, 0.3156, 0.3360, 0.3899, 0.4658, 0.5385, 0.5873,
                     0.6045, 0.5975, 0.5860, 0.5901, 0.6133, 0.6377, 0.6380, 0.5951, 0.5022, 0.3669, 0.2227,
                     0.1980, 0.3483, 0.5319, 0.6956, 0.8186, 0.8886, 0.8987, 0.8470, 0.7373, 0.5789, 0.3876,
                     0.1898, 0.1067, 0.2472, 0.3799, 0.4620, 0.4847, 0.4492, 0.3643, 0.2463, 0.1253, 0.1068,
                     0.2006, 0.2817, 0.3230, 0.3195, 0.2769, 0.2092, 0.1377, 0.0911, 0.0856, 0.0919, 0.0940,
                     0.1116, 0.1568, 0.2117, 0.2548, 0.2730, 0.2623, 0.2282, 0.1857, 0.1545, 0.1445, 0.1445,
                     0.1437, 0.1542]
        return np.array([gp_stddev[x] for x in self.convert_x_to_list(x)]).reshape(-1, 1)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        # The search space must be with 101 values in order for the motorcycle simulator to work..
        return np.linspace(0, 100, 101).reshape(-1, 1)


class McCormick(BaseSimulator):
    """
    McCormick function
    With the modification by Forrester et al. (2008).
    The original function has no noise. Following Picheny, we add 5% of the standard deviation of the modified objective
    function as noise, e.g., XX. The std. dev. is XX.
    All references can be found here: https://www.sfu.ca/~ssurjano/mccorm.html
    """

    def __init__(self, n_points=100, noise_sigma=1, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        mean = np.sin(x[:, 0] + x[:, 1]) + np.power(x[:, 0]-x[:, 1], 2) - \
               1.5 * x[:, 0] + 2.5 * x[:, 1] + 1
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(-1.5, 4, self.n_points)
        x2 = np.linspace(-3, 4, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space


class Santner(BaseSimulator):
    """
    Santner et al. (2003): [https://www.sfu.ca/~ssurjano/santetal03dc.html]
    The std. dev is 0.41.
    """

    def __init__(self, n_points=101, noise_sigma=0.41, n_inputs=1, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x): return np.exp(-1.4 * x) * np.cos(3.5 * np.pi * x)

    def stddev(self, x): return self.noise_sigma * np.ones_like(x)

    def noise(self, x): return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self): return np.linspace(0, 1, self.n_points).reshape(-1, 1)


class Welch(BaseSimulator):
    """
    Welch et al. (1992) [https://www.sfu.ca/~ssurjano/welchetal92.html]
    The std. dev. is 2.1
    """

    def __init__(self, n_points=100, noise_sigma=2.1, n_inputs=20, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)

    def mean(self, x):
        # the last two inputs are not active
        mean = (5 * x[:, 11]) / (1 + x[:, 0]) + 5 * np.power(x[:, 3] - x[:, 19], 2) + x[:, 4] \
               + 40 * np.power(x[:, 18], 3) - 5 * x[:, 18] + 0.05 * x[:, 1] + 0.08 * x[:, 2] - 0.03 * x[:, 5] \
               + 0.03 * x[:, 6] - 0.09 * x[:, 8] - 0.01 * x[:, 9] - 0.07 * x[:, 10] + 0.25 * np.power(x[:, 12], 2) \
               - 0.04 * x[:, 13] + 0.06 * x[:, 14] - 0.01 * x[:, 16] - 0.03 * x[:, 17]
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        rng = np.random.default_rng(seed=42)
        search_space = rng.uniform(low=-0.5, high=0.5, size=(1000000, 20))
        return search_space


class Zhou(BaseSimulator):
    """
    Zhou (1998) [https://www.sfu.ca/~ssurjano/zhou98.html]
    The std. dev. is 1.7
    """

    def __init__(self, n_points=100, noise_sigma=1.7, n_inputs=2, n_outputs=1):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs)
        self.d = 2

    def mean(self, x):
        def phi(x, d=self.d):
            return np.power(2 * np.pi, -d / 2) * np.exp(-0.5 * np.linalg.norm(x, 2, axis=1)**2)
        mean = np.power(10, self.d) / 2 * (phi(10 * (x - 1/3)) + phi(10 * (x - 2/3)))
        return mean.reshape(-1, 1)

    def stddev(self, x):
        stddev = self.noise_sigma * np.ones_like(x[:, 0])
        return stddev.reshape(-1, 1)

    def noise(self, x):
        return np.random.normal(loc=0, scale=1, size=[x.shape[0], self.n_outputs])

    def search_space(self):
        x1 = np.linspace(0, 1, self.n_points)
        x2 = np.linspace(0, 1, self.n_points)
        search_space = np.array(np.meshgrid(x1, x2, indexing='ij')).T.reshape(-1, 2)
        return search_space
