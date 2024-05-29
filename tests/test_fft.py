import unittest
from src.fft_simulation.fft_simulation import get_fourier_transform, simulate, double_well_potential, get_meshgrid
import numpy as np
import scipy
import matplotlib.pyplot as plt


def delta_function(x, eps):
    return 1.0 / eps if -eps / 2.0 < x < eps / 2.0 else 0


def gaussian_multiplied_by_phase(x, y):
    A = 0.7
    B = 0.3
    gaussian = lambda x, y: np.exp(-(x ** 2 + y ** 2))
    # return (A * np.exp(-100 * x * 1j) + B * np.exp(100 * x * 1j)) * gaussian(x, y)
    # return gaussian(x - 1, y) + gaussian(x + 1,y)
    # epsilon = 0.0004
    # return delta_function(x, epsilon) * delta_function(y, epsilon)
    # return np.cos(x)
    g_squeared_1_0 = lambda x, y: (x**2 + y**2)
    return gaussian(x, y) + np.exp(100 * 1j * x * y * np.sin(x) * np.sin(y))


class TestFFT(unittest.TestCase):
    def setUp(self) -> None:
        self.x_lim = (-100, 100)
        self.y_lim = (-100, 100)
        self.num_points = 1000

    @unittest.skip
    def test_simple_function(self):
        f = lambda x, y: np.exp(1j * x * y * np.exp(-(x ** 2 + y ** 2)))
        K_x, K_y, Z = get_fourier_transform(func=f,
                                            x_lim=self.x_lim,
                                            y_lim=self.y_lim,
                                            num_points=self.num_points)
        simulate(K_x, K_y, np.abs(Z))

    @unittest.skip
    def test_1d_ff(self):
        f = lambda x: np.exp(-x ** 2)
        num_of_points = 10000
        x = np.linspace(-10, 10, num_of_points)
        f_fft = np.abs(scipy.fft.fft(f(x)))
        k = scipy.fft.fftfreq(num_of_points, x[1] - x[0])
        plt.plot(x, f(x))
        plt.plot(k, f_fft)
        plt.xlim(-2, 2)
        plt.ylim(0, 1000)
        plt.show()

    @unittest.skip
    def test_double_well_potential_fft(self):
        x_lim = (-80, 80)
        y_lim = (-80, 80)
        num_points = 1000
        K_x, K_y, Z_fft = get_fourier_transform(func=double_well_potential,
                                                x_lim=x_lim,
                                                y_lim=y_lim,
                                                num_points=num_points)

        X, Y = get_meshgrid(x_lim=x_lim, y_lim=y_lim, num_points=num_points)
        Z = scipy.fft.fft2(Z_fft)

        # simulate(X, Y, np.abs(Z))
        Z_fft_log = np.log(np.abs(Z_fft))
        Z_fft_log[Z_fft_log < 0] = 0
        simulate(K_x, K_y, Z_fft_log)

    @unittest.skip
    def test_double_well_potential_plot(self):
        X, Y = get_meshgrid(x_lim=self.x_lim, y_lim=self.y_lim, num_points=self.num_points)
        F = np.vectorize(double_well_potential)
        simulate(X, Y, F(X, Y))

    def test_input_function(self):
        K_x, K_y, Z_fft = get_fourier_transform(func=gaussian_multiplied_by_phase,
                                                x_lim=self.x_lim,
                                                y_lim=self.y_lim,
                                                num_points=self.num_points)
        print(Z_fft)
        simulate(K_x, K_y, np.abs(Z_fft))

    @unittest.skip
    def test_how_input_func_should_look(self):
        X, Y = get_meshgrid(x_lim=self.x_lim, y_lim=self.y_lim, num_points=self.num_points)
        Z = np.vectorize(gaussian_multiplied_by_phase)(X, Y)
        simulate(X, Y, np.abs(Z))
