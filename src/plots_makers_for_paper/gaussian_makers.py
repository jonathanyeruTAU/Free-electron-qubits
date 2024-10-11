import matplotlib.pyplot as plt
import numpy as np


def gaussian(mu, sigma):
    return lambda x: np.exp((-((x - mu) ** 2)) / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)


def make_01_plots():
    left_gaussian = gaussian(mu=-1, sigma=0.3)
    right_gaussian = gaussian(mu=1, sigma=0.3)

    hadamard = lambda x: 1 / np.sqrt(2) * (gaussian(mu=-1, sigma=0.3)(x) + gaussian(mu=1, sigma=0.3)(x))

    x = np.linspace(-2, 2, 1000)

    plt.plot(x, left_gaussian(x), label=r"$|0\rangle$")
    plt.plot(x, right_gaussian(x), label=r"$|1\rangle$")
    plt.plot(x, hadamard(x), label=r"$\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$")
    plt.legend(loc='best', fontsize=14)  # Position legend in the upper left
    plt.axhline(0, color='black', linewidth=1)  # Horizontal axis line
    plt.axvline(0, color='black', linewidth=1)  # Vertical axis line
    plt.ylim(0, 1.75)
    plt.ylabel(r"$|\psi(x)|^2/\sigma$", fontsize=14)
    plt.xlabel(r"$x$", fontsize=14)
    plt.xticks([-1, 0, 1], [r'$-\mu$', '0', r'$\mu$'])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    make_01_plots()
