from typing import Callable

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


def main():
    lambda_0 = 1
    R_max = 12.5
    q_0 = 2 * np.pi / lambda_0
    NA = 0.01
    zf_zl = R_max / NA

    def target_phase(x: float) -> float:
        return x

    phi = lambda x: phase_function(target_phase=target_phase,
                                   lambda_0=lambda_0,
                                   R_max=R_max,
                                   x=x)
    psi = lambda x_f: real_wave_func_at_focus(x_f=x_f,
                                              q_0=q_0,
                                              zf_zl=zf_zl,
                                              phi=phi,
                                              R_max=R_max)

    vectorized_psi = np.vectorize(psi)
    x_f = np.linspace(-2, 2, 100)
    plt.plot(x_f, vectorized_psi(x_f))
    plt.show()


def real_wave_func_at_focus(x_f: float, q_0: float, zf_zl: float, phi: Callable, R_max: float):
    integrand = lambda x: np.cos(phi(x) - (q_0 * x * x_f / zf_zl))
    result, error = integrate.quad(integrand, -R_max, R_max)
    return result


def phase_function(target_phase: Callable, lambda_0: float, R_max: float, x: float):
    def inner_sine(arg):
        if abs(x) < 1e-3:
            return 4 * np.pi / lambda_0
        return np.sin(4 * np.pi * arg / lambda_0) / x

    integrand = lambda x_tag: inner_sine(arg=x - x_tag) * target_phase(x_tag)
    result, error = integrate.quad(integrand, -R_max, R_max)
    return 1 / np.pi * result


if __name__ == '__main__':
    main()
