from typing import Callable

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool




def main():
    R_max = 10 * 1e-6  # 10 micro meters
    lambda_0 = R_max / 12.5
    NA = 0.01
    lambda_e_perpendicular = lambda_0 / NA

    x_f = np.linspace(-3 * lambda_e_perpendicular, 3 * lambda_e_perpendicular, 500)

    focus_x = get_function_focus(target_phase=lambda x: 2 * x / R_max, R_max=R_max)
    plt.plot(x_f, focus_x(x_f), label="2x / R_max")

    focus_x_plus_1 = get_function_focus(target_phase=lambda x: 2 * (-x + 1) / R_max, R_max=R_max)
    plt.plot(x_f, focus_x_plus_1(x_f), label="2(-x+1) / R_max")


    sin_func = lambda x: 2 * x / R_max * np.sin(20 * np.pi * x / R_max)

    focus_sin = get_function_focus(target_phase=sin_func, R_max=R_max)
    plt.plot(x_f, focus_sin(x_f), label="2x/Rmax * sin(20*pi*x/Rmax)")

    move_by = 0.0000025
    asymmetric_sin_func = lambda x: 2 * (x + move_by) / R_max * np.sin(20 * np.pi * (x + move_by) / R_max) + 0.2
    focus_asymmetric_sin = get_function_focus(target_phase=asymmetric_sin_func, R_max=R_max)
    plt.plot(x_f, focus_asymmetric_sin(x_f), label="asymmetric sin")

    # x_sin = np.linspace(-R_max, R_max, 1000)
    # plt.plot(x_sin, asymetric_sin_func(x_sin))

    plt.grid()
    plt.legend()
    plt.show()

def double_gaussian_matching():
    R_max = 10 * 1e-6  # 10 micro meters
    lambda_0 = R_max / 12.5
    NA = 0.01
    lambda_e_perpendicular = lambda_0 / NA

    x_f = np.linspace(-3 * lambda_e_perpendicular, 3 * lambda_e_perpendicular, 500)

    well1 = derivative_of_a_gaussian(mu=lambda_e_perpendicular*2, sigma=R_max / 10)
    # well2 = derivative_of_a_gaussian(mu=-lambda_e_perpendicular, sigma=R_max / 10)
    phase_from_inverse = lambda x: np.arccos(well1(x))

    focus_inverse_phase = get_function_focus(target_phase=phase_from_inverse, R_max=R_max)
    plt.plot(x_f, focus_inverse_phase(x_f), label="2x / R_max")

    plt.grid()
    plt.legend()
    plt.show()



def derivative_of_a_gaussian(mu, sigma):
    return lambda x :(mu - x) / (sigma**3 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * (x-mu)**2 / sigma**2)


def abajo_paper_matching():
    R_max = 10 * 1e-6  # 10 micro meters
    lambda_0 = R_max / 12.5
    NA = 0.01
    lambda_e_perpendicular = lambda_0 / NA

    x_f = np.linspace(-3 * lambda_e_perpendicular, 3 * lambda_e_perpendicular, 500)

    focus_x = get_function_focus(target_phase=lambda x: 2 * x / R_max, R_max=R_max)
    plt.plot(x_f, focus_x(x_f), label="2x / R_max")

    focus_x_plus_1 = get_function_focus(target_phase=lambda x: 2 * (-x + 1) / R_max, R_max=R_max)
    plt.plot(x_f, focus_x_plus_1(x_f), label="2(-x+1) / R_max")

    sin_func = lambda x: 2 * x / R_max * np.sin(20 * np.pi * x / R_max)

    focus_sin = get_function_focus(target_phase=sin_func, R_max=R_max)
    plt.plot(x_f, focus_sin(x_f), label="2x/Rmax * sin(20*pi*x/Rmax)")

    x_sin = np.linspace(-R_max, R_max, 1000)

    plt.plot(x_sin, sin_func(x_sin))

    plt.grid()
    plt.legend()
    plt.show()


def get_function_focus(target_phase: Callable, R_max: float) -> Callable:
    lambda_0 = R_max / 12.5
    q_0 = 2 * np.pi / lambda_0
    NA = 0.01
    zf_zl = R_max / NA
    # phi = lambda x: phase_function(target_phase=target_phase,
    #                                lambda_0=lambda_0,
    #                                R_max=R_max,
    #                                x=x)
    psi = lambda x_f: real_wave_func_at_focus(x_f=x_f,
                                              q_0=q_0,
                                              zf_zl=zf_zl,
                                              phi=target_phase,
                                              R_max=R_max)

    vectorized_psi = np.vectorize(psi)
    return vectorized_psi


def real_wave_func_at_focus(x_f: float, q_0: float, zf_zl: float, phi: Callable, R_max: float):
    integrand = lambda x: np.cos(phi(x) - (q_0 * x * x_f / zf_zl))
    factor = 2.7
    result, error = integrate.quad(integrand, -factor*R_max, factor*R_max)
    print(error)
    return result


def phase_function(target_phase: Callable, lambda_0: float, R_max: float, x: float):
    def inner_sine(arg):
        if abs(arg) < 1e-5:
            c = 4 * np.pi / lambda_0
            return c - (c ** 3 * arg ** 2) / 6 + (c ** 5 * arg ** 4) / 120
        return np.sin(4 * np.pi * arg / lambda_0) / arg

    integrand = lambda x_tag: inner_sine(arg=x - x_tag) * target_phase(x_tag)
    result, error = integrate.quad(integrand, -R_max, R_max)
    return 1 / np.pi * result


if __name__ == '__main__':
    main()
    # double_gaussian_matching()