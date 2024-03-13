import numpy as np
from scipy.misc import derivative
from src.basic_functions import multi_root, timer
from scipy import integrate

def intensity_of_electron_with_rho_s_alpha_s(R, d, k, d_c, phase_func, rho_s, alpha_s):
    """
    :param R: the radius of the detector - units: meters
    :param d: the distance between the action point and the detector - units: meters
    :param k: the wave number of the electron
    :param d_c: i dont know ask yonatan
    :param phase_func: the phase function of the electron! input (x, y)
    :param rho_s, alpha_s: the point in the plane that we want to know the intensity.
    :return: the intensity of the electron at the detector
    we want to integrate over the measurement plane. we will use polar coordinates.
    so:
    x = rho * cos(theta)
    y = rho * sin(theta)
    But instead of using rho 0 -> R we will use
    zeta = (rho/R)**2
    there fore: Phase_Func(zeta) = phase_func(R*sqrt(zeta)*cos(theta), R*sqrt(zeta)*sin(theta))
    additionaly, we will move to kappa (dimensionless)
    kappa = k * R**2 / d
    delta = (d_c - d) / (2 * d_c)
    but we might have a varying phase in our setting, which renders direct numerical integration methods unstable.
    we will perform stationary phase approximation of the zeta integral in Eq(5) in Juffman.
    So wew will want to find for each theta (in the angular integral [-pi, pi]) the zeros of the function:
    F(zeta_n) = 2 * sqrt(zeta_n)[kappa*delta + Phase_Func'(zeta_n * cos(theta), zeta_n * sin(theta)) - kappa * sin(theta - alpha_s) * (rho_s / R)
    then for each theta we need to preform the sum:
    sum(sqrt(2*pi*i/F''(zeta_n))*exp(iF(zeta_n)
    ok so lets start
    """

    def Phase_Func(zeta, theta):
        if not isinstance(zeta, float):
            zeta[zeta < 0] = 10e-6
        rho = R * np.sqrt(zeta)
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return phase_func(x, y)

    def Phase_Func_derivative(zeta, theta, n=1):
        return derivative(Phase_Func, zeta, args=(theta,), n=n, dx=10e-6)

    kappa = k * R ** 2 * d
    delta = (d_c - d) / (2 * d_c)

    def F(zeta, theta):
        return kappa * delta * zeta - kappa * np.sqrt(zeta) * np.sin(theta - alpha_s) * rho_s / R * Phase_Func(zeta,
                                                                                                               theta)

    def F_deriv(zeta, theta):
        if not isinstance(zeta, float):
            zeta[zeta < 0] = 10e-6
        function = 2 * np.sqrt(zeta)
        function *= (kappa * delta + Phase_Func_derivative(zeta, theta))
        function -= (kappa * np.sin(theta - alpha_s) * rho_s / R)
        return function

    def integrate_over_zeta(theta):
        all_roots = multi_root(f=F_deriv, bracket=[0, 1], args=(theta,))

        def stationary_phase_argument(zeta_n):
            function = np.sqrt(2 * np.pi * 1j / derivative(F_deriv, zeta_n, args=(theta,), n=2))
            return function * np.exp(1j * F(zeta_n, theta))

        return np.sum(stationary_phase_argument(zeta_n=all_roots))
    theta = np.linspace(-np.pi, np.pi, 10)
    np_integrate_over_zeta = np.vectorize(integrate_over_zeta)
    y = np_integrate_over_zeta(theta)
    return np.abs(integrate.simpson(y, theta)) ** 2


def intensity_of_electron(R, d, k, d_c, phase_func):
    """
    same as the function above, but it returns a function that accepts x and y
    and returns the intensity at that point
    """

    def intensity_at_point_x_y(x, y):
        rho_s = np.sqrt(x ** 2 + y ** 2)
        alpha_s = np.arctan2(x, y)
        return intensity_of_electron_with_rho_s_alpha_s(R=R,
                                                        d=d,
                                                        d_c=d_c,
                                                        k=k,
                                                        phase_func=phase_func,
                                                        rho_s=rho_s,
                                                        alpha_s=alpha_s)

    return intensity_at_point_x_y
