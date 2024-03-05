import numpy as np
from src.basic_functions import E_L_electron_frame, lambda_L_electron_frame, square_f
from src.constants import fine_structure, rest_mass_energy
from scipy import integrate


def phase(x, y, v, E_L, lambda_L, g):
    """
    gives the phase of the electron in the xy plane, given a velocity v.
    :param x: x position  - normalized
    :param y: y position  - normalized
    :param v: units - meter / second
    :param E_L: energy of the laser. units - kg * (m/s)**2
    :param lambda_L: wave length of the laser. units - meter
    :param g: the spacial pulse profile - normalized by it's integral
    I made the function as they mention in section D, eq D11
    that the phase is lorentz variant so i work in the electron rest frame
    beta = 0
    E_e = rest masss
    lambda_L -> rest lambda L
    Energy_L -> rest energy L
    """
    electron_phase = -1 * fine_structure / (2 * np.pi)
    electron_phase *= E_L_electron_frame(E_L, v) / rest_mass_energy
    electron_phase *= lambda_L_electron_frame(lambda_L, v) ** 2

    square_g = square_f(g)
    electron_phase *= square_g(x, y) / integrate_g(g)
    return electron_phase


def phase_func(v, E_L, lambda_L, g):
    """
    same as the phase function above, but this one returns a function that accepts x and y are retruns the same output
    as the function above
    """
    return lambda x, y: phase(x=x, y=y, v=v, E_L=E_L, lambda_L=lambda_L, g=g)


def integrate_g(g):
    square_g = square_f(g)
    x_initial = -1
    x_final = 1
    y_initial = lambda x: 0
    y_final = lambda y: 1
    return integrate.dblquad(square_g, x_initial, x_final, y_initial, y_final)[0]
