from src.constants import speed_of_light
import numpy as np


def beta(v):
    """
    :param v: units - meter / second
    spped_of_light: units - meter / second
    """
    return v / speed_of_light


def gamma(v):
    """
    :param v: units - meter / second
    speed_of_light: units - meter / second
    :return:
    """
    return 1 / np.sqrt(1 - beta(v) ** 2)


def E_L_electron_frame(E_L, v):
    """
    laser energy in the electron wave frame
    :param E_L: laser energy - units: kilogram * (m**2 / s**2)
    :param v: electron speed
    :return:
    """
    return E_L * np.sqrt((1 - beta(v)) / (1 + beta(v)))


def lambda_L_electron_frame(lambda_L, v):
    """
    :param lambda_L: wave length of the laser meters
    :param v: electron speed m / s
    :return: the laser wave length in the electron frame
    """
    return lambda_L * np.sqrt((1 + beta(v)) / (1 - beta(v)))


def square_f(f):
    """
    :param f: a function that accepts x and y values
    :return: returns a function that is the square of f
    """
    return lambda x, y: f(x, y) ** 2