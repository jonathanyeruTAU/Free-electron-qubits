from src.constants import speed_of_light, electron_mass, hbar
import numpy as np
import warnings
from typing import Callable, Iterable
from scipy.optimize import root_scalar


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


def k(v):
    return gamma(v) * electron_mass * v / hbar


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


def multi_root(f: Callable, bracket: Iterable[float], args: Iterable = (), n: int = 30) -> np.ndarray:
    """ Find all roots of f in `bracket`, given that resolution `n` covers the sign change.
    Fine-grained root finding is performed with `scipy.optimize.root_scalar`.
    Parameters
    ----------
    f: Callable
        Function to be evaluated
    bracket: Sequence of two floats
        Specifies interval within which roots are searched.
    args: Iterable, optional
        Iterable passed to `f` for evaluation
    n: int
        Number of points sampled equidistantly from bracket to evaluate `f`.
        Resolution has to be high enough to cover sign changes of all roots but not finer than that.
        Actual roots are found using `scipy.optimize.root_scalar`.
    Returns
    -------
    roots: np.ndarray
        Array containing all unique roots that were found in `bracket`.
    """
    # Evaluate function in given bracket
    x = np.linspace(*bracket, n)
    y = f(x, *args)

    # Find where adjacent signs are not equal
    sign_changes = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]

    # Find roots around sign changes
    root_finders = (
        root_scalar(
            f=f,
            args=args,
            bracket=(x[s], x[s + 1])
        )
        for s in sign_changes
    )

    roots = np.array([
        r.root if r.converged else np.nan
        for r in root_finders
    ])

    if np.any(np.isnan(roots)):
        warnings.warn("Not all root finders converged for estimated brackets! Maybe increase resolution `n`.")
        roots = roots[~np.isnan(roots)]

    roots_unique = np.unique(roots)
    if len(roots_unique) != len(roots):
        warnings.warn("One root was found multiple times. "
                      "Try to increase or decrease resolution `n` to see if this warning disappears.")

    return roots_unique
