import numpy as np
import pint
import scipy.constants as constants
from scipy import integrate
import matplotlib.pyplot as plt


def gaussian(mu, sigma):
    return lambda x: np.exp((-((x - mu) ** 2)) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi * sigma ** 2))




ureg = pint.UnitRegistry(system='SI')

electron_mass = constants.electron_mass * ureg.kg

c = (1 * ureg.speed_of_light).to_base_units()

alpha = constants.fine_structure

lambda_laser = 1035 * pow(10, -9) * ureg.meter

epsilon_0 = constants.epsilon_0 * ureg.farad / ureg.meter

t_final = 280 * pow(10, -15)
t_initial = 0
u = gaussian(mu=t_final / 2, sigma=t_final / 3)
u_squared = lambda t: u(t) ** 2
integral_of_u_squared = integrate.quad(u_squared, t_initial, t_final)


def get_electron_wave_function(electron_speed, g_squared, electric_field_strength):
    """

    :param electron_speed: units meter / second
    :param g_squared: function of the laser pulse
    :param electric_field_strength: units volt / meter
    :return:
    """
    beta = electron_speed / c
    gamma = 1 / np.sqrt(1 - beta ** 2)
    E_e = gamma * electron_mass * c ** 2

    x_initial = 0
    x_final = 1
    y_initial = lambda x: 0  # TODO: make this a real value
    y_final = lambda y: 1  # TODO: make this a real value
    integral_of_g_squared = integrate.dblquad(g_squared, x_initial, x_final, y_initial,
                                              y_final)  # TODO: give this units

    # D10
    E_laser = c * epsilon_0 * electric_field_strength / 2 * integral_of_u_squared * integral_of_g_squared

    # D11
    psi_v = lambda x, y: -alpha * E_laser * (lambda_laser ** 2) * g_squared(x, y) / \
                         (2 * np.pi * (1 + beta) * E_e * integral_of_g_squared)

    return psi_v


def get_electron_probability_function(wave_function):
    return lambda x, y: np.abs(wave_function(x, y)) ** 2


def simulate_shape(electron_probability_function):
    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 1000)
    X, Y = np.meshgrid(x, y)
    Z = electron_probability_function(X, Y)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def make_2d_gaussian(mu_x, mu_y, sigma_x, sigma_y):
    """
    Create a 2D Gaussian function with specified parameters.

    Parameters:
        mu_x (float): x-coordinate of the center of the Gaussian.
        mu_y (float): y-coordinate of the center of the Gaussian.
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.

    Returns:
        function: A function representing the 2D Gaussian with the specified parameters.
    """
    def gaussian_2d(x, y):
        """
        Compute the value of the 2D Gaussian function at given coordinates (x, y).

        Parameters:
            x (float or numpy array): x-coordinate(s) of the point(s) to evaluate.
            y (float or numpy array): y-coordinate(s) of the point(s) to evaluate.

        Returns:
            float or numpy array: The value(s) of the 2D Gaussian function at the given coordinates.
        """
        exponent = -((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2))
        normalization = 1 / (2 * np.pi * sigma_x * sigma_y)
        return normalization * np.exp(exponent)

    return gaussian_2d
