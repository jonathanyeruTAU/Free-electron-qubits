import numpy as np
import pint
import scipy.constants as constants
from scipy import integrate

ureg = pint.UnitRegistry(system='SI')

electron_speed = (0.5 * ureg.speed_of_light).to_base_units()  # TODO: make this a real value

electron_mass = constants.electron_mass * ureg.kg

c = (1 * ureg.speed_of_light).to_base_units()

alpha = constants.fine_structure

beta = electron_speed / c

gamma = 1 / np.sqrt(1 - beta ** 2)

E_e = gamma * electron_mass * c ** 2

omega_laser = 1 / ureg.second  # TODO: make this a real value

lambda_laser = 2 * np.pi * c / omega_laser

electric_field_strength = 1 * ureg.volt / ureg.meter  # TODO: make this a real value

epsilon_0 = constants.epsilon_0 * ureg.farad / ureg.meter

u = lambda t: t  # TODO: make this a real value with real units

u_squared = lambda t: u(t) ** 2

g = lambda x, y: x + y  # TODO: make this a real value with real units

g_squared = lambda x, y: g(x, y) ** 2

t_final = 1  # TODO: make this a real value
t_initial = 0  # TODO: make this a real value
integral_of_u_squared = integrate.quad(u_squared, t_initial, t_final)  # TODO: give this units

x_initial = 0
x_final = 1
y_initial = lambda x: 0  # TODO: make this a real value
y_final = lambda y: 1  # TODO: make this a real value
integral_of_g_squared = integrate.dblquad(g_squared, x_initial, x_final, y_initial, y_final)  # TODO: give this units


# D10
E_laser = c * epsilon_0 * electric_field_strength / 2 * integral_of_u_squared * integral_of_g_squared

# D11
psi_v = lambda x, y: -alpha * E_laser * (lambda_laser ** 2) * g_squared(x, y) / \
                     (2 * np.pi * (1 + beta) * E_e * integral_of_g_squared)
