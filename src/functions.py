from typing import Callable

import numpy as np
import pint
import scipy.constants as constants

ureg = pint.UnitRegistry(system='SI')

electron_speed = (0.5 * ureg.speed_of_light).to_base_units()  # TODO: make this a real value

electron_mass = constants.electron_mass * ureg.kg

c = (1 * ureg.speed_of_light).to_base_units()

beta = electron_speed / c

gamma = 1 / np.sqrt(1 - beta ** 2)

E_e = gamma * electron_mass * c ** 2

omega_laser = 1 / ureg.second  # TODO: make this a real value

lambda_laser = 2 * np.pi * c / omega_laser

electric_field_strength = 1 * ureg.volt / ureg.meter  # TODO: make this a real value

epsilon_0 = constants.epsilon_0 * ureg.farad / ureg.meter

u = lambda t: 1  # TODO: make this a real value with real units

g = lambda x, y, v: 1  # TODO: make this a real value with real units

# enter the v, returns a function of g with a given v
g_v = lambda v: lambda x, y: g(x, y, v)

# integral_of_u =




print()
