import scipy.constants as physics_constants


# units: meter / second
speed_of_light = physics_constants.c

# units: none
fine_structure = physics_constants.fine_structure


# units: Farad / meter
vacuum_permittivity = physics_constants.epsilon_0


# units: kilograms
electron_mass = physics_constants.electron_mass

# units: kg * m**2 / s**2
rest_mass_energy = electron_mass * (speed_of_light ** 2)