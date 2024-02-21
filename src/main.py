from functions import get_electron_probability_function, get_electron_wave_function, ureg, make_2d_gaussian, \
    simulate_shape


def main():
    electron_speed = 1 * ureg.meter / ureg.second
    electric_field_strength = 1 * ureg.volt / ureg.meter
    wave_function = get_electron_wave_function(electron_speed=electron_speed,
                                               g_squared=g_squared,
                                               electric_field_strength=electric_field_strength)
    electron_probability = get_electron_probability_function(wave_function=wave_function)
    simulate_shape(electron_probability)


def g_squared(x, y):
    well1 = make_2d_gaussian(mu_x=1 / 4, mu_y=1 / 4, sigma_x=1 / 8, sigma_y=1 / 8)
    well2 = make_2d_gaussian(mu_x=3 / 4, mu_y=3 / 4, sigma_x=1 / 8, sigma_y=1 / 8)
    return (well1(x, y) + well2(x, y)) ** 2


if __name__ == '__main__':
    main()
