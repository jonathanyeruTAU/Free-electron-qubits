from intensity import intensity_of_electron
from src.basic_functions import k
from src.functions import make_2d_gaussian
from src.phase import phase_func, integrate_g
from src.simulate_intensity import simulate_intensity_electron_wave_function


def main():
    R = 1
    d = 1
    d_c = 2
    v = 1
    E_L = 1
    lambda_L = 1
    electron_wave_number = k(v)

    phase_func_according_to_x_y = phase_func(v=v, E_L=E_L, g=g, lambda_L=lambda_L,
                                             integral_of_g_squared=integral_of_g_squared)

    intensity_of_electron_func = intensity_of_electron(R=R, d=d, d_c=d_c, k=electron_wave_number,
                                                       phase_func=phase_func_according_to_x_y)

    simulate_intensity_electron_wave_function(intensity_of_electron_func)


def g(x, y):
    well1 = make_2d_gaussian(mu_x=1 / 4, mu_y=1 / 4, sigma_x=1 / 8, sigma_y=1 / 8)
    well2 = make_2d_gaussian(mu_x=3 / 4, mu_y=3 / 4, sigma_x=1 / 8, sigma_y=1 / 8)
    return well1(x, y) + well2(x, y)


integral_of_g_squared = integrate_g(g)

if __name__ == '__main__':
    main()
