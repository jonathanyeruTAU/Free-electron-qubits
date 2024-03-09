from intensity import intensity_of_electron
from src.basic_functions import k, v_from_electron_energy
from src.functions import make_2d_gaussian
from src.phase import phase_func, integrate_g
from src.simulate_intensity import simulate_intensity_electron_wave_function


def main():
    # units: meter
    R = 10e-3
    # units: meter
    d = 0.55
    #units: meter
    d_c = 20e-6
    # units: eV
    E_e = 10e3
    # units: m /s
    v = v_from_electron_energy(E_e=E_e)
    #units: Joule
    E_L = 10e-6

    # units: meter
    lambda_L = 1035 * 10e-9
    electron_wave_number = k(v)

    phase_func_according_to_x_y = phase_func(v=v, E_L=E_L, g=g, lambda_L=lambda_L,
                                             integral_of_g_squared=integral_of_g_squared)

    intensity_of_electron_func = intensity_of_electron(R=R, d=d, d_c=d_c, k=electron_wave_number,
                                                       phase_func=phase_func_according_to_x_y)

    simulate_intensity_electron_wave_function(intensity_of_electron_func)


def g(x, y):
    # units: meter
    sigma_x = sigma_y = 10e-6
    well1 = make_2d_gaussian(mu_x=1 / 4, mu_y=1 / 4, sigma_x=sigma_x, sigma_y=sigma_y)
    well2 = make_2d_gaussian(mu_x=3 / 4, mu_y=3 / 4, sigma_x=sigma_x, sigma_y=sigma_y)
    return well1(x, y) + well2(x, y)


integral_of_g_squared = integrate_g(g)

if __name__ == '__main__':
    main()
