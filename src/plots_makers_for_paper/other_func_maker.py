import matplotlib.pyplot as plt
import numpy as np


def g_10(rho):
    w_10 = 10e-5
    return (rho / w_10)**2 * np.exp(-2*(rho/w_10)**2)

def main():
    rho = np.linspace(0, 3*10e-5, 1000)
    plt.plot(rho, g_10(rho))
    plt.grid()
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$g^{2}(\rho)$")
    plt.title(r"$g_{10}^{2}(\rho)$")
    plt.show()

if __name__ == '__main__':
    main()