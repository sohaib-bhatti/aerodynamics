import numpy as np
from scipy.integrate import trapz
from airfoil_generation import Airfoil
import sympy as sp


def zero_lift_AoA(dz_dx, x):
    integrand = dz_dx * (np.cos(np.linspace(0, np.pi, len(x))) - 1)
    integral = trapz(integrand, x=x)

    return (-1/np.pi) * integral


def A(dz_dx, x, n):
    integrand = dz_dx * (np.cos(n*np.linspace(0, np.pi, len(x))) - 1)
    integral = trapz(integrand, x=x)

    return (2/np.pi) * integral


def main():
    airfoil = Airfoil(2412)
    print(np.degrees(zero_lift_AoA(airfoil)))


if __name__ == '__main__':
    main()