import numpy as np
from scipy.integrate import trapz
from airfoil_generation import airfoil_generation


def dz_dtheta(dx_dz, theta, c):
    # Calculate dx/dtheta using the given relation
    dx_dtheta = dx_dz / (0.5 * c * np.sin(theta))
    
    return dx_dtheta


def zero_lift_AoA(dz_dx, x):
    integrand = dz_dx * (np.cos(np.linspace(0, np.pi, len(x))) - 1)
    integral = trapz(integrand, x=x)

    return (-1/np.pi) * integral


def A(dz_dx, x, n):
    integrand = dz_dx * (np.cos(n*np.linspace(0, np.pi, len(x))) - 1)
    integral = trapz(integrand, x=x)

    return (2/np.pi) * integral


def main():
    af = airfoil_generation("2412", 30)
    print(np.degrees(zero_lift_AoA(af["dzc"], af["xc"])))
    print(A(af["dzc"], af["xc"], 2))


if __name__ == '__main__':
    main()