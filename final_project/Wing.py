import numpy as np
import sympy as sp
from scipy.integrate import cumtrapz
from sympy.utilities.lambdify import lambdify
from Airfoil import Airfoil
from Geometry import Geometry
from Fluid import Fluid


class Wing:
    def __init__(self, geometry, designation, fluid):
        self.b = geometry.wingspan
        self.S = geometry.surface_area
        self.AR = self.b ** 2 / self.S
        self.taper = geometry.taper
        self.root_twist = geometry.root_twist
        self.tip_twist = geometry.tip_twist
        self.root_chord = geometry.root_chord
        self.tip_chord = geometry.tip_chord
        self.mac = (self.root_chord + self.tip_chord) / 2
        self.taper_start = geometry.taper_start

        self.airfoil = Airfoil(designation)
        self.lift_slope = self.airfoil.lift_slope
        self.AoA_0 = self.airfoil.alpha_L0

        self.V_inf = fluid.V_inf
        self.rho = fluid.rho

        self.N = 3

    def vortex_distribution(self):
        y = sp.symbols('y')
        theta = sp.symbols('theta')

        alpha_y = (self.tip_twist - self.root_twist)/self.b * y\
            + self.root_twist
        alpha_theta = alpha_y.subs(y, self.b/2 * sp.cos(theta))

        y_list = np.linspace(0, self.b, self.N)
        theta_list = np.linspace(0.1, np.pi/2-0.1, self.N)

        rhs = np.zeros(self.N)
        lhs = np.zeros((self.N, self.N))

        for i in range(self.N):
            t = float(alpha_theta.subs(theta, theta_list[i]))
            rhs[i] = t
            for j in range(1, 2 * self.N + 1, 2):
                j = np.array(j)

                lhs[i][int((j - 1)/2)] = 4 * self.b / (self.lift_slope * self.mac) *\
                    np.sin(j * t) +\
                    j * np.sin(j*t)/np.sin(t)

        A = np.linalg.solve(lhs, rhs)

        print(A)

        gamma_theta = 2 * self.b * self.V_inf * A[0] * sp.sin(0*theta)

        for i in range(self.N):
            gamma_theta += 2 * self.b * self.V_inf * A[i] * sp.sin(i*theta)

        gamma_y = gamma_theta.subs(theta, sp.acos(y/self.b))

        return lambdify(y, gamma_y, modules='numpy')(y_list)

    def thin_airfoil_lift(self):
        return 1/2 * self.rho * self.V_inf * self.S**2 * self.airfoil.zero_Cl

    def lifting_line_lift(self):
        gamma = self.vortex_distribution()
        b_list = np.linspace(-self.b/2, self.b/2, self.N)
        b_list_1 = np.linspace(-self.b/2, self.b/2, self.N - 1)
        lift_per_span = self.rho * self.V_inf * cumtrapz(gamma, b_list)
        lift = self.rho * self.V_inf * np.trapz(lift_per_span, b_list_1)

        return lift


def test(designation):
    cessna = Geometry(11, 16.2, 0.5, np.radians(1.5),
                      np.radians(-1.5), 1.63, 1.13, 5)
    air = Fluid(63, 1.3)
    wing = Wing(cessna, designation, air)

    lift = wing.lifting_line_lift()

    # print(lift)


def main():
    test("2412")


if __name__ == '__main__':
    main()