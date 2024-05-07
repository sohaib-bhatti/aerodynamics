import numpy as np
import sympy as sp
# from sympy.utilities.lambdify import lambdify
from Airfoil import Airfoil
from Geometry import Airplane
from Fluid import Fluid
import matplotlib.pyplot as plt


class Wing:
    def __init__(self, geometry, designation, fluid, N):
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
        self.angle_of_inc = np.radians(15)

        self.V_inf = fluid.V_inf
        self.rho = fluid.rho
        self.mu = fluid.mu

        self.N = N

        self.RE = fluid.rho * fluid.V_inf * self.mac / fluid.mu

    def vortex_distribution(self, angle_of_inc):
        y = sp.symbols('y')
        theta = sp.symbols('theta')

        alpha_y = (self.tip_twist - self.root_twist)/self.b * y\
            + self.root_twist + angle_of_inc

        alpha_theta = alpha_y.subs(y, self.b/2 * sp.cos(theta))

        theta_list = np.linspace(0+0.1, np.pi/2-0.1, self.N)

        s = (self.root_chord - self.tip_chord)/(self.taper_start - self.b/2)
        b = self.root_chord - s * self.taper_start
        c_y = sp.Piecewise((self.root_chord, y <= self.taper_start),
                           (s*y + b, (y > self.taper_start) & (y <= self.b/2)))

        c_theta = c_y.subs(y, self.b/2 * sp.cos(theta))

        rhs = np.zeros(self.N)
        lhs = np.zeros((self.N, self.N))
        for i in range(self.N):
            t = theta_list[i]
            a = float(alpha_theta.subs(theta, t))
            rhs[i] = a - self.AoA_0
            c = float(c_theta.subs(theta, t))
            for j in range(1, 2 * self.N + 1, 2):
                lhs[i][int((j - 1)/2)] = 4*self.b / \
                    (self.lift_slope * c) *\
                    np.sin(j*t) + j*np.sin(j*t)/np.sin(t)

        A = np.linalg.solve(lhs, rhs)

        gamma_theta = 2 * self.b * self.V_inf * A[0] * sp.sin(0*theta)

        for i in range(self.N):
            gamma_theta += 2 * self.b * self.V_inf * A[i] * sp.sin(i*theta)

        """gamma_y = gamma_theta.subs(theta, sp.acos(y/self.b))

        return lambdify(y, gamma_y, modules='numpy')(y_list)

        return 1/2*(np.pi * A[0] * self.AR) * self.rho * self.V_inf**2 * self.S
        print(A)"""
        return np.pi * A[0] * self.AR

    def plot_CL(self, n_points):
        fig, ax = plt.subplots()
        alpha_i = np.linspace(np.radians(-20), np.radians(20), n_points)

        CL = np.zeros(n_points)

        for i in range(n_points):
            CL[i] = self.vortex_distribution(alpha_i[i])

        ax.plot(np.degrees(alpha_i), CL, color="red")
        ax.plot(np.degrees(alpha_i), self.airfoil.Cl, color="blue")
        ax.axvline(x=np.degrees(self.AoA_0), color="black",
                   linestyle=(0, (1, 3)))
        ax.axhline(y=0, color="black", linestyle=(0, (1, 3)))

        fig.legend(("Finite wing", "2D airfoil"))

        ax.set_xlabel("α, angle of attack (°)")
        ax.set_ylabel("$c_l$")

        ax.set_title("Lift Curve for NACA " + self.airfoil.code)
        ax.grid(visible=True)
        fig.savefig("NACA " + self.airfoil.code + " lift curve, finite wing")

    def plot_CLCD(self, n_points):
        fig, ax = plt.subplots()
        alpha_i = np.linspace(np.radians(-20), np.radians(20), n_points)

        CL = np.zeros(n_points)
        CD = np.zeros(n_points)

        for i in range(n_points):
            CL[i] = self.vortex_distribution(alpha_i[i])
            CD[i] = CL[i]**2/(np.pi*self.AR)

        ax.plot(CD, CL, color="red")

        ax.set_xlabel("$C_D,i$")
        ax.set_ylabel("$C_L$")

        ax.set_title("Lift vs Drag Curve for NACA " + self.airfoil.code)
        ax.grid(visible=True)
        fig.savefig("NACA " + self.airfoil.code + " Lift vs Drag Curve")

    def plot_CLCD_alpha(self, n_points):
        fig, ax = plt.subplots()
        alpha_i = np.linspace(np.radians(-20), np.radians(20), n_points)

        ratio = np.zeros(n_points)

        for i in range(n_points):
            CL = self.vortex_distribution(alpha_i[i])
            CD = CL**2/(np.pi*self.AR)

            ratio[i] = CL/CD

        ax.plot(np.degrees(alpha_i), ratio, color="red")

        ax.set_xlabel("$α$")
        ax.set_ylabel("$C_L/C_D,i$")

        ax.set_title("Lift/drag ratio vs AoA " + self.airfoil.code)
        ax.grid(visible=True)
        fig.savefig("NACA " + self.airfoil.code + " Lift-drag ratio vs AoA")

    def friction_drag(self):
        n = self.N - 1
        y_values = np.linspace(0, self.b/2, self.N)

        y = sp.symbols('y')
        s = (self.root_chord - self.tip_chord)/(self.taper_start - self.b/2)
        b = self.root_chord - s * self.taper_start
        c_y = sp.Piecewise((self.root_chord, y <= self.taper_start),
                           (s*y + b, (y > self.taper_start) & (y <= self.b/2)))

        drag = 0
        Re_local = 0
        for i in range(n):
            c_0 = float(c_y.subs(y, y_values[i]))
            c_1 = float(c_y.subs(y, y_values[i+1]))
            Re_local = np.average((c_0, c_1)) * self.V_inf**2 *\
                self.rho / self.mu
            Cf = 1.328 / np.sqrt(Re_local)
            area = self.b / self.N / 4 * (c_0 + c_1)
            drag += area * (1/2*Cf*self.rho*self.V_inf**2)

        return drag


def test(designation):
    cessna = Airplane(11, 16.2, 0.5, np.radians(1.5),
                      np.radians(-1.5), 1.63, 1.13, 2.2)
    air = Fluid(55, 1.3, 1.8*10**-5)
    wing = Wing(cessna, designation, air, 10)

    # wing.plot_CL(100)
    # wing.plot_CLCD(100)
    print(wing.friction_drag())


def main():
    test("2412")


if __name__ == '__main__':
    main()
