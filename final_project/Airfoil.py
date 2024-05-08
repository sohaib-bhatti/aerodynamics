import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.stats import linregress
from Fluid import Fluid

"""
AIRFOIL GENERATION CODE, ADAPTED FROM Divahar Jayaraman (j.divahar@yahoo.com)
"""


class Airfoil:
    def __init__(self, designation, c, fluid):
        # ALL AIRFOIL CHARACTERISTICS ARE IMPLEMENTED HERE

        self.code = str(designation)
        if len(self.code) != 4:
            raise Exception("4 digit NACA codes allowed only!")

        self.m = float(self.code[0])/100
        self.p = float(self.code[1])/10
        self.t = float(self.code[2:])/100

        self.MAC = c

        self.V_inf = fluid.V_inf
        self.rho = fluid.rho
        self.mu = fluid.mu

        x = sp.symbols('x')

        a0 = 0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = 0.2843
        a4 = -0.1015

        yc1 = self.m/self.p**2*(2*self.p*x - x**2)
        yc2 = self.m/(1-self.p)**2 * ((1 - 2*self.p) + 2*self.p*x - x**2)

        yc = sp.Piecewise((yc1, x <= self.p), (yc2, (x > self.p) & (x < 1)))

        theta = sp.atan(sp.diff(yc, x))

        yt = self.t/0.2 * (a0*sp.sqrt(x) + a1*x + a2*x**2 + a3*x**3 + a4*x**4)

        xu = x - yt*sp.sin(theta)
        yu = yc + yt*sp.cos(theta)
        xl = x + yt*sp.sin(theta)
        yl = yc - yt*sp.cos(theta)

        dy = sp.diff(yc, x)

        self.yc = lambdify(x, yc, modules='numpy')
        self.xu = lambdify(x, xu, modules='numpy')
        self.yu = lambdify(x, yu, modules='numpy')
        self.xl = lambdify(x, xl, modules='numpy')
        self.yl = lambdify(x, yl, modules='numpy')

        self.dy = lambdify(x, dy, modules='numpy')

        self.x = x

        self.theta = sp.symbols('theta')
        self.dtheta = dy.subs(self.x, 1/2*(1 - sp.cos(self.theta)))

        theta_n = np.linspace(0, np.pi-.01, 100)

        integrand = lambdify(self.theta,
                             self.dtheta * (sp.cos(self.theta) - 1),
                             modules='numpy')(theta_n)

        self.alpha_L0 = -1/np.pi * np.trapz(integrand, theta_n)

        self.dtheta_n = lambdify(self.theta, self.dtheta,
                                 modules='numpy')(theta_n)
        A0, A = self.fourier_terms(4)

        self.Cl = np.pi*(np.add(2*A0, A[1]))

        alpha_n = np.linspace(np.radians(-10), np.radians(10), 100)
        self.lift_slope = linregress(alpha_n, self.Cl).slope
        self.zero_Cl = linregress(alpha_n, self.Cl).intercept

    def fourier_terms(self, n):
        self.alpha = sp.symbols('alpha')
        theta_n = np.linspace(0, np.pi-.01, 100)
        A = np.zeros(n + 1)
        alpha_n = np.linspace(-0.175, 0.175, 100)
        A0 = lambdify(self.alpha,
                      self.alpha - 1/np.pi * np.trapz(self.dtheta_n, theta_n),
                      modules='numpy')(alpha_n)

        for i in range(1, n + 1):
            integrand = lambdify(self.theta,
                                 self.dtheta * sp.cos(i*self.theta),
                                 modules='numpy')(theta_n)
            A[i] = 2/np.pi * np.trapz(integrand, theta_n)

        return A0, A

    def plot_airfoil(self, n_points):
        x = np.linspace(0, 1, n_points)

        # Evaluate the functions at x_values to get y values
        yc = self.yc(x)
        xu = self.xu(x)
        yu = self.yu(x)
        xl = self.xl(x)
        yl = self.yl(x)

        dyc = self.dy(x)

        fig, axes = plt.subplots(2, sharex=True)
        # fig.subplots_adjust(hspace=-0.50)

        axes[0].tick_params(axis='x',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False)

        axes[0].plot(xu, yu, color="blue")
        axes[0].plot(xl, yl, color="blue")
        axes[0].plot(x, yc, color="red", linestyle=(0, (1, 1)))

        axes[0].set_aspect('equal')
        axes[0].axvline(x=self.p, color="black", linestyle=(0, (1, 3)))

        axes[0].set_ylim([-0.1, 0.1])
        axes[0].set_xlim([-0., 1.])

        axes[0].set_ylabel("z/c")

        axes[1].plot(x, dyc)
        axes[1].set_aspect('equal')

        axes[1].set_ylim([-0.2, 0.2])
        axes[1].axvline(x=self.p, color="black", linestyle=(0, (1, 3)))

        axes[1].set_ylim([-0.1, 0.15])
        axes[1].set_xlim([-0., 1.])

        axes[1].set_xlabel("$x/c$")
        axes[1].set_ylabel("$dz/c$")

        fig.suptitle("NACA " + self.code + " and camberline derivative")

        fig.savefig("NACA " + self.code)

    def plot_CL(self, n_points):
        fig, ax = plt.subplots()
        alpha_n = np.linspace(np.radians(-10), np.radians(10), n_points)

        ax.plot(np.degrees(alpha_n), self.Cl)
        ax.axvline(x=np.degrees(self.alpha_L0), color="black",
                   linestyle=(0, (1, 3)))
        ax.axhline(y=0, color="black", linestyle=(0, (1, 3)))

        ax.set_xlabel("α, angle of attack (°)")
        ax.set_ylabel("$c_l$")

        ax.set_title("Lift Curve for NACA " + self.code)
        ax.grid(visible=True)
        fig.savefig("NACA " + self.code + " lift curve, 2D")

    def lift(self, n_points):
        alpha_n = np.linspace(-10, 10, n_points)
        i = np.where(alpha_n > 4.1)
        i = i[0][0]

        y = sp.symbols('y')
        s = (1.63 - 1.13)/(2.2 - 11/2)
        b = 1.63 - s * 2.2
        c_y = sp.Piecewise((1.63, y <= 2.2),
                           (s*y + b, (y > 2.2) & (y <= 11/2)))

        Cl = self.Cl[i]
        print(i)
        lift_per_span = 1/2 * Cl * self.V_inf**2 * self.MAC
        lift_over_wing = 1/2 * Cl * self.V_inf**2 *\
            sp.integrate(c_y, (y, 0, 11/2))

        return Cl, lift_per_span, lift_over_wing


def test(designation):
    air = Fluid(55, 1.3, 1.8*10**-5)
    airfoil = Airfoil(designation, 1.375, air)
    airfoil.plot_CL(100)
    print(airfoil.lift(100))
    

def main():
    test("2412")


if __name__ == '__main__':
    main()
