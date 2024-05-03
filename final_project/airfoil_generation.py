import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify

"""
AIRFOIL GENERATION CODE, ADAPTED FROM Divahar Jayaraman (j.divahar@yahoo.com)
"""


def airfoil_generation(designation, n, half_cos_spacing=True, want_file=True,
                       is_finiteTE=True):
    code = str(designation)
    if len(code) != 4:
        raise Exception("4 digit NACA codes allowed only!")

    m = float(code[0])/100
    p = float(code[1])/10
    t = float(code[2:])/100

    a0 = 0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = 0.2843

    if is_finiteTE:
        a4 = -0.1015
    else:
        a4 = -0.1036

    if half_cos_spacing:
        beta = np.linspace(0, np.pi, n + 1)
        x = 0.5*(1 - np.cos(beta))
        header = "NACA" + code + ": " + str(2*n)\
            + " panels, half-cosine spacing"
    else:
        x = np.linspace(0, 1, n + 1)
        header = "NACA" + code + ": " + str(2*n)\
            + " panels, uniform spacing"

    yt = (t/0.2) * (a0 * np.sqrt(x) + a1 * x + a2 * np.square(x) +
                    a3 * np.power(x, 3) + a4 * np.power(x, 4))

    xc1 = x[x <= p]
    xc2 = x[x > p]
    xc = np.concatenate((xc1, xc2))

    if p == 0:
        xu = x
        yu = yt

        xl = x
        yl = -yt

        zc = np.zeros_like(xc)

    else:
        yc1 = (m / p ** 2) * (2 * p * xc1 - xc1 ** 2)
        yc2 = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xc2 - xc2 ** 2)
        zc = np.concatenate((yc1, yc2))
        dyc1 = (2 * m / p ** 2) * (p - xc1)
        dyc2 = (2 * m / (1 - p) ** 2) * (p - xc2)
        dzc = np.concatenate((dyc1, dyc2))

        dyc1_dx = (m / p ** 2) * (2 * p - 2 * xc1)
        dyc2_dx = (m / (1 - p) ** 2) * (2 * p - 2 * xc2)
        dyc_dx = np.concatenate((dyc1_dx, dyc2_dx))
        theta = np.arctan(dyc_dx)

        xu = x - yt * np.sin(theta)
        yu = zc + yt * np.cos(theta)

        xl = x + yt * np.sin(theta)
        yl = zc - yt * np.cos(theta)

    af_name = "NACA " + code
    af_x = np.concatenate((np.flipud(xu), xl[2:]))
    af_z = np.concatenate((np.flipud(yu), yl[2:]))

    index1 = np.arange(np.argmin(af_x) + 1)
    index2 = np.arange(np.argmin(af_x), len(af_x))

    af_xU = af_x[index1]
    af_zU = af_z[index1]

    af_xL = af_x[index2]
    af_zL = af_z[index2]

    af_rLE = 0.5 * (a0 * t / 0.2) ** 2

    le_offs = 0.5 / 100
    dyc_dx_le = (m / p ** 2) * (2 * p - 2 * le_offs)
    theta_le = np.arctan(dyc_dx_le)
    af_xLEcenter = af_rLE * np.cos(theta_le)
    af_yLEcenter = af_rLE * np.sin(theta_le)

    if want_file:
        filename = "NACA" + code + ".dat"
        f = open(filename, "w")
        f.write(header + "\n")
        np.savetxt(filename, np.round(np.c_[af_x, af_z], 4))
        f.close()

    return {"name": af_name,
            "x": af_x,
            "z": af_z,
            "xU": af_xU,
            "zU": af_zU,
            "xL": af_xL,
            "zL": af_zL,
            "xc": xc,
            "zc": zc,
            "dzc": dzc,
            "r_LE": af_rLE,
            "xLEcenter": af_xLEcenter,
            "yLEcenter": af_yLEcenter}


def airfoil_generation2(designation):
    code = str(designation)
    if len(code) != 4:
        raise Exception("4 digit NACA codes allowed only!")

    m = float(code[0])/100
    p = float(code[1])/10
    t = float(code[2:])/100

    x = sp.symbols('x')

    a0 = 0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1015

    yc1 = m/p**2*(2*p*x - x**2)
    yc2 = m/(1-p)**2 * ((1 - 2*p) + 2*p*x - x**2)

    yc = sp.Piecewise((yc1, x <= p), (yc2, (x > p) & (x < 1)))

    theta = sp.atan(sp.diff(yc, x))

    yt = t/0.2 * (a0*sp.sqrt(x) + a1*x + a2*x**2 + a3*x**3 + a4*x**4)

    xu = x - yt*sp.sin(theta)
    yu = yc + yt*sp.cos(theta)
    xl = x + yt*sp.sin(theta)
    yl = yc - yt*sp.cos(theta)

    dy = sp.diff(yc, x)

    yc_func = lambdify(x, yc, modules='numpy')
    xu_func = lambdify(x, xu, modules='numpy')
    yu_func = lambdify(x, yu, modules='numpy')
    xl_func = lambdify(x, xl, modules='numpy')
    yl_func = lambdify(x, yl, modules='numpy')

    dy_func = lambdify(x, dy, modules='numpy')

    print(xu)

    return {"name": code,
            "x": x,
            "yc": yc_func,
            "xU": xu_func,
            "yU": yu_func,
            "xL": xl_func,
            "yL": yl_func,
            "dyc": dy_func,
            "m": m,
            "p": p,
            "t": t,
            }


def test(designation, n):
    af = airfoil_generation2(designation)

    x = np.linspace(0, 1, 10000)

    # Evaluate the functions at x_values to get y values
    yc = af["yc"](x)
    xu = af["xU"](x)
    yu = af["yU"](x)
    xl = af["xL"](x)
    yl = af["yL"](x)

    dyc = af["dyc"](x)

    fig, axes = plt.subplots(2)
    axes[0].plot(xu, yu)
    axes[0].plot(xl, yl)
    axes[0].plot(x, yc)

    # axes[0].legend(["upper", "lower", "camber"])
    axes[0].set_aspect('equal', 'box')
    axes[0].axvline(x=af["p"], color="black", linestyle="--")

    axes[0].set_ylim([-0.1, 0.1])
    axes[0].set_xlim([0, 1])

    axes[1].plot(x, dyc)
    axes[1].set_aspect('equal', 'box')

    axes[1].set_ylim([-0.2, 0.2])

    axes[1].axvline(x=af["p"], color="black", linestyle="--")

    plt.show()


def main():
    test("2412", 30)


if __name__ == '__main__':
    main()
