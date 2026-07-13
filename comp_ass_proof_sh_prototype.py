import sympy as sp
import numpy as np

# Symbols
r, theta, mu, nu = sp.symbols("r theta mu nu", real=True)

# Current SR expression
f = (
    (-0.0001 * mu * sp.sin(mu) + 0.78017)
    * (0.00083 * mu - 0.16506 * nu + 4.03081)
    * sp.sin(r)
    * sp.sin(3.0191 * mu * nu - theta + 11.62159)
)

def polar_laplacian(expr):
    return (
        sp.diff(expr, r, 2)
        + (1 / r) * sp.diff(expr, r)
        + (1 / r**2) * sp.diff(expr, theta, 2)
    )

lap_f = polar_laplacian(f)
double_lap_f = polar_laplacian(lap_f)

# Steady SH residual
F = mu * f + nu * f**2 - f**3 - (f + 2 * lap_f + double_lap_f)

# Freeze one parameter pair
mu0 = 5.0
nu0 = 5.0

F_fixed = sp.simplify(F.subs({mu: mu0, nu: nu0}))
f_fixed = sp.simplify(f.subs({mu: mu0, nu: nu0}))

print("f_fixed:")
print(sp.factor(f_fixed))

print("\nResidual expression length:")
print(len(str(F_fixed)))

print("\nResidual factored/trig-simplified:")
print(sp.trigsimp(sp.factor(F_fixed)))

import sympy as sp
import numpy as np
from numpy.linalg import norm

r, theta, mu, nu = sp.symbols("r theta mu nu", real=True)

f = (
    (-0.0001 * mu * sp.sin(mu) + 0.78017)
    * (0.00083 * mu - 0.16506 * nu + 4.03081)
    * sp.sin(r)
    * sp.sin(3.0191 * mu * nu - theta + 11.62159)
)

def polar_laplacian(expr):
    return (
        sp.diff(expr, r, 2)
        + (1 / r) * sp.diff(expr, r)
        + (1 / r**2) * sp.diff(expr, theta, 2)
    )

lap_f = polar_laplacian(f)
double_lap_f = polar_laplacian(lap_f)

F = mu * f + nu * f**2 - f**3 - (f + 2 * lap_f + double_lap_f)

# Lambdify exact symbolic residual and formula
F_np = sp.lambdify((r, theta, mu, nu), F, "numpy")
f_np = sp.lambdify((r, theta, mu, nu), f, "numpy")

def test_symbolic_residual(mu0, nu0, Nr=400, Nt=400, r_min=0.1, r_max=10.0):
    rv = np.linspace(r_min, r_max, Nr)
    tv = np.linspace(0.0, 2 * np.pi, Nt, endpoint=False)
    R, T = np.meshgrid(rv, tv, indexing="ij")

    Res = F_np(R, T, mu0, nu0)
    U = f_np(R, T, mu0, nu0)

    Res = np.nan_to_num(Res)

    rms_res = norm(Res.ravel()) / np.sqrt(Res.size)
    rms_u = norm(U.ravel()) / np.sqrt(U.size)

    print(f"mu={mu0}, nu={nu0}")
    print("symbolic residual RMS:", rms_res)
    print("u RMS:", rms_u)
    print("relative residual RMS:", rms_res / max(rms_u, 1e-14))
    print("max abs residual:", np.max(np.abs(Res)))

    # Also check whether near r=0 is dominating
    for cutoff in [0.1, 0.25, 0.5, 1.0, 2.0]:
        mask = R >= cutoff
        rms_cut = norm(Res[mask].ravel()) / np.sqrt(np.count_nonzero(mask))
        print(f"  RMS for r >= {cutoff}: {rms_cut}")

test_symbolic_residual(5.0, 5.0)
import numpy as np
from numpy.linalg import norm

def inner(a, b):
    return np.mean(a * b)

def projection_coeff(res, mode):
    return inner(res, mode) / max(inner(mode, mode), 1e-14)

def analyze_residual_modes(mu0=5.0, nu0=5.0, Nr=600, Nt=600, r_min=0.1, r_max=10.0):
    rv = np.linspace(r_min, r_max, Nr)
    tv = np.linspace(0.0, 2*np.pi, Nt, endpoint=False)
    R, T = np.meshgrid(rv, tv, indexing="ij")

    Res = F_np(R, T, mu0, nu0)
    Res = np.nan_to_num(Res)

    phase = 3.0191 * mu0 * nu0 - T + 11.62159

    modes = {
        "sin(r) sin(phase)": np.sin(R) * np.sin(phase),
        "sin(r)^2 sin(phase)^2": np.sin(R)**2 * np.sin(phase)**2,
        "sin(r)^3 sin(phase)^3": np.sin(R)**3 * np.sin(phase)**3,
        "sin(2r) sin(phase)": np.sin(2*R) * np.sin(phase),
        "cos(r) sin(phase)": np.cos(R) * np.sin(phase),
        "sin(r) sin(2phase)": np.sin(R) * np.sin(2*phase),
        "sin(r) sin(3phase)": np.sin(R) * np.sin(3*phase),
        "sin(3r) sin(phase)": np.sin(3*R) * np.sin(phase),
    }

    print("Residual RMS:", norm(Res.ravel()) / np.sqrt(Res.size))
    print()

    for name, mode in modes.items():
        c = projection_coeff(Res, mode)
        captured = c * mode
        frac = norm(captured.ravel())**2 / max(norm(Res.ravel())**2, 1e-14)
        print(f"{name:30s} coeff={c: .6e}, energy fraction ={frac: .6f}")

analyze_residual_modes(5.0, 5.0)
