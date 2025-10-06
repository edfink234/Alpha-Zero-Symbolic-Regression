from sympy import *
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
from numpy import linalg as LA

sech=lambda x:1/cosh(x)

# Define the polar coordinates
SH = symbols('\\text{SwiftHohenberg} r theta mu nu')
r, theta = symbols('r theta')
mu, nu = 1, 1
# Define the function f as a function of r and theta
GENERIC = False
f = None
if GENERIC:
    f = Function('f')(r, theta)
else:
    f = ((theta / 8) - (sin(theta) * (1.0000085830688477 * sin(r))))
    
print(f"sp.latex(f) =", (latex_f := sp.latex(f)))
latex_f = latex_f.replace(r"(r", r"(\sqrt{x^2 + y^2}")
latex_f = latex_f.replace(r"\theta", r"\arctan{\dfrac{y}{x}}")
print(latex_f)

# Calculate the first Laplacian (Laplacian of f)
laplacian_f = diff(f, r, 2) + (1/r) * diff(f, r) + (1/(r**2)) * diff(f, theta, 2)

# Calculate the double Laplacian (Laplacian of the first Laplacian)
double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)

swift_hohenberg = mu*f + nu*f*f - f*f*f - (f + 2*laplacian_f + double_laplacian_f)

print(f"swift_hohenberg = {swift_hohenberg.evalf()}")

# print(*swift_hohenberg.args, sep="\n")
r_vals, theta_vals = [None]*2
func_vals = None
N = 33
if not GENERIC:
    r_vals, theta_vals = np.meshgrid(np.linspace(0.01, 10, N), np.linspace(0, 2*np.pi, N))
    func = lambdify((r, theta), swift_hohenberg)
    func_vals = func(r_vals, theta_vals)
    print(f"func_vals.size = {func_vals.size}")
    print(f"func_vals.shape = {func_vals.shape}")
    print(f"func_vals = {func_vals}");
    print(f"diff(func_vals, axis = 0) = {np.diff(func_vals, axis = 0)}") #diff(f, theta)
    print(f"diff(func_vals, axis = 1) = {np.diff(func_vals, axis = 1)}") #diff(f, r)
    squared_norm_error = LA.norm(func_vals.flatten())**2
    print(f"squared-norm error = {squared_norm_error}")
    print(sp.multiline_latex(SH, swift_hohenberg, 2).replace(r"\frac", r"\dfrac"))
    mean_squared_error = squared_norm_error / func_vals.size
    print(f"mean-squared_error = {mean_squared_error}")

#ROOT-FINDING#
##############
'''
Given `f` defined above, your assignment is
to use this vector as an initial seed to find a 
numerical root vector for the solution of the `swift_hohenberg`
equation object defined above. Feel free to use numpy and/or scipy
functions to accomplish this
'''

#TODO: Your code goes here:

#IDEA: We want to solve find a vector `f` s.t. F(f(x), x) ~ 0 as
#much as possible, where `F` is the equation `swift_hohenberg`
#we're trying to solve.

#MARK: Next time, Start looking at last response from here:
#https://chatgpt.com/share/e/68e2efd0-839c-8012-b832-5519f7059393

