import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from numpy import sqrt, exp, cos, sin
sech = lambda x: 1.0/np.cosh(x)

PrintFormula = False
# Define the polar coordinates
SH, r, theta, mu, nu = symbols('\\text{SwiftHohenberg} r theta \\mu \\nu')

# Define the function f as a function of r and theta
f = Function('f')(r, theta)

# Calculate the first Laplacian (Laplacian of f)
laplacian_f = diff(f, r, 2) + (1/r) * diff(f, r) + (1/(r**2)) * diff(f, theta, 2)

# Calculate the double Laplacian (Laplacian of the first Laplacian)
double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)

swift_hohenberg = mu*f + nu*f*f - f*f*f - (f + 2*laplacian_f + double_laplacian_f)

if PrintFormula:                
    print(sp.multiline_latex(SH, swift_hohenberg, 2).replace(r"\frac", r"\dfrac"))

# Create edges instead of centers
r_edges = np.linspace(0.0001, 10, 34)         # 34 = 33 + 1
theta_edges = np.linspace(0, 2*np.pi, 34)

# Compute 2D grid of cell edges
R_edges, Theta_edges = np.meshgrid(r_edges, theta_edges)

# Compute centers (optional, for evaluating function)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
R, Theta = np.meshgrid(r_centers, theta_centers)

# Evaluate function on cell centers
Z = ((((Theta * 10.000000) - 43.47847366333008) / 185.5887837532312) + (0.4000400020000667 - ((sin(Theta) * sin(R)) * 1.5707963267948966)))
# Convert edges to Cartesian
X_edges = R_edges * np.cos(Theta_edges)
Y_edges = R_edges * np.sin(Theta_edges)

# Plot using pcolormesh with edges
plt.figure(figsize=(6, 6))
ax_obj = plt.pcolormesh(X_edges, Y_edges, Z, cmap='viridis', norm = "log")
print(ax_obj)
plt.colorbar(label='f(r, θ)')
plt.axis('equal')
#print(*dir(ax_obj), sep = '\n')
#plt.title(r'$f(r, θ) = \dfrac{\theta}{10000^{1 / \sqrt{r}}}$')
plt.savefig("SwiftHohenberg2D.pdf")
from os import system 
system("open SwiftHohenberg2D.pdf")

