import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from numpy import linalg as LA
show = False
np.sech = lambda x: 1.0/np.cosh(x)

PERIODIC_IN_THETA = True
PRINT_LATEX_ONLY = False
PlotType = "2D"
# Create edges instead of centers
N = 1000
r_max = 10
r_edges = np.linspace(0.01, r_max, N)
theta_edges = np.linspace(0, 2*np.pi, N, endpoint=False)

round_floats = lambda expr, ndigits: expr.xreplace({f: Float(round(float(f), ndigits)) for f in expr.atoms(Float)})
f_per_idx = 10
mu, nu = 1, 1
r, theta = symbols('r theta')
f =  [sin(r)*sin(theta), \
      sin(r)*sin(theta)+0.604, \
      0.998846776839887*0.999950000416665**(r**4)*sin(r)*sin(theta) + 0.604, \
          0.88898139159952*0.999884875453817**(r**4.03)*sqrt(1 - cos(r)**2)*sin(theta) + 0.760176150613572, \
      -0.28580222883408**(r + 10)*(1.01 - sin(theta))*(167.620651926117*r**7.38905609893065 + 0.000105912014609458) + 0.833098208613807*0.999884875453817**(r**(17/4))*sqrt(1 - cos(r)**2)*sin(theta)  + 0.797073913381706, \
      -182.159206127457*0.28580222883408**(r + 10)*(1.02 - sin(theta))*(r + 0.00999991666708333)**7.50905609893065 + 0.745258709383936*0.999884875453817**(r**4.25)*sqrt(1 - cos(r)**2)*sin(theta + 6.28319)  + 0.815307524508096, \
      -0.28580222883408**(r + 10.02)*(2*r + 0.0137395477321287)**(0.01**(6.28319/(r + 0.01)) + 7.57016955826421)*(0.01**r - sin(theta) + 1) + 0.708762837941528*0.999884875453817**(1.3213487088109*r**4*(1.6*(tanh(.6*r))))*sqrt(1 - cos(r)**2)*sin(0.999999999988989*theta) + 0.845330825627302, \
        -0.285806921654494**(r + 10.0100907998593)*(r**1.00009081398177 + r**((r + 1)**0.0100001666741671))**(0.015**(6.28319/(r + 2)) + 7.59399521303526)*(- sin(theta + cos(theta) + 6.28319) + sin(log(r))) + 0.612417858855597*0.999884875453817**(1.4426686039141*(r + 0.01)**4*(1.6*(tanh(.6*r))))*sqrt(1 - cos(r)**2)*sin(theta) + 0.879032771381193, \
       -0.285806921654494**(r + 10.0139164646307)*(r + (r**0.999993025405072 - 5.00008333556817e-5)**(r**0.01))**(0.0166848951652189**(6.23978883640503/(r + 2)) + 0.000631778468553939*r + 7.59291602260893)*(-sin(theta + cos(theta) + 1/r) + sin(log(r))) + 0.606923362578475*sqrt(1 - cos(r)**2)*(sech(r + 10) + 0.999884875453817)**((r + 0.02)**4.03*(1.58*(tanh(.59*r)))/(sin(sech(r)) + 0.693147180559945))*sin(theta + 6.28319) + 0.886342906953379, \
        -0.285806921654494**(r + sech(r) + 10.0540737282843)*(r + (r**0.999993025405072 - 0.00373485491171487)**(r**0.01) + 0.00880895839265857)**(0.0166848851652189**(6.52587306756706/(r - 0.00781876960101768)) + 0.000702853356876745*r + 7.57798844600458)*(-sin(theta + cos(theta) + 1.25218047866732/r) + sin(log(r)) + sech(r)**(0.0611608608465381*r)) + 0.604143623800264*sqrt(1 - cos(r)**2)*(sech(r + cos(r) + 8.39984108535765) + 0.999884875453817)**((r + 0.0308839840501129)**4.03*(1.58*(tanh(.59*r)))/(tanh(sech(r)) + 0.657323425267437))*sin(theta + 6.32621184985741) + 0.890778403056392, \
        - ((1e-10 + (r + 0.0675028199851666*sin(theta) + 0.315589358780667)**(sqrt(r)*(r + 2.87892339678315)*(5953.65096806617 - r)/((5953.65096806617 - r)**2 + 1e10) + 0.980141037771426)/(0.013519701745416**r*(43.687622183442*r + 8.33814060981745) + r + 0.02*sin(r) + 1.82648253593279))**(((9.2348889286512)/(r + 0.55183450665909) + sin(theta + cos(theta + 0.519039044087815) + 5.8847752990135)*(.5*(1-tanh(1.025*(r-21.2)))))*(r + sin(r - 0.01) + (1.0e-10 + cos(sin(theta)))**(r - 1.58074387559245) + 0.37384427398835 + tanh(r)/(r + 7.97723076614237))))*((.5*(1-tanh(1.775e3*(r-10.01))))) + sqrt(1 - cos(r)**2)*(1.0e-10*0.68688067225485**(8.16109249232708*r) + 0.854229974212735)*(sech(r + cos(r) + 8.39614384384391) + 0.999884853180843)**(1.58799646315658*(r + 0.0308839840501129)**4.01549520152667*(1.57*(tanh(.62*r))))*(0.0100048594945809**(2*r + 5.29438341416157) + 0.7011748940086 - 45.6560816728088/(21934.7382737552))*sin(theta + 18.8962439891879)*(1) - ((1.5707963267949)**(-18.3837681892581) + 0.285811651486423)**(r + sech(r + 0.0561717295263584) + 10.0547134992516)*(r + (r**0.999950000416665 - 0.00364405505237706)**((0.376065617272839**r + r)**0.00999966667999946) + 0.0106243230277353)**(-r**2*exp(-r)/(1 + 1358.42254658947*exp(-r)) + (0.000469282041378069*r + 0.0160184860388267)**((sin(r) + 6.78974430415452)/(r - 0.00781876960101768)) + 7.58897670822754)*(-sin(theta + cos(theta - 0.0144023112886078) + 0.105413950813453) + sin(log(r + 0.390458429297535)) + ((-tanh(0.62*r)+1.01)*(pi/2))**(0.061275433230159*r + 0.00061275433230159)) + (0.00273233753019377**(6.19641677671904 - sin(theta + 0.089280925720443)) + 2.79499001433555e-13 + (6.12323399573677e-17)/(2.19270786451049 - 6.28221254344588*r))*(0.999329299739067*r + 0.453212918064574)**(sin(sqrt(r + 9.07998593378172e-5)) + 11.4130415650481 + 0.00010001/(1.01005016708417 - cos(theta)))*((.5*(1-tanh(1.775e3*(r-10.01))))) + 0.0101001582000134*tanh(10.0327249667171*r + 11.9956250261793) + 0.863191833358681][f_per_idx] \
        if PERIODIC_IN_THETA else \
        (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)

formula_label = latex(round_floats(f, 5), mul_symbol='dot')
print(f'f = {formula_label}')

if PRINT_LATEX_ONLY:
    exit()
# Calculate the first Laplacian (Laplacian of f)
laplacian_f = diff(f, r, 2) + (1/r) * diff(f, r) + (1/(r**2)) * diff(f, theta, 2)

# Calculate the double Laplacian (Laplacian of the first Laplacian)
double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)

swift_hohenberg = mu*f + nu*f*f - f*f*f - (f + 2*laplacian_f + double_laplacian_f)
r_vals, theta_vals = np.meshgrid(np.linspace(0.01, r_max, N), np.linspace(0, 2*pi, N))
func = lambdify((r, theta), swift_hohenberg)
func_vals = func(r_vals, theta_vals)
squared_norm_error = LA.norm(func_vals.flatten())**2
mean_squared_error = squared_norm_error / func_vals.size

# Compute centers
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
R, Theta = np.meshgrid(r_centers, theta_centers)

from numpy import *
# Evaluate function on cell centers
Z = [np.sin(R)*np.sin(Theta), \
     np.sin(R)*np.sin(Theta) + 0.604, \
     0.998846776839887*0.999950000416665**(R**4)*np.sin(R)*np.sin(Theta) + 0.604, \
     0.88898139159952*0.999884875453817**(R**4.03)*np.sqrt(1 - np.cos(R)**2)*np.sin(Theta) - (R/(R + 1.01))**((np.sin(Theta) + 10.0/R)*(R + np.sin(R) + 0.01)) + 0.760176150613572, \
     -0.28580222883408**(R + 10)*(1.01 - np.sin(Theta))*(167.620651926117*R**7.38905609893065 + 0.000105912014609458) + 0.833098208613807*0.999884875453817**(R**(17/4))*np.sqrt(1 - np.cos(R)**2)*np.sin(Theta)  + 0.797073913381706, \
     -182.159206127457*0.28580222883408**(R + 10)*(1.02 - np.sin(Theta))*(R + 0.00999991666708333)**7.50905609893065 + 0.745258709383936*0.999884875453817**(R**4.25)*np.sqrt(1 - np.cos(R)**2)*np.sin(Theta + 6.28319) + 0.815307524508096, \
      -0.28580222883408**(R + 10.02)*(2*R + 0.0137395477321287)**(0.01**(6.28319/(R + 0.01)) + 7.57016955826421)*(0.01**R - np.sin(Theta) + 1) + 0.708762837941528*0.999884875453817**(1.3213487088109*R**4*(1.6*(np.tanh(.6*R))))*np.sqrt(1 - np.cos(R)**2)*np.sin(0.999999999988989*Theta) + 0.845330825627302, \
      -0.285806921654494**(R + 10.0100907998593)*(R**1.00009081398177 + R**((R + 1)**0.0100001666741671))**(0.015**(6.28319/(R + 2)) + 7.59399521303526)*(- np.sin(Theta + np.cos(Theta) + 6.28319) + np.sin(np.log(R))) + 0.612417858855597*0.999884875453817**(1.4426686039141*(R + 0.01)**4*(1.6*(np.tanh(.6*R))))*np.sqrt(1 - np.cos(R)**2)*np.sin(Theta) + 0.879032771381193, \
     -0.285806921654494**(R + 10.0139164646307)*(R + (R**0.999993025405072 - 5.00008333556817e-5)**(R**0.01))**(0.0166848951652189**(6.23978883640503/(R + 2)) + 0.000631778468553939*R + 7.59291602260893)*(-np.sin(Theta + np.cos(Theta) + 1/R) + np.sin(np.log(R))) + 0.606923362578475*np.sqrt(1 - np.cos(R)**2)*(np.sech(R + 10) + 0.999884875453817)**((R + 0.02)**4.03*(1.58*(np.tanh(.59*R)))/(np.sin(np.sech(R)) + 0.693147180559945))*np.sin(Theta + 6.28319) + 0.886342906953379, \
     - ((1e-10 + (R + 0.0675028199851666*np.sin(Theta) + 0.315589358780667)**(np.sqrt(R)*(R + 2.87892339678315)*(5953.65096806617 - R)/((5953.65096806617 - R)**2 + 1e10) + 0.980141037771426)/(0.013519701745416**R*(43.687622183442*R + 8.33814060981745) + R + 0.02*sin(R) + 1.82648253593279))**(((9.2348889286512)/(R + 0.55183450665909) + np.sin(Theta + np.cos(Theta + 0.519039044087815) + 5.8847752990135)*(.5*(1-np.tanh(1.025*(R-21.2)))))*(R + np.sin(R - 0.01) + (1.0e-10 + np.cos(sin(Theta)))**(R - 1.58074387559245) + 0.37384427398835 + np.tanh(R)/(R + 7.97723076614237))))*((.5*(1-np.tanh(1.775e3*(R-10.01))))) + np.sqrt(1 - np.cos(R)**2)*(1.0e-10*0.68688067225485**(8.16109249232708*R) + 0.854229974212735)*(np.sech(R + np.cos(R) + 8.39614384384391) + 0.999884853180843)**(1.58799646315658*(R + 0.0308839840501129)**4.01549520152667*(1.57*(np.tanh(.62*R))))*(0.0100048594945809**(2*R + 5.29438341416157) + 0.7011748940086 - 45.6560816728088/(21934.7382737552))*np.sin(Theta + 18.8962439891879)*(1) - ((1.5707963267949)**(-18.3837681892581) + 0.285811651486423)**(R + np.sech(R + 0.0561717295263584) + 10.0547134992516)*(R + (R**0.999950000416665 - 0.00364405505237706)**((0.376065617272839**R + R)**0.00999966667999946) + 0.0106243230277353)**(-R**2*np.exp(-R)/(1 + 1358.42254658947*np.exp(-R)) + (0.000469282041378069*R + 0.0160184860388267)**((np.sin(R) + 6.78974430415452)/(R - 0.00781876960101768)) + 7.58897670822754)*(-np.sin(Theta + np.cos(Theta - 0.0144023112886078) + 0.105413950813453) + np.sin(np.log(R + 0.390458429297535)) + ((-tanh(0.62*R)+1.01)*(pi/2))**(0.061275433230159*R + 0.00061275433230159)) + (0.00273233753019377**(6.19641677671904 - np.sin(Theta + 0.089280925720443)) + 2.79499001433555e-13 + (6.12323399573677e-17)/(2.19270786451049 - 6.28221254344588*R))*(0.999329299739067*R + 0.453212918064574)**(np.sin(np.sqrt(R + 9.07998593378172e-5)) + 11.4130415650481 + 0.00010001/(1.01005016708417 - np.cos(Theta)))*((.5*(1-np.tanh(1.775e3*(R-10.01))))) + 0.0101001582000134*np.tanh(10.0327249667171*R + 11.9956250261793) + 0.863191833358681][f_per_idx] \
    if PERIODIC_IN_THETA else (((0.148475282221305 * Theta) - (np.sin(Theta) * (1.0000132758892615 * np.sin(R)))) - 0.0922858190550785)

# Convert to Cartesian
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

formula_label = r"$f(r,\theta) = " + formula_label + "$"

# 3D Plot
if PlotType == "3D":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$f(r,\theta)$")
    ax.set_title(f"Swift-Hohenberg 2D Pattern, MSE = {mean_squared_error:.3f}")
    fig.colorbar(surf, shrink=0.5, aspect=10, label=formula_label)

    # Improve viewing angle
    ax.view_init(elev=35, azim=235)
#2D Plot
else:
    fig, ax = plt.subplots()

    # Filled contour plot
    vmin, vmax = -1, 1
    min_, max_ = vmin, vmax

    levels = np.linspace(vmin, vmax, 100)

    # Close the periodic seam in theta for plotting
    Xc = np.vstack([X, X[0:1, :]])
    Yc = np.vstack([Y, Y[0:1, :]])
    Zc = np.vstack([Z, Z[0:1, :]])

    contour = ax.contourf(
        Xc, Yc, Zc,
        cmap="viridis",
        vmin=vmin, vmax=vmax,
        levels=levels,
        extend="both"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Swift-Hohenberg 2D Pattern, MSE = {mean_squared_error:.3f}")

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(formula_label, fontsize=7)
    cbar.set_ticks([min_, 0, max_])

    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
domain_str = f"_r_{int(r_edges[0])}_{int(r_edges[-1])}_theta_{int(theta_edges[0])}_{int(theta_edges[-1])}_N_{N}"
if show:
    plt.show()
else:
    filename = f"SwiftHohenbergPlots/SwiftHohenberg2D{'Periodic'+str(f_per_idx)+domain_str if PERIODIC_IN_THETA else 'NonPeriodic'}.pdf"
    plt.savefig(filename)
    print(f"Saved {filename}")
    from os import system
    filename = filename[:-4]
    system(f"open {filename}.pdf")
    system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 {filename}.pdf --out {filename}.png")
