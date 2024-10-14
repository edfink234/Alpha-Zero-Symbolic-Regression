import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import system

# Define sech function
def sech(x):
    return 1/np.cosh(x)

# Create a grid for x and y values
sigma = 0.2
x_vals = np.linspace(0.1, 2*np.pi, 1000)
y_vals = np.linspace(0.1, 2*np.pi, 1000)
x, y = np.meshgrid(x_vals, y_vals)
x0 = np.pi
y0 = np.pi

# Define the initial condition T0 (equivalent to I_val in your case)
T0 = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

# Define the function for T at a given time t
def T_func(T0, t):
    return T0 ** ((0.2 ** t) / (sech(np.log(np.log(np.pi))) + 0.2))

# Time values to generate plots for
time_values = [0.1, 3.4, 20]
print(time_values)

# Set limits for the plot (matching MATLAB)
z_limits = [0, 1]  # Equivalent to 'limits = [0, 1]' in MATLAB

# Loop over time values and generate the plots
for t in time_values:
    T_vals = T_func(T0, t)
    
    # Create the figure and two subplots
    fig = plt.figure(figsize=(14, 6))
    
    # Left subplot: 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(x, y, T_vals, cmap='jet', edgecolor='none')
    ax1.set_title(f'3D Plot of $T(x, y)$ for $t = {t}$')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_zlabel('$T(x, y)$')
    ax1.set_zlim(z_limits)  # Matching the zlim from MATLAB
    surf.set_clim(z_limits)  # Setting color limits

    # Right subplot: heatmap
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(x, y, T_vals, levels=100, cmap='jet')
    fig.colorbar(contour, ax=ax2, label='$T(x, y)$')
    ax2.set_title(f'Heatmap of $T(x, y)$ for $t = {t}$')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')

    # Save the figure as an SVG file (and optionally convert to PDF)
    plt.tight_layout()
    t_ = str(t).replace(".", "_")
    plt.savefig(f'T_plot_t{t_}_Case2.svg', format='svg')
    system(f"rsvg-convert -f pdf -o T_plot_t{t_}_Case2.pdf T_plot_t{t_}_Case2.svg")
    system(f"rm T_plot_t{t_}_Case2.svg")

    # Close the figure to free up memory
    plt.close(fig)

print(f"PDF files saved for time values: {time_values}")
