import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import system
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm as cm
import matplotlib.colors


# Define the function I and T
def I(x, y):
    return np.exp(-((x - 1.1)**2 + y**2)) / 0.08

def T(x, y, t):
    I_val = I(x, y)
    return I_val - (y * np.exp(x) / (4**I_val)) * (1 - np.tanh(20.2 - t))

# Create a grid for x and y values
x_vals = np.linspace(0.1, 2.1, 1000)
y_vals = np.linspace(-1.1, 1.1, 1000)
x, y = np.meshgrid(x_vals, y_vals)

# Time values to generate plots for
time_values = [0.1, 10.05, 20]
#cmap_ = 'Greys_r'
# Define the custom grey color (RGBA)
# Define the custom colors (RGBA)
grey_color = (0.8274509803921568, 0.8274509803921568, 0.8274509803921568, 1.0)
violet_color = (0.5, 0, 1, 1.0)  # Example color for violet
blue_color = (0, 0, 1, 1.0)  # Example color for blue
inverse_violet_color = [1-i for i in violet_color]  # Example color for inverse violet
inverse_blue_color = [1-i for i in blue_color]  # Example color for inverse blue

# Create a custom colormap that starts with the grey color
#cmap_ = LinearSegmentedColormap.from_list("", [grey_color, "violet", "blue"])
cmap_="coolwarm"

# Loop over time values and generate the plots
for t in time_values:
    T_vals = T(x, y, t)
    
    fig = plt.figure(figsize=(14, 6))
    
    # Left subplot: 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x, y, T_vals, cmap=cmap_, edgecolor='none')
    ax1.set_title(f'3D Plot of $T(x, y)$ for $t = {t}$')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_zlabel('$T(x, y)$')
    
    # Right subplot: heatmap
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(x, y, T_vals, levels=100, cmap=cmap_)
    fig.colorbar(contour, ax=ax2, label='$T(x, y)$')
    ax2.set_title(f'Heatmap of $T(x, y)$ for $t = {t}$')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')

    # Save the figure as an SVG file
    plt.tight_layout()
    t_ = str(t).replace(".","_")
    plt.savefig(f'T_plot_t{t_}_Case1.svg', format='svg')
    system(f"rsvg-convert -f pdf -o T_plot_t{t_}_Case1.pdf T_plot_t{t_}_Case1.svg")
    system(f"rm T_plot_t{t_}_Case1.svg")

    # Close the figure to free up memory
    plt.close(fig)

print(f"PDF files saved for t = {time_values}")

