import matplotlib.pyplot as plt
import numpy as np

# Data provided
iterations = np.arange(1, 9)
# symbolic regression mse progression
x_sym = [0.201, 0.149, 0.119, 0.078, 0.062, 0.047, 0.04, 0.027]  # 0 < r < 10
y_sym = [0.208, 0.15, 0.032, 0.025, 0.022, 0.019, 0.015, 0.011]   # 0 < r < 100

# finite-diff progression
x_fd = [70.505, 150.771, 142.679, 6.861e6, 4.73e9, 3.785e9, 2.992e9, 2.123e9] # 0 < r < 10
y_fd = [0.763, 0.706, 0.585, 9029.435, 7327.883, 5864.088, 5044.098, 3700.191] # 0 < r < 100

# Plotting configuration
plt.rcParams.update({'font.size': 11, 'axes.labelweight': 'bold'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def create_plot(ax, sym_data, fd_data, title):
    # Left Axis: Symbolic MSE (Linear)
    color_sym = 'tab:blue'
    ax.set_xlabel('Iteration (Model Complexity/Refinement)')
    ax.set_ylabel('Symbolic MSE', color=color_sym)
    lns1 = ax.plot(iterations, sym_data, marker='o', linestyle='-', color=color_sym,
                   linewidth=2, label='Symbolic MSE')
    ax.tick_params(axis='y', labelcolor=color_sym)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Right Axis: Finite Diff Error (Log Scale)
    ax_twin = ax.twinx()
    color_fd = 'tab:red'
    ax_twin.set_ylabel('Finite Diff Error (MSE)', color=color_fd)
    lns2 = ax_twin.plot(iterations, fd_data, marker='s', linestyle='--', color=color_fd,
                        linewidth=2, label='Finite Diff Error')
    ax_twin.set_yscale('log')
    ax_twin.tick_params(axis='y', labelcolor=color_fd)
    
    # Title and Legend
    ax.set_title(title, fontsize=14, pad=15)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Generate both plots
create_plot(ax1, x_sym, x_fd, 'Discrepancy ($0 < r < 10$)')
create_plot(ax2, y_sym, y_fd, 'Discrepancy ($0 < r < 100$)')

plt.tight_layout()
plt.show()
