from functools import lru_cache
import matplotlib.pyplot as plt
from os import system

@lru_cache(maxsize=None)
def getNumUnique(depth, N_B, N_U, N_L):
    """
    Number of unique expression trees of exactly `depth`.
    Assumes ordered binary children.
    """
    if depth < 0:
        return 0

    if depth == 0:
        return N_L

    prev = getNumUnique(depth - 1, N_B, N_U, N_L)
    smaller_sum = sum(getNumUnique(j, N_B, N_U, N_L) for j in range(depth - 1))

    return (
        N_U * prev
        + N_B * prev**2
        + 2 * N_B * prev * smaller_sum
    )

N_L = 8  # number of leaves
N_U = 10   # number of unary operators
N_B = 3   # number of binary operators

depths = range(7)
num_binary = range(3, 6)
num_unary = range(8, 11)
num_leaves = range(8, 11)

for N_B in num_binary:
    plt.plot(depths, [getNumUnique(i, N_B, N_U, N_L) for i in depths], label = r"$N_{\mathrm{binary}} = $"f"{N_B}")

plt.yscale('log')
plt.xlabel("Expression-Tree Depth")
plt.ylabel("Number of Unique Expression Trees")
plt.title(r"Unique Expressions, $N_{\mathrm{leaves}} = $"f"{N_L}, "r"$N_{\mathrm{unary}} = $"f"{N_U}")
plt.legend()
plt.savefig(f"/Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/figures_chapter_2/NumUniqueEquationsWith{N_L}Leaves{N_U}Unaries.pdf")
system(f"open /Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/figures_chapter_2/NumUniqueEquationsWith{N_L}Leaves{N_U}Unaries.pdf")
plt.close()

N_B = 3
for N_U in num_unary:
    plt.plot(depths, [getNumUnique(i, N_B, N_U, N_L) for i in depths], label = r"$N_{\mathrm{unary}} = $"f"{N_U}")

plt.yscale('log')
plt.xlabel("Expression-Tree Depth")
plt.ylabel("Number of Unique Expression Trees")
plt.title(r"Unique Expressions, $N_{\mathrm{leaves}} = $"f"{N_L}, "r"$N_{\mathrm{binary}} = $"f"{N_B}")
plt.legend()
plt.savefig(f"/Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/figures_chapter_2/NumUniqueEquationsWith{N_L}Leaves{N_B}Binaries.pdf")
system(f"open /Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/figures_chapter_2/NumUniqueEquationsWith{N_L}Leaves{N_B}Binaries.pdf")
plt.close()

N_U = 10
for N_L in num_leaves:
    plt.plot(depths, [getNumUnique(i, N_B, N_U, N_L) for i in depths], label = r"$N_{\mathrm{leaves}} = $"f"{N_L}")

plt.yscale('log')
plt.xlabel("Expression-Tree Depth")
plt.ylabel("Number of Unique Expression Trees")
plt.title(r"Unique Expressions, $N_{\mathrm{unary}} = $"f"{N_U}, "r"$N_{\mathrm{binary}} = $"f"{N_B}")
plt.legend()
plt.savefig(f"/Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/figures_chapter_2/NumUniqueEquationsWith{N_U}Unaries{N_B}Binaries.pdf")
system(f"open /Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/figures_chapter_2/NumUniqueEquationsWith{N_U}Unaries{N_B}Binaries.pdf")
plt.close()

#for N_U in num_binary:

#print(f'{getNumUnique(depth = 7):e}')
