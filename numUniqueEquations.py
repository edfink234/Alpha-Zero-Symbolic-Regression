from functools import lru_cache
import matplotlib.pyplot as plt
from os import system


@lru_cache(maxsize=None)
def getNumUnique(depth, N_B, N_BC, N_U, N_L):
    """
    Number of unique expressions of exactly `depth`.

    Parameters
    ----------
    depth : int
        Exact expression-tree depth.
    N_B : int
        Number of non-commuting binary operators.
    N_BC : int
        Number of commuting binary operators (+, *, etc.).
    N_U : int
        Number of unary operators.
    N_L : int
        Number of leaves.

    Non-commuting binaries distinguish (a,b) from (b,a).
    Commuting binaries count those two expressions only once.
    """
    if depth < 0:
        return 0

    if depth == 0:
        return N_L

    prev = getNumUnique(depth - 1, N_B, N_BC, N_U, N_L)

    smaller_sum = sum(
        getNumUnique(j, N_B, N_BC, N_U, N_L)
        for j in range(depth - 1)
    )

    # Unary operators:
    unary = N_U * prev

    # Non-commuting binaries:
    # At least one child must have depth exactly depth-1.
    noncommuting_binary = (
        N_B * prev**2
        + 2 * N_B * prev * smaller_sum
    )

    # Commuting binaries:
    #
    # Case 1: both children have depth depth-1.
    # Number of unordered pairs with repetition:
    # prev * (prev + 1) / 2
    #
    # Case 2: one child has depth depth-1 and the other has
    # smaller depth. Since the operator commutes, there is
    # no factor of 2.
    commuting_binary = N_BC * (
        prev * (prev + 1) // 2
        + prev * smaller_sum
    )

    return unary + noncommuting_binary + commuting_binary


# ------------------------------------------------------------
# Baseline configuration
# ------------------------------------------------------------

N_L = 8
N_U = 10
N_B = 3
N_BC = 2

depths = range(7)

num_binary = range(3, 6)
num_commuting_binary = range(0, 3)
num_unary = range(8, 11)
num_leaves = range(8, 11)


# ------------------------------------------------------------
# Vary number of NON-COMMUTING binary operators
# ------------------------------------------------------------

for n_binary in num_binary:
    plt.plot(
        depths,
        [
            getNumUnique(i, n_binary, N_BC, N_U, N_L)
            for i in depths
        ],
        label=rf"$N_B = {n_binary}$"
    )

plt.yscale("log")
plt.xlabel("Expression-Tree Depth")
plt.ylabel("Number of Unique Expressions")
plt.title(
    rf"Unique Expressions, "
    rf"$N_L={N_L}$, $N_U={N_U}$, $N_{{BC}}={N_BC}$"
)
plt.legend()

filename = (
    f"/Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/"
    f"figures_chapter_2/"
    f"NumUniqueEquationsWith{N_L}Leaves"
    f"{N_U}Unaries{N_BC}CommutingBinaries.pdf"
)

plt.savefig(filename)
system(f"open {filename}")
plt.close()


# ------------------------------------------------------------
# Vary number of COMMUTING binary operators
# ------------------------------------------------------------

for n_commuting in num_commuting_binary:
    plt.plot(
        depths,
        [
            getNumUnique(i, N_B, n_commuting, N_U, N_L)
            for i in depths
        ],
        label=rf"$N_{{BC}} = {n_commuting}$"
    )

plt.yscale("log")
plt.xlabel("Expression-Tree Depth")
plt.ylabel("Number of Unique Expressions")
plt.title(
    rf"Unique Expressions, "
    rf"$N_L={N_L}$, $N_U={N_U}$, $N_B={N_B}$"
)
plt.legend()

filename = (
    f"/Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/"
    f"figures_chapter_2/"
    f"NumUniqueEquationsWith{N_L}Leaves"
    f"{N_U}Unaries{N_B}NonCommutingBinaries.pdf"
)

plt.savefig(filename)
system(f"open {filename}")
plt.close()


# ------------------------------------------------------------
# Vary number of unary operators
# ------------------------------------------------------------

for n_unary in num_unary:
    plt.plot(
        depths,
        [
            getNumUnique(i, N_B, N_BC, n_unary, N_L)
            for i in depths
        ],
        label=rf"$N_U = {n_unary}$"
    )

plt.yscale("log")
plt.xlabel("Expression-Tree Depth")
plt.ylabel("Number of Unique Expressions")
plt.title(
    rf"Unique Expressions, "
    rf"$N_L={N_L}$, $N_B={N_B}$, $N_{{BC}}={N_BC}$"
)
plt.legend()

filename = (
    f"/Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/"
    f"figures_chapter_2/"
    f"NumUniqueEquationsWith{N_L}Leaves"
    f"{N_B}Binaries{N_BC}CommutingBinaries.pdf"
)

plt.savefig(filename)
system(f"open {filename}")
plt.close()


# ------------------------------------------------------------
# Vary number of leaves
# ------------------------------------------------------------

for n_leaves in num_leaves:
    plt.plot(
        depths,
        [
            getNumUnique(i, N_B, N_BC, N_U, n_leaves)
            for i in depths
        ],
        label=rf"$N_L = {n_leaves}$"
    )

plt.yscale("log")
plt.xlabel("Expression-Tree Depth")
plt.ylabel("Number of Unique Expressions")
plt.title(
    rf"Unique Expressions, "
    rf"$N_U={N_U}$, $N_B={N_B}$, $N_{{BC}}={N_BC}$"
)
plt.legend()

filename = (
    f"/Users/edwardfinkelstein/SDSU_UCI/PhD-Thesis/"
    f"figures_chapter_2/"
    f"NumUniqueEquationsWith{N_U}Unaries"
    f"{N_B}Binaries{N_BC}CommutingBinaries.pdf"
)

plt.savefig(filename)
system(f"open {filename}")
plt.close()


# ------------------------------------------------------------
# Example value
# ------------------------------------------------------------

print(
    f"{getNumUnique(5, N_B, N_BC, N_U, N_L):e}"
)
