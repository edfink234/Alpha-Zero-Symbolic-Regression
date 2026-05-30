import numpy as np

#ADAGRAD
#=======
numElements = 5
w = np.random.randint(1, 10, numElements) #Epoch 0: Start by randomly initalizing weights
g_t = np.random.randint(1, 10, numElements) #Epoch 1: Compute gradient (vector of derivatives of the loss function wrt each of the weights, i.e., [∂L/∂w_1, ∂L/∂w_2, ∂L/∂w_3, ∂L/∂w_4, ∂L/∂w_5]
g_t_squared = g_t*g_t #Epoch 1: Squaring g_t, i.e., [(∂L/∂w_1)^2, (∂L/∂w_2)^2, (∂L/∂w_3)^2, (∂L/∂w_4)^2, (∂L/∂w_5)^2], A VECTOR OF THE SAME SIZE AS g_t AND w
eps = 1e-8
alpha = 1

N = 5
print("\nADAGRAD\n=======")
print(f"Iteration 0: w = {w}")
for i in range(1, N+1):
    g_sigma = (eps + g_t_squared)**(-1/2) #Epoch 1, 2, ..., N
    w = w - alpha * g_sigma * g_t
    print(f"Iteration {i}: w = {w}, g_t = {g_t}")
    g_t = np.random.randint(1, 10, numElements) #gradient at next epoch, i.e.,  #Epoch 2, 3, ..., N+1
    g_t_squared += (g_t*g_t) #gradient squared at next epoch: A VECTOR OF THE SAME SIZE AS g_t AND w, i.e.,  #Epoch 2, 3, ..., N+1

#REGULAR GRADIENT DESCENT
#========================
numElements = 5
w = np.random.randint(1, 10, numElements) #Epoch 0: Start by randomly initalizing weights
g_t = np.random.randint(1, 10, numElements) #Epoch 1: Compute gradient (vector of derivatives of the loss function wrt each of the weights, i.e., [∂L/∂w_1, ∂L/∂w_2, ∂L/∂w_3, ∂L/∂w_4, ∂L/∂w_5]
eps = 1e-8
alpha = 1
N = 5

print("\nREGULAR GRADIENT DESCENT\n========================")
print(f"Iteration 0: w = {w}")
for i in range(1, N+1):
    w = w - alpha * g_t
    print(f"Iteration {i}: w = {w}, g_t = {g_t}")
    g_t = np.random.randint(1, 10, numElements) #gradient at next epoch, i.e.,  #Epoch 2, 3, ..., N+1
    g_t_squared += (g_t*g_t) #gradient squared at next epoch: A VECTOR OF THE SAME SIZE AS g_t AND w, i.e.,  #Epoch 2, 3, ..., N+1
