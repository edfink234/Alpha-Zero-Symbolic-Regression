
import numpy as np
import matplotlib.pyplot as plt
from os import system
np.random.seed(42) #For Reproducibility
N = 10000
iteration_counter = range(1, N+1)

#ADAGRAD
#=======
numElements = 5
w = np.random.randint(1, 10, numElements) #Epoch 0: Start by randomly initalizing weights
g_t = np.random.randint(1, 10, numElements) #Epoch 1: Compute gradient (vector of derivatives of the loss function wrt each of the weights, i.e., [∂L/∂w_1, ∂L/∂w_2, ∂L/∂w_3, ∂L/∂w_4, ∂L/∂w_5]
g_t_squared = g_t*g_t #Epoch 1: Squaring g_t, i.e., [(∂L/∂w_1)^2, (∂L/∂w_2)^2, (∂L/∂w_3)^2, (∂L/∂w_4)^2, (∂L/∂w_5)^2], A VECTOR OF THE SAME SIZE AS g_t AND w
eps = 1e-8
alpha = 1
average_weight_changes_adagrad = []

print("\nADAGRAD\n=======")
#print(f"Iteration 0: w = {w}")
for i in iteration_counter:
    g_sigma = (eps + g_t_squared)**(-1/2) #Epoch 1, 2, ..., N
    temp = w.copy()
    w = w - alpha * g_sigma * g_t
    average_weight_changes_adagrad.append(np.sum(np.abs(temp-w))/N)

    #print(f"Iteration {i}: w = {w}, g_t = {g_t}")
    g_t = np.random.randint(1, 10, numElements) #gradient at next epoch, i.e.,  #Epoch 2, 3, ..., N+1
    g_t_squared += (g_t*g_t) #gradient squared at next epoch: A VECTOR OF THE SAME SIZE AS g_t AND w, i.e.,  #Epoch 2, 3, ..., N+1

#REGULAR GRADIENT DESCENT
#========================
numElements = 5
w = np.random.randint(1, 10, numElements) #Epoch 0: Start by randomly initalizing weights
g_t = np.random.randint(1, 10, numElements) #Epoch 1: Compute gradient (vector of derivatives of the loss function wrt each of the weights, i.e., [∂L/∂w_1, ∂L/∂w_2, ∂L/∂w_3, ∂L/∂w_4, ∂L/∂w_5]
eps = 1e-8
alpha = 1
average_weight_changes_gradient_descent = []

print("\nREGULAR GRADIENT DESCENT\n========================")
#print(f"Iteration 0: w = {w}")
for i in iteration_counter:
    temp = w.copy()
    w = w - alpha * g_t
    average_weight_changes_gradient_descent.append(np.sum(np.abs(temp-w))/N)

    #print(f"Iteration {i}: w = {w}, g_t = {g_t}")
    g_t = np.random.randint(1, 10, numElements) #gradient at next epoch, i.e.,  #Epoch 2, 3, ..., N+1
    g_t_squared += (g_t*g_t) #gradient squared at next epoch: A VECTOR OF THE SAME SIZE AS g_t AND w, i.e.,  #Epoch 2, 3, ..., N+1


#RMS Prop
#========
print("\nRMSProp\n========================")
numElements = 5
w = np.random.randint(1, 10, numElements) #Epoch 0: Start by randomly initalizing weights
g_t = np.random.randint(1, 10, numElements) #Epoch 1: Compute gradient (vector of derivatives of the loss function wrt each of the weights, i.e., [∂L/∂w_1, ∂L/∂w_2, ∂L/∂w_3, ∂L/∂w_4, ∂L/∂w_5]
eps = 1e-8
gamma = 0.9
alpha = 1
E_g_t_squared = 0 + (1-gamma)*g_t*g_t
average_weight_changes_rms_prop = []

#print(f"Iteration 0: w = {w}")
for i in iteration_counter:
    temp = w.copy()
    Delta_w = (-alpha / (E_g_t_squared + eps)**.5)*g_t
    w = w + Delta_w
    average_weight_changes_rms_prop.append(np.sum(np.abs(temp-w))/N)
    #print(f"Iteration {i}: w = {w}, g_t = {g_t}")
    g_t = np.random.randint(1, 10, numElements) #gradient at next epoch, i.e.,  #Epoch 2, 3, ..., N+1
    E_g_t_squared = gamma*E_g_t_squared + (1-gamma)*g_t*g_t
    

plt.plot(iteration_counter, average_weight_changes_adagrad, label = "Adagrad")
plt.plot(iteration_counter, average_weight_changes_gradient_descent, label = "Gradient Descent")
plt.plot(iteration_counter, average_weight_changes_rms_prop, label = "RmsProp")
plt.xlabel("Iteration Number")
plt.ylabel("Average weight change")
plt.title("Average Weight Change vs Iteration #")
plt.legend()
plt.tight_layout()
plt.yscale("log")
plt.savefig("AverageWeightChangeVsIterationNumberDifferentWeightUpdateRules.png", dpi = 5*96)
system("open AverageWeightChangeVsIterationNumberDifferentWeightUpdateRules.png")

print("\n"*3)









