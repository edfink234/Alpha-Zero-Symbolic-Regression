'''
 ===============
 NUM_EPOCHS = 10
 ===============
 1. 10 benchmarks -> 1410 established weight-update rule configs
    A. For each benchmark, 3 neural nets -> 141 established weight-update rule configs
        I.  Each neural net has 5, 6, 7 layers (including the input layer) with N inputs and 1 output
            a. Neural net 1: {N, {2, "sigmoid"}, {7, "sigmoid"}, {6, "sigmoid"}, {1, "none"}}
                i. Established Weight-Update Rules -> 5+6+6+6+8+4+6+6 = 47 configs
                    - Gradient-Descent: η ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} -> 5 configs
                    - Heavy Ball: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3} -> 6 configs
                    - Nesterov: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3} -> 6 configs
                    - AdaGrad: ε ∈ {1e-8, 1e-6}, η ∈ {1e-3, 3e-3, 1e-2} -> 6 configs
                    - RMSProp: ε ∈ {1e-8}, η ∈ {1e-5, 3e-5, 1e-4, 3e-4}, Ɣ ∈ {0.9, 0.99} -> 8 configs
                    - AdaDelta: ε ∈ {1e-6, 1e-8}, Ɣ ∈ {0.95, 0.99} -> 4 configs
                    - Adam: η ∈ {1e-5, 3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9, 0.95}, β_2 ∈ {0.999} -> 6 configs
                    - AdamW: λ ∈ {1e-5, 1e-4, 1e-3}, η ∈ {3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9}, β_2 ∈ {0.999} -> 6 configs
            b. Neural net 2: {N, {6, "sigmoid"}, {8, "sigmoid"}, {1, "sigmoid}, {5, "none"}, {1, "none"}}
                i. Established Weight-Update Rules
                    - Gradient-Descent: η ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
                    - Heavy Ball: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3}
                    - Nesterov: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3}
                    - AdaGrad: ε ∈ {1e-8, 1e-6}, η ∈ {1e-3, 3e-3, 1e-2}
                    - RMSProp: ε ∈ {1e-8}, η ∈ {1e-5, 3e-5, 1e-4, 3e-4}, Ɣ ∈ {0.9, 0.99}
                    - AdaDelta: ε ∈ {1e-6, 1e-8}, Ɣ ∈ {0.95, 0.99}
                    - Adam: η ∈ {1e-5, 3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9, 0.95}, β_2 ∈ {0.999}
                    - AdamW: λ ∈ {1e-5, 1e-4, 1e-3}, η ∈ {3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9}, β_2 ∈ {0.999}
            c. Neural net 3: {N, {10, "sigmoid}, {9, "sigmoid"}, {8, "sigmoid"}, {10, "none"}, {8, "none"}, {1, "none"}}
                i. Established Weight-Update Rules
                    - Gradient-Descent: η ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
                    - Heavy Ball: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3}
                    - Nesterov: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3}
                    - AdaGrad: ε ∈ {1e-8, 1e-6}, η ∈ {1e-3, 3e-3, 1e-2}
                    - RMSProp: ε ∈ {1e-8}, η ∈ {1e-5, 3e-5, 1e-4, 3e-4}, Ɣ ∈ {0.9, 0.99}
                    - AdaDelta: ε ∈ {1e-6, 1e-8}, Ɣ ∈ {0.95, 0.99}
                    - Adam: η ∈ {1e-5, 3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9, 0.95}, β_2 ∈ {0.999}
                    - AdamW: λ ∈ {1e-5, 1e-4, 1e-3}, η ∈ {3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9}, β_2 ∈ {0.999}
'''

#TODO: See if you can create a list of the configs above in the manner below (`example_config`) and implement a for-loop that loops over each config above and does the below operation

from os import system

#for benchmark in benchmarks:
#    for neural_network in neural_networks:
#        for weight_update_rule in weight_update_rules:
#            for hyper_parameter_combo in hyper_parameter_combos[weight_update_rule]:
#                with open(whateverFileName, "w") as f:
#                    neural_network_layers = f'{benchmarks[benchmark]}' + neural_network[0][1:]
#                    f.write(neural_network_layers+"\n")
#                    f.write(neural_network[1]+"\n")
#                    f.write(benchmark+"\n")
#                    f.write(weight_update_rule+"\n")
#                    f.write(
                    
                
benchmarks = ('Hemberg_1', 'Hemberg_2', 'Hemberg_3', 'Hemberg_4', 'Hemberg_5', 'Feynman_1', 'Feynman_2', 'Feynman_3', 'Feynman_4', 'Feynman_5')
benchmark_Ns = (2, 2, 2, 2, 2, 5, 9, 7, 8, 6)
benchmarks = {'Hemberg_1': 2, 'Hemberg_2': 2, 'Hemberg_3': 2, 'Hemberg_4': 2, 'Hemberg_5': 2, 'Feynman_1': 5, 'Feynman_2': 9, 'Feynman_3': 7, 'Feynman_4': 8, 'Feynman_5': 6}
print(benchmarks)
neural_networks = (('N 2 7 6 1', 'sigmoid sigmoid sigmoid none'), ('N 6 8 1 5 1', 'sigmoid sigmoid sigmoid none none'), ('N 10 9 8 10 8 1', 'sigmoid sigmoid sigmoid none none none'))
weight_update_rules = ("basic", "heavy ball", "NAG", "AdaGrad", "RMSProp", "AdaDelta", "Adam", "AdamW")
default_eta = 0.5
default_theta = 0.01
default_gamma = 0.9
default_epsilon = 0.1
default_beta_1 = 0.9
default_beta_2 = 0.999
default_lambda = 0.01
hyper_parameter_combos = (\
    ({"eta": 1e-5} ,{"eta": 3e-5}, {"eta": 1e-4}, {"eta": 3e-4}, {"eta": 1e-3}), \
    ({"theta": 0.8, "eta": 1e-4}, {"theta": 0.8, "eta": 3e-4}, {"theta": 0.8, "eta": 1e-3}, {"theta": 0.9, "eta": 1e-4}, {"theta": 0.9, "eta": 3e-4}, {"theta": 0.9, "eta": 1e-3}), \
    ({"theta": 0.8, "eta": 1e-4}, {"theta": 0.8, "eta": 3e-4}, {"theta": 0.8, "eta": 1e-3}, {"theta": 0.9, "eta": 1e-4}, {"theta": 0.9, "eta": 3e-4}, {"theta": 0.9, "eta": 1e-3}), \
    ({"epsilon": 1e-8, "eta": 1e-3}, {"epsilon": 1e-8, "eta": 3e-3}, {"epsilon": 1e-8, "eta": 1e-2}, {"epsilon": 1e-6, "eta": 1e-3}, {"epsilon": 1e-6, "eta": 3e-3}, {"epsilon": 1e-6, "eta": 1e-2}), \
    ({"epsilon": 1e-8, "eta": 1e-5, "gamma": 0.9}, {"epsilon": 1e-8, "eta": 1e-5, "gamma": 0.99}, {"epsilon": 1e-8, "eta": 3e-5, "gamma": 0.9}, {"epsilon": 1e-8, "eta": 3e-5, "gamma": 0.99}, {"epsilon": 1e-8, "eta": 1e-4, "gamma": 0.9}, {"epsilon": 1e-8, "eta": 1e-4, "gamma": 0.99}, {"epsilon": 1e-8, "eta": 3e-4, "gamma": 0.9}, {"epsilon": 1e-8, "eta": 3e-4, "gamma": 0.99}), \
    ({"epsilon": 1e-6, "gamma": 0.95}, {"epsilon": 1e-6, "gamma": 0.99}, {"epsilon": 1e-8, "gamma": 0.95}, {"epsilon": 1e-8, "gamma": 0.99}), \
    ({"eta": 1e-5, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999}, {"eta": 1e-5, "epsilon": 1e-8, "beta_1": 0.95, "beta_2": 0.999}, {"eta": 3e-5, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999}, {"eta": 3e-5, "epsilon": 1e-8, "beta_1": 0.95, "beta_2": 0.999}, {"eta": 1e-4, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999}, {"eta": 1e-4, "epsilon": 1e-8, "beta_1": 0.95, "beta_2": 0.999}), \
    ({"lambda": 1e-5, "eta": 3e-5, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999}, {"lambda": 1e-5, "eta": 1e-4, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999}, {"lambda": 1e-4, "eta": 3e-5, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999}, {"lambda": 1e-4, "eta": 1e-4, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999}, {"lambda": 1e-3, "eta": 3e-5, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999}, {"lambda": 1e-3, "eta": 1e-4, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999}))
    
for hyper_parameter_combo in hyper_parameter_combos:
    print(len(hyper_parameter_combo), hyper_parameter_combo, end='\n\n')

example_config = [('layers', '2 2 7 6 1'), ('layer_types', 'sigmoid sigmoid sigmoid none'), ('func_type', 'Hemberg_1'), ('weight_update_rule', 'NAG'), ('eta', '1e-5'), ('theta', '.8'), ('gamma', '.9'), ('epsilon', '1e-8'), ('beta_1', '.9'), ('beta_2', '.999'), ('lambda', '1e-5')]
#print(list(zip(example_config.keys(), example_config.values())))
with open("RunTestsNeuralNetworksVecSR.txt", "w") as f:
    for config_pair in example_config:
        f.write(f'{config_pair[1]}\n')
    
system("cat RunTestsNeuralNetworksVecSR.txt")

