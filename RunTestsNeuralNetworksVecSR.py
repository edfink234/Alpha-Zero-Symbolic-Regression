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

example_config = [('layers', '2 2 7 6 1'), ('layer_types', 'sigmoid sigmoid sigmoid none'), ('func_type', 'Hemberg_1'), ('weight_update_rule', 'NAG'), ('eta', '1e-5'), ('theta', '.8'), ('gamma', '.9'), ('epsilon', '1e-8'), ('beta_1', '.9'), ('beta_2', '.999'), ('lambda', '1e-5')]
#print(list(zip(example_config.keys(), example_config.values())))
with open("RunTestsNeuralNetworksVecSR.txt", "w") as f:
    for config_pair in example_config:
        f.write(f'{config_pair[1]}\n')

system("cat RunTestsNeuralNetworksVecSR.txt")

