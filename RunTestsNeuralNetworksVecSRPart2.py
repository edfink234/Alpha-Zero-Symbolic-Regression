from os import system
import pandas as pd
#First get results that we gotta beat for each benchmark
df = pd.read_csv("results_established_weight_update_rules_grid_search_best_by_benchmark_nn.csv")
print(df)
df = df[["Benchmark", "NeuralNet", "MSE"]]
Benchmarks, NeuralNet, MSE = [df[i].tolist() for i in ["Benchmark", "NeuralNet", "MSE"]]
config_idxs = range(1,len(Benchmarks)+1)
benchmarks = {'Hemberg_1': 2, 'Hemberg_2': 2, 'Hemberg_3': 2, 'Hemberg_4': 2, 'Hemberg_5': 2, 'Feynman_1': 5, 'Feynman_2': 9, 'Feynman_3': 7, 'Feynman_4': 8, 'Feynman_5': 6}
print(Benchmarks, NeuralNet, MSE); #exit()
filename = "temp_config.txt"
executable_name = 'NeuralNetworks_VecSR'
for idx, benchmark, nn, mse in zip(config_idxs, Benchmarks, NeuralNet, MSE):
    print("idx, benchmark, nn, mse =", idx, benchmark, nn, mse)
    with open(filename, "w") as f:
        f.write(f'{idx}\n')
        f.write(f'{benchmark}\n')
        f.write(f'{nn}\n')
        f.write(f'{mse}\n')
    #TODO: Write neural net and mse to the file, run the executable
    system(f"./{executable_name}")
    exit()

