from os import system
import pandas as pd
#First get results that we gotta beat for each benchmark
df = pd.read_csv("results_established_weight_update_rules_grid_search_best_by_benchmark_nn.csv")
print(df)
df = df[["Benchmark", "NeuralNet", "MSE"]]
Benchmarks, NeuralNet, MSE = [df[i].tolist() for i in ["Benchmark", "NeuralNet", "MSE"]]
config_idxs = range(1,len(Benchmarks)+1)
benchmarks = {'Hemberg_1': 2, 'Hemberg_2': 2, 'Hemberg_3': 2, 'Hemberg_4': 2, 'Hemberg_5': 2, 'Feynman_1': 5, 'Feynman_2': 9, 'Feynman_3': 7, 'Feynman_4': 8, 'Feynman_5': 6}
print(Benchmarks, NeuralNet, MSE); #['Feynman_1', 'Feynman_1', 'Feynman_1', 'Feynman_2', 'Feynman_2', 'Feynman_2', 'Feynman_3', 'Feynman_3', 'Feynman_3', 'Feynman_4', 'Feynman_4', 'Feynman_4', 'Feynman_5', 'Feynman_5', 'Feynman_5', 'Hemberg_1', 'Hemberg_1', 'Hemberg_1', 'Hemberg_2', 'Hemberg_2', 'Hemberg_2', 'Hemberg_3', 'Hemberg_3', 'Hemberg_3', 'Hemberg_4', 'Hemberg_4', 'Hemberg_4', 'Hemberg_5', 'Hemberg_5', 'Hemberg_5'] [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3] [3.51061, 3.54013, 4.02907, 12.5245, 14.9046, 4.82552, 288984000.0, 292646000.0, 291545000.0, 5.41257, 7.76881, 2.09216, 18.22, 20.7264, 16.4511, 0.316765, 0.388911, 0.379245, 546.92, 490.559, 408.566, 14.9353, 9.0529, 6.96706, 5560.6, 6009.35, 3338.46, 5668.95, 5859.24, 3767.03]
start_idx = 1

filename = "temp_config.txt"
executable_name = 'NeuralNetworks_VecSR'
for idx, benchmark, nn, mse in list(zip(config_idxs, Benchmarks, NeuralNet, MSE))[start_idx:]:
    print("idx, benchmark, nn, mse =", idx, benchmark, nn, mse)
    with open(filename, "w") as f:
        f.write(f'{idx}\n')
        f.write(f'{benchmark}\n')
        f.write(f'{nn}\n')
        f.write(f'{mse}\n')
    #TODO: Write neural net and mse to the file, run the executable
    system(f"./{executable_name}")
#    exit()

#Kill python script by getting pid from `ps -ef | grep python`
#pgrep NeuralNetworks_VecSR
