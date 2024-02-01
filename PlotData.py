import csv
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to read CSV file and calculate average max_score
def process_csv(file_path):
    time_values = []
    avg_max_scores = []
    std_devs = []

    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        
        for row in csvreader:
            time_values.append(int(row[0]))
            avg_max_scores.append(np.mean([(1/float(value))-1 for value in row[1:]]))
            
            std_devs.append(np.std([(1/float(value))-1 for value in row[1:]]))
    
    return time_values, np.array(avg_max_scores), np.array(std_devs), file_path.strip(".txt")

# Function to plot results
def plot_results(time_values, avg_max_scores, std_devs, file_path):
    fig, ax = plt.subplots()
    ax.plot(time_values, avg_max_scores, marker='o', color = "black")
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.fill_between(time_values, avg_max_scores - std_devs, avg_max_scores + std_devs)
    plt.title('Average MSE over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Average MSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}.svg")
    plt.close()
    os.system(f"rsvg-convert -f pdf -o {file_path}.pdf {file_path}.svg")
    os.system(f"rm {file_path}.svg")

def process_files(files):
    for file_path in files:
        # Process the CSV file and plot the results
        time_values, avg_max_scores, std_devs, file_path = process_csv(file_path)
        plot_results(time_values, avg_max_scores, std_devs, file_path)
        
# Specify the path to your CSV file
Hemberg_Files = 'Hemberg_Benchmarks/Hemberg_1PreRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_1PostRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_2PreRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_2PostRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_3PreRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_3PostRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_4PreRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_4PostRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_5PreRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_5PostRandomSearch.txt', 'Hemberg_Benchmarks/Hemberg_1PreMCTS.txt', 'Hemberg_Benchmarks/Hemberg_1PostMCTS.txt', 'Hemberg_Benchmarks/Hemberg_2PreMCTS.txt', 'Hemberg_Benchmarks/Hemberg_2PostMCTS.txt', 'Hemberg_Benchmarks/Hemberg_3PreMCTS.txt', 'Hemberg_Benchmarks/Hemberg_3PostMCTS.txt', 'Hemberg_Benchmarks/Hemberg_4PreMCTS.txt', 'Hemberg_Benchmarks/Hemberg_4PostMCTS.txt', 'Hemberg_Benchmarks/Hemberg_5PreMCTS.txt', 'Hemberg_Benchmarks/Hemberg_5PostMCTS.txt', 'Hemberg_Benchmarks/Hemberg_1PrePSO.txt', 'Hemberg_Benchmarks/Hemberg_1PostPSO.txt', 'Hemberg_Benchmarks/Hemberg_2PrePSO.txt', 'Hemberg_Benchmarks/Hemberg_2PostPSO.txt', 'Hemberg_Benchmarks/Hemberg_3PrePSO.txt', 'Hemberg_Benchmarks/Hemberg_3PostPSO.txt', 'Hemberg_Benchmarks/Hemberg_4PrePSO.txt', 'Hemberg_Benchmarks/Hemberg_4PostPSO.txt', 'Hemberg_Benchmarks/Hemberg_5PrePSO.txt', 'Hemberg_Benchmarks/Hemberg_5PostPSO.txt', 'Hemberg_Benchmarks/Hemberg_1PreGP.txt', 'Hemberg_Benchmarks/Hemberg_1PostGP.txt', 'Hemberg_Benchmarks/Hemberg_2PreGP.txt', 'Hemberg_Benchmarks/Hemberg_2PostGP.txt','Hemberg_Benchmarks/Hemberg_3PreGP.txt', 'Hemberg_Benchmarks/Hemberg_3PostGP.txt', 'Hemberg_Benchmarks/Hemberg_4PreGP.txt', 'Hemberg_Benchmarks/Hemberg_4PostGP.txt', 'Hemberg_Benchmarks/Hemberg_5PreGP.txt', 'Hemberg_Benchmarks/Hemberg_5PostGP.txt', 'Hemberg_Benchmarks/Hemberg_1PreSimulatedAnnealing.txt', 'Hemberg_Benchmarks/Hemberg_1PostSimulatedAnnealing.txt', 'Hemberg_Benchmarks/Hemberg_2PreSimulatedAnnealing.txt', 'Hemberg_Benchmarks/Hemberg_2PostSimulatedAnnealing.txt', 'Hemberg_Benchmarks/Hemberg_3PreSimulatedAnnealing.txt', 'Hemberg_Benchmarks/Hemberg_3PostSimulatedAnnealing.txt', 'Hemberg_Benchmarks/Hemberg_4PreSimulatedAnnealing.txt', 'Hemberg_Benchmarks/Hemberg_4PostSimulatedAnnealing.txt', 'Hemberg_Benchmarks/Hemberg_5PreSimulatedAnnealing.txt', 'Hemberg_Benchmarks/Hemberg_5PostSimulatedAnnealing.txt'

def PaperPlots(files):
    for i in range(0, len(files), 2):
        fig, ax = plt.subplots()
        title = f'Prefix vs Postfix: {files[i][files[i].index("/")+1:].strip(".txt").replace("Pre"," ")}'

        time_values, avg_max_scores, std_devs, file_path = process_csv(files[i])
        ax.plot(time_values, avg_max_scores, marker='o', color = "black", label = "prefix")
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.fill_between(time_values, avg_max_scores - std_devs, avg_max_scores + std_devs, alpha = 0.5)

        time_values, avg_max_scores, std_devs, file_path = process_csv(files[i+1])
        ax.plot(time_values, avg_max_scores, marker='o', color = "red", label = "postfix")
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.fill_between(time_values, avg_max_scores - std_devs, avg_max_scores + std_devs, alpha = 0.5)

        file_path = f'{files[i].strip(".txt").replace("Pre","").replace("/", "/PrePost")}'
        plt.title(title)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Average MSE')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{file_path}.svg")
        plt.close()
        os.system(f"rsvg-convert -f pdf -o {file_path}.pdf {file_path}.svg")
        os.system(f"rm {file_path}.svg")

AIFeynman_Files = 'Feynman_1PreRandomSearch.txt', 'Feynman_1PostRandomSearch.txt', 'Feynman_2PreRandomSearch.txt', 'Feynman_2PostRandomSearch.txt', 'Feynman_3PreRandomSearch.txt', 'Feynman_3PostRandomSearch.txt', 'Feynman_4PreRandomSearch.txt', 'Feynman_4PostRandomSearch.txt', 'Feynman_5PreRandomSearch.txt', 'Feynman_5PostRandomSearch.txt', 'Feynman_1PreMCTS.txt', 'Feynman_1PostMCTS.txt', 'Feynman_2PreMCTS.txt', 'Feynman_2PostMCTS.txt', 'Feynman_3PreMCTS.txt', 'Feynman_3PostMCTS.txt', 'Feynman_4PreMCTS.txt', 'Feynman_4PostMCTS.txt', 'Feynman_5PreMCTS.txt', 'Feynman_5PostMCTS.txt', 'Feynman_1PrePSO.txt', 'Feynman_1PostPSO.txt', 'Feynman_2PrePSO.txt', 'Feynman_2PostPSO.txt', 'Feynman_3PrePSO.txt', 'Feynman_3PostPSO.txt', 'Feynman_4PrePSO.txt', 'Feynman_4PostPSO.txt', 'Feynman_5PrePSO.txt', 'Feynman_5PostPSO.txt', 'Feynman_1PreGP.txt', 'Feynman_1PostGP.txt', 'Feynman_2PreGP.txt', 'Feynman_2PostGP.txt','Feynman_3PreGP.txt', 'Feynman_3PostGP.txt', 'Feynman_4PreGP.txt', 'Feynman_4PostGP.txt', 'Feynman_5PreGP.txt', 'Feynman_5PostGP.txt', 'Feynman_1PreSimulatedAnnealing.txt', 'Feynman_1PostSimulatedAnnealing.txt', 'Feynman_2PreSimulatedAnnealing.txt', 'Feynman_2PostSimulatedAnnealing.txt', 'Feynman_3PreSimulatedAnnealing.txt', 'Feynman_3PostSimulatedAnnealing.txt', 'Feynman_4PreSimulatedAnnealing.txt', 'Feynman_4PostSimulatedAnnealing.txt', 'Feynman_5PreSimulatedAnnealing.txt', 'Feynman_5PostSimulatedAnnealing.txt'

#process_files(Hemberg_Files)
process_files(AIFeynman_Files[:46])
#PaperPlots(Hemberg_Files)


