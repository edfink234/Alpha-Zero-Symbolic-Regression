import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter, FormatStrFormatter



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

def save_plot(file_path):
    plt.savefig(f"{file_path}.svg", bbox_inches='tight')
    plt.close()
    os.system(f"rsvg-convert -f pdf -o {file_path}.pdf {file_path}.svg")
    os.system(f"rm {file_path}.svg")
    
# Function to plot results
def plot_results(time_values, avg_max_scores, std_devs, file_path, save = True, label = "", ax = "new", legend = False, y_scale = 'linear'):
    if ax == "new":
        fig, ax = plt.subplots()
    ax.plot(time_values, avg_max_scores, marker='o', label = label)
    if y_scale == 'linear':
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.fill_between(time_values, avg_max_scores - std_devs, avg_max_scores + std_devs)
    ax.set_title('Average MSE over Time')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Average MSE')
    ax.set_yscale(y_scale)
    ax.grid(True)
    
    
    if legend:
        ax.legend()
    plt.tight_layout()
    if save:
        save_plot(file_path)

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

AIFeynman_Files = 'AIFeynman_Benchmarks/Feynman_1PreRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_1PostRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_2PreRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_2PostRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_3PreRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_3PostRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_4PreRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_4PostRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_5PreRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_5PostRandomSearch.txt', 'AIFeynman_Benchmarks/Feynman_1PreMCTS.txt', 'AIFeynman_Benchmarks/Feynman_1PostMCTS.txt', 'AIFeynman_Benchmarks/Feynman_2PreMCTS.txt', 'AIFeynman_Benchmarks/Feynman_2PostMCTS.txt', 'AIFeynman_Benchmarks/Feynman_3PreMCTS.txt', 'AIFeynman_Benchmarks/Feynman_3PostMCTS.txt', 'AIFeynman_Benchmarks/Feynman_4PreMCTS.txt', 'AIFeynman_Benchmarks/Feynman_4PostMCTS.txt', 'AIFeynman_Benchmarks/Feynman_5PreMCTS.txt', 'AIFeynman_Benchmarks/Feynman_5PostMCTS.txt', 'AIFeynman_Benchmarks/Feynman_1PrePSO.txt', 'AIFeynman_Benchmarks/Feynman_1PostPSO.txt', 'AIFeynman_Benchmarks/Feynman_2PrePSO.txt', 'AIFeynman_Benchmarks/Feynman_2PostPSO.txt', 'AIFeynman_Benchmarks/Feynman_3PrePSO.txt', 'AIFeynman_Benchmarks/Feynman_3PostPSO.txt', 'AIFeynman_Benchmarks/Feynman_4PrePSO.txt', 'AIFeynman_Benchmarks/Feynman_4PostPSO.txt', 'AIFeynman_Benchmarks/Feynman_5PrePSO.txt', 'AIFeynman_Benchmarks/Feynman_5PostPSO.txt', 'AIFeynman_Benchmarks/Feynman_1PreGP.txt', 'AIFeynman_Benchmarks/Feynman_1PostGP.txt', 'AIFeynman_Benchmarks/Feynman_2PreGP.txt', 'AIFeynman_Benchmarks/Feynman_2PostGP.txt','AIFeynman_Benchmarks/Feynman_3PreGP.txt', 'AIFeynman_Benchmarks/Feynman_3PostGP.txt', 'AIFeynman_Benchmarks/Feynman_4PreGP.txt', 'AIFeynman_Benchmarks/Feynman_4PostGP.txt', 'AIFeynman_Benchmarks/Feynman_5PreGP.txt', 'AIFeynman_Benchmarks/Feynman_5PostGP.txt', 'AIFeynman_Benchmarks/Feynman_1PreSimulatedAnnealing.txt', 'AIFeynman_Benchmarks/Feynman_1PostSimulatedAnnealing.txt', 'AIFeynman_Benchmarks/Feynman_2PreSimulatedAnnealing.txt', 'AIFeynman_Benchmarks/Feynman_2PostSimulatedAnnealing.txt', 'AIFeynman_Benchmarks/Feynman_3PreSimulatedAnnealing.txt', 'AIFeynman_Benchmarks/Feynman_3PostSimulatedAnnealing.txt', 'AIFeynman_Benchmarks/Feynman_4PreSimulatedAnnealing.txt', 'AIFeynman_Benchmarks/Feynman_4PostSimulatedAnnealing.txt', 'AIFeynman_Benchmarks/Feynman_5PreSimulatedAnnealing.txt', 'AIFeynman_Benchmarks/Feynman_5PostSimulatedAnnealing.txt'

#process_files(Hemberg_Files)
#process_files(AIFeynman_Files)
#PaperPlots(Hemberg_Files)
#PaperPlots(AIFeynman_Files)

def DiscoverySciencePlots(*Benchmark_File_Lists):

    for Benchmark_File_List, PlotFilePrefix in Benchmark_File_Lists:
        for i in range(1, 6):
            Benchmark_Files = filter(lambda file: f'{i}' in file, Benchmark_File_List)
            fig, ax = plt.subplots(1, 2, width_ratios=[2, 1])
            xscale = 'log'
            ax[1].grid(True)
            ax[1].tick_params(axis='x', which="both", rotation=50)

            for j, file_path in enumerate(Benchmark_Files):
                time_values, avg_max_scores, std_devs, file_path = process_csv(file_path)
                label = 'Prefix' if 'Pre' in file_path else 'Postfix'
                if 'RandomSearch' in file_path:
                    label += " Random Search"
                elif 'MCTS' in file_path:
                    label += " MCTS"
                elif 'PSO' in file_path:
                    label += " PSO"
                elif 'GP' in file_path:
                    label += " GP"
                elif 'SimulatedAnnealing' in file_path:
                    label += " Simulated Annealing"
                
                plot_results(time_values, avg_max_scores, std_devs, file_path, save = False, label = label, ax = ax[0], y_scale = 'log', legend = True)
                
                xscale = 'symlog' if avg_max_scores[-1] == 0.0 else xscale
                
                ax[1].errorbar(x = avg_max_scores[-1], y = j, xerr = std_devs[-1], linestyle = 'dotted', marker='o', markersize=6, capsize = 3)
                ax[1].set_title(r'Final $\overline{\mathrm{MSE}}\pm\mathrm{std}$')
                ax[1].set_xlabel('Final Average MSE')
#
#
            ax[1].yaxis.set_tick_params(left = False, labelleft = False)

            ax[1].set_yscale('linear')
            ax[1].set_xscale(xscale)
            save_plot(f"{PlotFilePrefix}{i}")
        
        #Feynman Benchmarks

DiscoverySciencePlots((Hemberg_Files, "Hemberg_Benchmarks/Hemberg_Benchmark_"), (AIFeynman_Files, "AIFeynman_Benchmarks/Feynman_Benchmark_"))

#TODO: MAYBE Make Plots of Final Means and Stds if you can get Discovery Science Paper to 13 pages
