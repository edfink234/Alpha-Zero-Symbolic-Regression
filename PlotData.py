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

# Function to plot the results
def plot_results(time_values, avg_max_scores, std_devs, file_path):
    plt.plot(time_values, avg_max_scores, marker='o', color = "black")
    plt.fill_between(time_values, avg_max_scores - std_devs, avg_max_scores + std_devs)
    plt.title('Average MSE over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Average MSE')
    plt.grid(True)
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
files = 'Hemberg_1PreRandomSearch.txt', 'Hemberg_1PostRandomSearch.txt', 'Hemberg_2PreRandomSearch.txt', 'Hemberg_2PostRandomSearch.txt', 'Hemberg_3PreRandomSearch.txt'
process_files(files)



