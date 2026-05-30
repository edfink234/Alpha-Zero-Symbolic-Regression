#!/usr/bin/env python3

import pandas as pd


def main():
    input_file = "results_established_weight_update_rules_grid_search.csv"
    output_file = "results_established_weight_update_rules_grid_search_best_by_benchmark_nn.csv"

    # Read CSV
    df = pd.read_csv(input_file)

    # Make sure MSE is numeric
    df["MSE"] = pd.to_numeric(df["MSE"], errors="coerce")

    # Drop rows with missing MSE, if any
    df = df.dropna(subset=["MSE"])

    # For each (Benchmark, NeuralNet), find index of row with minimum MSE
    idx = df.groupby(["Benchmark", "NeuralNet"])["MSE"].idxmin()

    # Select those rows and sort for readability
    best_df = df.loc[idx].sort_values(["Benchmark", "NeuralNet"]).reset_index(drop=True)

    # Write result
    best_df.to_csv(output_file, index=False)

    print(f"Wrote {len(best_df)} rows to {output_file}")
    print()
    print(best_df)


if __name__ == "__main__":
    main()
