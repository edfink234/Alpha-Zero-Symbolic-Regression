import pandas as pd
df = pd.read_csv("data.txt", sep = " & ", engine = "python")
print(df.sort_values(by = 'MSE').head(5).to_latex(index = False, caption = "Best 5 Configurations for Case 1", label = "tab:Best_5_Case_1"))
df1 = pd.read_csv("data1.txt", sep = " & ", engine = "python")
print(df1.sort_values(by = 'MSE').head(5).to_latex(index = False, caption = "Best 5 Configurations for Case 2", label = "tab:Best_5_Case_2"))
