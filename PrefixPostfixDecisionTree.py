Alg_Map = {1: 'RS', 2: 'MCTS', 3: 'PSO', 4: 'GP', 5: 'SA'}
Label_Map = {1: "postfix did better", 2: "postfix did the same", 3: "postfix did worse"}

import pandas as pd
from io import StringIO
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

data = '''
Algorithm,Depth,NumberOfInputVariables,NumberOfTokensPerUnitDepth,Label
1,4,2,2.5,3
1,4,2,3.5,3
1,5,2,2.8,3
1,9,2,4.44,1
1,10,2,5.2,1
1,4,5,3,2
1,5,9,4.4,1
1,6,7,3.5,3
1,7,8,3.14,2
1,8,6,3.75,2
2,4,2,2.5,3
2,4,2,3.5,1
2,5,2,2.8,1
2,9,2,4.44,1
2,10,2,5.2,1
2,4,5,3,3
2,5,9,4.4,3
2,6,7,3.5,1
2,7,8,3.14,3
2,8,6,3.75,3
3,4,2,2.5,2
3,4,2,3.5,2
3,5,2,2.8,2
3,9,2,4.44,2
3,10,2,5.2,2
3,4,5,3,3
3,5,9,4.4,1
3,6,7,3.5,3
3,7,8,3.14,2
3,8,6,3.75,2
4,4,2,2.5,2
4,4,2,3.5,2
4,5,2,2.8,2
4,9,2,4.44,1
4,10,2,5.2,1
4,4,5,3,1
4,5,9,4.4,1
4,6,7,3.5,3
4,7,8,3.14,2
4,8,6,3.75,2
5,4,2,2.5,2
5,4,2,3.5,3
5,5,2,2.8,2
5,9,2,4.44,2
5,10,2,5.2,2
5,4,5,3,2
5,5,9,4.4,3
5,6,7,3.5,3
5,7,8,3.14,2
5,8,6,3.75,2
'''

# Creating a DataFrame
df = pd.read_csv(StringIO(data))
print(df)
# Extracting features and labels
X = df.drop('Label', axis=1)
y = df['Label']

# Creating and training the Decision Tree model
model = DecisionTreeClassifier(max_depth = 4, criterion = "entropy", ccp_alpha = 0.09, random_state = 3)
model.fit(X, y)

# Visualizing the decision tree
plt.figure(figsize=(15, 7))
annotations = plot_tree(model, filled = True, feature_names=list(X.columns), class_names=['better than prefix', 'same as prefix', 'worse than prefix'], rounded = True, proportion = True, fontsize = 10, precision = 1, impurity = True, label = 'root')
print(*(i for i in annotations), sep='\n')
#print(*dir(annotations[0]), sep='\n')
plt.savefig("PrefixPostfixDecisionTree.svg")
os.system(f"rsvg-convert -f pdf -o PrefixPostfixDecisionTree.pdf PrefixPostfixDecisionTree.svg")
os.system(f"rm PrefixPostfixDecisionTree.svg")

# Predicting on the training data
y_pred = model.predict(X)

# Calculating accuracy on the training data
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy on the training data: {accuracy}")
