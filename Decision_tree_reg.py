from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.tree import export_graphviz
import pydot

# Generate a synthetic regression dataset for demonstration
X, y = make_regression(n_samples=100, n_features=1, noise=0.2, random_state=42)

# Create a Decision Tree Regressor
tree_reg = DecisionTreeRegressor(max_depth=3)
tree_reg.fit(X, y)

# Export the decision tree to a Graphviz format
dot_data = export_graphviz(tree_reg, out_file=None, filled=True, rounded=True, special_characters=True)

# Create a Graphviz graph from the DOT data
graph = pydot.graph_from_dot_data(dot_data)[0]

# Create a Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Convert the graph to a Matplotlib image and display it
ax.imshow(graph.create_png(), aspect='equal')

# Remove the axis labels
ax.axis('off')

# Show the plot
plt.show()
