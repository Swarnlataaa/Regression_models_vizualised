import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Generate some example data for visualization
np.random.seed(0)
X = np.random.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# Create an SVM classifier
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X, y)

# Create a meshgrid to plot the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the class for each meshgrid point
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a contour plot to visualize the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

# Highlight the support vectors
support_vectors = svm_classifier.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
            facecolors='none', edgecolors='k')

# Set labels and title
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Classifier')

# Show the plot
plt.show()
