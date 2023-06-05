import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Fit the linear regression model
coefficients = np.polyfit(x, y, 1)
m = coefficients[0]  # slope
b = coefficients[1]  # intercept

# Create a scatter plot of the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Plot the linear regression line
plt.plot(x, m * x + b, color='red', label='Linear Regression Line')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')

# Add legend
plt.legend()

# Display the plot
plt.show()
