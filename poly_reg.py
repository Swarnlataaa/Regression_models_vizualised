import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 5, 8, 12, 18, 26, 36, 48, 62, 78])

# Define the degree of the polynomial
degree = 3

# Fit the polynomial regression model
coefficients = np.polyfit(x, y, degree)
p = np.poly1d(coefficients)

# Generate points on the polynomial curve for plotting
x_curve = np.linspace(x.min(), x.max(), 100)
y_curve = p(x_curve)

# Create a scatter plot of the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Plot the polynomial curve
plt.plot(x_curve, y_curve, color='red', label='Polynomial Regression Curve')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression')

# Add legend
plt.legend()

# Display the plot
plt.show()
