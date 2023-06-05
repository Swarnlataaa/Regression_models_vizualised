import matplotlib.pyplot as plt
import numpy as np

# Generate some example data for visualization
X = np.arange(0, 10, 0.1).reshape(-1, 1)
y_true = np.sin(X).ravel()

# Train the Random Forest model (replace this with your actual training code)
rf_reg.fit(X, y_true)

# Make predictions on the data points
y_pred = rf_reg.predict(X)

# Plot the true values and predicted values
plt.plot(X, y_true, label='True Values', color='blue')
plt.plot(X, y_pred, label='Predicted Values', color='red')

# Set labels and title
plt.xlabel('X')
plt.ylabel('y')
plt.title('Random Forest Regression')

# Add legend
plt.legend()

# Display the plot
plt.show()
