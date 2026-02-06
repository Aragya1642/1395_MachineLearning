# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn

# Define globals
D = 58

# Toy Dataset Verification
X = np.array([
    [1, 0, 1],
    [1, 1, 1.5],
    [1, 2, 4],
    [1, 3, 2]
])

y = np.array([
    [1.5 + (D/100)],
    [4],
    [8.5],
    [8.5 + (D/50)]
])

# Cost Computation Verification
computeCost_result = computeCost(X, y, np.array([[0.5], [2], [1]]))
print("Computed Cost J:", computeCost_result)

# Gradient Descent Verification
alpha = 0.01
theta_init = np.array([[0], [0], [0]])
num_iters = 100 + (D * 10)
theta_minimized, cost_history = gradientDescent(X, y, theta_init, alpha, num_iters)
print("Theta from Gradient Descent:\n", theta_minimized)

# Plot the cost history
plt.plot(range(num_iters), cost_history, 'b-')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost History Over Iterations')
plt.savefig('./output/ps2-1-d.png')
plt.show()

# Normal Equation Verification
theta_normal = normalEqn(X, y)
print("Theta from Normal Equation:\n", theta_normal)

# Real Dataset Application - https://www.kaggle.com/datasets/sudhirsingh27/electricity-consumption-based-on-weather-data/data
# Set path to dataset
path = "./input/electricity_consumption_based_weather_dataset.csv"

# Load dataset
data = np.genfromtxt(path, delimiter=',', skip_header=1)

# Get rid of the dates
data = data[:, 1:]
print("NaNs per column:", np.isnan(data).sum(axis=0))
# Remove rows with NaN values
data = data[~np.isnan(data).any(axis=1)]
print("Data shape after removing NaNs:", data.shape)

# Visualize the data
# Average Daily Wind Speed vs. Energy Consumption
plt.plot(data[:, 0], data[:, -1], 'rx')
plt.xlabel('Average Daily Wind Speed (m/s)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Average Daily Wind Speed')
plt.savefig('./output/ps2-2-b-1.png')
plt.show()
# Daily percipitation vs. Energy Consumption
plt.plot(data[:, 1], data[:, -1], 'rx')
plt.xlabel('Daily Percipitation (mm)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Daily Percipitation')
plt.savefig('./output/ps2-2-b-2.png')
plt.show()
# Daily max temperature vs. Energy Consumption
plt.plot(data[:, 2], data[:, -1], 'rx')
plt.xlabel('Daily Max Temperature (°C)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Daily Max Temperature')
plt.savefig('./output/ps2-2-b-3.png')
plt.show()
# Daily min temperature vs. Energy Consumption
plt.plot(data[:, 3], data[:, -1], 'rx')
plt.xlabel('Daily Min Temperature (°C)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Daily Min Temperature')
plt.savefig('./output/ps2-2-b-4.png')
plt.show()

# Preprocess the data
# Get number of training examples
m = data.shape[0]
print("Number of training examples:", m)
# Perform standardization
data_mean = np.mean(data[:, :-1], axis=0)
data_std = np.std(data[:, :-1], axis=0)
data[:, :-1] = (data[:, :-1] - data_mean) / data_std
print("Data means:", data_mean)
print("Data stds:", data_std)
# Add bias term
X = np.hstack((np.ones((m, 1)), data[:, :-1]))
y = data[:, -1].reshape(m, 1)
# Split the data into training and test sets randomly
np.random.seed(D)
row_indices = np.random.permutation(m)
X_shuffled = X[row_indices]
y_shuffled = y[row_indices]
split_index = int(0.8 * m)
X_train = X_shuffled[:split_index]
y_train = y_shuffled[:split_index]
X_test = X_shuffled[split_index:]
y_test = y_shuffled[split_index:]
print("X_train size:", X_train.shape)
print("y_train size:", y_train.shape)
print("X_test size:", X_test.shape)
print("y_test size:", y_test.shape)

# Univariate Linear Regression using Gradient Descent
# Using only Daily Min Temperature (4th feature)
X_univariate = X_train[:, [0, 4]]
alpha_univariate = 0.5
iterations_univariate = 500 + (D*5)
theta_init_univariate = np.array([[0], [0]])
theta_univariate, cost_history_univariate = gradientDescent(X_univariate, y_train, theta_init_univariate, alpha_univariate, iterations_univariate)
print("Theta from Gradient Descent (Univariate):\n", theta_univariate)
# Plot the cost history for univariate regression
plt.plot(range(iterations_univariate), cost_history_univariate, 'b-')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost History Over Iterations')
plt.savefig('./output/ps2-2-d-1.png')
plt.show()
# Plot linear regression line on training data
plt.plot(X_train[:, 4], y_train, 'rx', label='Training Data')
plt.plot(X_train[:, 4], X_univariate @ theta_univariate, 'b-', label='Linear Regression Fit')
plt.xlabel('Daily Min Temperature (°C)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Univariate Linear Regression Fit')
plt.legend()
plt.savefig('./output/ps2-2-d-2.png')
plt.show()

# Multivariate Linear Regression
alpha_multivariate = 0.5
iterations_multivariate = 750 + (D*5)
theta_init_multivariate = np.zeros((X_train.shape[1], 1))
theta_multivariate, cost_history_multivariate = gradientDescent(X_train, y_train, theta_init_multivariate, alpha_multivariate, iterations_multivariate)
theta_multivariate_normal = normalEqn(X_train, y_train)
# Compare theta from gradient descent and normal equation
print("Theta from Gradient Descent (Multivariate):\n", theta_multivariate)
print("Theta from Normal Equation (Multivariate):\n", theta_multivariate_normal)
# Plot the cost history for multivariate regression gradient descent
plt.plot(range(iterations_multivariate), cost_history_multivariate, 'b-')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost History Over Iterations')
plt.savefig('./output/ps2-2-e-1.png')
plt.show()

# Model evaluation and comparison
# Predict on test data using multivariate theta
y_predict = X_test @ theta_multivariate
# Calculate mean squared error on test data
mean_squared_error = 2*computeCost(X_test, y_test, theta_multivariate)
print("Mean Squared Error (Gradient Descent):", mean_squared_error)
# Predict on test data using multivariate theta from normal equation
y_predict_normal = X_test @ theta_multivariate_normal
# Calculate mean squared error on test data
mean_squared_error_normal = 2*computeCost(X_test, y_test, theta_multivariate_normal)
print("Mean Squared Error (Normal Equation):", mean_squared_error_normal)
# Calculate MSE for univariate model
y_predict_univariate = X_test[:, [0, 4]] @ theta_univariate
mean_squared_error_univariate = 2*computeCost(X_test[:, [0, 4]], y_test, theta_univariate)
print("Mean Squared Error (Univariate):", mean_squared_error_univariate)

# Learning rate analysis
learning_rates = [0.001, 0.5, 0.865, 1]
for lr in learning_rates:
    theta_init_lr = np.zeros((X_train.shape[1], 1))
    iterations_lr = 300
    theta_lr, cost_history_lr = gradientDescent(X_train, y_train, theta_init_lr, lr, iterations_lr)
    plt.plot(range(iterations_lr), cost_history_lr, label=f'Alpha={lr}')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.title('Learning Rate Analysis')
    plt.legend()
    plt.savefig(f'./output/ps2-2-g-{learning_rates.index(lr)+1}.png')
    plt.show()
    plt.close()