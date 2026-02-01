# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os
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
plt.savefig('ps2-1-d.png')
plt.show()

# Normal Equation Verification
theta_normal = normalEqn(X, y)
print("Theta from Normal Equation:\n", theta_normal)

# Real Dataset Application - https://www.kaggle.com/datasets/sudhirsingh27/electricity-consumption-based-on-weather-data/data
# Download latest version
path = kagglehub.dataset_download("sudhirsingh27/electricity-consumption-based-on-weather-data")
path = os.path.join(path, "electricity_consumption_based_weather_dataset.csv")

# Load dataset
data = np.genfromtxt(path, delimiter=',', skip_header=1)

# Get rid of the dates
data = data[:, 1:]

# Visualize the data
# Average Daily Wind Speed vs. Energy Consumption
plt.plot(data[:, 0], data[:, -1], 'rx')
plt.xlabel('Average Daily Wind Speed')
plt.ylabel('Energy Consumption')
plt.title('Average Daily Wind Speed vs. Energy Consumption')
plt.savefig('ps2-2-b-1.png')
plt.show()
# Daily percipitation vs. Energy Consumption
plt.plot(data[:, 1], data[:, -1], 'rx')
plt.xlabel('Daily Percipitation')
plt.ylabel('Energy Consumption')
plt.title('Daily Percipitation vs. Energy Consumption')
plt.savefig('ps2-2-b-2.png')
plt.show()
# Daily max temperature vs. Energy Consumption
plt.plot(data[:, 2], data[:, -1], 'rx')
plt.xlabel('Daily Max Temperature')
plt.ylabel('Energy Consumption')
plt.title('Daily Max Temperature vs. Energy Consumption')
plt.savefig('ps2-2-b-3.png')
plt.show()
# Daily min temperature vs. Energy Consumption
plt.plot(data[:, 3], data[:, -1], 'rx')
plt.xlabel('Daily Min Temperature')
plt.ylabel('Energy Consumption')
plt.title('Daily Min Temperature vs. Energy Consumption')
plt.savefig('ps2-2-b-4.png')
plt.show()