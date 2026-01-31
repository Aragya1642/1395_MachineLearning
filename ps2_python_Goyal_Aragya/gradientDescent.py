import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta_init, alpha, num_iters): 
    # Initialize
    theta = theta_init.copy()
    cost_history = np.zeros((num_iters, 1))
    m = X.shape[0]

    # Perform gradient descent
    for i in range(num_iters):
        # Compute predictions and errors
        hypothesis = X @ theta
        err = hypothesis - y

        # Calculate gradient
        gradient = (1/m) * (np.transpose(X) @ err)
        
        # Calculate new thetas
        theta = theta - (alpha * gradient)

        # Calculate and save cost
        cost_history[i] = computeCost(X, y, theta)
    
    # Return theta and cost history
    return theta, cost_history