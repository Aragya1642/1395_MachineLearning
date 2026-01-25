import numpy as np

def computeCost(X, y, theta):
    # Get number of training samples
    m = X.shape[0]

    # Using theta and X, calculate hypothesis
    hypothesis = np.transpose(X) @ theta

    # Calculate error using hypothesis and true value
    err = hypothesis - y
    err = err ** 2

    # Calculate final cost
    J = (1/(2*m)) * (sum(err))

    return J