import numpy as np

def normalEqn(X, y):
    # Calculate X_transpose
    X_transpose = np.transpose(X)

    # Calculate theta
    theta = np.linalg.pinv((X_transpose @ X)) @ (X_transpose @ y)

    return theta