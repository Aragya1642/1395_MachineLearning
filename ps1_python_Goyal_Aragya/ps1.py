# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import time

# Constants
D = 58.0

# Calculate mean and std dev.
mean = 2.0 + (D/100)
stddev = 0.5 + (D/200)

# Generate x vector
x = stddev * np.random.randn(1000000, 1) + mean

# Generate z vector
low_bound = -(D/50.0)
high_bound = D/100.0
z = np.random.uniform(low_bound, high_bound, (1000000,1))

# Print arrays
print(f"X Array: {x}")
print(f"Z Array: {z}")

# Create figure for x
plt.figure()
plt.hist(x, bins=100, density=True, label='x (Gaussian)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig("ps1-2-c-1.png")
plt.close()

# Create figure for z
plt.figure()
plt.hist(z, bins=100, density=True, label='z (Uniform)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig("ps1-2-c-2.png")
plt.close()

# Add 2 to x using loop
init_time = time.time()
for i in range(len(x)):
    x[i] += 2
loop_time = time.time() - init_time

# Add 2 to x using matrix operation
init_time = time.time()
x += 2
add_time = time.time() - init_time

# Print out each timings
print(f"Loop time: {loop_time}")
print(f"Add time: {add_time}")

# Define and populate y vector
y = z[np.where((z > 0) & (z < 0.8))]
print(f"Y Array: {y}")
print(f"Retrieved {y.shape} elements")

