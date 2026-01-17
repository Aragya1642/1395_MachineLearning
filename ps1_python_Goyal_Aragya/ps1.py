# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import time

# Constants
D = 58.0

######################################################
## Question 2 ##
print("## Question 2 ##")
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

######################################################
## Question 3 ##
print("\n\n## Question 3 ##")
# Define matrix A
a = np.array([[2, 10, 8],
              [3, 5, 2],
              [6, 4, 4]])
print(f"A Array: {a}")

# Find column minimums
column_mins = a.min(axis=0) # axis=0=columns, axis=1=rows
print(f"Column Minimums: {column_mins}")

# Find row maximums
row_maxs = a.max(axis=1)
print(f"Row Maxiumums: {row_maxs}")

# Find smallest value in the matrix
print(f"Full matrix minimum: {a.min()}")

# Sum each row
a_sums = a.sum(axis=1)
print(f"Row sums: {a_sums}")

# Sum all elements
print(f"Sum of all elements: {a.sum()}")

# Define matrix B
b = np.square(a)
print(f"B Array: {b}")

# Verification of square
if (b[0,1] == a[0,1]**2):
    print("B[0,1] is the square of A[0,1]")
else:
    print("Failed match")

# Solve system of equations
a = np.array([[2,5,-2],
              [2,6,4],
              [6,8,18]])
b = np.array([D,6,15])
a_inv = np.linalg.inv(a)
x = a_inv @ b
print(f"[x, y, z] = {x}")

######################################################
## Question 4 ##
print("\n\n## Question 4 ##")
# Create matrix x
i = np.arange(10)
x = np.column_stack((i, i**2, i*D))
print(f"X Array: {x}")

# Create matrix y
y = (3*i) + D
print(f"Y Array: {y}")
