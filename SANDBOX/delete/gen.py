import numpy as np

# Load the existing matrix
matrix = np.genfromtxt('SANDBOX/label.csv', delimiter=',')

# Create 10 more matrices proportional to the existing one
for i in range(1, 11):
    scaled_matrix = matrix * i  # Scale the matrix by a factor of i
    np.savetxt(f'SANDBOX/delete/label_{i}.csv', scaled_matrix, delimiter=',', fmt='%d')