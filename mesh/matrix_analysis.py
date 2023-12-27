# import numpy as np

# A = np.array([[2, 1], [1, 3]])

# print(np.dot(A,A.T))
# U, Sigma, VT = np.linalg.svd(A)

# # print("U:")
# # print(U)
# # print("\nSigma:")
# print(Sigma)
# # print("\nV^T:")
# # print(VT)

# ATA = np.dot(A.T,A)
# eigenvalues, eigenvectors = np.linalg.eig(ATA)

# print("Eigenvalues:")
# print(eigenvalues)
# # print("\nEigenvectors:")
# print(eigenvectors)


import numpy as np

A = np.array([[2, 1], [1, 3]])

# Perform SVD
U, Sigma, VT = np.linalg.svd(A)

# Calculate A^T A
ATA = np.dot(A.T, A)

# Calculate eigenvalues of A^T A
eigenvalues_ATA, _ = np.linalg.eig(ATA)

# Compare the square of singular values with the eigenvalues of A^T A
print("Square of singular values:")
print(Sigma**2)
print("\nEigenvalues of A^T A:")
print(eigenvalues_ATA)
