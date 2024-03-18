import numpy as np
from numpy.linalg import norm

from colors import bcolors
from matrix_utility import is_diagonally_dominant, DominantDiagonalFix

"""
Performs Jacobi iterations to solve the line system of equations, Ax=b, 
starting from an initial guess, ``x0``.

Terminates when the change in x is less than ``tol``, or
if ``N`` [default=200] iterations have been exceeded.

Receives 5 parameters:
    1.  a, the NxN matrix that method is being performed on.
    2.  b, vector of solution. 
    3.  X0,  the desired initial guess.
        if x is None, the initial guess will bw determined as a vector of 0's.
    4.  TOL, tolerance- the desired limitation of tolerance of solution's anomaly.
        if tolerance is None, the default value will set as 1e-16.
    5.  N, the maxim number of possible iterations to receive the most exact solution.
        if N is None, the default value will set as 200.

Returns variables:
    1.  x, the estimated solution

"""

def jacobi_iterative(A, b, X0, TOL=1e-16, N=200):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming jacobi algorithm\n')
    else:
        A,b = DominantDiagonalFix(A,b)

    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    while k <= N:
        x = np.zeros(n, dtype=np.double)
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * X0[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)



def make_diagonally_dominant(matrix, result_vector):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    # Calculate the absolute sum of each row without the diagonal element
    row_sum_without_diag = np.sum(np.abs(matrix), axis=1) - np.abs(np.diag(matrix))

    # Calculate the diagonal elements to make the matrix diagonally dominant
    diagonal_elements = np.maximum(row_sum_without_diag, np.abs(np.diag(matrix)))

    # Create the new diagonally dominant matrix
    new_matrix = np.diag(diagonal_elements) - matrix

    # Adjust the result vector if needed to maintain the equality
    result_vector = result_vector + np.sum(matrix - new_matrix, axis=1)

    return new_matrix, result_vector


if __name__ == "__main__":
    A = np.array([[2, 3, 4, 5, 6], [-5, 3, 4, -2, 3], [4, -5, -2, 2, 6], [4, 5, -1, -2, -3], [5, 5, 3, -3, 5]])
    b = np.array([70, 20, 26, -12, 37])

    x = np.zeros_like(b, dtype=np.double)
    solution = jacobi_iterative(A, b, x)

    print(bcolors.OKBLUE,"\nApproximate solution:", solution)
    print(
        "date:18.03.24 \n the git link: https://github.com/yagelbatito/TEST_2_NUMERIC.git\ngroup:Almog Babila 209477678, Hay Carmi 207265687, Yagel Batito 318271863, Meril Hasid 324569714\nstudent:Yagel Batito 318271863")


