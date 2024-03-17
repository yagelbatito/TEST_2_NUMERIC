import numpy as np
from numpy.linalg import norm

from colors import bcolors
from matrix_utility import is_diagonally_dominant,DominantDiagonalFix


def gauss_seidel(A, b, X0, TOL=1e-16, N=200):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming gauss seidel algorithm\n')
    else:
        A,b = DominantDiagonalFix(A,b)

    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


if __name__ == '__main__':

    A = np.array([[3, 1, 1], [1, 1, 3], [1, 3, 1]])
    b = np.array([7.0, 7.0, 11.0])
    X0 = np.zeros_like(b)

    solution =gauss_seidel(A, b, X0)
    print(bcolors.OKBLUE,"\nApproximate solution:", solution)
    print(
        "date:18.03.24 \n the git link: https://github.com/yagelbatito/TEST_2_NUMERIC.git\ngroup:Almog Babila 209477678, Hay Carmi 207265687, Yagel Batito 318271863, Meril Hasid 324569714\nstudent:Yagel Batito 318271863")

