import time
time_start = time.perf_counter()
import numpy as np

from numpy.random import seed
from numpy.random import randint
seed(1)
import matplotlib.pyplot as plt


N = 10**4 + 1 #matrix has N-1 dimensions

x = np.linspace(0,1,N+1) #N+1 equally spaced x values from 0 to 1
h = (x[N] - x[0])/N #step-size


b_ = np.zeros(N+1); u = np.zeros(N+1)

for i in range(1,N):
    "inserting h**2*fi into b_ for all x values"
    b_[i] = h**2*100*np.exp(-10*x[i]) #FLOPS: 5N per loop

    "closed form solution u(x) = 1 - (1 - e^-10)x - e^-10x"
    u[i] = 1 - (1 - np.exp(-10))*x[i] - np.exp(-10*x[i]) #FLOPS: 7N per loop

b_ = np.array(b_).tolist()
del b_[0]; del b_[-1]
u = np.array(u).tolist()
del u[0]; del u[-1]



#creating tridiagonal matrix
from scipy.sparse import diags

diagonals = [np.full(N-1, 2), np.full(N-2, -1), np.full(N-2, -1)]
mat = diags(diagonals, [0, -1, 1]).toarray()

import scipy
import scipy.linalg   # SciPy Linear Algebra Library

P, L, U = scipy.linalg.lu(mat)

X = np.zeros(N-1)
Y = np.zeros(N-1)
#1 e)


from scipy.linalg import solve_triangular

X = solve_triangular(L, b_, lower=True)
Y = solve_triangular(U, b_, lower=False)



eps = np.zeros(N-1)
total_eps = 0
for i in range(N-1):
    eps[i] = np.log10(abs((Y[i] - u[i])/u[i]))
    total_eps = total_eps + eps[i]



time_elapsed = (time.perf_counter() - time_start)
print(time_elapsed) #3.23 before adding u, 5.95 after



#testing
def test_len():
    assert len(b_) == N-1, "Should be length N-1"
    assert len(X) == N-1, "Should be length N-1"
    assert len(Y) == N-1, "Should be length N-1"
    assert len(u) == N-1, "Should be length N-1"

if __name__ == "__main__":
    test_len()
    print("Everything passed")
