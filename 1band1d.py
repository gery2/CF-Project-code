import time
time_start = time.perf_counter()
import numpy as np
from numpy.random import seed
from numpy.random import randint
seed(1)
import matplotlib.pyplot as plt



N = 10**1 + 1

x = np.linspace(0,1,N+1) #N+1 equally spaced x values from 0 to 1
h = (x[N] - x[0])/N #step-size
h = np.log10(h) #for 1d)


b_ = np.zeros(N+1); v = np.zeros(N+1); u = np.zeros(N+1)

for i in range(1,N+1):
    "inserting h**2*fi into b_ for all x values"
    b_[i] = h**2*100*np.exp(-10*x[i])

    "closed form solution u(x) = 1 - (1 - e^-10)x - e^-10x"
    u[i] = 1 - (1 - np.exp(-10))*x[i] - np.exp(-10*x[i]) #FLOPS: 7N per loop



#creating tridiagonal matrix
from scipy.sparse import diags
diagonals = [randint(1,2,N-1), randint(1,2,N-2), randint(1,2,N-2)]
mat = diags(diagonals, [0, -1, 1]).toarray()

import scipy
import scipy.linalg   # SciPy Linear Algebra Library

P, L, U = scipy.linalg.lu(mat) #LU decomposition solver


b_ = np.array(b_).tolist()
del b_[0]; del b_[-1] #making this vector the same length as the others


#forward substitution with known b_ values
def forward(L, b_):
    x = []
    for i in range(len(b_)):
        x.append(b_[i])
        for j in range(i):
            x[i]=x[i]-(L[i, j]*x[j])
        x[i] = x[i]/L[i, i]
    return x

X = forward(L, b_)

#backwards substitution with the found v values
def backward(U, X):
    n = np.size(X)
    x = np.zeros_like(X)

    x[-1] = 1 / U[-1, -1] * X[-1]
    for i in range(n-2, -1, -1):
        x[i] = 1 / U[i, i] * (X[i] - np.sum(U[i, i+1:] * x[i+1:]))

    return x

Y = backward(U,X)

x = np.array(x).tolist()

del x[0]; del x[-1]

u = np.array(u).tolist()
del u[0]; del u[-1]

'''
plt.plot(x,Y)
plt.plot(x,u)
plt.show()
'''
eps = np.zeros(N-1)
total_eps = 0
for i in range(N-1):
    eps[i] = np.log10(abs((Y[i] - u[i])/u[i]))
    total_eps = total_eps + eps[i]



time_elapsed = (time.perf_counter() - time_start)
print(time_elapsed)


#testing
def test_len():
    assert len(b_) == N-1, "Should be length N-1"
    assert len(X) == N-1, "Should be length N-1"
    assert len(Y) == N-1, "Should be length N-1"
    assert len(u) == N-1, "Should be length N-1"
    assert len(x) == N-1, "Should be length N-1"

if __name__ == "__main__":
    test_len()
    print("Everything passed")
