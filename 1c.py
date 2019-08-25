import time
time_start = time.perf_counter()
import numpy as np
from numpy.random import seed
from numpy.random import randint
seed(1)
import matplotlib.pyplot as plt


N = 10**5

x = np.linspace(0,1,N+1) #N+1 equally spaced x values from 0 to 1
h = (x[N] - x[0])/N #step-size


b_ = np.zeros(N+1); v = np.zeros(N+1); u = np.zeros(N+1)

for i in range(N+1):
    "inserting h**2*fi into b_ for all x values"
    b_[i] = h**2*100*np.exp(-10*x[i])

#decomposition:
'''
#eliminating a[i]
v[1] = 4; v[2] = 3

for i in range(1,N-1):

    v[i+1] = v[i+1] - (b[i]/a[i])*v[i]

#eliminating c[i]
for i in range(3,N+1): #different indexing here

    v[-i] = v[-i] - (b[-i+2]/c[-i+1])*v[-i+1]
'''


ac = -1 #same values for a and c diagonal
b = 2   #Middle diagonal


#forward substitution with known b_ values
for i in range(1,N):
    "forward substitution to find v values"
    "can remove one FLOPS when knowing ac = -1"
    #v[i] = (b_[i] - ac*v[i-1])/b #for varying a,b
    v[i] = (b_[i] + v[i-1])/b


#backward substitution with known v values
for i in range(2,N+1):
    "backward substitution to find u values"
    "can remove one FLOPS per loop when knowing ac = -1"
    #u[-i] = (b_[-i] - ac*u[-i+1])/b #for varying c, b
    u[-i] = (v[-i] + u[-i+1])/b

"FLOPS now: 2(N-1) per method down from 3(N-1)"


time_elapsed = (time.perf_counter() - time_start)
print(time_elapsed)



#testing
def test_len():
    assert len(b_) == N+1, "Should be length N+1"
    assert len(v) == N+1, "Should be length N+1"
    assert len(u) == N+1, "Should be length N+1"
    assert len(x) == N+1, "Should be length N+1"

if __name__ == "__main__":
    test_len()
    print("Everything passed")
