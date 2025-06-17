import sys
import scipy.io as sio

N=int(sys.argv[1])

# Load the J matrix
J = sio.loadmat(f'Jmat/Ising2d_N{N}_alpha1.0.mat')['J']

print(f'RealScalar Jmat[{N}][{N}] = {{')
for i in range(N):
    print('{', end='')
    for j in range(N):
        if j != N-1:
            print(J[i,j], end=',')
        else:
            print(J[i,j], end='}')
    if i != N-1:
        print(',')
    else:
        print('};')
