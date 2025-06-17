import sys
import numpy as np
from scipy.io import savemat
from equilibrium_positions import equilibrium_positions

N = int(sys.argv[1]) # ion number
J0 = 1
alpha = 1.0

# compute equilibrium positions of N Yb-171 ions in a harmonic trap
r = equilibrium_positions(N, V=[[], *(2 * np.pi 
                                      * np.array([0.60, 2.164, 0.144]))**2
                                ], m=171, e=1, 
                          method='cooling', collective_mode=False)
# iterate until convergence of positions
while True:
    r2 = equilibrium_positions(N, V=[[], *(2 * np.pi 
                                           * np.array([0.60, 2.164, 0.144]))**2
                                     ], m=171, e=1, 
                               method='cooling', collective_mode=False,
                               r0=r)
    shift = np.max(np.abs(r - r2))
    r = r2
    print(shift)
    if shift < 0.1:
        break
# further improve accuracy
r = equilibrium_positions(N, V=[[], *(2 * np.pi 
                                      * np.array([0.60, 2.164, 0.144]))**2
                                ], m=171, e=1, 
                           method='iterate', collective_mode=False,
                           r0=r)
# sort ions by z coordinates
ind = np.argsort(r[:, 2])
r = r[ind, :]

# compute ion-ion distance
dist = np.sqrt(np.sum((r[np.newaxis, :, :] - r[:, np.newaxis, :])**2,
                      axis=2))
np.fill_diagonal(dist, np.max(dist) * 10)
dmin = np.min(dist, axis=1) # distance between nearest-neighbor ions
inv_dist = 1 / dist
np.fill_diagonal(inv_dist, 0)

J = J0 * (inv_dist * np.mean(dmin))**alpha # use average dmin for normalization

# save generated J matrix
savemat('Ising2d_N{}_alpha{}.mat'.format(N, alpha), 
        {"N" : N, "r" : r, "J" : J})
