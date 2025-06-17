# -*- coding: utf-8 -*-
"""
@author: wuyukai
Given static trapping potential in 3D, return equilibrium positions
Input:
    N: ion number
    [V, [Vx, Vy, Vz]]: 
        Total potential, pass as [V], or [[], Vx, Vy, Vz]
        Each Vx, Vy, Vz can be numerical values or functions,
            but V must be functions
        If values are provided, potential is 0.5 * V * r**2 * (q * m0 / q0)
            in other words V represents square of trap frequency for ion 0
        If functions are provided, they are given in the format of
            [fun, jac, hess]
            to give gradient and hessian of the potential V(r) * (q * m0 / q0)
            V accepts 3D input, while Vx, Vy, Vz accept 1D input each
        Coordinate r in unit of um
        To be consistent with Coulomb energy, scale potential here
            by 1 / m0
    [V_DC, [Vx_DC, Vy_DC, Vz_DC]]: 
        DC potential in similar format
    [V_RF, [Vx_RF, Vy_RF, Vz_RF]]: 
        RF pseudopotential given by
            0.5 * V_RF * r**2 * (q * m0 / q0)**2 / m
        or
            V_RF(r) * (q * m0 / q0)**2 / m
    m: ion mass in atomic mass unit, default to Yb-171
    e: ion charge in unit of |e|, default to 1
        If m or e are arrays, V_RF and omegaRF are required
        If m or e are arrays, their size must match ion number N
    r0: initial guess of equilibrium positions
        If not provided, use random values
    method: which mode to use to solve equilibrium positions
        'cooling': time evolution under damping term F=-gamma * v,
                where gamma is an input argument
        'minimize': minimize total potential 
                    by standard Newton-CG algorithm in scipy.optimize
                options: input arguments for Newton-CG method
                    including xtol, maxiter
        'global': basinhopping algorithm to try to find global minimum
                options: input arguments for basinhopping method
                    including niter, stepsize
        'iterate': manually iterate, can be used to improve accuracy
                    alpha: input argument between 0 and 1
                    step: input argument for number of steps
    args: a dictionary for additional arguments used by individual methods
    collective_mode: whether to compute mode frequencies and vectors
Output:
    r: N by 3 array for equilibrium positions of ions
    omega_k: mode frequencies in MHz, with 2pi factor kept
    b_jk: mode vectors
"""
import numpy as np
from scipy.linalg import block_diag, eigh
from scipy.optimize import minimize, basinhopping
from scipy.integrate import solve_ivp
from scipy.io import savemat
def decode_potential_single(V):
    if isinstance(V, (float, int)):
        fun = lambda x: 0.5 * V * x**2
        jac = lambda x: V * x
        hess = lambda x: V * np.eye(x.size)
        return([fun, jac, hess])
    else: # a tuple of functions for fun, jac and hess
        return(V)
def decode_potential_all(V):
    if len(V) == 1:
        return(decode_potential_single(V[0]))
    else:
        Vx = decode_potential_single(V[1])
        Vy = decode_potential_single(V[2])
        Vz = decode_potential_single(V[3])
        def fun(r):
            r = np.reshape(r, (-1, 3))
            return(Vx[0](r[:, 0]) + Vy[0](r[:, 1]) + Vz[0](r[:, 2]))
        def jac(r):
            r = np.reshape(r, (-1, 3))
            return(np.stack((Vx[1](r[:, 0]), Vy[1](r[:, 1]), Vz[1](r[:, 2])),
                            axis=-1))
        def hess(r):
            r = np.reshape(r, (-1, 3))
            n = r.shape[0]
            H = np.zeros((n, 3, n, 3))
            H[:, 0, :, 0] = Vx[2](r[:, 0])
            H[:, 1, :, 1] = Vy[2](r[:, 1])
            H[:, 2, :, 2] = Vz[2](r[:, 2])
            return(H.reshape((n * 3, n * 3)))
        return([fun, jac, hess])
def VCoulomb(r, e=1):
    r = np.reshape(r, (-1, 3))
    # dif_ijk = r_ik - r_jk
    dif = r[:, np.newaxis, :] - r[np.newaxis, :, :]
    # dist_ij = \sqrt(\sum_k (r_ik - r_jk)^2)
    dist = np.sqrt(np.sum(dif**2, axis=2))
    # set diagonal elements to 1 for convenience of element inverse
    np.fill_diagonal(dist, 1)
    invdist = 1 / dist
    # set diagonal elements to 0
    np.fill_diagonal(invdist, 0)
    matee = np.reshape(e, (-1, 1)) * np.reshape(e, (1, -1))
    V = 0.5 * np.sum(invdist * matee)
    return(V)
def GradCoulomb(r, e=1):
    r = np.reshape(r, (-1, 3))
    # dif_ijk = r_ik - r_jk
    dif = r[:, np.newaxis, :] - r[np.newaxis, :, :]
    # dist_ij = \sqrt(\sum_k (r_ik - r_jk)^2)
    dist = np.sqrt(np.sum(dif**2, axis=2))
    # set diagonal elements to 1 for convenience of element inverse
    np.fill_diagonal(dist, 1)
    invdist = 1 / dist
    # set diagonal elements to 0
    np.fill_diagonal(invdist, 0)
    matee = np.reshape(e, (-1, 1)) * np.reshape(e, (1, -1))
    grad = -np.sum(dif * (invdist**3 * matee)[:, :, np.newaxis],
                   axis=1).flatten()
    return(grad)
def HessCoulomb(r, e=1):
    r = np.reshape(r, (-1, 3))
    N = r.shape[0]
    # dif_ijk = r_ik - r_jk
    dif = r[:, np.newaxis, :] - r[np.newaxis, :, :]
    # dist_ij = \sqrt(\sum_k (r_ik - r_jk)^2)
    dist = np.sqrt(np.sum(dif**2, axis=2))
    # set diagonal elements to 1 for convenience of element inverse
    np.fill_diagonal(dist, 1)
    invdist = 1 / dist
    # set diagonal elements to 0
    np.fill_diagonal(invdist, 0)
    matee = np.reshape(e, (-1, 1)) * np.reshape(e, (1, -1))
    hess = -3 * dif[:, :, :, np.newaxis] * dif[:, :, np.newaxis, :] \
                    * (invdist**5 * matee)[:, :, np.newaxis, np.newaxis]
    hess[:, :, range(3), range(3)] += (invdist**3 * matee)[:, :, np.newaxis]
    hess[range(N), range(N), :, :] = -np.sum(hess, axis=1)
    hess = np.swapaxes(hess, 1, 2).reshape((N*3, N*3))
    return(hess)
def equilibrium_positions(N, V=None, V_DC=None, V_RF=None,
                          m=171, e=1, r0=None, method='minimize', args={},
                          collective_mode=False):
    if isinstance(m, (int, float)) and isinstance(e, (int, float)):
        m = m * np.ones(N)
        e = e * np.ones(N)
    else:
        m = np.array(m)
        e = np.array(e)
        assert m.size == N and e.size == N
        assert V_DC is not None and V_RF is not None
    coef = 8.9876e9 * 1.6022e-19**2 / (m[0] * 1.6605e-27 * 1e-6)
    if V is not None:
        fun0, jac0, hess0 = decode_potential_all(V)
        fun = lambda r: np.sum(fun0(r) * (e / e[0])) + VCoulomb(r, e) * coef
        jac = lambda r: (jac0(r).reshape((-1, 3)) 
                         * (e / e[0]).reshape((-1, 1))).flatten() \
                        + GradCoulomb(r, e) * coef
        def hess(r):
            H = hess0(r).reshape((N, 3, N, 3))
            H[range(N), :, range(N), :] *= (e / e[0]).reshape((-1, 1, 1))
            return(H.reshape((N * 3, N * 3)) 
                   + HessCoulomb(r, e) * coef)
    if V_DC is not None and V_RF is not None:
        fun1, jac1, hess1 = decode_potential_all(V_DC)
        fun2, jac2, hess2 = decode_potential_all(V_RF)
        fun = lambda r: np.sum(fun1(r) * (e / e[0])
                               + fun2(r) * ((e / e[0])**2 * m[0] / m)) \
                        + VCoulomb(r, e) * coef
        jac = lambda r: (jac1(r).reshape((-1, 3)) 
                         * (e / e[0]).reshape((-1, 1))
                         + jac2(r).reshape((-1, 3)) * ((e / e[0])**2 * m[0] 
                                      / m).reshape((-1, 1))).flatten() \
                        + GradCoulomb(r, e) * coef
        def hess(r):
            H1 = hess1(r).reshape((N, 3, N, 3))
            H1[range(N), :, range(N), :] *= (e / e[0]).reshape((-1, 1, 1))
            H2 = hess2(r).reshape((N, 3, N, 3))
            H2[range(N), :, range(N), :] *= ((e / e[0])**2 * m[0]
                                             / m).reshape((-1, 1, 1))
            return((H1 + H2).reshape((N * 3, N * 3)) 
                   + HessCoulomb(r, e) * coef)
    if r0 is None:
        r0 = np.random.rand(N * 3)
    assert method in ['cooling', 'minimize', 'global', 'iterate']
    if method == 'minimize':
        options = {key: args[key] for key in args.keys() 
                                           & {'xtol', 'maxiter'}}
        sol = minimize(fun, r0, method='Newton-CG', jac=jac, hess=hess,
               options=options)
        print(sol)
        r = sol.x.reshape((N, 3))
    if method == 'cooling':
        if 'gamma' in args.keys():
            gamma = args['gamma']
        else:
            gamma = 0.1
        if 'T' in args.keys():
            T = args['T']
        else:
            T = 100 # us
        if 'v0' in args.keys():
            v0 = np.copy(args['v0'])
        else:
            v0 = np.zeros((N, 3))
        fun_RK = lambda t, x: np.concatenate((x[3*N:], 
                                              -jac(x[:3*N]) * m[0]  
                                                  / np.repeat(m, 3)
                                              -gamma * x[3*N:]))
        sol = solve_ivp(fun_RK, (0, T), np.concatenate((r0.flatten(), 
                                                        v0.flatten())),
                        method='RK45', t_eval=np.array([0, T]))
        print(sol)
        r = sol.y[:3*N, -1].reshape((N, 3))
    if method == 'global':
        if 'niter' in args.keys():
            niter = args['niter']
        else:
            niter = 1000
        if 'stepsize' in args.keys():
            stepsize = args['stepsize']
        else:
            stepsize = 1
        x0 = minimize(fun, r0, jac=jac, method="L-BFGS-B").x
        minimizer_kwargs = {"method":"Newton-CG", "jac":jac, "hess":hess}
        def save(x, f, accept):
            save.count += 1
            if save.count % 100 == 0:
                print('Basinhopping step ' + str(save.count))
            if f < save.fmin:
                save.fmin = f
                print('New minimum at potential ' + str(f))
                print('Configuration saved to tmp.mat')
                savemat('tmp.mat', {'N': N, 'r': x.reshape((N, 3))})
        save.fmin = np.inf
        save.count = 0
        sol = basinhopping(fun, x0, minimizer_kwargs=minimizer_kwargs,
                   niter=niter, stepsize=stepsize, callback=save)
        print(sol)
        r = sol.x.reshape((N, 3))
    if method == 'iterate':
        if 'alpha' in args.keys():
            alpha = args['alpha']
        else:
            alpha = 0.1
        if 'step' in args.keys():
            step = args['step']
        else:
            step = 100
        print('Iterate mode is used to improve accuracy of existing solution')
        x_new = r0.flatten()
        for i in range(step):
            grad = jac(x_new)
            H = hess(x_new)
            dx = np.linalg.solve(H, grad)
            x_new = x_new - alpha * dx
            print(fun(x_new))
        r = x_new.reshape((N, 3))
    if collective_mode:
        H = hess(r.flatten())
        T = np.diag(np.repeat(m, 3)) / m[0]
        omega2, b_jk = eigh(H, T)
        omega_k = np.sqrt(omega2)
        b_jk = b_jk / np.sqrt(np.sum(b_jk**2, axis=0, keepdims=True))
        return(r, omega_k, b_jk)
    else:
        return(r)
    
if __name__ == '__main__':
    # Two Yb-171 ions
    r, omega_k, b_jk = equilibrium_positions(2, V=[[], *(2 * np.pi 
                                    * np.array([2.5, 2.6, 0.545]))**2],
                                             m=171, e=1,
                                             collective_mode=True)
    
    # Yb-171 and Ba-137 in a harmonic trap
    # trap frequency given for the first ion (Yb)
    r, omega_k, b_jk = equilibrium_positions(2, V_DC=[[], *(2 * np.pi * 0.1)**2 
                                             * np.array([-1, 0, 1])],
                                             V_RF=[[], *(2 * np.pi * 2.5)**2 
                                             * np.array([1, 1, 0])],
                                             m=[171, 137], e=[1, 1],
                                             collective_mode=True)
    
    # Yb^+ and Yb^2+ in a harmonic trap passed as functions
    # trap frequency given for the first ion (Yb^+)
    V_RF = (2 * np.pi * 2.5)**2
    fun = lambda z: 0.5 * V_RF * z**2
    jac = lambda z: V_RF * z
    hess = lambda z: V_RF * np.eye(z.size)
    r, omega_k, b_jk = equilibrium_positions(2, V_DC=[[], *(2 * np.pi * 0.1)**2 
                                             * np.array([-1, 0, 1])],
                                             V_RF=[[], [fun, jac, hess],
                                                   [fun, jac, hess], 0],
                                             m=[171, 171], e=[1, 2],
                                             collective_mode=True)
    
    # N = 19 example for anharmonic trap in PRA 97, 062325
    l0 = 40e-6
    gamma4 = 4.3
    m = 171 * 1.66e-27
    alpha2 = 8.9876e9 * 1.6022e-19**2 / l0**3 * 1e-6**2 / m
    alpha4 = gamma4 * alpha2 / l0**2 * 1e-6**2
    fun = lambda z: -0.5 * alpha2 * z**2 + 0.25 * alpha4 * z**4
    jac = lambda z: -alpha2 * z + alpha4 * z**3
    hess = lambda z: np.diag(-alpha2 + 3 * alpha4 * z**2)
    r = equilibrium_positions(19, [[], (2 * np.pi * 3)**2, 
                                   (2 * np.pi * 2.5)**2, [fun, jac, hess]])
    # cooling method
    r2 = equilibrium_positions(19, [[], (2 * np.pi * 3)**2, 
                                    (2 * np.pi * 2.5)**2, [fun, jac, hess]],
                               method='cooling')
    r3 = equilibrium_positions(19, [[], (2 * np.pi * 3)**2, 
                                    (2 * np.pi * 2.5)**2, [fun, jac, hess]],
                               method='cooling', r0=r2)
    print(np.max(np.abs(r2 - r3)))
    # improve precision and compute collective modes
    r4, omega_k, b_jk = equilibrium_positions(19, [[], (2 * np.pi * 3)**2, 
                                    (2 * np.pi * 2.5)**2, [fun, jac, hess]],
                                    method='iterate', r0=r2, 
                                    collective_mode=True)
    
    # global minimum of 50 ion in 2D in a harmonic trap
    r = equilibrium_positions(50, [[], *(2 * np.pi * np.array([0.2, 0.4, 2]))**2],
                               method='global')
    
    # 2D crystal with nonlinear terms
    omega = 2 * np.pi * np.array([0.6, 2.164, 0.144])
    def func(r):
        r = np.reshape(r, (-1, 3))
        res = 0.5 * (omega[0]**2 * r[:, 0]**2 + omega[1]**2 * r[:, 1]**2
                     + omega[2]**2 * r[:, 2]**2)
        return(res)
    def jac(r):
        r = np.reshape(r, (-1, 3))
        res = (omega**2 * r).ravel()
        return(res)
    def hess(r):
        r = np.reshape(r, (-1, 3))
        n = r.shape[0]
        H = np.zeros((n, 3, n, 3))
        H[:, 0, :, 0] = omega[0]**2 * np.eye(n)
        H[:, 1, :, 1] = omega[1]**2 * np.eye(n)
        H[:, 2, :, 2] = omega[2]**2 * np.eye(n)
        res = H.reshape((3 * n, 3 * n))
        return(res)
    r = equilibrium_positions(500, [[func, jac, hess]], method='cooling')
    r2, omega_k, b_jk = equilibrium_positions(500, [[func, jac, hess]], 
                                              method='iterate', r0=r, 
                                              collective_mode=True)
    eps = 1e-4
    def func2(r):
        r = np.reshape(r, (-1, 3))
        #res = eps * r[:, 2]**3
        res = eps * r[:, 2]**2 * r[:, 0]
        return(res + func(r))
    def jac2(r):
        r = np.reshape(r, (-1, 3))
        n = r.shape[0]
        #res = eps * 3 * r[:, 2]**2
        #res = np.concatenate((np.zeros((n, 2)), res.reshape((n, 1))), axis=1)
        res = eps * np.stack((r[:, 2]**2, np.zeros(n), 2 * r[:, 2] * r[:, 0]), axis=1)
        return(res.ravel() + jac(r))
    def hess2(r):
        r = np.reshape(r, (-1, 3))
        n = r.shape[0]
        H = np.zeros((n, 3, n, 3))
        #H[:, 2, :, 2] = np.diag(eps * 6 * r[:, 2])
        H[:, 2, :, 2] = np.diag(eps * 2 * r[:, 0])
        H[:, 0, :, 2] = np.diag(eps * 2 * r[:, 2])
        H[:, 2, :, 0] = np.diag(eps * 2 * r[:, 2])
        res = H.reshape((3 * n, 3 * n))
        return(res + hess(r))
    r3, omega_k2, b_jk2 = equilibrium_positions(500, [[func2, jac2, hess2]], 
                                                method='iterate', r0=r2, 
                                                collective_mode=True)