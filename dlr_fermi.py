import numpy.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slinalg
from scipy.linalg import lu_factor, lu_solve


def kernelT(E, tau, beta):
    x = beta*E / 2
    y = 2.0*tau / beta - 1
    if -100.0 < x < 100.0:
        G = np.exp(-x * y) / (2 * np.cosh(x))
    elif x >= 100.0:
        G = np.exp(-x * (y + 1))
    else:  # x<=-100.0
        G = np.exp(x * (1 - y))
    return G


def kernelWn(E, n, beta):
    wn = (2.0*n+1.0)*np.pi/beta
    return 1.0/(1j*wn+E)


beta = 1000.0
wmax = 10.0

dlr = np.loadtxt("dlr.dat")
wGrid = dlr[:, 1]  # real frequency grid
tauGrid = dlr[:, 2]  # tau grid
wnGrid = dlr[:, 3]  # Matsubara frequency grid

# transfer tau from (-beta/2, beta/2) to (0, beta)
for ti, t in enumerate(dlr[:, 2]):
    if t < 0.0:
        tauGrid[ti] = t+beta
tauGrid = np.sort(tauGrid)

# demonstrate how to use the tau grid
transfer = np.zeros((len(tauGrid), len(wGrid)))
for wi, w in enumerate(wGrid):
    transfer[:, wi] = kernelT(w, tauGrid, beta)

u, s, v = linalg.svd(transfer)
print("singular value of the transfer matrix: ", s)

# test with the bare Green's function
G = np.zeros(len(tauGrid))
E = 0.1
for ti, t in enumerate(tauGrid):
    G[ti] = np.exp(-E*t)/(1.0+np.exp(-E*beta))

lu, piv = lu_factor(transfer)
# calculate the dlr coefficients
coeff = lu_solve((lu, piv), G)
plt.figure()
plt.plot(coeff)
# plt.show()

print("G(tau) difference: ", np.max(transfer @ coeff - G))

# test Matsurbara frequency representation:
nlist = np.array(range(20000))-10000
Gw = 1.0/(1j*(2.0*nlist+1.0)*np.pi/beta+E)

Gwdlr = np.zeros(len(nlist))+0.0*1j
for ni, n in enumerate(nlist):
    Gwdlr[ni] = kernelWn(wGrid, n, beta) @ coeff

plt.figure()
plt.plot(nlist, Gw.imag, label="Gw")
plt.plot(nlist, Gwdlr.imag, label="Gwdlr")
plt.legend()
# plt.show()

print("G(iwn) difference: ", np.max(np.abs(Gwdlr-Gw)))
