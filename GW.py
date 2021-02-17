#!/usr/bin/env python
from scipy import *
from scipy import interpolate
from scipy import integrate
from pylab import *
from pylab import *
import sys
from mpi4py import MPI


def Sigma_x(EF, k, lam, eps):
    kF = sqrt(EF)
    pp = 0
    if lam > 0:
        pp = lam/kF*(arctan((k+kF)/lam)-arctan((k-kF)/lam))
    qq = 1 - pp - (lam**2+kF**2-k**2)/(4*k*kF) * \
        log((lam**2+(k-kF)**2)/(lam**2+(k+kF)**2))
    return -2*kF/(pi*eps)*qq


def EpsEx(rs, lam, eps):
    fx = 1.
    if lam > 0:
        x = (9*pi/4)**(1./3.) * 1./(lam*rs)
        fx = 1 - 1./(6*x**2) - 4*arctan(2*x)/(3*x) + \
            (1.+1./(12*x**2))*log(1+4*x**2)/(2*x**2)
    return -(3./(2.*pi*eps))*(9*pi/4.)**(1./3.)/rs*fx


def Polarisi_orig(q, w, EF):
    """ Polarization P(q,iW) on imaginary axis. Note that the result is real.
        It works on arrays of frequency, i.e., w can be array of bosonic Matsubara points.
    """
    iw = w*1j
    kF = sqrt(EF)
    q2 = q**2
    wmq2 = iw-q2
    wpq2 = iw+q2
    kFq = 2*kF*q
    C1 = log(wmq2-kFq)-log(wmq2+kFq)
    C2 = log(wpq2-kFq)-log(wpq2+kFq)
    D = 1./(8.*kF*q)
    res = -kF/(4*pi**2) * (1. - D*(wmq2**2/q**2-4*EF)
                           * C1 + D*(wpq2**2/q**2-4*EF)*C2)

    # careful for small q or large w
    if type(w) == ndarray:
        b2 = q2 * (q2 + 12./5. * EF)  # b2==b^2
        c = 2*EF*kF*q2/(3*pi**2)
        for i in range(len(w)):
            if w[i] > 20*(q2+kFq):
                res[i] = -c/(w[i]**2 + b2)
    else:
        if w > 20*(q2+kFq):
            b2 = q2 * (q2 + 12./5. * EF)
            c = 2*EF*kF*q2/(3*pi**2)
            res = -c/(w**2 + b2)

    return real(res)


def Polarisi(q, w, EF):
    """ Polarization P(q,iW) on imaginary axis. Note that the result is real.
        It works on arrays of frequency, i.e., w can be array of bosonic Matsubara points.
    """
    kF = sqrt(EF)
    q2 = q**2
    kFq = 2*kF*q
    D = 1./(8.*kF*q)

    if type(w) == ndarray:
        res = zeros(len(w), dtype=float)

        # careful for small q or large w
        # for w[i>=iw_start] we should use power expansion
        is_w_large = w > 20*(q2+kFq)
        # if this was newer true, is_w_large contains only False => iw_start=len(w)
        iw_start = len(w)
        # If at least the last frequency is larger than the cutoff, we can find the index
        if is_w_large[-1]:
            iw_start = argmax(is_w_large)

        # if w < cutoff use exact expression
        iw = w[:iw_start]*1j
        wmq2 = iw-q2
        wpq2 = iw+q2
        C1 = log(wmq2-kFq)-log(wmq2+kFq)
        C2 = log(wpq2-kFq)-log(wpq2+kFq)
        res[:iw_start] = real(-kF/(4*pi**2) * (1. - D *
                                               (wmq2**2/q**2-4*EF)*C1 + D*(wpq2**2/q**2-4*EF)*C2))
        # if w < cutoff use proper power expansion
        b2 = q2 * (q2 + 12./5. * EF)  # b2==b^2
        c = 2*EF*kF*q2/(3*pi**2)
        res[iw_start:] = -c/(w[iw_start:]**2 + b2)
    else:
        # careful for small q or large w
        if w <= 20*(q2+kFq):
            iw = w*1j
            wmq2 = iw-q2
            wpq2 = iw+q2
            C1 = log(wmq2-kFq)-log(wmq2+kFq)
            C2 = log(wpq2-kFq)-log(wpq2+kFq)
            res = real(-kF/(4*pi**2) * (1. - D*(wmq2**2/q**2-4*EF)
                                        * C1 + D*(wpq2**2/q**2-4*EF)*C2))
        else:
            b2 = q2 * (q2 + 12./5. * EF)
            c = 2*EF*kF*q2/(3*pi**2)
            res = -c/(w**2 + b2)
    return res


def GivePolyMesh(x0, L, Nw, power=3, negative=True):
    """ polynomial mesh of the type
        mesh = a * x + b * x**power
        where x=[-1,1] with 2*Nw+1 points (including zero)
        and a and b are determed by x0 & L.
        Requires : x0 > L/Nw**power
                    L > Nw*x0
    """

    den = 1-1./Nw**(power-1)
    alpha = (Nw*x0 - L/Nw**(power-1))/den
    beta = (L - Nw*x0)/den
    if negative:
        x = linspace(-1, 1, 2*Nw+1)
    else:
        x = linspace(0, 1, Nw+1)
    om = alpha*x + beta*x**power
    return om


def Phi_Functional_inside(beta, EF, q, lam, eps, plt=False):
    """computes the Phi functional and the potential energy of
    electron gas within G0W0 approximation.
    """
    def GiveLimits(M_max):
        "Returns the value of the function at this Matsubara point"
        Wm = M_max*2*pi*T
        Pq = Polarisi(q, Wm, EF)
        return abs(log(1-vq*Pq) + vq*Pq)

    T = 1./beta
    kF = sqrt(EF)
    vq = 8*pi/(eps*(q**2+lam**2))
    # First we compute how many Matsubara points is needed
    M_max = 1000  # First start with 1000 points.
    limits = GiveLimits(M_max)  # How small is the function at this frequency?
    if limits > 1e-6:  # The value is larger than 1e-6. We want to increase M.
        while (limits > 1e-6 and M_max > 50):
            M_max = int(M_max*1.2)
            limits = GiveLimits(M_max)
    elif limits < 1e-7:  # The value is too small. We can safely decrease M.
        while (limits < 1e-7 and M_max > 50):
            M_max = int(M_max/1.2)
            limits = GiveLimits(M_max)
    else:
        pass
    Nw_max = 100                  # Number of frequencies in non-uniform mesh
    # The number of all Matsubara points should always be larger than those in poly mesh.
    M_max = max(M_max, Nw_max)
    # and should be smaller than M**3, because we use cubic mesh.
    M_max = min(M_max, Nw_max**3-1)
    # Produces non-uniform mesh of formula y = a*x + b*x^3, which a and b choosen such that the first
    # few points have distance 1 (0,1,...) and we have in total M_max~100 points extended up to M_max~1000.
    Wm_small = array([round(x)
                      for x in GivePolyMesh(1., M_max, Nw_max, 3, False)])
    Wm_small *= 2*pi*T  # small bosonic Matsubara mesh prepared

    # Now we evaluate Polarization on this mesh of ~100 points
    Pq = Polarisi(q, Wm_small, EF)
    Sq_x = -vq*Pq                # Exchange part
    # Phi functional withouth exchange part
    # We keep only correlation part, because exchange part does not converge in this formula
    Sq1 = log(1-vq*Pq) - Sq_x
    # 1/2*Tr(Sigma*G) part withouth exchange part
    # We keep only correlation part, because exchange part does not converge in this formula
    Sq2 = Sq_x/(1+Sq_x) - Sq_x

    # All Matsubara points up to cutoff (~1000 points)
    Wm = 2*pi*T*arange(0, M_max+1)
    # We now interpolate on the entire mesh, using information from the smaller poly mesh.
    Sq1_tot = interpolate.UnivariateSpline(Wm_small, real(Sq1), s=0)(Wm)
    # And now we just perform the sum. The zero point is special in bosonic case. The rest of the points appear twice.
    phi = T*(Sq1_tot[0] + 2*sum(Sq1_tot[1:]))
    # We also interpolate the potential energy.
    Sq2_tot = interpolate.UnivariateSpline(Wm_small, real(Sq2), s=0)(Wm)
    trSG2 = T*(Sq2_tot[0] + 2*sum(Sq2_tot[1:]))

    if plt:
        plot(Wm_small/(2*pi*T), -Sq1, 'o-')
        plot(Wm/(2*pi*T), -Sq1_tot, '-')
        xlim([0, 100])
        show()

    return array([phi, trSG2])


def Phi_Functional(beta, EF, lam, eps, plt=False):
    kF = sqrt(EF)
    rho = kF**3/(3*pi**2)
    qs = linspace(1e-6, 6*kF, 2**7+1)  # reasonable q-mesh with 129 points
    phis = array([Phi_Functional_inside(
        beta, EF, q, lam, eps, plt=False)*q**2 for q in qs])
    Phi = integrate.romb(phis[:, 0], dx=qs[1]-qs[0])/(4*pi**2)
    trSG2 = integrate.romb(phis[:, 1], dx=qs[1]-qs[0])/(4*pi**2)

    if plt:
        plot(qs, phis[:, 0], 'o-', label='Phi')
        plot(qs, phis[:, 1], 's-', label='trSG2')
        legend(loc='best')
        show()
    return (Phi/rho, trSG2/rho)


class ExchangeCorrelation:
    """*************************************************************************/
    Calculates Exchange&Correlation Energy and Potential                       */ 
             S.H.Vosko, L.Wilk, and M.Nusair, Can.J.Phys.58, 1200 (1980)       */
    ****************************************************************************/
    """

    def __init__(self):
        self.alphax = 0.610887057710857  # //(3/(2 Pi))^(2/3)
        self.Ap = 0.0621814
        self.xp0 = -0.10498
        self.bp = 3.72744
        self.cp = 12.9352
        self.Qp = 6.1519908
        self.cp1 = 1.2117833
        self.cp2 = 1.1435257
        self.cp3 = -0.031167608

    def Vx(self, rs):  # Vx
        return -2*self.alphax/rs

    def ExVx(self, rs):  # Ex-Vx
        return 0.5*self.alphax/rs

    def Ex(self, rs):
        return -1.5*self.alphax/rs

    def Vc(self, rs):  # Vc
        x = sqrt(rs)
        xpx = x*x + self.bp*x + self.cp
        atnp = arctan(self.Qp/(2*x+self.bp))
        ecp = 0.5*self.Ap*(log(x*x/xpx)+self.cp1*atnp -
                           self.cp3*(log((x-self.xp0)**2/xpx)+self.cp2*atnp))
        return 2*(ecp - self.Ap/6.*(self.cp*(x-self.xp0)-self.bp*x*self.xp0)/((x-self.xp0)*xpx))

    def EcVc(self, rs):  # Ec-Vc
        x = sqrt(rs)
        return 2*(self.Ap/6.*(self.cp*(x-self.xp0)-self.bp*x*self.xp0)/((x-self.xp0)*(x*x+self.bp*x+self.cp)))


def n_bose(x, beta):
    if x*beta > 100:
        return 0.0
    elif x*beta < -100:
        return -1.0
    else:
        return 1./(exp(x*beta)-1.)


def ferm(x, beta):
    if x*beta > 100:
        return 0.0
    elif x*beta < -100:
        return 1.0
    else:
        return 1./(exp(x*beta)+1.)


def Rkwq(beta, EF, n, k, q, lam, eps, plt=False):
    """ Calculates the Matsubara sum of
       R(k,iw,q) = T*sum_{iW} (1/(1-v_q*P_q)-1) * Integrate[G(iw+iW,sqrt(k^2+q^2-2kq*cost)),{cost,-1,1}]
       At high frequency we subtract the approximation, which goes as 1/iW^3, as explained above.
    """
    def GiveLimits(M_max):
        Wm_limits = array([(-(M_max+2)-n)*2*pi*T, (M_max+2-n)*2*pi*T])
        PqW = Polarisi(q, Wm_limits, EF)
        Wwmn = (Wm_limits + wn)*1j + EF
        C1_limits = (log(Wwmn - kmq2) - log(Wwmn - kpq2)) / \
            (2*k*q) * (1./(1.-vq*PqW)-1.)
        C2_limits = -2*vq*cq/((Wwmn - k2q2)*(Wm_limits**2+aq**2))
        dC = C1_limits-C2_limits
        return sqrt(dC[0].real**2+dC[1].real**2)

    T = 1./beta
    vq = 8*pi/(eps*(q**2+lam**2))
    wn = (2*n+1)*pi*T
    k2q2 = k**2+q**2

    kmq2 = (k-q)**2
    kpq2 = (k+q)**2

    bq2 = q**2 * (q**2 + 12./5. * EF)
    kF = sqrt(EF)
    cq = 2*EF*kF*q**2/(3*pi**2)
    aq = sqrt(bq2 + vq*cq)

    small = 1e-6
    M_max = int((8./3.*k2q2/small)**(1./3.)/(2*pi*T) + n-1)
    if M_max < 50:
        M_max = 50
    limits = GiveLimits(M_max)
    if limits > 1e-4:
        while (limits > 1e-4 and M_max > 50):
            M_max = int(M_max*1.2)
            limits = GiveLimits(M_max)
    elif limits < 1e-5:
        while (limits < 1e-5 and M_max > 50 and M_max > n/2):
            M_max = int(M_max/1.2)
            limits = GiveLimits(M_max)
    else:
        pass

    # GivePolyMesh requires : x0 > L/Nw**power  and L > Nw*x0 hence
    #   M_max+1 > Nw_max
    #   M_max+1 < Nw_max**3
    Nw_max = 30  # This is the number of points we actually evaluate
    M_max = max(M_max, n/2)
    M_max = max(M_max, Nw_max)
    M_max = min(M_max, Nw_max**3-1)
    Wm_small_ = [round(x) for x in GivePolyMesh(1., M_max+1, Nw_max)]
    # Where is zero on this mesh?
    n_0 = Wm_small_.index(0)
    # Creating a combine mesh centered around 0 and around iw.
    Wm_small = []
    for x in Wm_small_:
        if x <= n/2:
            Wm_small.append(x-n)
    for x in Wm_small_:
        if x > -n/2:
            Wm_small.append(x)

    Wm_small = array(Wm_small)*(2*pi*T)

    PqW = Polarisi(q, Wm_small, EF)
    Wwmn = (Wm_small + wn)*1j + EF
    # both frequency dependent quantities
    C1_small = (log(Wwmn - kmq2) - log(Wwmn - kpq2)) / \
        (2*k*q) * (1./(1.-vq*PqW)-1.)
    C2_small = -2*vq*cq/((Wwmn - k2q2)*(Wm_small**2+aq**2))

    # The difference needs to be interpolate on the entire mesh, to sum over Matsubara mesh
    dCrl = interpolate.UnivariateSpline(
        Wm_small[:n_0+1], real(C1_small[:n_0+1]-C2_small[:n_0+1]), s=0)
    dCil = interpolate.UnivariateSpline(
        Wm_small[:n_0+1], imag(C1_small[:n_0+1]-C2_small[:n_0+1]), s=0)
    dCrr = interpolate.UnivariateSpline(
        Wm_small[n_0:], real(C1_small[n_0:]-C2_small[n_0:]), s=0)
    dCir = interpolate.UnivariateSpline(
        Wm_small[n_0:], imag(C1_small[n_0:]-C2_small[n_0:]), s=0)
    # entire Matsubara mesh
    Wml = 2*pi*T*arange(-M_max-n-1, -n)
    Wmr = 2*pi*T*arange(-n, M_max+1)
    # Here is finally the sum over the entire Matsubara mesh
    R1 = sum(dCrl(Wml))*T + sum(dCrr(Wmr))*T + 1j * \
        sum(dCil(Wml))*T + 1j*sum(dCir(Wmr))*T
    # We still need to add the high frequency correction
    wnkq = wn*1j+EF-k2q2
    xi = k2q2-EF
    R3 = vq*cq/aq * ((n_bose(-aq, beta)+ferm(xi, beta)) /
                     (wnkq-aq)-(n_bose(aq, beta)+ferm(xi, beta))/(wnkq+aq))

    if plt:
        print 'n=', n, 'q=', q, 'k=', k
        figure()
        col = ['b', 'r', 'm', 'k', 'g', 'c', 'y', 'b', 'r']
        #plot(Wm_small/(2*pi*T), real(C1_small), 'o', label='C1.real')
        #plot(Wm_small/(2*pi*T), imag(C1_small), 'o', label='C1.imag')
        plot(Wm_small/(2*pi*T), real(C1_small-C2_small),
             'o'+col[0], label='C1-C2.real')
        plot(Wm_small/(2*pi*T), imag(C1_small-C2_small),
             'o'+col[1], label='C1-C2.imag')
        plot(Wml/(2*pi*T), dCrl(Wml), '-'+col[0], label='C1-C2.real')
        plot(Wml/(2*pi*T), dCil(Wml), '-'+col[1], label='C1-C2.imag')
        plot(Wmr/(2*pi*T), dCrr(Wmr), '-'+col[0], label='C1-C2.real')
        plot(Wmr/(2*pi*T), dCir(Wmr), '-'+col[1], label='C1-C2.imag')
        legend(loc='best')
        show()

    return R1+R3


def Sigc_High_Frequency_or_Small_q(q, beta, EF, n, k, lam, eps):
    def GiveQDep(q):
        vq = 8*pi/(eps*(q**2+lam**2))      # v_q
        b2 = q**2 * (q**2 + 12./5. * EF)  # b_q^2
        c = 2*kF**3*q**2/(3*pi**2)         # c_q
        a = sqrt(b2 + vq*c)               # a_q
        kmq2 = (k-q)**2
        kpq2 = (k+q)**2
        amiw = iw+EF-kmq2
        apiw = iw+EF-kpq2
        vqq = vq*q**2
        vqq2 = (vqq)**2/(4*a*k*q)
        return (vq, a, amiw, apiw, vqq, vqq2)

    kF = sqrt(EF)
    wn = (2*n+1)*pi/beta
    iw = wn*1j
    pref = kF**3/(6*pi**4)

    # cc -- correlation part
    # xx -- exchange part
    (vq, a, amiw, apiw, vqq, vqq2) = GiveQDep(q)
    # first part : -kF^3/(6*pi^4) * (v_q*q^2)^2/(4*a*k*q) * ( n(-a) * log((iw+EF-(k-q)^2-a)/(iw+EF-(k+q)^2-a)) - n(a) *  log((iw+EF-(k-q)^2+a)/(iw+EF-(k+q)^2+a)) )
    n_ma = array([n_bose(-aq, beta) for aq in a])
    n_a = array([n_bose(aq, beta) for aq in a])
    cc = n_ma * log((amiw-a)/(apiw-a)) + n_a * log((apiw+a)/(amiw+a))
    xx = zeros(len(cc))

    if k < kF:
        for i, _q in enumerate(q):
            if _q < kF-k:  # q < kF-k and k<kF
                cc[i] += log((amiw[i]-a[i])/(amiw[i]+a[i])) - \
                    log((apiw[i]-a[i])/(apiw[i]+a[i]))
                xx[i] += -vqq[i]/(pi**2) * 0.5

            elif _q < kF+k:  # q is in [kF-k,kF+k] and k<kF
                cc[i] += log((amiw[i]-a[i])/(amiw[i]+a[i])) - \
                    log((iw-a[i])/(iw+a[i]))
                xx[i] += -vqq[i]/(pi**2) * (kF**2 - (_q-k)**2)/(8*k*_q)
    else:
        # first part : -kF^3/(6*pi^4) * (v_q*q^2)^2/(4*a*k*q) * ( log((iw+EF-(k-q)^2-a)/(iw+EF-(k-q)^2+a)) - log((iw-a)/(iw+a)) )
        for i, _q in enumerate(q):
            if _q > k-kF and _q < k+kF:  # q must satisfy : q in [k-kF, k+kF]
                cc[i] += log((amiw[i]-a[i])/(amiw[i]+a[i])) - \
                    log((iw-a[i])/(iw+a[i]))
                xx[i] += -vqq[i]/(pi**2) * (kF**2 - (_q-k)**2)/(8*k*_q)
    res = -pref * vqq2 * cc  # + xx
    return res


def Sigma_c(beta, EF, n, k, lam, eps, plt=False):
    """Numeric calculation of Sigma_correlation"""
    kF = sqrt(EF)

    lowest_point = 1./beta
    kend = max(kF+k+0.5, min(5*(kF+k), 8*kF))

    if k > 0.1:
        ab = sort([lowest_point, abs(kF-k)])
        lowest_point = ab[0]
        qxs = [linspace(ab[0], ab[1], 2**4+1),
               linspace(ab[1], kF+k,  2**6+1),
               linspace(kF+k, kend, 2**5+1)]
    else:
        dk = min(0.05, kF/2.)
        qxs = [linspace(lowest_point, abs(kF-k)-dk, 2**4+1),
               linspace(abs(kF-k)-dk, kF+k+dk,  2**6+1),
               linspace(kF+k+dk, kend, 2**5+1)]

    res = 0
    # Very small q should be computed analytically
    qx = linspace(1e-6, lowest_point, 2**2+1)

    cc2 = Sigc_High_Frequency_or_Small_q(qx, beta, EF, n, k, lam, eps)
    res += integrate.romb(cc2, dx=qx[1]-qx[0])

    if plt:
        plot(qx, real(cc2)*(-pi/2), 's-', label='Re: mesh -1')
        plot(qx, imag(cc2)*(-pi/2), 's-', label='Im: mesh -1')

    max_val = -1e100
    # The rest of the points computed numerically
    for ii, qx in enumerate(qxs):
        cc = array([Rkwq(beta, EF, n, k, q, lam, eps) *
                    q**2/(eps*(q**2+lam**2)) for q in qx])
        max_val = max(max_val, max(abs(cc)))
        res += -2/pi*integrate.romb(cc, dx=qx[1]-qx[0])

        if plt:
            plot(qx, real(cc), 'o-', label='Re: mesh '+str(ii))
            plot(qx, imag(cc), 'o-', label='Im: mesh '+str(ii))

    val = abs(cc[-1])
    value_at_the_end = val/max_val
    # print 'How far = ', value_at_the_end
    if value_at_the_end > 1e-2:  # we need to extend the mesh
        q = qxs[-1][-1]
        while val/max_val > 1e-2:
            q *= 2.
            v = Rkwq(beta, EF, n, k, q, lam, eps) * q**2/(eps*(q**2+lam**2))
            val = abs(v)
        # print 'Found how far we need to go ', q, qx[-1], val/max_val
        qx2 = linspace(qxs[-1][-1], q, 2**5+1)
        cc2 = array([Rkwq(beta, EF, n, k, q, lam, eps) *
                     q**2/(eps*(q**2+lam**2)) for q in qx2])
        res += -2/pi*integrate.romb(cc2, dx=qx2[1]-qx2[0])

        if plt:
            plot(qx2, real(cc2), 'o-', label='Re: mesh '+str(ii))
            plot(qx2, imag(cc2), 'o-', label='Im: mesh '+str(ii))

    if plt:
        legend(loc='best')
        grid()
        show()
    return res


def Sigma_c_high_w(beta, EF, n, k, lam, eps, plt=False):
    """Calculation of Sigma_correlation for high frequency"""
    kF = sqrt(EF)
    lowest_point = 1./beta
    kend = max(kF+k+0.5, min(5*(kF+k), 8*kF))
    if k > 0.1:
        ab = sort([lowest_point, abs(kF-k)])
        lowest_point = ab[0]
        qxs = [linspace(1e-6, lowest_point, 2**2+1),
               linspace(ab[0], ab[1], 2**4+1),
               linspace(ab[1], kF+k,  2**6+1),
               linspace(kF+k, kend, 2**5+1)]
    else:
        dk = min(0.05, kF/2.)
        qxs = [linspace(1e-6, lowest_point, 2**2+1),
               linspace(lowest_point, abs(kF-k)-dk, 2**4+1),
               linspace(abs(kF-k)-dk, kF+k+dk,  2**6+1),
               linspace(kF+k+dk, kend, 2**5+1)]

    res = 0
    max_val = -1e100
    for ii, qx in enumerate(qxs):
        cc = Sigc_High_Frequency_or_Small_q(qx, beta, EF, n, k, lam, eps)
        max_val = max(max_val, max(abs(cc)))
        res += integrate.romb(cc, dx=qx[1]-qx[0])
        if plt:
            plot(qx, real(cc)*(-pi/2), 'o-', label='Re: mesh '+str(ii))
            plot(qx, imag(cc)*(-pi/2), 'o-', label='Im: mesh '+str(ii))

    val = abs(cc[-1])
    value_at_the_end = val/max_val
    if value_at_the_end > 1e-2:  # we need to extend the mesh
        q = qxs[-1][-1]
        while val/max_val > 1e-2:
            q *= 2.
            v = Sigc_High_Frequency_or_Small_q(
                array([q]), beta, EF, n, k, lam, eps)[-1]
            val = abs(v)
        # print 'Found how far we need to go ', q, qx[-1], val/max_val
        qx2 = linspace(qxs[-1][-1], q, 2**5+1)
        cc2 = Sigc_High_Frequency_or_Small_q(qx2, beta, EF, n, k, lam, eps)
        res += integrate.romb(cc2, dx=qx2[1]-qx2[0])

        if plt:
            plot(qx2, real(cc2)*(-pi/2), 'o-', label='Re: mesh '+str(ii))
            plot(qx2, imag(cc2)*(-pi/2), 'o-', label='Im: mesh '+str(ii))

    if plt:
        legend(loc='best')
        grid()
        show()
    return res


def Compute_and_Save_SigmaC(rs, lam, eps, beta=100., nom_low=40, nom_high=40, nom_max=10000):
    "Computes Sigma_c on a 2D mesh of k,iw, and saves it to SigC_rs_xx.dat, mesh_rs_xx.dat"
    def Give2DMesh(EF, beta, nom_low=40, nom_high=40, nom_max=10000):
        " Gives k,iw mesh"
        kF = sqrt(EF)
        # Mesh for variable k
        km = hstack(([1e-10], linspace(0.01, 0.9*kF, 10), linspace(0.9*kF,
                                                                   1.2333*kF, 20)[1:], linspace(1.2333*kF, 2.1*kF, 5)[1:]))
        # cutoff of Matsubara frequency treated numerically
        n_cutoff = int((round((sqrt(kF)*100*beta)/pi)-1)/2)
        nw1 = array([int(round(x))
                     for x in GivePolyMesh(1., n_cutoff, nom_low-1, 3, False)])
        # Second Matsubara mesh computed analytically in the range [n_cutoff, nom_max+n_cutoff] with nom points
        nw2 = array([int(round(x))+nw1[-1]
                     for x in GivePolyMesh(nw1[-1]-nw1[-2], nom_max, nom_high-1, 3, False)])
        nw = hstack((nw1, nw2[1:]))
        return (nw, km)

    CF = (9*pi/4.)**(1./3.)
    kF = CF/rs
    EF = kF**2

    # LDA exchange correlation
    # exc = ExchangeCorrelation()
    # EC_LDA = exc.EcVc(rs) + exc.Vc(rs)
    # G0W0 parts of the functional
    # (Phi, trSG2) = Phi_Functional(beta=beta, EF=EF, lam=lam, eps=eps)

    (nw, km) = Give2DMesh(EF, beta=beta,
                          nom_low=nom_low, nom_high=nom_high, nom_max=nom_max)
    headertxt = '%d %d %d  %f %f %f %f  %f %f %f # Nw,Nk,nom_low, kF,beta,lam,eps, PhiC,EpotC0,EC_LDA' % (
        len(nw), len(km), nom_low, kF, beta, lam, eps, Phi, trSG2, EC_LDA)
    savetxt('mesh_rs_'+str(rs)+'_eps_'+str(eps)+'_lam_'+str(lam) +
            '.dat', hstack((nw, km)).transpose(), header=headertxt)
    SigC_all = zeros((len(km), len(nw)), dtype=complex)
    for ik, k in enumerate(km):
        SigC = [Sigma_c(beta, EF, n, k, lam=lam, eps=eps)
                for n in nw[:nom_low]]
        SigCH = [Sigma_c_high_w(beta, EF, n, k, lam=lam, eps=eps)
                 for n in nw[nom_low:]]
        SigC_ = SigC+SigCH  # concatanating the low and high frequency part
        SigC_all[ik, :] = array(SigC_)  # saving it
    savetxt('SigC_rs_'+str(rs)+'_eps_'+str(eps)+'_lam_' +
            str(lam)+'.dat', SigC_all.view(float).T)


def parallel_limits(rsx, size):
    pr_proc = len(rsx)/size
    if pr_proc*size < len(rsx):
        pr_proc += 1
    a = min(pr_proc*rank, len(rsx))
    b = min(pr_proc*(rank+1), len(rsx))
    return (pr_proc, a, b)


if __name__ == '__main__':

    # rs=8.
    # lam=3.
    # eps=1.
    # beta=100
    #kF = (9*pi/4.)**(1./3.) /rs
    # print 'kF=', kF
    #EF = kF**2
    #k= kF/3.
    # n=1
    #sc = Sigma_c(beta, EF, n, k, lam, eps, plt=True)
    # print sc
    #
    #sc = Sigma_c_high_w(beta, EF, n, k, lam, eps, plt=True)
    # print sc
    # sys.exit(0)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    print size
    # size = 8
    rank = comm.Get_rank()

    if True:  # loop over rs at constant eps,lam
        rsx = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
        # rsx = [1, ]

        pr_proc, a, b = parallel_limits(rsx, size)
        my_rs = rsx[a:b]
        print 'rank=', rank, 'my_rs=', my_rs

        for lam in [0.0]:
            for rs in my_rs:
                Compute_and_Save_SigmaC(rs=rs, lam=lam, eps=1, beta=100.)

        print 'Finished'
    else:  # loop over eps at constant rs,lam
        epsx = [5, 6]
        rs = 0.5

        pr_proc, a, b = parallel_limits(epsx, size)
        #pr_proc = len(epsx)/size
        # if pr_proc*size<len(epsx):
        #    pr_proc += 1
        #a = min( pr_proc*rank, len(epsx) )
        #b = min( pr_proc*(rank+1), len(epsx) )
        my_epsx = epsx[a:b]
        print 'rank=', rank, 'my_epsx=', my_epsx

        for eps in my_epsx:
            Compute_and_Save_SigmaC(rs=rs, lam=0, eps=eps, beta=100.)
        print 'Finished'
