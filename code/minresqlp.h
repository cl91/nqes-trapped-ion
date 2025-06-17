#pragma once

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <concepts>
#include <complex>
#include <algorithm>

#if DBG
#include <cstdio>
#endif

/*
 * Enumeration type for the result of the minres-QLP procedure.
 *
 * -2 Internal error. The procedure encountered an internal programming error.
 * -1 (beta2=0)  B and X are eigenvectors of (A - SHIFT*I).
 * 0 (beta1=0)  B = 0.  The exact solution is X = 0.
 * 1 X solves the compatible (possibly singular) system (A - SHIFT*I)X = B
 * to the desired tolerance:
 * RELRES = RNORM / (ANORM*XNORM + NORM(B)) <= RTOL,
 * where
 * R = B - (A - SHIFT*I)X and RNORM = norm(R).
 * 2 X solves the incompatible (singular) system (A - SHIFT*I)X = B
 * to the desired tolerance:
 * RELARES = ARNORM / (ANORM * RNORM) <= RTOL,
 * where
 * AR = (A - SHIFT*I)R and ARNORM = NORM(AR).
 * 3 Same as 1 with RTOL = EPS.
 * 4 Same as 2 with RTOL = EPS.
 * 5 X converged to an eigenvector of (A - SHIFT*I).
 * 6 XNORM exceeded MAXXNORM.
 * 7 ACOND exceeded ACONDLIM.
 * 8 MAXIT iterations were performed before one of the previous
 * conditions was satisfied.
 * 9 The system appears to be exactly singular.  XNORM does not
 * yet exceed MAXXNORM, but would if further iterations were
 * performed.
 */
enum ResultFlag {
    InternalError = -2,
    EigenVector = -1,
    Zero = 0,
    Converged = 1,
    ConvergedLeastSquare = 2,
    MachinePrecisionReached = 3,
    MachinePrecisionReachedLeastSquare = 4,
    ConvergedToEigenVector = 5,
    NormLimitExceeded = 6,
    ConditionLimitExceeded = 7,
    MaxIterationsReached = 8,
    ExactlySingular = 9
};

template<typename FpType>
struct MinresQlpResult {
    ResultFlag flag;
#if DBG
    std::size_t niters;
    std::size_t minres_iter;
    std::size_t qlp_iter;
    FpType relres;
    FpType relAres;
    FpType Anorm;
    FpType Acond;
    FpType xnorm;
    FpType Axnorm;
#endif
};

template<typename Vector>
concept VectorType = requires (Vector v) {
    typename Vector::RealScalar;
    { v.size() } -> std::convertible_to<std::size_t>;
    v.norm();
    v.dot(v);
};

template<typename FpType>
FpType sign(FpType a) {
    return std::copysign(1.0, a);
}

template<typename FpType>
static void sym_ortho(FpType a, FpType b, FpType &c, FpType &s, FpType &r) {
    if (b == 0) {
        if (a == 0) {
            c = 1;
	} else {
            c = sign(a);
	}
        s = 0;
        r = std::fabs(a);
    } else if (a == 0) {
        c = 0;
        s = sign(b);
        r = std::fabs(b);
    } else if (std::fabs(b) > std::fabs(a)) {
        FpType t = a / b;
        s = sign(b) / std::sqrt(1 + t * t);
        c = s * t;
        r = b / s;
    } else {
        FpType t = b / a;
        c = sign(a) / std::sqrt(1 + t * t);
        s = c * t;
        r = a / c;
    }
}

template<VectorType Vector, typename Callable>
MinresQlpResult<typename Vector::RealScalar>
minresqlp(Callable &&A, const Vector &b, Vector &x,
	  typename Vector::RealScalar rtol, std::size_t maxit,
	  typename Vector::RealScalar shift = 0,
	  typename Vector::RealScalar maxxnorm = 1e7,
	  typename Vector::RealScalar Acondlim = 1e15,
	  typename Vector::RealScalar TranCond = 1e7)
{
    using FpType = Vector::RealScalar;
    size_t n = b.size();
    assert(b.size() == x.size());
    Vector r1(b), r2(b), r3(b);
    FpType beta1 = r2.norm();

    // Initialize
    ResultFlag flag0 = ResultFlag::InternalError, flag = flag0;
    std::size_t iters = 0, QLPiter = 0;
    FpType beta = 0, tau = 0, taul = 0, phi = beta1, betan = beta1, gmin = 0, gminl = 0;
    FpType cs = -1, sn = 0, cr1 = -1, sr1 = 0, cr2 = -1, sr2 = 0, dltan = 0, eplnn = 0;
    FpType gama = 0, gamal = 0, gamal2 = 0, eta = 0, etal = 0, etal2 = 0;
    FpType vepln = 0, veplnl = 0, veplnl2 = 0, ul3 = 0, ul2 = 0, ul = 0, u = 0;
    FpType gama_QLP = 0, gamal_QLP = 0, vepln_QLP = 0, u_QLP = 0, ul_QLP = 0;
    FpType rnorm = betan, xnorm = 0, xl2norm = 0, Axnorm = 0, Anorm = 0, Acond = 1;
    FpType relres = rnorm / (beta1 + 1e-50);
    x = Vector::Zero(n);
    Vector w = Vector::Zero(n);
    Vector wl = Vector::Zero(n);
    Vector wl2 = Vector::Zero(n);
    Vector xl2 = Vector::Zero(n);

#if DBG
    printf("Enter Minres-QLP: \n");
    printf("  Min-length solution of symmetric(singular) (A-sI)x = b or min ||(A-sI)x - b||\n");
    // ||Ax - b|| is ||(A-sI)x - b|| if shift != 0 here
    printf("      n = %8zd    ||Ax - b|| = %8.2e     shift = %8.2e       rtol = %8g\n",
	   n, beta1, shift, rtol);
    printf("  maxit = %8zd      maxxnorm = %8.2e  Acondlim = %8.2e   TranCond = %8g\n",
	   maxit, maxxnorm, Acondlim, TranCond);
#endif

    // b = 0 --> x = 0 skip the main loop
    if (beta1 == 0) {
	flag = ResultFlag::Zero;
    }

    while (flag == flag0 && iters < maxit) {
        // Lanczos
        iters++;
        FpType betal = beta;
        beta = betan;
        Vector v(r3/beta);
        r3 = A(v);
        if (shift != 0) {
            r3 -= shift*v;
	}

        if (iters > 1) {
            r3 -= r1 * (beta/betal);
	}

        FpType alfa = std::real(r3.dot(v));
        r3 -= r2 * (alfa/beta);
        r1 = r2;
        r2 = r3;

        betan = r3.norm();
        if (iters == 1) {
            if (betan == 0) {
                if (alfa == 0) {
                    flag = ResultFlag::Zero;
		} else {
                    flag = ResultFlag::EigenVector;
                    x = b/alfa;
		}
                break;
	    }
	}
        FpType pnorm = std::sqrt(betal * betal + alfa * alfa + betan * betan);

        // previous left rotation Q_{k-1}
        FpType dbar = dltan;
        FpType dlta = cs*dbar + sn*alfa;
        FpType epln = eplnn;
        FpType gbar = sn*dbar - cs*alfa;
        eplnn = sn*betan;
        dltan = -cs*betan;
        FpType dlta_QLP = dlta;
        // current left plane rotation Q_k
        FpType gamal3 = gamal2;
        gamal2 = gamal;
        gamal = gama;
        sym_ortho(gbar, betan, cs, sn, gama);
        FpType gama_tmp = gama;
        FpType taul2 = taul;
        taul = tau;
        tau = cs*phi;
        Axnorm = std::sqrt(Axnorm * Axnorm + tau * tau);
        phi = sn*phi;
        // previous right plane rotation P_{k-2,k}
        if (iters > 2) {
            veplnl2 = veplnl;
            etal2 = etal;
            etal = eta;
            FpType dlta_tmp = sr2*vepln - cr2*dlta;
            veplnl = cr2*vepln + sr2*dlta;
            dlta = dlta_tmp;
            eta = sr2*gama;
            gama = -cr2 *gama;
	}
        // current right plane rotation P{k-1,k}
        if (iters > 1) {
            sym_ortho(gamal, dlta, cr1, sr1, gamal);
            vepln = sr1*gama;
            gama = -cr1*gama;
	}

        // update xnorm
        FpType xnorml = xnorm;
        FpType ul4 = ul3;
        ul3 = ul2;
        if (iters > 2) {
            ul2 = (taul2 - etal2*ul4 - veplnl2*ul3)/gamal2;
	}
        if (iters > 1) {
            ul = (taul - etal*ul3 - veplnl *ul2)/gamal;
	}
        FpType xnorm_tmp = std::sqrt(xl2norm* xl2norm + ul2 * ul2 + ul * ul);
        if (std::fabs(gama) > std::numeric_limits<FpType>::min() && xnorm_tmp < maxxnorm) {
            u = (tau - eta*ul2 - vepln*ul)/gama;
            if (std::sqrt(xnorm_tmp * xnorm_tmp + u * u) > maxxnorm) {
                u = 0;
                flag = ResultFlag::NormLimitExceeded;
	    }
	} else {
            u = 0;
            flag = ResultFlag::ExactlySingular;
	}
        xl2norm = std::sqrt(xl2norm * xl2norm + ul2 * ul2);
        xnorm = std::sqrt(xl2norm * xl2norm + ul * ul + u * u);
        // update w&x
        if (Acond < TranCond && flag != flag0 && QLPiter == 0) {
            // Minres
            wl2 = wl;
            wl = w;
            w = (v - epln*wl2 - dlta_QLP*wl)/gama_tmp;
            if (xnorm < maxxnorm) {
                x += tau*w;
	    } else {
                flag = ResultFlag::NormLimitExceeded;
	    }
	} else {
            // Minres-QLP
            QLPiter++;
            if (QLPiter == 1) {
                if (iters > 1) {
		    // construct w_{k-3}, w_{k-2}, w_{k-1}
                    if (iters > 3) {
                        wl2 = gamal3*wl2 + veplnl2*wl + etal*w;
		    }
                    if (iters > 2) {
                        wl = gamal_QLP*wl + vepln_QLP*w;
		    }
                    w = gama_QLP*w;
		    xl2 = x - wl*ul_QLP - w*u_QLP;
		}
	    }

            if (iters == 1) {
                wl2 = wl;
                wl = v*sr1;
                w = -v*cr1;
	    } else if (iters == 2) {
                wl2 = wl;
                wl = w*cr1 + v*sr1;
                w = w*sr1 - v*cr1;
	    } else {
                wl2 = wl;
                wl = w;
                w = wl2*sr2 - v*cr2;
                wl2 = wl2*cr2 +v*sr2;
                v = wl*cr1 + w*sr1;
                w = wl*sr1 - w*cr1;
                wl = v;
	    }
	    xl2 += wl2*ul2;
            x = xl2 + wl*ul + w*u;
	}

        // next right plane rotation P{k-1,k+1}
        FpType gamal_tmp = gamal;
        sym_ortho(gamal, eplnn, cr2, sr2, gamal);
        // transfering from Minres to Minres-QLP
        gamal_QLP = gamal_tmp;
        // printf("gamal_QLP=\n", gamal_QLP);
        vepln_QLP = vepln;
        gama_QLP = gama;
        ul_QLP = ul;
        u_QLP = u;
        // Estimate various norms
        FpType abs_gama = std::fabs(gama);
        FpType Anorml = Anorm;
        Anorm = std::max({Anorm, pnorm, gamal, abs_gama});
        if (iters == 1) {
            gmin = gama;
            gminl = gmin;
	} else if (iters > 1) {
            FpType gminl2 = gminl;
            gminl = gmin;
            gmin = std::min({gminl2, gamal, abs_gama});
	}
	FpType Acondl = Acond;
        Acond = Anorm / gmin;
        FpType rnorml = rnorm;
        FpType relresl = relres;
        if (flag != ResultFlag::ExactlySingular) {
            rnorm = phi;
	}
        relres = rnorm / (Anorm * xnorm + beta1);
        FpType rootl = std::sqrt(gbar * gbar + dltan * dltan);
        FpType Arnorml = rnorml * rootl;
        FpType relAresl = rootl / Anorm;
        // See if any of the stopping criteria are satisfied.
        FpType epsx = Anorm * xnorm * std::numeric_limits<FpType>::epsilon();
        if (flag == flag0 || flag == ResultFlag::ExactlySingular) {
            FpType t1 = 1 + relres;
            FpType t2 = 1 + relAresl;
            if (iters >= maxit) {
                flag = ResultFlag::MaxIterationsReached;
	    }
            if (Acond >= Acondlim) {
                flag = ResultFlag::ConditionLimitExceeded; // Huge Acond
	    }
            if (xnorm >= maxxnorm) {
                flag = ResultFlag::NormLimitExceeded; // xnorm exceeded
	    }
            if (epsx >= beta1) {
                flag = ResultFlag::ConvergedToEigenVector; // x = eigenvector
	    }
            if (t2 <= 1) {
		// Accurate Least Square Solution
                flag = ResultFlag::MachinePrecisionReachedLeastSquare;
	    }
            if (t1 <= 1) {
                flag = ResultFlag::MachinePrecisionReached; // Accurate Ax = b Solution
	    }
            if (relAresl <= rtol) {
                flag = ResultFlag::ConvergedLeastSquare; // Trustful Least Square Solution
	    }
            if (relres <= rtol) {
                flag = ResultFlag::Converged; // Trustful Ax = b Solution
	    }
	}
        if (flag == ResultFlag::ConvergedLeastSquare ||
	    flag == ResultFlag::MachinePrecisionReachedLeastSquare ||
	    flag == ResultFlag::NormLimitExceeded ||
	    flag == ResultFlag::ConditionLimitExceeded) {
            // possibly singular
            iters--;
            Acond = Acondl;
            rnorm = rnorml;
            relres = relresl;
	} else {
#if DBG
            if ((iters % 10) == 1) {
                printf("       iter   rnorm     Arnorm     relres"
		       "   relAres    Anorm     Acond     xnorm\n");
	    }
            if (QLPiter == 1) {
                printf("QLP");
            } else {
                printf("   ");
	    }
            printf("%8zd  %8.2e  %8.2e   %8.2e  %8.2e  %8.2e  %8.2e  %8.2e\n",
		   iters-1, rnorml, Arnorml, relresl, relAresl, Anorml, Acondl, xnorml);
#endif
	}
    }

    // exited the main loop
    MinresQlpResult<FpType> res;
    res.flag = flag;

#if DBG
    std::size_t Miter = iters - QLPiter;
    res.niters = iters;
    res.minres_iter = Miter;
    res.qlp_iter = QLPiter;

    if (QLPiter == 1) {
        printf("QLP");
    } else {
        printf("   ");
    }
    // final quantities, these are only computed in debug mode
    r1 = b - A(x) + shift*x;
    rnorm = r1.norm();
    FpType Arnorm = (A(r1) - shift*r1).norm();
    xnorm = x.norm();
    relres = rnorm/(Anorm*xnorm + beta1);
    FpType relAres = 0;
    if (rnorm > std::numeric_limits<FpType>::min() && Anorm*rnorm != 0.0) {
        relAres = Arnorm/(Anorm*rnorm);
    }
    res.relres = relres;
    res.relAres = relAres;
    res.Anorm = Anorm;
    res.Acond = Acond;
    res.xnorm = xnorm;
    res.Axnorm = Axnorm;

    if (rnorm > std::numeric_limits<FpType>::min()) {
        printf("%8zd  %8.2e  %8.2eD  %8.2e  %8.2eD %8.2e  %8.2e  %8.2e\n",
	       iters, rnorm, Arnorm, relres, relAres, Anorm, Acond, xnorm);
    } else {
        printf("%8zd  %8.2e  %8.2eD  %8.2e           %8.2e  %8.2e  %8.2e\n",
	       iters, rnorm, Arnorm, relres, Anorm, Acond, xnorm);
    }

    printf("\nExit Minres-QLP: \n");
    const char *msg[] = {
	" beta2 = 0.  b and x are eigenvectors                   ",  // -1
	" beta1 = 0.  The exact solution is  x = 0               ",  // 0
	" A solution to Ax = b found, given rtol                 ",  // 1
	" Min-length solution for singular LS problem, given rtol",  // 2
	" A solution to Ax = b found, given eps                  ",  // 3
	" Min-length solution for singular LS problem, given eps ",  // 4
	" x has converged to an eigenvector                      ",  // 5
	" xnorm has exceeded maxxnorm                            ",  // 6
	" Acond has exceeded Acondlim                            ",  // 7
	" The iteration limit was reached                        ",  // 8
	" Least-squares problem but no converged solution yet    ",  // 9
    };

    printf("  Flag = %8d     %8s\n", flag, msg[int(flag + 1)]);
    printf("  Iter = %8zd      Minres = %8zd   Minres-QLP =   %8zd\n",
	   iters, Miter, QLPiter);
    printf("  relres = %8.2e    relAres = %8.2e  rnorm = %8.2e      Arnorm = %8.2e\n",
	   relres, relAres, rnorm, Arnorm);
    printf("  Anorm = %8.2e     Acond = %8.2e    xnorm = %8.2e      Axnorm = %8.2e\n\n",
	   Anorm, Acond, xnorm, Axnorm);
#endif

    return res;
}
