#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <matio.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <bitset>
#include <complex>
#include <concepts>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <cfenv>
#include <csignal>
#include <execinfo.h>
#include "minresqlp.h"
#include "except.h"
#include "argparse.h"
#include "pcg_random.h"

using namespace std::complex_literals;

using RealScalar = double;
using ComplexScalar = std::complex<RealScalar>;

#pragma omp declare					\
    reduction(+:ComplexScalar : omp_out += omp_in)	\
    initializer(omp_priv = omp_orig)

#pragma omp declare					\
    reduction(*:ComplexScalar : omp_out *= omp_in)	\
    initializer(omp_priv = omp_orig)

constexpr int MAX_STACK_ALLOCATED_VEC_SIZE = 8;

constexpr int compile_time_size(std::size_t n)
{
    return n > MAX_STACK_ALLOCATED_VEC_SIZE ? Eigen::Dynamic : n;
}

constexpr auto storage_order(std::size_t n, std::size_t m)
{
    return m == 1 ? Eigen::ColMajor : Eigen::RowMajor;
}

template<std::size_t N>
using RealVector = Eigen::Vector<RealScalar, compile_time_size(N)>;

template<std::size_t N, std::size_t M>
using RealMatrix = Eigen::Matrix<RealScalar, compile_time_size(N),
				 compile_time_size(M), storage_order(N, M)>;

template<std::size_t N>
using ComplexVector = Eigen::Vector<ComplexScalar, compile_time_size(N)>;

template<std::size_t N, std::size_t M>
using ComplexMatrix = Eigen::Matrix<ComplexScalar, compile_time_size(N),
				    compile_time_size(M), storage_order(N, M)>;

template<std::size_t N>
ComplexVector<N> zeros() { return ComplexVector<N>::Zero(N); }

template<std::size_t N, std::size_t M>
ComplexMatrix<N,M> zeros() { return ComplexMatrix<N, M>::Zero(N,M); }

// Defines the K-by-K matrix that represents the value of the grand
// wave function at a given point of spin configuration and weights.
template<std::size_t K>
using Phi = ComplexMatrix<K, K>;

// Represents a given spin configuration with N spins.
template<std::size_t N>
using Spin = std::bitset<N>;

// Represents K spin configurations, each having N spins.
template<std::size_t N, std::size_t K>
using MultiSpin = std::array<std::bitset<N>, K>;

// Computes the total number of weight parameters for one single RBM
constexpr std::size_t num_weights(std::size_t n, std::size_t m)
{
    return n + m + n*m;
}

// Computes the total number of weight parameters for a multi-RBM
constexpr std::size_t num_weights_multi(std::size_t n, std::size_t m, std::size_t k)
{
    return k * num_weights(n, m);
}

// Represents the derivatives of a scalar function of the multi-RBM state with respect
// to the weights of the Multi-RBM state, evaluated at a given multi-spin configuration.
template<std::size_t N, std::size_t M, std::size_t K>
using Deriv = ComplexVector<num_weights_multi(N, M, K)>;

// Generates a random initial spin configuration
template<std::size_t N>
Spin<N> init_spin(auto &rg)
{
    Spin<N> s;
    std::bernoulli_distribution d(0.5);
    for (unsigned int i = 0; i < N; i++) {
	s[i] = d(rg);
    }
    return s;
}

// Generates a random initial multi-spin configuration of K spins
template<std::size_t N, std::size_t K>
MultiSpin<N,K> init_multi_spin(auto &rg)
{
    MultiSpin<N,K> s;
    for (unsigned int i = 0; i < K; i++) {
	s[i] = init_spin<N>(rg);
    }
    return s;
}

// Represents one single restricted Boltzmann machine
template<std::size_t N, std::size_t M>
struct RBM {
    ComplexVector<N> a;
    ComplexVector<M> b;
    ComplexMatrix<N, M> w;
};

// Represents the K restricted Boltzmann machines used to parameterize
// the grand wave function
template<std::size_t N, std::size_t M, std::size_t K>
using MultiRBM = std::array<RBM<N,M>, K>;

// Computes the value of theta_j given an RMB and a spin configuration
template<std::size_t N, std::size_t M>
ComplexScalar theta(int j, const Spin<N> &s, const RBM<N,M> &rbm)
{
    ComplexScalar res = rbm.b[j];
#pragma omp simd reduction (+:res)
    for (unsigned int i = 0; i < N; i++) {
	res += s[i] ? -rbm.w(i,j) : rbm.w(i,j);
    }
    return res;
}

// Computes the value of the grand wave function parameterized by K RBMs at a
// given spin configuration and the corresponding tanh(theta), using multi-threading.
template<std::size_t N, std::size_t M, std::size_t K> requires ((M/N)*N == M)
Phi<K> state_fcn(const MultiSpin<N,K> &s, const MultiRBM<N,M,K> &rbm,
		 ComplexMatrix<M,K*K> *tanh_theta = nullptr)
{
    auto phi = zeros<K,K>();
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int kk = 0; kk < K; kk++) {
	    for (unsigned int i = 0; i < N; i++) {
		phi(k,kk) += s[k][i] ? -rbm[kk].a[i] : rbm[kk].a[i];
	    }
	    phi(k,kk) = std::exp(phi(k,kk));
	    for (unsigned int j = 0; j < M; j++) {
		// Technically there is a factor of two before the cosh, but since we are
		// computing an unnormalized wave function it's ok to leave it out.
		auto th = theta(j, s[k], rbm[kk]);
		phi(k,kk) *= std::cosh(th);
		if (tanh_theta) {
		    (*tanh_theta)(j,k*K+kk) = std::tanh(th);
		}
	    }
	}
    }
    return phi;
}

// Compute the (multi-)state function with the i-th spin of the k-th system flipped.
// This function modifies the k-th row of the given grand wave function Phi and the
// corresponding entries of the tanh_theta matrix.
template<std::size_t N, std::size_t M, std::size_t K>
void flip_state_fcn(unsigned int k, unsigned int i, const MultiSpin<N,K> &s,
		    const MultiRBM<N,M,K> &rbm, Phi<K> &phi, ComplexMatrix<M,K*K> &tanh_theta)
{
    for (unsigned int kk = 0; kk < K; kk++) { // kk-th column
	phi(k,kk) *= std::exp(s[k][i] ? 2.0 * rbm[kk].a[i] : -2.0 * rbm[kk].a[i]);
	for (unsigned int j = 0; j < M; j++) {
	    ComplexScalar ws = s[k][i] ? -2.0 * rbm[kk].w(i,j) : 2.0 * rbm[kk].w(i,j);
	    ComplexScalar th = tanh_theta(j,k*K+kk);
	    ComplexScalar denom = std::cosh(ws) - th * std::sinh(ws);
	    tanh_theta(j,k*K+kk) = (th * std::cosh(ws) - std::sinh(ws)) / denom;
	    phi(k,kk) *= denom;
	}
    }
}

// Computes the derivatives of the determinant of the grand wave function
// with respect to the Multi-RBM weights divided by the determinant, ie.
//
//   O_{k,l}(S) = (det Phi(S))^(-1) (\partial_{W_{k,l}} det Phi)(S)
//
// Note: caller must supply phi and tanh_theta that correspond to the given
// multi-RBM state and the multi-spin configuration.
template<std::size_t N, std::size_t M, std::size_t K>
Deriv<N,M,K> deriv(const MultiSpin<N,K> &s, const Phi<K> &phi,
		   const Phi<K> &phi_inv, const ComplexMatrix<M,K*K> &tanh_theta)
{
    auto L = num_weights(N,M);
    Deriv<N,M,K> res(num_weights_multi(N,M,K));
    // Compute the derivatives with respect to the weights of the k-th RBM.
    // We do so by multiplying the k-th column of the grand wave function Phi
    // by their derivatives and then taking the determinant.
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int i = 0; i < N; i++) {
	    Phi<K> derphi(phi);
	    for (unsigned int kk = 0; kk < K; kk++) { // kk-th row
		if (s[kk][i]) {
		    derphi(kk, k) = -derphi(kk, k);
		}
	    }
	    res[k*L + i] = (phi_inv * derphi).determinant();
	}
	for (unsigned int j = 0; j < M; j++) {
	    Phi<K> derphi(phi);
	    for (unsigned int kk = 0; kk < K; kk++) { // kk-th row
		derphi(kk, k) *= tanh_theta(j, kk*K+k);
	    }
	    res[k*L + N + j] = (phi_inv * derphi).determinant();
	}
	for (unsigned int i = 0; i < N; i++) {
	    for (unsigned int j = 0; j < M; j++) {
		Phi<K> derphi(phi);
		for (unsigned int kk = 0; kk < K; kk++) { // kk-th row
		    derphi(kk, k) *= tanh_theta(j, kk*K+k);
		    if (s[kk][i]) {
			derphi(kk, k) = -derphi(kk, k);
		    }
		}
		res[k*L + N + M + i*M + j] = (phi_inv * derphi).determinant();
	    }
	}
    }
    return res;
}

// Generates an initial RMB with random weights, sampled from a Gaussian distribution.
template<std::size_t N, std::size_t M>
RBM<N,M> init_rbm(auto &rg)
{
    RBM<N,M> rbm{.a{N}, .b{M}, .w{N,M}};
    std::normal_distribution<RealScalar> norm;
    for (unsigned int i = 0; i < N; i++) {
	rbm.a[i] = ComplexScalar{norm(rg), norm(rg)} / ComplexScalar{std::sqrt(N)};
    }
    for (unsigned int i = 0; i < M; i++) {
	rbm.b[i] = ComplexScalar{norm(rg), norm(rg)} / ComplexScalar{std::sqrt(M)};
    }
    for (unsigned int i = 0; i < N; i++) {
	for (unsigned int j = 0; j < M; j++) {
	    rbm.w(i,j) = ComplexScalar{norm(rg), norm(rg)} / ComplexScalar{std::sqrt(N*M)};
	}
    }
    return rbm;
}

// Generates K random initial RBMs
template<std::size_t N, std::size_t M, std::size_t K>
MultiRBM<N,M,K> init_multi_rbm(auto &rg)
{
    MultiRBM<N,M,K> rbm;
    for (unsigned int i = 0; i < K; i++) {
	rbm[i] = init_rbm<N,M>(rg);
    }
    return rbm;
}

// Load the weights of the K RBMs from a MATLAB .mat file (v5)
template<std::size_t N, std::size_t M, std::size_t K>
MultiRBM<N,M,K> load_rbm(const char *fname)
{
    MultiRBM<N,M,K> rbm;
    for (unsigned int k = 0; k < K; k++) {
        rbm[k] = RBM<N,M>{.a{N}, .b{M}, .w{N,M}};
    }
    mat_t *mf = Mat_Open(fname, MAT_ACC_RDONLY);
    if (!mf) {
	ThrowException(MatioCreateFileError, fname);
    }

    auto dataty = sizeof(RealScalar) == 4 ? MAT_T_SINGLE : MAT_T_DOUBLE;
    matvar_t *ma = Mat_VarRead(mf, "a");
    if (!ma) {
	ThrowException(MatioReadVarError, "a");
    }
    if (ma->data_type != dataty) {
	ThrowException(MatioInvalidVarError, "a" , "floating point type mismatch");
    }
    if (!ma->isComplex) {
	ThrowException(MatioInvalidVarError, "a" , "not a complex matrix");
    }
    if (ma->rank != 2) {
	ThrowException(MatioInvalidVarError, "a" , "not a rank 2 matrix");
    }
    if (ma->dims[0] != K) {
	ThrowException(MatioInvalidVarError, "a" , "invalid row dimension");
    }
    if (ma->dims[1] != N) {
	ThrowException(MatioInvalidVarError, "a" , "invalid column dimension");
    }

    matvar_t *mb = Mat_VarRead(mf, "b");
    if (!mb) {
	ThrowException(MatioReadVarError, "b");
    }
    if (!mb->isComplex) {
	ThrowException(MatioInvalidVarError, "b" , "not a complex matrix");
    }
    if (mb->data_type != dataty) {
	ThrowException(MatioInvalidVarError, "b" , "floating point type mismatch");
    }
    if (mb->rank != 2) {
	ThrowException(MatioInvalidVarError, "b" , "not a rank 2 matrix");
    }
    if (mb->dims[0] != K) {
	ThrowException(MatioInvalidVarError, "b" , "invalid row dimension");
    }
    if (mb->dims[1] != M) {
	ThrowException(MatioInvalidVarError, "b" , "invalid column dimension");
    }

    matvar_t *mw = Mat_VarRead(mf, "w");
    if (!mw) {
	ThrowException(MatioReadVarError, "w");
    }
    if (mw->data_type != dataty) {
	ThrowException(MatioInvalidVarError, "w" , "floating point type mismatch");
    }
    if (!mw->isComplex) {
	ThrowException(MatioInvalidVarError, "w" , "not a complex matrix");
    }
    if (mw->rank != 3) {
	ThrowException(MatioInvalidVarError, "w" , "not a rank 3 matrix");
    }
    if (mw->dims[0] != K) {
	ThrowException(MatioInvalidVarError, "w" , "invalid dimension 0");
    }
    if (mw->dims[1] != N) {
	ThrowException(MatioInvalidVarError, "w" , "invalid dimension 1");
    }
    if (mw->dims[2] != M) {
	ThrowException(MatioInvalidVarError, "w" , "invalid dimension 2");
    }

    auto mac = (const mat_complex_split_t *)ma->data;
    auto ar = (const RealScalar *)mac->Re;
    auto ai = (const RealScalar *)mac->Im;
#pragma omp parallel for collapse(2)
    // IMPORTANT: MATLAB matrices are stored in COLUMN-major order!
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int i = 0; i < N; i++) {
	    rbm[k].a[i] = ComplexScalar{ar[i*K+k], ai[i*K+k]};
	}
    }
    auto mbc = (const mat_complex_split_t *)mb->data;
    auto br = (const RealScalar *)mbc->Re;
    auto bi = (const RealScalar *)mbc->Im;
#pragma omp parallel for collapse(2)
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int i = 0; i < M; i++) {
	    rbm[k].b[i] = ComplexScalar{br[i*K+k], bi[i*K+k]};
	}
    }
    auto mwc = (const mat_complex_split_t *)mw->data;
    auto wr = (const RealScalar *)mwc->Re;
    auto wi = (const RealScalar *)mwc->Im;
#pragma omp parallel for collapse(3)
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int i = 0; i < N; i++) {
	    for (unsigned int j = 0; j < M; j++) {
		rbm[k].w(i,j) = ComplexScalar{wr[j*K*N+i*K+k], wi[j*K*N+i*K+k]};
	    }
	}
    }

    Mat_VarFree(ma);
    Mat_VarFree(mb);
    Mat_VarFree(mw);
    Mat_Close(mf);
    return rbm;
}

// Load the weights of the single RBM from a MATLAB .mat file (v5) for all K
// RBMs and apply the spin flip transformation to all RBMs except the first.
template<std::size_t N, std::size_t M, std::size_t K>
MultiRBM<N,M,K> load_single_rbm(const char *fname)
{
    auto single_rbm = load_rbm<N, M, 1>(fname);
    MultiRBM<N,M,K> rbm;
    rbm[0].a = single_rbm[0].a;
    rbm[0].b = single_rbm[0].b;
    rbm[0].w = single_rbm[0].w;
    for (unsigned int k = 1; k < K; k++) {
        rbm[k].a = -single_rbm[0].a;
	rbm[k].b = single_rbm[0].b;
	rbm[k].w = -single_rbm[0].w;
    }
    return rbm;
}

// Writes the weights of the K RBMs to a MATLAB .mat file (v5)
template<std::size_t N, std::size_t M, std::size_t K>
void dump_rbm(const MultiRBM<N,M,K> &rbm, const char *fname_prefix)
{
    std::stringstream ss;
    ss << fname_prefix << "N" << N << "M" << M << "K" << K << ".mat";
    mat_t *mf = Mat_CreateVer(ss.str().c_str(), NULL, MAT_FT_MAT5);
    if (!mf) {
	ThrowException(MatioCreateFileError, ss.str().c_str());
    }

    std::vector<RealScalar> ar(K*N), ai(K*N), br(K*M), bi(K*M), wr(K*N*M), wi(K*N*M);
    // IMPORTANT: MATLAB matrices are stored in COLUMN-major order!
#pragma omp parallel for collapse(2)
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int i = 0; i < N; i++) {
	    ar[i*K+k] = rbm[k].a[i].real();
	    ai[i*K+k] = rbm[k].a[i].imag();
	}
    }
#pragma omp parallel for collapse(2)
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int i = 0; i < M; i++) {
	    br[i*K+k] = rbm[k].b[i].real();
	    bi[i*K+k] = rbm[k].b[i].imag();
	}
    }
#pragma omp parallel for collapse(3)
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int i = 0; i < N; i++) {
	    for (unsigned int j = 0; j < M; j++) {
		wr[j*K*N+i*K+k] = rbm[k].w(i,j).real();
		wi[j*K*N+i*K+k] = rbm[k].w(i,j).imag();
	    }
	}
    }

    auto classty = sizeof(RealScalar) == 4 ? MAT_C_SINGLE : MAT_C_DOUBLE;
    auto dataty = sizeof(RealScalar) == 4 ? MAT_T_SINGLE : MAT_T_DOUBLE;
    size_t adims[] = { K, N };
    struct mat_complex_split_t a = { ar.data(), ai.data() };
    matvar_t *mv = Mat_VarCreate("a", classty, dataty, 2, adims, &a, MAT_F_COMPLEX);
    if (!mv) {
	ThrowException(MatioCreateVarError, "a");
    }
    Mat_VarWrite(mf, mv, MAT_COMPRESSION_NONE);
    Mat_VarFree(mv);

    size_t bdims[] = { K, M };
    struct mat_complex_split_t b = { br.data(), bi.data() };
    mv = Mat_VarCreate("b", classty, dataty, 2, bdims, &b, MAT_F_COMPLEX);
    if (!mv) {
	ThrowException(MatioCreateVarError, "b");
    }
    Mat_VarWrite(mf, mv, MAT_COMPRESSION_NONE);
    Mat_VarFree(mv);

    size_t wdims[] = { K, N, M };
    struct mat_complex_split_t w = { wr.data(), wi.data() };
    mv = Mat_VarCreate("w", classty, dataty, 3, wdims, &w, MAT_F_COMPLEX);
    if (!mv) {
	ThrowException(MatioCreateVarError, "w");
    }
    Mat_VarWrite(mf, mv, MAT_COMPRESSION_NONE);
    Mat_VarFree(mv);
    Mat_Close(mf);
}

// Writes the values of the grand wave function Phi for each spin configuration to a mat file
template<std::size_t N, std::size_t M, std::size_t K> requires ((K*N) < 32)
void dump_states(const MultiRBM<N,M,K> &rbm, const char *fname_prefix, auto &&fcn)
{
    std::stringstream ss;
    ss << fname_prefix << "N" << N << "M" << M << "K" << K << ".mat";
    mat_t *mf = Mat_CreateVer(ss.str().c_str(), NULL, MAT_FT_MAT5);
    if (!mf) {
	ThrowException(MatioCreateFileError, ss.str().c_str());
    }

    size_t dim0 = 1ULL << (K*N);
    int rows = fcn(MultiSpin<N,K>{}, rbm).rows();
    int cols = fcn(MultiSpin<N,K>{}, rbm).cols();
    std::vector<size_t> dims{dim0};
    if (rows > 1) {
	dims.push_back(rows);
    }
    if (cols > 1) {
	dims.push_back(cols);
    }
    size_t dim = std::ranges::fold_left(dims, 1, std::multiplies());
    std::vector<RealScalar> sr(dim), si(dim);
    // IMPORTANT: this only works on little-endian machines. Fortunately, the only
    // big-endian machines you can find as of year 2024 are in a museum somewhere.
    for (unsigned long long spin = 0; spin < dim0; spin++) {
	MultiSpin<N,K> ms;
	for (unsigned int k = 0; k < K; k++) {
	    ms[k] = spin >> (k*N);
	}
	auto phi = fcn(ms, rbm);
	for (int i = 0; i < rows; i++) {
	    for (int j = 0; j < cols; j++) {
		// Again: MATLAB matrices are stored in COLUMN-major order!
		sr[j*rows*dim0 + i * dim0 + spin] = phi(i,j).real();
		si[j*rows*dim0 + i * dim0 + spin] = phi(i,j).imag();
	    }
	}
    }

    auto classty = sizeof(RealScalar) == 4 ? MAT_C_SINGLE : MAT_C_DOUBLE;
    auto dataty = sizeof(RealScalar) == 4 ? MAT_T_SINGLE : MAT_T_DOUBLE;
    struct mat_complex_split_t s = { sr.data(), si.data() };
    matvar_t *mv = Mat_VarCreate("s", classty, dataty, dims.size(), dims.data(),
				 &s, MAT_F_COMPLEX);
    if (!mv) {
	ThrowException(MatioCreateVarError, "s");
    }
    Mat_VarWrite(mf, mv, MAT_COMPRESSION_NONE);
    Mat_VarFree(mv);
    Mat_Close(mf);
}

// Performs one step of the Monte-Carlo sampling, where each spin is flipped according
// to the probability min(1, |phi(s')/phi(s)|^2), starting from the least significant bit.
// Note: the caller must specify the phi, det_phi, tanh_theta that correspond to the given
// multi-spin configuration in s.
template<std::size_t N, std::size_t M, std::size_t K>
void monte_carlo_step(MultiSpin<N,K> &s, const MultiRBM<N,M,K> &rbm, auto &rg,
		      Phi<K> &phi, Phi<K> &phi_inv, ComplexMatrix<M,K*K> &tanh_theta)
{
    ComplexScalar det_phi = phi.determinant();
    std::uniform_real_distribution<RealScalar> unif_real(0,1);
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int i = 0; i < N; i++) {
	    Phi<K> phip(phi);
	    ComplexMatrix<M,K*K> tanh_thetap(tanh_theta);
	    flip_state_fcn(k, i, s, rbm, phip, tanh_thetap);
	    ComplexScalar det_phip = phip.determinant();
	    RealScalar prob = det_phi == 0.0 ? 1.0 : std::norm(det_phip) / std::norm(det_phi);
	    if (unif_real(rg) <= std::min(1.0, prob)) {
		s[k].flip(i);
		det_phi = det_phip;
		phi = std::move(phip);
		phi_inv = phi.completeOrthogonalDecomposition().pseudoInverse();
		tanh_theta.swap(tanh_thetap);
	    }
	}
    }
}

#pragma omp declare							\
    reduction(+:ComplexVector<num_weights_multi(NUM_SPINS,		\
						NUM_HIDDEN_UNITS,	\
						NUM_EIGENSTATES)>	\
	      : omp_out += omp_in)					\
    initializer(omp_priv = omp_orig)

// Performs the whole Markov chain of length m, with the number of burn-in steps specified.
// The derivative matrix is calculated by averaging over all Monte-Carlo samples and the
// weights are updated using stochastic reconfiguration, given the learning rate.
template<std::size_t N, std::size_t M, std::size_t K>
Phi<K> monte_carlo(MultiRBM<N,M,K> &rbm, std::size_t seed, int ncpus, int mc_steps_percpu,
		   int burn_in_steps, RealScalar learning_rate, RealScalar reglam,
		   RealScalar rtol, std::size_t maxit, auto &&hamiltonian)
{
    std::vector<pcg64> rg;
    for (int i = 0; i < ncpus; i++) {
	rg.push_back(seed + i);
    }

    int mc_steps = ncpus * mc_steps_percpu;
    constexpr std::size_t len = num_weights_multi(N,M,K);
    auto o = zeros<len>();
    auto f = zeros<len>();
    auto skk_diag = zeros<len>();
    auto e_mat = zeros<K,K>();
    ComplexScalar eloc{};
#if MATFREE
    // Each std::vector is a matrix with dimension (ncpus, mc_steps_percpu)
    std::vector<MultiSpin<N,K>> spins(mc_steps);
    std::vector<Phi<K>> phis(mc_steps);
    std::vector<Phi<K>> phi_invs(mc_steps);
    std::vector<ComplexMatrix<M,K*K>> tanh_thetas(mc_steps);
#else
    auto oo = zeros<len, len>();
#endif

#pragma omp declare				\
    reduction(+:Phi<K> : omp_out += omp_in)	\
    initializer(omp_priv = omp_orig)

#if MATFREE
#pragma omp parallel for reduction (+:o,f,skk_diag,e_mat,eloc)
#else
#pragma omp declare						\
    reduction(+:ComplexMatrix<len,len> : omp_out += omp_in)	\
    initializer(omp_priv = omp_orig)
#pragma omp parallel for reduction (+:o,f,skk_diag,e_mat,eloc,oo)
#endif
    for (int cpu = 0; cpu < ncpus; cpu++) {
	// Generate an initial random multi-spin configuration and compute the state function.
	auto spin = init_multi_spin<N, K>(rg[cpu]);
	ComplexMatrix<M,K*K> tanh_theta(M,K*K);
	auto phi = state_fcn(spin, rbm, &tanh_theta);
	Phi<K> phi_inv = phi.completeOrthogonalDecomposition().pseudoInverse();
	// Perform some burn-in steps by doing the Monte-Carlo step but discarding the results,
	// so the system is fully thermalized and the final results do not depend on the
	// (random) initial state.
	for (int i = 0; i < burn_in_steps; i++) {
	    monte_carlo_step(spin, rbm, rg[cpu], phi, phi_inv, tanh_theta);
	}
	// Do the actual Monte-Carlo calculation, by generating the Markov chain and summing
	// the derivative matrix and the local energy vector computed from each spin config.
	for (int i = 0; i < mc_steps_percpu; i++) {
	    monte_carlo_step(spin, rbm, rg[cpu], phi, phi_inv, tanh_theta);
	    Phi<K> ham = hamiltonian(spin, rbm, phi, phi_inv, tanh_theta);
	    e_mat += ham;
	    ComplexScalar energy(ham.trace());
	    eloc += energy;
	    auto der = deriv<N,M,K>(spin, phi, phi_inv, tanh_theta);
	    o += der;
	    skk_diag += der.cwiseProduct(der.conjugate());
	    f += energy * der.conjugate();
#if MATFREE
	    int idx = cpu * mc_steps_percpu + i;
	    spins[idx] = spin;
	    phis[idx] = phi;
	    phi_invs[idx] = phi_inv;
	    tanh_thetas[idx] = tanh_theta;
#else
	    for (std::size_t l = 0; l < len; l++) {
		oo.row(l) += std::conj(der[l]) * der;
	    }
#endif
	}
    }

    eloc /= mc_steps;
#pragma omp parallel for simd
    for (std::size_t l = 0; l < len; l++) {
	o[l] /= mc_steps;
	skk_diag[l] /= mc_steps;
	skk_diag[l] -= std::norm(o[l]);
	skk_diag[l] *= reglam;
	f[l] /= mc_steps;
	f[l] -= eloc * std::conj(o[l]);
    }
#ifndef MATFREE
#pragma omp parallel for
    for (std::size_t l = 0; l < len; l++) {
	oo.row(l) /= mc_steps;
	oo.row(l) -= std::conj(o.coeff(l)) * o;
    }
#endif
    auto wdiff = zeros<len>();

#if MATFREE
    minresqlp([&](auto &&v){
	auto res = zeros<len>();
#pragma omp parallel for reduction (+:res)
	for (int cpu = 0; cpu < ncpus; cpu++) {
	    for (int i = 0; i < mc_steps_percpu; i++) {
		int idx = cpu * mc_steps_percpu + i;
		auto der = deriv<N,M,K>(spins[idx], phis[idx],
					phi_invs[idx], tanh_thetas[idx]);
		res += (der.transpose() * v) * der.conjugate();
	    }
	}
	ComplexScalar ov{};
#pragma omp parallel for reduction(+:ov)
	for (std::size_t l = 0; l < len; l++) {
	    ov += o[l] * v[l];
	}
#pragma omp parallel for
	for (std::size_t i = 0; i < len; i++) {
	    res[i] /= mc_steps;
	    res[i] += skk_diag[i] * v[i] - ov * std::conj(o[i]);
	}
	return res;
    }, f, wdiff, rtol, maxit);

#else  // !MATFREE
    minresqlp([&](auto &&v){
	ComplexVector<len> res(len);
#pragma omp parallel for
	for (unsigned int l = 0; l < len; l++) {
	    res[l] = skk_diag[l] * v[l];
	    for (unsigned int ll = 0; ll < len; ll++) {
		res[l] += oo(l,ll) * v[ll];
	    }
	}
	return res;
    }, f, wdiff, rtol, maxit);
#endif	// MATFREE

    std::size_t L = num_weights(N, M);
    constexpr unsigned int A = M/N;
#pragma omp parallel for
    for (unsigned int i = 0; i < N; i++) {
	for (unsigned int k = 0; k < K; k++) { // k-th RBM
	    rbm[k].a[i] -= learning_rate * wdiff[k*L + i];
	}
	for (unsigned int a = 0; a < A; a++) {
	    unsigned int j = i * A + a;		   // j-th hidden unit
	    for (unsigned int k = 0; k < K; k++) { // k-th RBM
		rbm[k].b[j] -= learning_rate * wdiff[k*L + N + j];
	    }
	}
	for (unsigned int j = 0; j < M; j++) {
	    for (unsigned int k = 0; k < K; k++) {
		rbm[k].w(i,j) -= learning_rate * wdiff[k*L + N + M + i*M + j];
	    }
	}
    }
    return e_mat / mc_steps;
}

// Defines the concept that constraints the callable passed as the correlation function.
template<typename Callable, std::size_t N, std::size_t M, std::size_t K>
concept CorrFcn = std::invocable<Callable, std::size_t, MultiSpin<N,K>,
				 MultiRBM<N,M,K>, Phi<K>> &&
    std::same_as<std::invoke_result_t<Callable, std::size_t, MultiSpin<N,K>,
				      MultiRBM<N,M,K>, Phi<K>>, Phi<K>>;

// Computes the correlation function by sampling the configuration space with
// the given number of Monte-Carlo samples.
template<std::size_t N, std::size_t M, std::size_t K, std::size_t ResMatSize>
RealMatrix<ResMatSize,K> correlation_fcn(const MultiRBM<N,M,K> &rbm, std::size_t seed,
					 int ncpus, int nsamples_percpu, int burn_in_steps,
					 CorrFcn<N,M,K> auto &&fcn, const Phi<K> &evecs)
{
    std::vector<pcg64> rg;
    for (int i = 0; i < ncpus; i++) {
	rg.push_back(seed + i);
    }

#pragma omp declare							\
    reduction(+:ComplexMatrix<ResMatSize, K*K> : omp_out += omp_in)	\
    initializer(omp_priv = omp_orig)

    auto cor = zeros<ResMatSize, K*K>();
#pragma omp parallel for reduction (+:cor)
    for (int cpu = 0; cpu < ncpus; cpu++) {
	auto spin = init_multi_spin<N, K>(rg[cpu]);
	ComplexMatrix<M,K*K> tanh_theta(M,K*K);
	auto phi = state_fcn(spin, rbm, &tanh_theta);
	Phi<K> phi_inv = phi.completeOrthogonalDecomposition().pseudoInverse();
	for (int i = 0; i < burn_in_steps; i++) {
	    monte_carlo_step(spin, rbm, rg[cpu], phi, phi_inv, tanh_theta);
	}
	for (int i = 0; i < nsamples_percpu; i++) {
	    monte_carlo_step(spin, rbm, rg[cpu], phi, phi_inv, tanh_theta);
	    for (unsigned int i = 0; i < ResMatSize; i++) {
		Phi<K> mat = phi_inv * fcn(i, spin, rbm, phi);
		cor.row(i) += mat.template reshaped<Eigen::RowMajor>().transpose();
	    }
	}
    }
    int nsamples = ncpus * nsamples_percpu;
    RealMatrix<ResMatSize,K> res(ResMatSize,K);
#pragma omp parallel for
    for (unsigned int i = 0; i < ResMatSize; i++) {
	Phi<K> mat = cor.row(i).template reshaped<Eigen::RowMajor>(K,K);
	mat /= nsamples;
	res.row(i) = (evecs.inverse() * mat * evecs).diagonal().real();
    }
    return res;
}

// Computes the result of sigma^x_i acting on the given Multi-RBM state.
template<std::size_t N, std::size_t M, std::size_t K>
Phi<K> corrfcn_x(std::size_t i, MultiSpin<N,K> spin,
		 const MultiRBM<N,M,K> &rbm, const Phi<K> &phi)
{
    for (unsigned int k = 0; k < K; k++) {
	spin[k].flip(i);
    }
    return state_fcn(spin, rbm);
}

// Computes the result of sigma^x_i sigma^x_j acting on the given Multi-RBM state.
template<std::size_t N, std::size_t M, std::size_t K>
Phi<K> corrfcn_xx(std::size_t idx, MultiSpin<N,K> spin,
		  const MultiRBM<N,M,K> &rbm, const Phi<K> &phi)
{
    // idx = i * N + j
    std::size_t i = idx / N;
    std::size_t j = idx % N;
    for (unsigned int k = 0; k < K; k++) {
	spin[k].flip(i);
	spin[k].flip(j);
    }
    return state_fcn(spin, rbm);
}

// Computes the result of sigma^z_i acting on the given Multi-RBM state.
template<std::size_t N, std::size_t M, std::size_t K>
Phi<K> corrfcn_z(std::size_t i, MultiSpin<N,K> spin,
		 const MultiRBM<N,M,K> &rbm, Phi<K> phi)
{
    for (unsigned int k = 0; k < K; k++) {
	if (spin[k][i]) {
	    phi.row(k) = -phi.row(k);
	}
    }
    return phi;
}

// Computes the result of sigma^z_i sigma^z_j acting on the given Multi-RBM state.
template<std::size_t N, std::size_t M, std::size_t K>
Phi<K> corrfcn_zz(std::size_t idx, MultiSpin<N,K> spin,
		  const MultiRBM<N,M,K> &rbm, Phi<K> phi)
{
    // idx = i * N + j
    std::size_t i = idx / N;
    std::size_t j = idx % N;
    for (unsigned int k = 0; k < K; k++) {
	if (spin[k][i] ^ spin[k][j]) {
	    phi.row(k) = -phi.row(k);
	}
    }
    return phi;
}

template<std::size_t N, std::size_t K>
void dump_corrs(const RealMatrix<N, K> &corx, const RealMatrix<N, K> &corz,
		const RealMatrix<N*N, K> &corxx, const RealMatrix<N*N, K> &corzz,
		const std::string &fname)
{
    mat_t *mf = Mat_CreateVer(fname.c_str(), NULL, MAT_FT_MAT5);
    if (!mf) {
	ThrowException(MatioCreateFileError, fname.c_str());
    }

    std::vector<RealScalar> x(K*N), z(K*N), xx(K*N*N), zz(K*N*N);
    // IMPORTANT: MATLAB matrices are stored in COLUMN-major order!
#pragma omp parallel for collapse(2)
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int i = 0; i < N; i++) {
	    x[k*N + i] = corx(i,k);
	    z[k*N + i] = corz(i,k);
	}
    }
#pragma omp parallel for collapse(3)
    for (unsigned int k = 0; k < K; k++) {
	for (unsigned int j = 0; j < N; j++) {
	    for (unsigned int i = 0; i < N; i++) {
		xx[k*N*N + j*N + i] = corxx(i*N+j,k);
		zz[k*N*N + j*N + i] = corzz(i*N+j,k);
	    }
	}
    }

    auto classty = sizeof(RealScalar) == 4 ? MAT_C_SINGLE : MAT_C_DOUBLE;
    auto dataty = sizeof(RealScalar) == 4 ? MAT_T_SINGLE : MAT_T_DOUBLE;
    size_t dims1[] = { N, K };
    matvar_t *mv = Mat_VarCreate("x", classty, dataty, 2, dims1, x.data(), 0);
    if (!mv) {
	ThrowException(MatioCreateVarError, "x");
    }
    Mat_VarWrite(mf, mv, MAT_COMPRESSION_NONE);
    Mat_VarFree(mv);

    mv = Mat_VarCreate("z", classty, dataty, 2, dims1, z.data(), 0);
    if (!mv) {
	ThrowException(MatioCreateVarError, "z");
    }
    Mat_VarWrite(mf, mv, MAT_COMPRESSION_NONE);
    Mat_VarFree(mv);

    size_t dims[] = { N, N, K };
    mv = Mat_VarCreate("xx", classty, dataty, 3, dims, xx.data(), 0);
    if (!mv) {
	ThrowException(MatioCreateVarError, "xx");
    }
    Mat_VarWrite(mf, mv, MAT_COMPRESSION_NONE);
    Mat_VarFree(mv);

    mv = Mat_VarCreate("zz", classty, dataty, 3, dims, zz.data(), 0);
    if (!mv) {
	ThrowException(MatioCreateVarError, "zz");
    }
    Mat_VarWrite(mf, mv, MAT_COMPRESSION_NONE);
    Mat_VarFree(mv);
    Mat_Close(mf);
}

enum ModelEnum {
    Ising,
    PeriodicIsing,
    HS,
    XXZ,
    LongRange,
    TrappedIon
};

struct CmdArgs : public argparse::Args {
    ModelEnum &model = kwarg("model", "Model to simulate");
    RealScalar &h = kwarg("h", "Specify the external field h.");
    int &niters = kwarg("n,niters", "Specify the number of iterations to perform.");
    int &mc_steps_percpu = kwarg("m,mc-steps-percpu", "Specify the number of Monte Carlo "
				 "samples per CPU core.");
    int &ncpus = kwarg("c,ncpus", "Specify the number of CPU cores to use.");
    int &burn_in_steps = kwarg("burn-in-steps",
			       "Specify the number of burn-in steps.").set_default(64);
    RealScalar &tol = kwarg("tol", "Specify the stopping tolerance for the per-partile "
			    "energy.").set_default(1e-6);
    RealScalar &lrate0 = kwarg("lrate-init", "Specify the initial learning rate. Learning "
			       "rate decreases as training goes on.").set_default(0.1);
    RealScalar &lratef = kwarg("lrate-decay", "Specify the decay factor of the learning rate, "
			       "ie. lrate(t) = lrate(0) * f^t.").set_default(0.99);
    RealScalar &lrate_cutoff = kwarg("lrate-cutoff", "Specify the minimal value of the learning "
			       "rate.").set_default(0.01);
    RealScalar &reglam0 = kwarg("reglam-init", "Specify the initial value for the lambda "
				    "used in regulating S_kk.").set_default(0.01);
    RealScalar &reglam_decay = kwarg("reglam-decay", "Specify the decay factor for the lambda "
				     "used in regulating S_kk.").set_default(0.9);
    RealScalar &reglam_cutoff = kwarg("reglam-cutoff", "Specify the minimal value for the lambda "
				      "used in regulating S_kk.").set_default(0.001);
    RealScalar &minres_rtol = kwarg("minres-rtol", "Specify the tolerance of the "
				    "MINRES-QLP algorithm.").set_default(1e-10);
    std::size_t &minres_maxit = kwarg("minres-maxit", "Specify the max number of iterations "
				      "for the MINRES-QLP algorithm.").set_default(~0ULL);
    std::optional<std::size_t> &seed = kwarg("seed", "Specify the seed of the RNG.");
    std::string &load_rbm = kwarg("load-rbm", "If specified, the initial RBM weights will "
				  "be loaded from the given .mat file.").set_default("");
    std::string &load_single_rbm = kwarg("load-single-rbm", "If specified, the initial weights "
					 "will be loaded from the given single RBM with spins "
					 "flipped except for the first RBM.").set_default("");
    std::string &dump_rbm = kwarg("dump-rbm", "If specified, will dump the weights of the "
				  "RBMs to the file with the given prefix.").set_default("");
    bool &throw_fperror = flag("throw-fperror", "Throw an exception when a floating point "
			       "error has occurred.");
    bool &apply_gauge = flag("apply-gauge", "When dumping the RBM weights, apply the local "
			     "gauge transformation a_i -> a_i - I PI / 2 for odd i");
    bool &inverse_gauge = flag("inverse-gauge", "When loading RBM weights, apply the inverse "
			     "gauge transformation a_i -> a_i + I PI / 2 for odd i");
    std::string &correlation_data = kwarg("correlation-data", "Specify the file name to save the "
					  "correlation function data.");
#if NUM_SPINS * NUM_EIGENSTATES < 32
    std::string &dump_states = kwarg("dump-states", "If specified, will dump the "
				     "states of the RBMs to the file with the given "
				     "prefix.").set_default("");
    std::string &dump_ham = kwarg("dump-ham", "If specified, will dump the Hamiltonian "
				  "for each spin config to the file with the given "
				  "prefix.").set_default("");
    std::string &dump_deriv = kwarg("dump-deriv", "If specified, will dump the derivatives "
				    "for each spin config to the file with the given "
				    "prefix.").set_default("");
#endif

    void welcome() {
        std::cout << "This program computes the lowest " << NUM_EIGENSTATES
		  << " energy eigenstates of the transverse field Ising model with "
		  << NUM_SPINS << " spins using RBMs with "
		  << NUM_HIDDEN_UNITS << " hidden units." << std::endl;
    }
};

// Computes the sum \sum_{ij} J_{ij} (-1)^(s_i+s_j) for the Ising model with
// Dirichlet boundary condition.
template<std::size_t N>
RealScalar ising_model(const Spin<N> &s)
{
    constexpr RealScalar J = 1;
    Spin<N> sp(s << 1);
    sp ^= s;
    sp.reset(0);
    return -J * static_cast<int>(N - 1 - 2*sp.count());
}

// Computes the sum \sum_{ij} J_{ij} (-1)^(s_i+s_j) for the Ising model with
// periodic boundary condition.
template<std::size_t N>
RealScalar periodic_ising_model(const Spin<N> &s)
{
    constexpr RealScalar J = 1;
    Spin<N> sp(s << 1);
    sp[0] = s[N-1];
    sp ^= s;
    return -J * static_cast<int>(N - 2*sp.count());
}

// Computes the sum \sum_{ij} J_{ij} (-1)^(s_i+s_j) for the long-range correlation model.
template<std::size_t N>
RealScalar long_range_model(const Spin<N> &s)
{
    constexpr auto powdiff{[](){
	std::array<RealScalar,N-1> res;
	for (unsigned int i = 0; i < N-1; i++) {
	    res[i] = std::pow(i+1, -0.2);
	}
	return res;
    }()};
    RealScalar res{};
    for (unsigned int i = 1; i < N; i++) {
	for (unsigned int j = 0; j < i; j++) {
	    RealScalar p = powdiff[i - j - 1];
	    if (s[i] ^ s[j]) {
		p = -p;
	    }
	    res += p;
	}
    }
    return res * 2.0;
}

// Computes the sum \sum_{ij} J_{ij} (-1)^(s_i+s_j) for the trapped-ion model.
template<std::size_t N>
RealScalar trapped_ion_model(const Spin<N> &s)
{
#include JMAT_INC
    static_assert(sizeof(Jmat) == N * N * sizeof(RealScalar));
    static_assert(sizeof(Jmat[0]) == N * sizeof(RealScalar));
    RealScalar res{};
    for (unsigned int i = 1; i < N; i++) {
	for (unsigned int j = 0; j < i; j++) {
	    if (i != j) {
		RealScalar p = Jmat[i][j];
		if (s[i] ^ s[j]) {
		    p = -p;
		}
		res += p;
	    }
	}
    }
    return res * 2.0;
}

// Computes the value of the Hamiltonian acting on the Multi-RBM state, ie. the matrix
//                         <s^0|H|phi^0>      <s^0|H|phi^1>      ...  <s^0|H|phi^(K-1)>
//    H Phi = Phi^(-1)  *  <s^1|H|phi^0>      <s^1|H|phi^1>      ...  <s^1|H|phi^(K-1)>
//                         ...
//                         <s^(K-1)|H|phi^0>  <s^(K-1)|H|phi^1>  ...  <s^(K-1)|H|phi^(K-1)>
// Here Phi^(-1) is the Penrose-pseudo-inverse of the grand wave function (a K-by-K matrix).
// The Hamiltonian is defined as
//    H = \sum_{ij} J_{ij} sigma^z_i sigma^z_j - h \sum_i sigma^x_i
// Note: you must supply the Phi that corresponds to the given multi-spin configuration.
template<std::size_t N, std::size_t M, std::size_t K>
Phi<K> ising_like_hamiltonian(auto &&J, const CmdArgs &args, const MultiSpin<N,K> &s,
			      const MultiRBM<N,M,K> &rbm, const Phi<K> &phi,
			      const Phi<K> &phi_inv, const ComplexMatrix<M,K*K> &tanh_theta)
{
    Phi<K> res(phi);
    for (unsigned k = 0; k < K; k++) {
	res.row(k) *= J(s[k]);
    }
    auto sx_sum = zeros<K,K>();
    for (unsigned int i = 0; i < N; i++) {
	Phi<K> phip(phi);
	ComplexMatrix<M, K*K> tanh_thetap(tanh_theta);
	for (unsigned int k = 0; k < K; k++) {
	    flip_state_fcn(k, i, s, rbm, phip, tanh_thetap);
	}
	sx_sum += phip;
    }
    res -= args.h * sx_sum;
    return phi_inv * res;
}

// Computes the J_ij for the Haldane-Shashtry model, ie. J_ij = 1/d_ij^2 with
// d_ij = (N/\pi) sin[pi (i-j)/N].
template<std::size_t N>
constexpr auto hs_model(std::size_t i, std::size_t j)
{
    constexpr auto dist{[](){
	std::array<RealScalar,N-1> res;
	for (unsigned int i = 0; i < N-1; i++) {
	    RealScalar d = std::sin(M_PI * (i+1) / N) * N / M_PI;
	    res[i] = 1.0 / (d*d);
	}
	return res;
    }()};
    return dist[i - j - 1];
}

// Computes the J_ij for the Heisenberg XXX and XXZ model, ie, J_{ij} = \delta_{i,i+1}
// with periodic boundary condition.
template<std::size_t N>
constexpr auto xxz_model(std::size_t i, std::size_t j)
{
    if (i - j == 1 || (i == (N-1) && j == 0)) {
	return -1;
    }
    return 0;
}

// Computes the Hamiltonian for a Heisenberg-like model, ie.
// H = \sum_{i<j} J_{ij} (sx_i sx_j + sy_i sy_j + delta * sz_i sz_j) given J_ij
template<std::size_t N, std::size_t M, std::size_t K>
Phi<K> hs_like_hamiltonian(auto &&J, RealScalar delta, const CmdArgs &args,
			   const MultiSpin<N,K> &s, const MultiRBM<N,M,K> &rbm,
			   const Phi<K> &phi, const Phi<K> &phi_inv,
			   const ComplexMatrix<M,K*K> &tanh_theta)
{
    auto res = zeros<K,K>();
    for (unsigned int i = 1; i < N; i++) {
	for (unsigned int j = 0; j < i; j++) {
	    RealScalar p = J(i,j);
	    MultiSpin<N, K> sp(s);
	    Phi<K> phip(phi);
	    ComplexMatrix<M, K*K> tanh_thetap(tanh_theta);
	    for (unsigned k = 0; k < K; k++) {
		if (s[k][i] ^ s[k][j]) {
		    flip_state_fcn(k, i, sp, rbm, phip, tanh_thetap);
		    sp[k].flip(i);
		    flip_state_fcn(k, j, sp, rbm, phip, tanh_thetap);
		    res.row(k) += p * (2 * phip.row(k) - delta * phi.row(k));
		} else {
		    res.row(k) += p * delta * phi.row(k);
		}
	    }
	}
    }
    return phi_inv * res;
}

template<std::size_t N, std::size_t M, std::size_t K, typename Args>
void run(Args &&args, auto &&hamiltonian)
{
    std::random_device rd;
    std::size_t seed{args.seed.value_or(rd())};
    std::cout << "Random seed is " << seed << std::endl;
    pcg64 rg(seed-1);
    MultiRBM<N,M,K> rbm;
    if (!args.load_rbm.empty()) {
	rbm = load_rbm<N,M,K>(args.load_rbm.c_str());
	if (args.apply_gauge) {
	    for (unsigned int k = 0; k < K; k++) {
		for (unsigned int i = 0; i < N; i++) {
		    if (i & 1) {
			if (args.inverse_gauge) {
			    rbm[k].a[i] += ComplexScalar{0, M_PI_2};
			} else {
			    rbm[k].a[i] -= ComplexScalar{0, M_PI_2};
			}
		    }
		}
	    }
	}
    } else if (!args.load_single_rbm.empty()) {
	rbm = load_single_rbm<N,M,K>(args.load_single_rbm.c_str());
    } else {
	rbm = init_multi_rbm<N,M,K>(rg);
    }
#if NUM_SPINS * NUM_EIGENSTATES < 32
    if (!args.dump_states.empty()) {
	dump_states(rbm, args.dump_states.c_str(),
		    [](auto &&ms, auto &&rbm){ return state_fcn(ms, rbm); });
    }
    if (!args.dump_ham.empty()) {
	dump_states(rbm, args.dump_ham.c_str(),
		    [&](auto &&ms, auto &&rbm){
			ComplexMatrix<M, K*K> tanh_theta(M, K*K);
			auto phi = state_fcn(ms, rbm, &tanh_theta);
			Phi<K> phi_inv = phi.completeOrthogonalDecomposition().pseudoInverse();
			return hamiltonian(ms, rbm, phi, phi_inv, tanh_theta);
		    });
    }
    if (!args.dump_deriv.empty()) {
	dump_states(rbm, args.dump_deriv.c_str(),
		    [&](auto &&ms, auto &&rbm){
			ComplexMatrix<M, K*K> tanh_theta(M, K*K);
			auto phi = state_fcn(ms, rbm, &tanh_theta);
			Phi<K> phi_inv = phi.completeOrthogonalDecomposition().pseudoInverse();
			return deriv<N,M,K>(ms, phi, phi_inv, tanh_theta);
		    });
    }
#endif

    RealScalar reglam = args.reglam0;
    RealScalar lrate = args.lrate0;
    RealVector<K> res(K);
    Phi<K> e_mat(K,K);
    for (int iter = 0; iter < args.niters; iter++) {
	e_mat = monte_carlo(rbm, seed + iter * args.ncpus, args.ncpus, args.mc_steps_percpu,
			    args.burn_in_steps, std::max(lrate, args.lrate_cutoff),
			    std::max(reglam, args.reglam_cutoff),
			    args.minres_rtol, args.minres_maxit, hamiltonian);
	e_mat /= N;
	reglam *= args.reglam_decay;
	lrate *= args.lratef;
	RealVector<K> evals(e_mat.eigenvalues().real());
	std::sort(evals.begin(), evals.end());
	std::cout << evals.transpose() << std::endl;
	if (iter && (evals - res).norm() < args.tol) {
	    std::cout << "Converged after " << iter+1 << " steps." << std::endl;
	    break;
	}
	if (!iter || evals.sum() < res.sum()) {
	    res = evals;
	}
    }

    Eigen::ComplexEigenSolver<Phi<K>> solver(e_mat);
    Phi<K> evecs = solver.eigenvectors();
    auto evals = solver.eigenvalues().real();
    std::array<int, K> indices;
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
	return evals[i] < evals[j];
    });
    auto corx = correlation_fcn<N,M,K,N>(rbm, seed + args.niters * args.ncpus, args.ncpus,
					 args.mc_steps_percpu, args.burn_in_steps,
					 corrfcn_x<N,M,K>, evecs);
    auto corz = correlation_fcn<N,M,K,N>(rbm, seed + args.niters * args.ncpus, args.ncpus,
					 args.mc_steps_percpu, args.burn_in_steps,
					 corrfcn_z<N,M,K>, evecs);
    auto corxx = correlation_fcn<N,M,K,N*N>(rbm, seed + args.niters * args.ncpus, args.ncpus,
					    args.mc_steps_percpu, args.burn_in_steps,
					    corrfcn_xx<N,M,K>, evecs);
    auto corzz = correlation_fcn<N,M,K,N*N>(rbm, seed + args.niters * args.ncpus, args.ncpus,
					    args.mc_steps_percpu, args.burn_in_steps,
					    corrfcn_zz<N,M,K>, evecs);
    RealMatrix<N,K> sorted_corx(N,K), sorted_corz(N,K);
    for (unsigned int i = 0; i < N; i++) {
	for (unsigned k = 0; k < K; k++) {
	    sorted_corx(i,k) = corx(i, indices[k]);
	    sorted_corz(i,k) = corz(i, indices[k]);
	}
    }
    RealMatrix<N*N,K> sorted_corxx(N*N,K), sorted_corzz(N*N,K);
    for (unsigned int i = 0; i < N*N; i++) {
	for (unsigned k = 0; k < K; k++) {
	    sorted_corxx(i,k) = corxx(i, indices[k]);
	    sorted_corzz(i,k) = corzz(i, indices[k]);
	}
    }
    dump_corrs<N,K>(sorted_corx, sorted_corz, sorted_corxx, sorted_corzz, args.correlation_data);

    if (!args.dump_rbm.empty()) {
	if (args.apply_gauge) {
	    for (unsigned int k = 0; k < K; k++) {
		for (unsigned int i = 0; i < N; i++) {
		    if (i & 1) {
			rbm[k].a[i] -= ComplexScalar{0, M_PI_2};
		    }
		}
	    }
	}
	dump_rbm(rbm, args.dump_rbm.c_str());
    }
}

#define CALLSTACK_SIZE 32
void print_stack(void) {
    void *buf[CALLSTACK_SIZE + 1];
    int nptrs = backtrace(buf, CALLSTACK_SIZE);
    char **strings = backtrace_symbols(buf, nptrs);

    if (!strings) {
        return;
    }

    std::cerr << "Backtrace:" << std::endl;
    for (int i = 0; i < nptrs; i++) {
	std::cerr << strings[i] << std::endl;
    }

    free(strings);
}

void handle_fpexcept(int signal) {
    assert(signal == SIGFPE);
    std::cerr << "Floating point error occured. Exiting." << std::endl;
    print_stack();
    exit(127);
}

int main(int argc, char *argv[])
{
    constexpr unsigned int N = NUM_SPINS; // Number of spins of the model
    constexpr unsigned int M = NUM_HIDDEN_UNITS; // Number of hidden units of the RBM
    constexpr unsigned int K = NUM_EIGENSTATES;  // Number of eigenstates to solve for

    auto args = argparse::parse<CmdArgs>(argc, argv);
    if (!args.ncpus) {
	std::cerr << "Number of CPU cores cannot be zero." << std::endl;
	return 1;
    }
#ifdef _OPENMP
    omp_set_num_threads(args.ncpus);
#endif

    // Enable floating-point exceptions if specified
    if (args.throw_fperror) {
	std::signal(SIGFPE, handle_fpexcept);
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    }

    switch (args.model) {
    case Ising:
	run<N,M,K>(args, [&](auto...ts) {
	    return ising_like_hamiltonian(ising_model<N>, args, ts...);
	});
	break;
    case PeriodicIsing:
	run<N,M,K>(args, [&](auto...ts) {
	    return ising_like_hamiltonian(periodic_ising_model<N>, args, ts...);
	});
	break;
    case LongRange:
	run<N,M,K>(args, [&](auto...ts) {
	    return ising_like_hamiltonian(long_range_model<N>, args, ts...);
	});
	break;
    case TrappedIon:
	run<N,M,K>(args, [&](auto...ts) {
	    return ising_like_hamiltonian(trapped_ion_model<N>, args, ts...);
	});
	break;
    case HS:
	run<N,M,K>(args, [&](auto...ts) {
	    return hs_like_hamiltonian(hs_model<N>, 1, args, ts...);
	});
	break;
    case XXZ:
	run<N,M,K>(args, [&](auto...ts) {
	    return hs_like_hamiltonian(xxz_model<N>, -1, args, ts...);
	});
	break;
    }
    return 0;
}
