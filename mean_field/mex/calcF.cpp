#include "mex.h"
#include <math.h>

inline size_t idx(size_t n, size_t i, size_t j) {
	return i*n - (i+1)*(i+2)/2 + j;
}

double calcF(double *mu, uint32_t *feats_aa, size_t seqlen,
			double *theta_tri, double *gamma, double theta_dist, double theta_seqlen, 
			double theta_prior) {

	double F = 0;
	double mu_ij, mu_ik, mu_jk;
	double prob_0, prob_1, prob_2, prob_3;

	size_t n = seqlen;
	double seq_feat = seqlen*theta_seqlen;

	// i, j are less equals, but k is less than to deal with sum over two versus sum over three elements
	for (size_t i = 0; i <= n-2; i++) {
		for (size_t j = i+1; j <= n-1; j++) {
			mu_ij = mu[idx(n, i, j)];
			F += mu_ij*(gamma[feats_aa[idx(n, i, j)]] + theta_dist*(j - i) + seq_feat + theta_prior);
			F -= (mu_ij*log(mu_ij) + (1 - mu_ij)*log(1 - mu_ij));
			for (size_t k = j+1; k < n; k++) {
				mu_ik = mu[idx(n, i, k)];
				mu_jk = mu[idx(n, j, k)];
				prob_0 = (1 - mu_ij)*(1 - mu_ik)*(1 - mu_jk);
				prob_1 = mu_ij*(1 - mu_ik)*(1 - mu_jk) + (1 - mu_ij)*mu_ik*(1 - mu_jk) + (1 - mu_ij)*(1 - mu_ik)*mu_jk;
				prob_2 = mu_ij*mu_ik*(1 - mu_jk) + mu_ij*(1 - mu_ik)*mu_jk + (1 - mu_ij)*mu_ik*mu_jk;
				prob_3 = mu_ij*mu_ik*mu_jk;
				F += theta_tri[0]*prob_0 + theta_tri[1]*prob_1 + theta_tri[2]*prob_2 + theta_tri[3]*prob_3;
			}
		}
	}

	return F;

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	if (nrhs != 8)
		mexErrMsgTxt("calcF: Requires eight arguments.");
	if ( (!mxIsClass(prhs[1], "uint32")) || (!mxIsClass(prhs[2], "uint32")) )
		mexErrMsgTxt("calcF: Arguments 2 and 3 must be UINT32.");

	double *mu = mxGetPr(prhs[0]);
	uint32_t *feats_aa = (uint32_t *) mxGetData(prhs[1]);
	uint32_t seqlen = *((uint32_t *) mxGetData(prhs[2]));
	double *theta_tri = mxGetPr(prhs[3]);
	double *gamma = mxGetPr(prhs[4]);
	double theta_dist = *(mxGetPr(prhs[5]));
	double theta_seqlen = *(mxGetPr(prhs[6]));
	double theta_prior = *(mxGetPr(prhs[7]));

	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *F = mxGetPr(plhs[0]);
	(*F) = calcF(mu, feats_aa, seqlen, theta_tri, gamma, 
					theta_dist, theta_seqlen, theta_prior);
}
