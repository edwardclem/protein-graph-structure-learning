/*
 * Perform coordinate ascent to calculate mu_hat = argmax_mus F(theta, mus), for a given protein.
 * Used in mean-field CRF approximation.
 */
#include "mex.h"
#include <cmath>
#include <string.h>
#include <cstdint>

#define NUM_ITER 20
#define CONV_TOL 0.0001

// Get index of mu_ij given seqence length. Have to offset for upper triangular matrix
inline size_t get_idx(size_t seqlen, size_t i, size_t j) {
	size_t first = fmin(i,j);
	size_t second = fmax(i,j);
	return first*seqlen - (first+1)*(first+2)/2 + second;
}

/*
 * Uses coordinate ascent to calculate optimal value for mus given theta and features
 * for a particular protein. Mus is initialized in mexFunction to be a vector of 0.5. 
 * feats_aa are amino acid features for that protein, seqlen is the number of amino acids.
 * Full theta vector is given by theta = [theta_tri, gamma, theta_dist, theta_seqlen, theta_prior].
 * Vector is broken up in MATLAB code for ease.
 */
void calc_muhat(double *mus, uint32_t *feats_aa, size_t seqlen,
				double *theta_tri, double *gamma, double theta_dist, double theta_seqlen,
				double theta_prior) {

	size_t mu_idx;
	double alpha;
	double mu_ik, mu_jk;
	double prob_0, prob_1, prob_2, prob_3;
	double diff = 0;
	// coordinate ascent for NUM_ITER iterations
	for (size_t iter = 0; iter < NUM_ITER; iter++) {
		for (size_t i = 0; i <= seqlen - 2; i++) {
			for (size_t j = i+1; j <= seqlen - 1; j++) {

				mu_idx = get_idx(seqlen, i, j);
				alpha = 0;

				// Calculation for sequence features. Depends on ij features.

				// for edge ij, the amino acid indicator will be nonzero at only one location, use that to index gammas
				// Also distance feature (note: j - i = abs(i - j) b/c of the way the loop works) and prior
				alpha += gamma[feats_aa[mu_idx]] + (j - i)*theta_dist + theta_prior;

				// Calculation for triplet factors. Depends on mu_ik, mu_jk.
				for (size_t k = 0; k < seqlen; k++) {
					if ((k == i) || (k == j))
						continue;

					mu_ik = mus[get_idx(seqlen, i, k)];
					mu_jk = mus[get_idx(seqlen, j, k)];

					prob_0 = -(1 - mu_ik)*(1 - mu_jk);
					prob_1 = (1 - mu_ik)*(1 - mu_jk) - mu_ik*(1 - mu_jk) - (1 - mu_ik)*mu_jk;
					prob_2 = mu_ik*(1 - mu_jk) + (1 - mu_ik)*mu_jk - mu_ik*mu_jk;
					prob_3 = mu_ik*mu_jk;

					alpha += theta_tri[0]*prob_0 + theta_tri[1]*prob_1 + theta_tri[2]*prob_2 + theta_tri[3]*prob_3;
				}
				alpha = exp(alpha);
				diff += std::abs(mus[mu_idx] - (alpha / (1 + alpha)));
				mus[mu_idx] = alpha / (1 + alpha);
			}
		}
		if (diff < CONV_TOL)
			break;
		diff = 0;
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 8)
		mexErrMsgTxt("calc_muhat: Requires eight arguments.");
	if ( (!mxIsClass(prhs[0], "uint32")) || (!mxIsClass(prhs[1], "uint32")) || (!mxIsClass(prhs[2], "uint32")) )
		mexErrMsgTxt("calc_muhat: Arguments 1 - 3 must be UINT32.");
	uint32_t n_mus = *((uint32_t *) mxGetData(prhs[0]));
	uint32_t *feats_aa = (uint32_t *) mxGetData(prhs[1]);
	uint32_t seqlen = *((uint32_t *) mxGetData(prhs[2]));
	double *theta_tri = mxGetPr(prhs[3]);
	double *gamma = mxGetPr(prhs[4]);
	double theta_dist = *(mxGetPr(prhs[5]));
	double theta_seqlen = *(mxGetPr(prhs[6]));
	double theta_prior = *(mxGetPr(prhs[7]));

	// Initialize mus array and set everything to 0.5.
	double *mus = (double *) mxCalloc(n_mus, sizeof(double));
	for (size_t i = 0; i < n_mus; i++) {
		mus[i] = 0.5;
	}

	calc_muhat(mus, feats_aa, seqlen, theta_tri, gamma, 
				theta_dist, theta_seqlen, theta_prior);

	plhs[0] = mxCreateDoubleMatrix(n_mus, 1, mxREAL);
	mxSetData(plhs[0], mus);


	//passing in array of thetas:
	//20^2 amino acid pair thetas
	//1 distance theta
	//1 seqlen theta
	//1 prior theta (# edges)
	//4 edge potential thetas
	//407 by 1 

	//thetas:
	//4 triplet thetas
	//

	// double *thetas;
	// int nthetas;
	// thetas = mexGetPr(prhs[1]);
	// nthetas = mexGetN(prhs[1]);

	// if (nThetas != NUM_THETAS){
	// 	mexErrMsgTxt("Theta matrix is of incorrect size.");
	// }

	// //N*(N - 1)/2 x 1 array of mus
	// //N equals number of amino acids in this training example
	// double *mus;
	// int nedges;
	// mus = mxGetPr(prhs[2]);
	// nedges = mxGetN(prhs[2]);

	// //403 x N*(N - 1)/2 array of features
	// double *feats;
	// int *dims;

	// feats = mxGetPr(prhs[3]);
	// dims = mxGetDimensions(prhs[3]);

	// int nfeats = dims[0];

	// if (nfeats != NUM_FEATS){
	// 	mexErrMsgTxt("Feature vector must be 403 x N*(N - 1)/2, where N is the number of possible edges in the protein")
	// }

	// //4 x 1 array of triplet sufficient stats
	// double *triplet_ss;
	// int nss;
	// triplet_ss = mexGetPr(prhs[4]);
	// nss = mexGtN(prhs[4]);

	// if (nss != 4){
	// 	mexErrMsgTxt("There must be 4 triplet sufficient statistics.")
	// }

	// //creating output array for mus
	// plhs[0] = mxCreateDoubleMatrix(1, nedges, mxREAL);
	// mu_hats = mxGetPr(plhs[0]);

	// //TODO: GET SEQLEN SOMEHOW
	// calc_muhat(nthetas, thetas, mus, nedges, seqlen, feats,triplet_ss, mu_hats);

}
