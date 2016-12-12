/*
 * Perform coordinate ascent to calculate mu_hat = argmax_mus F(theta, mus), for a given protein.
 * Used in mean-field CRF approximation.
 */
#include "mex.h"
#include <cmath>
#include <string.h>
#include <cstdint>
#include "matrix.h"

#define NUM_ITER 20
#define CONV_TOL 0.0001

// Get index of mu_ij given seqence length. Have to offset for upper triangular matrix
inline int get_idx(int seqlen, int i, int j) {
	int first = fmin(i,j);
	int second = fmax(i,j);
	return first*seqlen - (first+1)*(first+2)/2 + second;
}

/*
 * Uses coordinate ascent to calculate optimal value for mus given theta and features
 * for a particular protein. Mus is initialized in mexFunction to be a vector of 0.5. 
 * feats_aa are amino acid features for that protein, seqlen is the number of amino acids.
 * Full theta vector is given by theta = [theta_tri, gamma, theta_dist, theta_seqlen, theta_prior].
 * Vector is broken up in MATLAB code for ease.
 */
void calc_muhat(
	double *mus, 
	const int *feats_aa, 
	const int seqlen,
	const double *theta_tri, 
	const double *gamma, 
	const double theta_dist, 
	const double theta_seqlen,
	const double theta_prior) {

	int mu_idx;
	double alpha;
	double mu_ik, mu_jk;
	double prob_0, prob_1, prob_2, prob_3;
	double diff = 0;
	// coordinate ascent for NUM_ITER iterations
	for (int iter = 0; iter < NUM_ITER; iter++) {
		for (int i = 0; i <= seqlen - 2; i++) {
			for (int j = i+1; j <= seqlen - 1; j++) {

				mu_idx = get_idx(seqlen, i, j);
				alpha = 0;

				// Calculation for sequence features. Depends on ij features.

				// for edge ij, the amino acid indicator will be nonzero at only one location, use that to index gammas
				// Also distance feature (note: j - i = abs(i - j) b/c of the way the loop works) and prior
				alpha += gamma[feats_aa[mu_idx]] + (j - i)*theta_dist + theta_prior;
				// Calculation for triplet factors. Depends on mu_ik, mu_jk.
				for (int k = 0; k < seqlen; k++) {
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
	if (nrhs != 9)
		mexErrMsgTxt("calc_muhat: Requires nine arguments.");
	if ( (!mxIsClass(prhs[0], "uint32")) || 
		 (!mxIsClass(prhs[2], "uint32")) || 
		 (!mxIsClass(prhs[8], "uint32")) )
		mexErrMsgTxt("calc_muhat: Arguments 1 - 3 must be UINT32.");

	const int *n_mus = (int *) mxGetData(prhs[0]); // L - dimension vector
	const mxArray *feats_aa = prhs[1]; // (L x 1) cell array of int vectors
	const int *seqlen = (int *) mxGetData(prhs[2]); // L - dimension vector
	const double *theta_tri = mxGetPr(prhs[3]); // 4 dimension vector of three factor weights
	const double *gamma = mxGetPr(prhs[4]); // 20*(20+1)/2 dimension vector of amino acid weights
	const double theta_dist = *(mxGetPr(prhs[5])); // distance weight
	const double theta_seqlen = *(mxGetPr(prhs[6])); // seqlen weight
	const double theta_prior = *(mxGetPr(prhs[7])); // prior weight
	const int L = *((int *) mxGetData(prhs[8])); // Number of training examples
	
	//int dims[2]; dims[0] = L; dims[1] = 1;
	const mwSize dims[2] = {(mwSize)L, 1};
	//const int dims[2] = {L, 1};
	plhs[0] = mxCreateCellArray(2, dims); // output mus, (L x 1) cell array of double vectors

	// Iterate through each training example
	for (int l = 0; l < L; l++) {
		// Initialize mus array and set everything to 0.5.
		double *mus = (double *) mxCalloc(n_mus[l], sizeof(double));
		for (int i = 0; i < n_mus[l]; i++) {
			mus[i] = 0.5;
		}

		// Obtain correct feats array
		const int *feats_aa_l = (int *) mxGetData(mxGetCell(feats_aa, l));

		// Run calc_muhat
		calc_muhat(mus, feats_aa_l, seqlen[l], theta_tri, gamma,
					theta_dist, theta_seqlen, theta_prior);

		// Put mus into an mxArray to return
		mxArray *mus_tmp = mxCreateDoubleMatrix(n_mus[l], 1, mxREAL);
		mxSetData(mus_tmp, mus);

		// Set that cell l to the mxArray above.
		mxSetCell(plhs[0], l, mus_tmp);
	}
	
	
}