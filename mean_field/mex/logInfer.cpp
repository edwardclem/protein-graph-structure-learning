/*
 * Calculating approximation F(theta, mus) of log-normalizer for a given protein.
 * Used in mean-field CRF approximation.
 */
#include "mex.h"
#include <cmath>
#include <cstdint>

#define NUM_FEATURES 213
#define NUM_INTERACTIONS 0
#define NUM_AA_FEATS 210

// Get index of mu_ij given seqence length. Have to offset for upper triangular matrix
inline int get_idx(int seqlen, int i, int j) {
	return i*seqlen - (i+1)*(i+2)/2 + j;
}

inline double sigmoid(double a) {
	return 1/(1 + exp(-a));
}

/*
 * Calculates pseudolikelihood approximation of log-normalizer for a given protein. Mus are current mus for that protein, 
 * feats_aa are amino acid features for that protein, seqlen is number of amino acids.
 *
 * Full theta vector is given by theta = [theta_tri, gamma, theta_dist, theta_seqlen, theta_prior]
 * Vector is broken up in MATLAB code for ease.
 *
 * Also calculates gradient of Pll wrt theta and stores it in gradPll.
 */ 
void logInfer(
	double *scores, 
	const int *feats_aa, 
	const int seqlen,
	const double *gamma, 
	const double theta_dist, 
	const double theta_seqlen, 
	const double theta_prior,
	const int condition_dist
	) 
{
	double seq_feat = seqlen*theta_seqlen;
	for (int i = 0; i < seqlen-1; i++) {
		for (int j = i+1; j < seqlen; j++) {
			scores[get_idx(seqlen, i, j)] = sigmoid(gamma[feats_aa[get_idx(seqlen, i, j)]] + theta_dist*(j - i) + seq_feat + theta_prior);
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	if (nrhs != 7)
		mexErrMsgTxt("calcLogistic: Requires nine arguments.");
	if ( (!mxIsClass(prhs[0], "int32")) || (!mxIsClass(prhs[1], "int32")) )
		mexErrMsgTxt("calcLogistic: Arguments 1 and 2 must be INT32.");

	const int *feats_aa = (int *) mxGetData(prhs[0]);
	const int seqlen = *((int *) mxGetData(prhs[1]));
	const double *gamma = mxGetPr(prhs[2]);
	const double theta_dist = *(mxGetPr(prhs[3]));
	const double theta_seqlen = *(mxGetPr(prhs[4]));
	const double theta_prior = *(mxGetPr(prhs[5]));
	const int condition_dist = *((int *) mxGetData(prhs[6]));


	double *scores = (double *) mxCalloc(seqlen*(seqlen-1)/2, sizeof(double));

	logInfer(scores, feats_aa, seqlen, gamma, 
					theta_dist, theta_seqlen, theta_prior, condition_dist);
	plhs[0] = mxCreateDoubleMatrix(seqlen*(seqlen-1)/2, 1, mxREAL);
	mxSetData(plhs[0], scores);
}
