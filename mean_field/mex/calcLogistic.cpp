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
double calcLogistic(
	const int *x, 
	const int *feats_aa, 
	const int seqlen,
	const double *theta_tri, 
	const double *gamma, 
	const double theta_dist, 
	const double theta_seqlen, 
	const double theta_prior, 
	double *gradLogistic,
	const int condition_dist) {

	double cost = 0;
	double x_ij;

	double seq_feat = seqlen*theta_seqlen;
	double alpha_ij;
	// i, j are less equals, but k is less than to deal with sum over two versus sum over three elements
	for (int i = 0; i < seqlen-1; i++) {
		for (int j = i+1; j < seqlen; j++) {
			//conditioning
			//if (j - i > condition_dist){

			// Calculation for sequence features.
			x_ij = x[get_idx(seqlen, i, j)];
			alpha_ij = sigmoid(gamma[feats_aa[get_idx(seqlen, i, j)]] + theta_dist*(j - i) + seq_feat + theta_prior);
			cost += x_ij*log(alpha_ij) + (1 - x_ij)*log(1 - alpha_ij);

			gradLogistic[feats_aa[get_idx(seqlen, i, j)]] += x_ij - alpha_ij; // aa feature
			gradLogistic[NUM_AA_FEATS] += ((double)(j - i))*(x_ij - alpha_ij); // dist feature
			gradLogistic[NUM_AA_FEATS + 1] += seqlen*(x_ij - alpha_ij); // seqlen feature
			gradLogistic[NUM_AA_FEATS + 2] += x_ij - alpha_ij; // prior
		}
	}
	return cost;

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	if (nrhs != 9)
		mexErrMsgTxt("calcLogistic: Requires nine arguments.");
	if ( (!mxIsClass(prhs[1], "int32")) || (!mxIsClass(prhs[2], "int32")) )
		mexErrMsgTxt("calcLogistic: Arguments 2 and 3 must be INT32.");

	const int *x = (int *) mxGetData(prhs[0]);
	const int *feats_aa = (int *) mxGetData(prhs[1]);
	const int seqlen = *((int *) mxGetData(prhs[2]));
	const double *theta_tri = mxGetPr(prhs[3]);
	const double *gamma = mxGetPr(prhs[4]);
	const double theta_dist = *(mxGetPr(prhs[5]));
	const double theta_seqlen = *(mxGetPr(prhs[6]));
	const double theta_prior = *(mxGetPr(prhs[7]));
	const int condition_dist = *((int *) mxGetData(prhs[8]));


	double *gradLogistic = (double *) mxCalloc(NUM_FEATURES, sizeof(double));
	for (int i = 0; i < NUM_FEATURES; i++) {
		gradLogistic[i] = 0;
	}

	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *cost = mxGetPr(plhs[0]);
	(*cost) = calcLogistic(x, feats_aa, seqlen, theta_tri, gamma, 
					theta_dist, theta_seqlen, theta_prior, gradLogistic, condition_dist);
	plhs[1] = mxCreateDoubleMatrix(NUM_FEATURES, 1, mxREAL);
	mxSetData(plhs[1], gradLogistic);
}
