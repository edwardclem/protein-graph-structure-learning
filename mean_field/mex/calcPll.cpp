/*
 * Calculating approximation F(theta, mus) of log-normalizer for a given protein.
 * Used in mean-field CRF approximation.
 */
#include "mex.h"
#include <cmath>
#include <cstdint>

#define NUM_FEATURES 217
#define NUM_INTERACTIONS 4
#define NUM_AA_FEATS 210

// Get index of mu_ij given seqence length. Have to offset for upper triangular matrix
inline int get_idx(int seqlen, int i, int j) {
	return i*seqlen - (i+1)*(i+2)/2 + j;
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
double calcPll(
	const int *x, 
	const int *feats_aa, 
	const int seqlen,
	const double *theta_tri, 
	const double *gamma, 
	const double theta_dist, 
	const double theta_seqlen, 
	const double theta_prior, 
	double *gradPll,
	const int condition_dist) {

	double Pll = 0;
	double x_ij, x_ik, x_jk;
	double prob_0, prob_1, prob_2, prob_3;

	double seq_feat = seqlen*theta_seqlen;
	double edge_feats = 0;
	double alpha_ij, beta_ij, a_plus_b, prob_ij;
	double nBeta2, nAlpha2, nAlpha3; //double to ensure division works
	int num_present_alpha;
	int num_present_beta;

	int total2_false, total2_true, total3;
	double probData = 0;
	// i, j are less equals, but k is less than to deal with sum over two versus sum over three elements
	for (int i = 0; i < seqlen-1; i++) {
		for (int j = i+1; j < seqlen; j++) {
			//conditioning
			//if (j - i > condition_dist){

			// Calculation for sequence features. Depends only on mu_ij
			//for edge ij, the amino acid indicator will be nonzero at only one location, so use that to index into gammas
			alpha_ij = (gamma[feats_aa[get_idx(seqlen, i, j)]] + theta_dist*(j - i) + seq_feat + theta_prior);
			beta_ij = 0;
			x_ij = x[get_idx(seqlen, i, j)];

			nBeta2 = 0;
			nAlpha2 = 0;
			nAlpha3 = 0;
			total2_false = 0;
			total2_true = 0;
			total3 = 0;
			for (int k = 0; k < seqlen; k++){
				if ((k == i) || (k == j))
					continue;

				x_jk = x[get_idx(seqlen, j, k)];
				x_ik = x[get_idx(seqlen, i, k)];

				if (x_ik + x_jk == 2) {
					beta_ij += theta_tri[2];
					alpha_ij += theta_tri[3];
					nBeta2 += 1;
					nAlpha3 += 1;
					if (x_ij)
						total3 += 1;
					else
						total2_false += 1;
				} else if (x_ik + x_jk == 1) {
					alpha_ij += theta_tri[2];
					nAlpha2 += 1;
					if (x_ij)
						total2_true += 1;
				}
			}

			a_plus_b = exp(alpha_ij) + exp(beta_ij);
			//mexPrintf("alpha: %0.2f, beta: %0.2f\n", alpha_ij, beta_ij);
			Pll +=  log(a_plus_b);
			prob_ij = exp(alpha_ij)/a_plus_b;
			if (x[get_idx(seqlen, i, j)])
				probData += log(prob_ij);
			else
				probData += log((1 - prob_ij));

			gradPll[2] += total2_false - nBeta2*(1 - prob_ij);
			gradPll[2] += total2_true - nAlpha2*prob_ij;
			gradPll[3] += total3 - nAlpha3*prob_ij;

			gradPll[NUM_INTERACTIONS + feats_aa[get_idx(seqlen, i, j)]] += x_ij - prob_ij; // aa feature
			gradPll[NUM_INTERACTIONS + NUM_AA_FEATS] += ((double)(j - i))*(x_ij - prob_ij); // dist feature
			gradPll[NUM_INTERACTIONS + NUM_AA_FEATS + 1] += seqlen*(x_ij - prob_ij); // seqlen feature
			gradPll[NUM_INTERACTIONS + NUM_AA_FEATS + 2] += x_ij - prob_ij; // prior
		}
	}
	//mexPrintf("%0.2f\n", probData);
	//mexPrintf("total2: %d, total3: %d\n", total2, total3);
	return probData - Pll;

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	if (nrhs != 9)
		mexErrMsgTxt("calcPll: Requires nine arguments.");
	if ( (!mxIsClass(prhs[0], "int32")) || (!mxIsClass(prhs[1], "int32")) || (!mxIsClass(prhs[2], "int32")) )
		mexErrMsgTxt("calcPll: Arguments 1, 2, and 3 must be INT32.");

	const int *x = (int *) mxGetData(prhs[0]);
	const int *feats_aa = (int *) mxGetData(prhs[1]);
	const int seqlen = *((int *) mxGetData(prhs[2]));
	const double *theta_tri = mxGetPr(prhs[3]);
	const double *gamma = mxGetPr(prhs[4]);
	const double theta_dist = *(mxGetPr(prhs[5]));
	const double theta_seqlen = *(mxGetPr(prhs[6]));
	const double theta_prior = *(mxGetPr(prhs[7]));
	const int condition_dist = *((int *) mxGetData(prhs[8]));

	// char buf[100];
	// sprintf(buf, "Condition distance: %u \n", condition_dist);

	// mexPrintf(buf);


	double *gradPll = (double *) mxCalloc(NUM_FEATURES, sizeof(double));
	for (int i = 0; i < NUM_FEATURES; i++) {
		gradPll[i] = 0;
	}

	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *Pll = mxGetPr(plhs[0]);
	(*Pll) = calcPll(x, feats_aa, seqlen, theta_tri, gamma, 
					theta_dist, theta_seqlen, theta_prior, gradPll, condition_dist);
	plhs[1] = mxCreateDoubleMatrix(NUM_FEATURES, 1, mxREAL);
	mxSetData(plhs[1], gradPll);
}
