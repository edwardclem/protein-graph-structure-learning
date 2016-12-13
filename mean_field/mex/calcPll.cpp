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
	int condition_dist) {

	double Pll = 0;
	double x_ij, x_ik, x_jk;
	double prob_0, prob_1, prob_2, prob_3;

	double seq_feat = seqlen*theta_seqlen;
	double edge_feats = 0;
	double alpha_ij;
	double beta_ij;
	double a_plus_b;
	double num_present[] = {0, 0, 0, 0}; //double to ensure division works
	int num_present_alpha;
	int num_present_beta;


	// i, j are less equals, but k is less than to deal with sum over two versus sum over three elements
	for (int i = 0; i < seqlen-1; i++) {
		for (int j = i+1; j < seqlen; j++) {
			//conditioning
			//if (j - i > condition_dist){

				// Calculation for sequence features. Depends only on mu_ij
				//for edge ij, the amino acid indicator will be nonzero at only one location, so use that to index into gammas
				alpha_ij = (gamma[feats_aa[get_idx(seqlen, i, j)]] + theta_dist*(j - i) + seq_feat + theta_prior);
				beta_ij = 0;

				// for (int k = 0; k < seqlen; k++){
				// 	if ((k == i) || (k == j))
				// 		continue;

				// 	x_jk = x[get_idx(seqlen, j, k)];
				// 	x_ik = x[get_idx(seqlen, i, k)];

				// 	num_present_beta = x_jk + x_ik; //if x_ij = 0
				// 	num_present_alpha = 1 + num_present_beta; //if x_ij = 1


				// 	alpha_ij += theta_tri[num_present_alpha];
				// 	beta_ij += theta_tri[num_present_beta];
				// 	num_present[num_present_alpha] += 1;
				// 	num_present[num_present_beta] += 1;
				// }

			a_plus_b = exp(alpha_ij); //+ exp(beta_ij);
			Pll +=  log(a_plus_b);

			// for (int n_edge = 0; n_edge < 3; n_edge++){
			// 	gradPll[n_edge] = num_present[n_edge]/a_plus_b;
			// 	num_present[n_edge] = 0; //reset to zero
			// }
			// gradPll[0] = num_present[0]/a_plus_b;
			// gradPll[1] = num_present[1]/a_plus_b;
			// gradPll[2] = num_present[2]/a_plus_b;
			// gradPll[3] = num_present[3]/a_plus_b;

			gradPll[NUM_INTERACTIONS + feats_aa[get_idx(seqlen, i, j)]] += 1/a_plus_b; // aa feature
			gradPll[NUM_INTERACTIONS + NUM_AA_FEATS] += ((double)(j - i))/a_plus_b; // dist feature
			gradPll[NUM_INTERACTIONS + NUM_AA_FEATS + 1] += seqlen/a_plus_b; // seqlen feature
			gradPll[NUM_INTERACTIONS + NUM_AA_FEATS + 2] += 1/a_plus_b; // prior


			
		}
	}

	return Pll;

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	if (nrhs != 9)
		mexErrMsgTxt("calcPll: Requires nine arguments.");
	if ( (!mxIsClass(prhs[1], "uint32")) || (!mxIsClass(prhs[2], "uint32")) )
		mexErrMsgTxt("calcPll: Arguments 2 and 3 must be UINT32.");

	int *x = (int *) mxGetData(prhs[0]);
	int *feats_aa = (int *) mxGetData(prhs[1]);
	int seqlen = *((int *) mxGetData(prhs[2]));
	double *theta_tri = mxGetPr(prhs[3]);
	double *gamma = mxGetPr(prhs[4]);
	double theta_dist = *(mxGetPr(prhs[5]));
	double theta_seqlen = *(mxGetPr(prhs[6]));
	double theta_prior = *(mxGetPr(prhs[7]));
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