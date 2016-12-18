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
 * Calculates approximation of log-normalizer for a given protein. Mus are current mus for that protein, 
 * feats_aa are amino acid features for that protein, seqlen is number of amino acids.
 *
 * Full theta vector is given by theta = [theta_tri, gamma, theta_dist, theta_seqlen, theta_prior]
 * Vector is broken up in MATLAB code for ease.
 *
 * Also calculates gradient of F wrt theta and stores it in gradF.
 */ 
double calcF(
	const double *mus, 
	const int *feats_aa, 
	const int seqlen,
	const double *theta_tri, 
	const double *gamma, 
	const double theta_dist, 
	const double theta_seqlen, 
	const double theta_prior, 
	double *gradF,
	const int condition_dist,
	const int *x) {

	double F = 0;
	double mu_ij, mu_ik, mu_jk;
	double x_ij, x_ik, x_jk;
	double prob_0, prob_1, prob_2, prob_3;

	double seq_feat = seqlen*theta_seqlen;
	double nBeta2, nAlpha2, nAlpha3; //double to ensure division works
	int num_present_alpha;
	int num_present_beta;

	int total2_false, total2_true, total3;
	double probData = 0;

	// i, j are less equals, but k is less than to deal with sum over two versus sum over three elements
	for (int i = 0; i <= seqlen-2; i++) {
		for (int j = i+1; j <= seqlen-1; j++) {
			//conditioning
			if (j - i > condition_dist){
				mu_ij = mus[get_idx(seqlen, i, j)];
				x_ij = x[get_idx(seqlen, i, j)];

				// Calculation for sequence features. Depends only on mu_ij
				//for edge ij, the amino acid indicator will be nonzero at only one location, so use that to index into gammas
				// DEBUG!!
				F += mu_ij*(gamma[feats_aa[get_idx(seqlen, i, j)]] + theta_dist*(j - i) + seq_feat + theta_prior);
				F -= (mu_ij*log(mu_ij) + (1 - mu_ij)*log(1 - mu_ij));
				probData += x_ij*(gamma[feats_aa[get_idx(seqlen, i, j)]] + theta_dist*(j - i) + seq_feat + theta_prior);
				
				gradF[NUM_INTERACTIONS + feats_aa[get_idx(seqlen, i, j)]] += x_ij - mu_ij; // aa feature
				gradF[NUM_INTERACTIONS + NUM_AA_FEATS] += (x_ij - mu_ij)*(j - i); // dist feature
				gradF[NUM_INTERACTIONS + NUM_AA_FEATS + 1] += (x_ij - mu_ij)*seqlen; // seqlen feature
				gradF[NUM_INTERACTIONS + NUM_AA_FEATS + 2] += x_ij - mu_ij; // prior



				// Calculation for triplet factors. Depends on mu_ij, mu_ik, mu_jk
				
				for (int k = j+1; k < seqlen; k++) {
					if (k - i > condition_dist){ //if all three are too close, then don't do anything
						mu_ik = mus[get_idx(seqlen, i, k)];
						mu_jk = mus[get_idx(seqlen, j, k)];

						x_ik = x[get_idx(seqlen, i, k)];
						x_jk = x[get_idx(seqlen, j, k)];
						//prob_0 = (1 - mu_ij)*(1 - mu_ik)*(1 - mu_jk);
						//prob_1 = mu_ij*(1 - mu_ik)*(1 - mu_jk) + (1 - mu_ij)*mu_ik*(1 - mu_jk) + (1 - mu_ij)*(1 - mu_ik)*mu_jk;
						prob_2 = mu_ij*mu_ik*(1 - mu_jk) + mu_ij*(1 - mu_ik)*mu_jk + (1 - mu_ij)*mu_ik*mu_jk;
						prob_3 = mu_ij*mu_ik*mu_jk;
						
						//F += theta_tri[0]*prob_0 + theta_tri[1]*prob_1 + theta_tri[2]*prob_2 + theta_tri[3]*prob_3;
						F += theta_tri[2]*prob_2 + theta_tri[3]*prob_3;
						//gradF[0] += prob_0;
						//gradF[1] += prob_1;
						gradF[2] += (x_ij + x_ik + x_jk) - prob_2;
						gradF[3] += (x_ij + x_ik + x_jk) - prob_3;
						
					}
				}
				
			}
			
		}
	}

	return F;

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	if (nrhs != 10)
		mexErrMsgTxt("calcF: Requires nine arguments.");
	if ( (!mxIsClass(prhs[1], "uint32")) || (!mxIsClass(prhs[2], "uint32")) )
		mexErrMsgTxt("calcF: Arguments 2 and 3 must be UINT32.");

	const double *mus = mxGetPr(prhs[0]);
	const int *feats_aa = (int *) mxGetData(prhs[1]);
	const int seqlen = *((int *) mxGetData(prhs[2]));
	const double *theta_tri = mxGetPr(prhs[3]);
	const double *gamma = mxGetPr(prhs[4]);
	const double theta_dist = *(mxGetPr(prhs[5]));
	const double theta_seqlen = *(mxGetPr(prhs[6]));
	const double theta_prior = *(mxGetPr(prhs[7]));
	const int condition_dist = *((int *) mxGetData(prhs[8]));
	const int *x = (int *) mxGetData(prhs[9]);


	// char buf[100];
	// sprintf(buf, "Condition distance: %u \n", condition_dist);

	// mexPrintf(buf);


	double *gradF = (double *) mxCalloc(NUM_FEATURES, sizeof(double));
	for (int i = 0; i < NUM_FEATURES; i++) {
		gradF[i] = 0;
	}

	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *F = mxGetPr(plhs[0]);
	(*F) = calcF(mus, feats_aa, seqlen, theta_tri, gamma, 
					theta_dist, theta_seqlen, theta_prior, gradF, condition_dist, x);
	plhs[1] = mxCreateDoubleMatrix(NUM_FEATURES, 1, mxREAL);
	mxSetData(plhs[1], gradF);
}
