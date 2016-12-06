//calculating alpha values for use in mean-field CRF approximation
#include "mex.h"
#include <math.h>


#define NUM_THETAS 407
#define NUM_FEATS 403
#define NUM_STATS 4
#define NUM_ITER 20



int coords_to_linear(int c1, int c2, int len){
	return c1*len + c2 - (c1 + 1)*(c1 + 2)/2; //offset for upper triangular
}

//given parameters theta, mus for a specific training example, and the features for a specific training example
//return array of mu hats (highest likelihood mu)
//performing coordinate ascent; for each iteration, calculate mu_hat given other mus for every mu

void calc_muhat(double *mus, size_t *feats_aa, size_t seqlen, size_t nedges, 
				double *thetas_tri, double *gamma, double theta_dist, double theta_seqlen, double theta_prior){

	//coordinate ascent for NUM_ITER iterations
	for (int iter = 0; iter < NUM_ITER; iter++){
		int mu_idx;
		for (int i = 0; i <= seqlen - 2; i++){
			for (int j = i + 1; j <= seqlen - 1; j++){

			mu_idx = coords_to_linear(i, j, seqlen);

			double alpha = 0; 

			//updating feature stats

			//for edge ij, the amino acid indicator will be nonzero at only one location;
			//storing this location only and using it to index into thetas
			alpha += gamma[feats_aa[mu_idx]];
			//distance feature
			alpha += abs(i - j)*theta_dist;
			//prior
			alpha += theta_prior;

			double mu_ik;
			double mu_jk;

			for (int k = j + 1; k < seqlen; k++){
				ik = coords_to_linear(i, k, seqlen);
				jk = coords_to_linear(j, k, seqlen);
					
				alpha -= thetas_tri[0]*(1 - mu_ik)*(1 - mu_jk);
				alpha += thetas_tri[1]*((1 - mu_ik)*(1 - mu_jk) - mu_ik*(1 - mu_jk) - (1 - mu_ik)*mu_jk);
				alpha += thetas_tri[2]*(mu_ik*(1 - mu_jk)  + (1 - mu_ik)*mu_jk - mu_ik*mu_jk);
				alpha += thetas_tri[3]*mu_ik*mu_jk;
			}
			

			//compute mu_hat

			mus[mu_idx] = exp(alpha/(1 - alpha));

			}
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nhrs != 5){
		mexErrMsgTxt("Insufficient inputs.");
	}


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

	double *thetas;
	int nthetas;
	thetas = mexGetPr(prhs[1]);
	nthetas = mexGetN(prhs[1]);

	if (nThetas != NUM_THETAS){
		mexErrMsgTxt("Theta matrix is of incorrect size.");
	}

	//N*(N - 1)/2 x 1 array of mus
	//N equals number of amino acids in this training example
	double *mus;
	int nedges;
	mus = mxGetPr(prhs[2]);
	nedges = mxGetN(prhs[2]);

	//403 x N*(N - 1)/2 array of features
	double *feats;
	int *dims;

	feats = mxGetPr(prhs[3]);
	dims = mxGetDimensions(prhs[3]);

	int nfeats = dims[0];

	if (nfeats != NUM_FEATS){
		mexErrMsgTxt("Feature vector must be 403 x N*(N - 1)/2, where N is the number of possible edges in the protein")
	}

	//4 x 1 array of triplet sufficient stats
	double *triplet_ss;
	int nss;
	triplet_ss = mexGetPr(prhs[4]);
	nss = mexGtN(prhs[4]);

	if (nss != 4){
		mexErrMsgTxt("There must be 4 triplet sufficient statistics.")
	}

	//creating output array for mus
	plhs[0] = mxCreateDoubleMatrix(1, nedges, mxREAL);
	mu_hats = mxGetPr(plhs[0]);

	//TODO: GET SEQLEN SOMEHOW
	calc_muhat(nthetas, thetas, mus, nedges, seqlen, feats,triplet_ss, mu_hats);

}