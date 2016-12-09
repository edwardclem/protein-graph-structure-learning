/*
 * Perform coordinate ascent to calculate mu_hat = argmax_mus F(theta, mus), for a given protein.
 * Used in mean-field CRF approximation.
 */
#include "mex.h"
#include <cmath>
#include <string.h>
#include <cstdint>
#include <pthread.h>
#include <unistd.h>
#include "matrix.h"

#define NUM_ITER 20
#define CONV_TOL 0.0001

// Keep a count of the total number of running threads
volatile size_t n_running_threads = 0;
pthread_mutex_t n_running_mutex = PTHREAD_MUTEX_INITIALIZER;

// Data for calc_muhat
struct calc_muhat_data_t {
  double *mus;
  const uint32_t *feats_aa;
  size_t seqlen;
  const double *theta_tri;
  const double *gamma;
  double theta_dist;
  double theta_seqlen;
  double theta_prior;
};

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
void calc_muhat(
	double *mus, 
	const uint32_t *feats_aa, 
	const size_t seqlen,
	const double *theta_tri, 
	const double *gamma, 
	const double theta_dist, 
	const double theta_seqlen,
	const double theta_prior) {

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

// Wrapper function. Calls calc_muhat and decrements n_running_threads when done.
void *calc_muhat_wrapper(void *args) {
	struct calc_muhat_data_t *data = (struct calc_muhat_data_t *) args;
	calc_muhat(data->mus, data->feats_aa, data->seqlen, data->theta_tri, 
				data->gamma, data->theta_dist, data->theta_seqlen, 
				data->theta_prior);
	pthread_mutex_lock(&n_running_mutex);
	n_running_threads--;
	pthread_mutex_unlock(&n_running_mutex);
	pthread_exit(NULL);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 10)
		mexErrMsgTxt("calc_muhat: Requires ten arguments.");
	if ( (!mxIsClass(prhs[0], "uint32")) || 
		 (!mxIsClass(prhs[2], "uint32")) || 
		 (!mxIsClass(prhs[8], "uint32")) ||
		 (!mxIsClass(prhs[9], "uint32")) )
		mexErrMsgTxt("calc_muhat: Arguments 1, 3, 9, 10 must be UINT32.");

	const uint32_t *n_mus = (uint32_t *) mxGetData(prhs[0]); // L - dimension vector
	const mxArray *feats_aa = prhs[1]; // (L x 1) cell array of uint32_t vectors
	const uint32_t *seqlen = (uint32_t *) mxGetData(prhs[2]); // L - dimension vector
	const double *theta_tri = mxGetPr(prhs[3]); // 4 dimension vector of three factor weights
	const double *gamma = mxGetPr(prhs[4]); // 20*(20+1)/2 dimension vector of amino acid weights
	const double theta_dist = *(mxGetPr(prhs[5])); // distance weight
	const double theta_seqlen = *(mxGetPr(prhs[6])); // seqlen weight
	const double theta_prior = *(mxGetPr(prhs[7])); // prior weight
	const uint32_t L = *((uint32_t *) mxGetData(prhs[8])); // Number of training examples
	const uint32_t n_max_threads = *((uint32_t *) mxGetData(prhs[9])); // Number of threads to use
	
	const mwSize dims[2] = {L, 1};


	plhs[0] = mxCreateCellArray(2, dims); // output mus, (L x 1) cell array of double vectors

	pthread_t threads[L];
	struct calc_muhat_data_t mh_data[L];
	pthread_attr_t attr;
	int rc;
	void *status;

	// Initialize and set thread joinable
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED); // So you don't have to wait to join the threads

	// Iterate through each training example
	for (size_t l = 0; l < L; l++) {
		// Initialize mus array and set everything to 0.5.
		double *mus = (double *) mxCalloc(n_mus[l], sizeof(double));
		for (size_t i = 0; i < n_mus[l]; i++) {
			mus[i] = 0.5;
		}

		// Obtain correct feats array
		const uint32_t *feats_aa_l = (uint32_t *) mxGetData(mxGetCell(feats_aa, l));

		// Set data
		mh_data[l].mus = mus;
		mh_data[l].feats_aa = feats_aa_l;
		mh_data[l].seqlen = seqlen[l];
		mh_data[l].theta_tri = theta_tri;
		mh_data[l].gamma = gamma;
		mh_data[l].theta_dist = theta_dist;
		mh_data[l].theta_seqlen = theta_seqlen;
		mh_data[l].theta_prior = theta_prior;


		// Don't launch too many threads
		while (n_running_threads >= n_max_threads) {
			sleep(1);
		}

		// Increment number of currently running threads
		pthread_mutex_lock(&n_running_mutex);
		n_running_threads++;
		pthread_mutex_unlock(&n_running_mutex);
		
		// Launch thread
		rc = pthread_create(&threads[l], &attr, calc_muhat_wrapper, (void *) &mh_data[l]);
		if (rc) {
			char buf[100];
			sprintf(buf, "Error: unable to create thread, %d\n", rc);
			mexErrMsgTxt(buf);
			exit(-1);
		}
	}

	// Wait until threads finish
	while (n_running_threads > 0) {
		sleep(1);
	}

	// Set data into mxArray
	for (size_t l = 0; l < L; l++) {
		// Put mus into an mxArray to return
		mxArray *mus_tmp = mxCreateDoubleMatrix(n_mus[l], 1, mxREAL);
		mxSetData(mus_tmp, mh_data[l].mus);

		// Set that cell l to the mxArray above.
		mxSetCell(plhs[0], l, mus_tmp);
	}
	
	
}