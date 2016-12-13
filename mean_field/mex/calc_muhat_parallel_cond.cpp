/*
 * Perform coordinate ascent to calculate mu_hat = argmax_mus F(theta, mus), for a given protein.
 * Used in mean-field CRF approximation.
 * conditioning on edge distance
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
volatile int n_running_threads = 0;
pthread_mutex_t n_running_mutex = PTHREAD_MUTEX_INITIALIZER;

// Data for calc_muhat
struct calc_muhat_data_t {
  double *mus;
  const int *feats_aa;
  int seqlen;
  const double *theta_tri;
  const double *gamma;
  double theta_dist;
  double theta_seqlen;
  double theta_prior;
  int condition_dist; 
};

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
	const double theta_prior,
	const int condition_dist) {



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
				if (j - i <= condition_dist){ //conditioning on this edge
					mus[mu_idx] = 1; //q(x_ij) = 1 -> observed
				} else{
					
					alpha = 0;

					// Calculation for sequence features. Depends on ij features.

					// for edge ij, the amino acid indicator will be nonzero at only one location, use that to index gammas
					// Also distance feature (note: j - i = abs(i - j) b/c of the way the loop works) and prior
					// DEBUG!!
					//alpha += gamma[feats_aa[mu_idx]] + (j - i)*theta_dist + theta_prior;

					// Calculation for triplet factors. Depends on mu_ik, mu_jk.
					
					for (int k = 0; k < seqlen; k++) {
						if ((k == i) || (k == j))
							continue;

						mu_ik = mus[get_idx(seqlen, i, k)];
						mu_jk = mus[get_idx(seqlen, j, k)];

						//prob_0 = -(1 - mu_ik)*(1 - mu_jk);
						//prob_1 = (1 - mu_ik)*(1 - mu_jk) - mu_ik*(1 - mu_jk) - (1 - mu_ik)*mu_jk;
						prob_2 = mu_ik*(1 - mu_jk) + (1 - mu_ik)*mu_jk - mu_ik*mu_jk;
						prob_3 = mu_ik*mu_jk;

						//alpha += theta_tri[0]*prob_0 + theta_tri[1]*prob_1 + theta_tri[2]*prob_2 + theta_tri[3]*prob_3;
						alpha += theta_tri[2]*prob_2 + theta_tri[3]*prob_3;
					}
					
					alpha = exp(alpha);
					diff += std::abs(mus[mu_idx] - (alpha / (1 + alpha)));
					mus[mu_idx] = alpha / (1 + alpha);
				}
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
				data->theta_prior, data->condition_dist);
	pthread_mutex_lock(&n_running_mutex);
	n_running_threads--;
	pthread_mutex_unlock(&n_running_mutex);
	pthread_exit(NULL);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 11)
		mexErrMsgTxt("calc_muhat_parallel_cond: Requires 11 arguments.");
	if ( (!mxIsClass(prhs[0], "int32")) || 
		 (!mxIsClass(prhs[2], "int32")) || 
		 (!mxIsClass(prhs[8], "int32")) ||
		 (!mxIsClass(prhs[9], "int32")) ||
		 (!mxIsClass(prhs[10], "int32")) )
		mexErrMsgTxt("calc_muhat: Arguments 1, 3, 9, 10, 11 must be INT32.");

	const int *n_mus = (int *) mxGetData(prhs[0]); // L - dimension vector
	const mxArray *feats_aa = prhs[1]; // (L x 1) cell array of int vectors
	const int *seqlen = (int *) mxGetData(prhs[2]); // L - dimension vector
	const double *theta_tri = mxGetPr(prhs[3]); // 4 dimension vector of three factor weights
	const double *gamma = mxGetPr(prhs[4]); // 20*(20+1)/2 dimension vector of amino acid weights
	const double theta_dist = *(mxGetPr(prhs[5])); // distance weight
	const double theta_seqlen = *(mxGetPr(prhs[6])); // seqlen weight
	const double theta_prior = *(mxGetPr(prhs[7])); // prior weight
	const int L = *((int *) mxGetData(prhs[8])); // Number of training examples
	const int n_max_threads = *((int *) mxGetData(prhs[9])); // Number of threads to use
	const int condition_dist = *((int *) mxGetData(prhs[10])); //conditioning on these edges being true
	
	const mwSize dims[2] = {(mwSize)L, 1};

	// char buf[100];
	// sprintf(buf, "Condition distance: %u \n", condition_dist);

	// mexPrintf(buf);


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
	for (int l = 0; l < L; l++) {
		// Initialize mus array and set everything to 0.5.
		double *mus = (double *) mxCalloc(n_mus[l], sizeof(double));
		for (int i = 0; i < n_mus[l]; i++) {
			mus[i] = 0.5;
		}

		// Obtain correct feats array
		const int *feats_aa_l = (int *) mxGetData(mxGetCell(feats_aa, l));

		// Set data
		mh_data[l].mus = mus;
		mh_data[l].feats_aa = feats_aa_l;
		mh_data[l].seqlen = seqlen[l];
		mh_data[l].theta_tri = theta_tri;
		mh_data[l].gamma = gamma;
		mh_data[l].theta_dist = theta_dist;
		mh_data[l].theta_seqlen = theta_seqlen;
		mh_data[l].theta_prior = theta_prior;
		mh_data[l].condition_dist = condition_dist;


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
	for (int l = 0; l < L; l++) {
		// Put mus into an mxArray to return
		mxArray *mus_tmp = mxCreateDoubleMatrix(n_mus[l], 1, mxREAL);
		mxSetData(mus_tmp, mh_data[l].mus);

		// Set that cell l to the mxArray above.
		mxSetCell(plhs[0], l, mus_tmp);
	}
	
	
}