/*
 * potts3.cu
 */

/*
 * potts3, an optimized CUDA implementation of q-state Potts model.
 * For an L*L system of the Q-state Potts model, this code starts from an
 * initial ordered state (blacks=0, whites=0), it fixes the temperature temp
 * to TEMP_MIN and run TRAN Monte Carlo steps (MCS) to attain equilibrium,
 * then it runs TMAX MCS taking one measure each DELTA_T steps to perform
 * averages. After that, it keeps the last configuration of the system and
 * use it as the initial state for the next temperature, temp+DELTA_TEMP.
 * This process is repeated until some maximum temperature TEMP_MAX is reached.
 * The whole loop is repeated SAMPLES times to average over different
 * realizations of the thermal noise.
 * The outputs are the averaged energy, magnetization and their related second
 * and fourth moments for each temperature.
 * Copyright (C) 2010 Ezequiel E. Ferrero, Juan Pablo De Francesco,
 * Nicolás Wolovick, Sergio A. Cannas
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * This code was originally implemented for: "q-state Potts model metastability
 * study using optimized GPU-based Monte Carlo algorithms",
 * Ezequiel E. Ferrero, Juan Pablo De Francesco, Nicolás Wolovick,
 * Sergio A. Cannas
 * http://arxiv.org/abs/1101.0876
 */


#include <stddef.h> // NULL, size_t
#include <math.h> // expf
#include <stdio.h> // printf
#include <time.h> // time
#include <sys/time.h> // gettimeofday
#include <assert.h>
#include <limits.h> // UINT_MAX
#include "cutil.h" // CUDA_SAFE_CALL, CUT_CHECK_ERROR


// Default parameters
#ifndef Q
#define Q 9 // spins
#endif

#ifndef L
#define L 2048 // linear system size
#endif

#ifndef SAMPLES
#define SAMPLES 1 // number of samples
#endif

#ifndef TEMP_MIN
#define TEMP_MIN 0.70f // minimum temperature
#endif

#ifndef TEMP_MAX
#define TEMP_MAX 0.75f // maximum temperature
#endif

#ifndef DELTA_TEMP
#define DELTA_TEMP 0.0005f // temperature step
#endif

#ifndef TRAN
#define TRAN 20000 // equilibration time
#endif

#ifndef TMAX
#define TMAX 10000 // measurement time
#endif

#ifndef DELTA_T
#define DELTA_T 50 // sampling period for energy and magnetization
#endif

// Functions

// maximum
#define MAX(a,b) (((a)<(b))?(b):(a))
// minimum
#define MIN(a,b) (((a)<(b))?(a):(b))
// integer ceiling division
#define DIV_CEIL(a,b) (((a)+(b)-1)/(b))
// highest power of two less than x
// Thanks to Pablo Dal Lago for pointing this out, a minor variation of
// http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
#define ROUND_DOWN_POWER_OF_2(x) (( ( (x)>>0 | (x)>>1 | (x)>>2 | (x)>>3 | (x)>>4 | (x)>>5 | (x)>>6 | (x)>>7 | (x)>>8 | (x)>>9 | (x)>>10 | (x)>>11 | (x)>>12 | (x)>>13 | (x)>>14 | (x)>>15 | (x)>>16 | (x)>>17 | (x)>>18 | (x)>>19 | (x)>>20 ) +1 ) >> 1)

// here we cannot use __CUDA_ARCH__, since it is only available on kernels,
// simply uncomment the desired option

// Hardware parameters for GTX 280 (GT200)
#define SHARED_PER_BLOCK 16384
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID 65535
// Hardware parameters for GTX 470/480 (GF100)
//#define SHARED_PER_BLOCK 49152
//#define WARP_SIZE 32
//#define THREADS_PER_BLOCK 1024
//#define BLOCKS_PER_GRID 65535

// Auto adjusting parameters
// Block size for sumupECUDA, autoadjust to fill up max allowable threads per block
// Block size should be less than THREAD_PER_BLOCK and
// less than the SHARED_PER_BLOCK bytes of shared per block
#define BLOCK_E_TMP (MIN(THREADS_PER_BLOCK, SHARED_PER_BLOCK/sizeof(unsigned int)))
#define BLOCK_E (ROUND_DOWN_POWER_OF_2(BLOCK_E_TMP))
// BLOCKS_PER_GRID limits the amount of linear blocks, we divide in GRID_E
#define GRID_E 64
// block size for sumupMCUDA, autoadjust to fill up shared memory per block
// 16 is a little slack because shared is not purely for __shared__
#define BLOCK_M_TMP (MIN(THREADS_PER_BLOCK, (SHARED_PER_BLOCK-16)/(Q*sizeof(unsigned int))))
#define BLOCK_M (ROUND_DOWN_POWER_OF_2(BLOCK_M_TMP))

// Tweakeable parameters
#define CUDA_DEVICE 0	// card number
#define FRAME 512	// the whole thing is framed for the RNG
#define TILE 16		// each block of threads is a tile
#undef DETERMINISTIC_WRITE // spin write is deterministic or probabilistic

// Internal definitions and functions
// out vector size, it is +1 since we reach TEMP_MAX
#define NPOINTS (1+(int)((TEMP_MAX-TEMP_MIN)/DELTA_TEMP))
#define N (L*L) // system size
#define SAFE_PRIMES_FILENAME "safeprimes_base32.txt"
#define SEED (time(NULL)) // random seed
#define MICROSEC (1E-6)
#define WHITE 0
#define BLACK 1

// cells are bytes
typedef unsigned char byte;

// temperature, E, E^2, E^4, M, M^2, M^4
struct statpoint {
	double t;
	double e; double e2; double e4;
	double m; double m2; double m4;
};

static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y);

// we frame the grid in FRAME*FRAME/2
#define NUM_THREADS (FRAME*FRAME/2)
// state of the random number generator, last number (x_n), last carry (c_n) packed in 64 bits
__device__ static unsigned long long d_x[NUM_THREADS];
// multipliers (constants)
__device__ static unsigned int d_a[NUM_THREADS];


/***
 * Device functions
 ***/

// RNG: multiply with carry
#include "CUDAMCMLrng.cu"

__global__ void updateCUDA(const float temp,
			   const unsigned int color,
			   byte* __restrict__ const write,
			   const byte* __restrict__ const read) {

	const unsigned int jOriginal = blockIdx.x*TILE + threadIdx.x;
	const unsigned int iOriginal = blockIdx.y*TILE + threadIdx.y;
	const unsigned int tid = iOriginal*FRAME + jOriginal;

	int h_before, h_after, delta_E;
	byte spin_old, spin_new;
	byte spin_neigh_n, spin_neigh_e, spin_neigh_s, spin_neigh_w;

	unsigned int i;
	// move thread RNG state to registers. Thanks to Carlos Bederián for pointing this out.
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];
	for (unsigned int iFrame=0; iFrame<(L/FRAME); iFrame++) {
		i = iOriginal + (FRAME/2)*iFrame;
		unsigned int j;
		for (unsigned int jFrame=0; jFrame<(L/FRAME); jFrame++) {
			j = jOriginal + FRAME*jFrame;

			spin_old = write[i*L + j];

			// computing h_before
			spin_neigh_n = read[i*L + j];
			spin_neigh_e = read[i*L + (j+1)%L];
			spin_neigh_w = read[i*L + (j-1+L)%L];
			spin_neigh_s = read[((i+(2*(color^(j%2))-1)+L/2)%(L/2))*L + j];
			// using !(spin_old^mag_neigh.x) + ... is slightly slower
			h_before = - (spin_old==spin_neigh_n) - (spin_old==spin_neigh_e)
				   - (spin_old==spin_neigh_w) - (spin_old==spin_neigh_s);

			// new spin
			spin_new = (spin_old + (byte)(1 + rand_MWC_co(&rng_state, &rng_const)*(Q-1))) % Q;

			// h after taking new spin
			h_after = - (spin_new==spin_neigh_n) - (spin_new==spin_neigh_e)
				  - (spin_new==spin_neigh_w) - (spin_new==spin_neigh_s);

			delta_E = h_after - h_before;
			float p = rand_MWC_co(&rng_state, &rng_const);
#ifdef DETERMINISTIC_WRITE
			int change = delta_E<=0 || p<=expf(-delta_E/temp);
			write[i*L + j] = (change)*spin_new + (1-change)*spin_old;
#else
			if (delta_E<=0 || p<=expf(-delta_E/temp)) {
				write[i*L + j] = spin_new;
			}
#endif
		}
	}
	d_x[tid] = rng_state; // store RNG state into global again
}


__global__ void calculateCUDA(const byte* __restrict__ const white,
			      const byte* __restrict__ const black,
			      unsigned int* __restrict__ const Ematrix,
			      unsigned int* __restrict__ const Mmatrix) {
	const unsigned int j = blockIdx.x*TILE + threadIdx.x;
	const unsigned int i = blockIdx.y*TILE + threadIdx.y;

	// per-block vector of spin counters
	__shared__ unsigned int M[Q];
	// per block sum of energy
	__shared__ unsigned int E;

	// linear coordinates (threadId, blockId)
	const unsigned int tid = threadIdx.y*TILE+threadIdx.x;
	const unsigned int bid = blockIdx.y*(L/TILE)+blockIdx.x;

	if (tid==0) { // per-block reset of partial sum of energies
		E = 0;
	}
	if (tid<Q) { // per-block reset of vector of spin counters
		M[tid] = 0;
	}
	__syncthreads();

	byte spin;
	uchar4 spin_neigh;

	spin = white[i*L + j];
	spin_neigh.x = black[i*L + j];
	spin_neigh.y = black[i*L + (j+1)%L];
	spin_neigh.z = black[i*L + (j-1+L)%L];
	spin_neigh.w = black[((i+(2*(j%2)-1)*1+L/2)%(L/2))*L + j];

	// energy is pairwise, is enough to add white.
	atomicAdd(&E, (spin==spin_neigh.x)+(spin==spin_neigh.y)+(spin==spin_neigh.z)+(spin==spin_neigh.w));
	atomicAdd(&(M[spin]), 1);

	spin = black[i*L + j];
	atomicAdd(&(M[spin]), 1);

	// store per-block accumulations
	__syncthreads();
	if (tid<Q) {
		Mmatrix[bid*Q + tid] = M[tid];
	}
	if (tid==0) {
		Ematrix[bid] = E;
	}
}


// Input: vector of BLOCKS_CALCULATE energies
//	where BLOCKS_CALCULATE = DIV_CEIL((L*L/2), (TILE*TILE))
// Output: vector of (BLOCKS_CALCULATE/BLOCK_E) partial sums.
__global__ void sumupECUDA(unsigned int* __restrict__ const Ematrix) {
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ unsigned int block_E[BLOCK_E];

	block_E[tid] = Ematrix[bid*BLOCK_E + tid];

	// Kirk&Hwu, Programming Massively Parallel Processors, p.100
	for (unsigned int stride=BLOCK_E/2; 0<stride; stride/=2) {
		__syncthreads();
		if (tid<stride) {
			block_E[tid] += block_E[tid+stride];
		}
	}
	// write result
	if (tid==0) {
		Ematrix[bid] = block_E[0];
	}
}


// Input: vector of Q * BLOCKS_CALCULATE magnetizations: m0,m1,...,mq-1,m0,...
//	where BLOCKS_CALCULATE = DIV_CEIL((L*L/2), (TILE*TILE))
// Output: vector of Q * (BLOCKS_CALCULATE/BLOCK_M) partial histogram.
__global__ void sumupMCUDA(unsigned int* __restrict__ const Mmatrix) {
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ unsigned int block_M[BLOCK_M*Q];

	unsigned int i = 0;
	for (i=0; i<Q; i++) {
		block_M[Q*tid + i] = Mmatrix[Q*BLOCK_M*bid + Q*tid + i];
	}

	// Kirk&Hwu, Programming Massively Parallel Processors, p.100
	for (unsigned int stride=BLOCK_M/2; 0<stride; stride/=2) {
		__syncthreads();
		if (tid<stride) {
			for (i=0; i<Q; i++) {
				block_M[Q*tid + i] += block_M[Q*(tid+stride) + i];
			}
		}
	}
	// write result
	if (tid==0) {
		for (i=0; i<Q; i++) {
			Mmatrix[Q*bid + i] = block_M[i];
		}
	}
}


/***
 * Host functions
 ***/

static void update(const float temp, byte* const white, byte* const black) {
	dim3 dimBlock(TILE, TILE);
	dim3 dimGrid(FRAME/TILE, (FRAME/2)/TILE);
	assert(0.0f<=temp);
	assert(white!=NULL && black!=NULL);
	assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

	// white update, read from black
	updateCUDA<<<dimGrid, dimBlock>>>(temp, WHITE, white, black);
	CUT_CHECK_ERROR("Kernel updateCUDA(WHITE) execution failed");

	// black update, read from white
	updateCUDA<<<dimGrid, dimBlock>>>(temp, BLACK, black, white);
	CUT_CHECK_ERROR("Kernel updateCUDA(BLACK) execution failed");
}


static double calculate(const byte* const white, const byte* const black, unsigned int* const M_max) {
	double energy = 0.0;
	assert(white!=NULL && black!=NULL);
	assert(M_max!=NULL);

	const unsigned int numBlocksC = DIV_CEIL(L,TILE) * DIV_CEIL(L/2,TILE);

	unsigned int *E_matrix = NULL;
	size_t size = numBlocksC * sizeof(unsigned int);
	CUDA_SAFE_CALL(cudaMalloc((void**) &E_matrix, size));

	unsigned int *M_matrix = NULL;
	size = numBlocksC * Q * sizeof(unsigned int);
	CUDA_SAFE_CALL(cudaMalloc((void**) &M_matrix, size));

	// per block TILE*TILE accumulation of Energy and Magnetization
	assert((L/2)%TILE==0);
	dim3 dimBlockC(TILE, TILE);
	dim3 dimGridC(L/TILE, (L/2)/TILE);
	assert(Q<L/2); // at least one thread per magnetization level (Q)
	assert(dimBlockC.x*dimBlockC.y<=THREADS_PER_BLOCK);
	assert(dimGridC.x<=BLOCKS_PER_GRID && dimGridC.y<=BLOCKS_PER_GRID);
	calculateCUDA<<<dimGridC, dimBlockC>>>(white, black, E_matrix, M_matrix);
	CUT_CHECK_ERROR("Kernel calculateCUDA execution failed");

	// partially sum up the energy matrix in the GPU
	assert(numBlocksC%BLOCK_E==0);
	const unsigned int numBlocksE = DIV_CEIL(numBlocksC, BLOCK_E);
	dim3 dimBlockE(BLOCK_E);
	dim3 dimGridE(numBlocksE);
	assert(dimBlockE.x*dimBlockE.y<=THREADS_PER_BLOCK);
	assert(dimGridE.x<=BLOCKS_PER_GRID && dimGridE.y<=BLOCKS_PER_GRID);
	assert(BLOCK_E*sizeof(unsigned int)<=SHARED_PER_BLOCK); // shared memory per block
	sumupECUDA<<<dimGridE, dimBlockE>>>(E_matrix);
	CUT_CHECK_ERROR("Kernel sumupECUDA execution failed");
	// device to host of energy
	static unsigned int E_blocks[numBlocksE];
	CUDA_SAFE_CALL(cudaMemcpy(E_blocks, E_matrix, numBlocksE*sizeof(unsigned int),
				  cudaMemcpyDeviceToHost));
	// CPU accumulating the remaining numBlocksE energies
	unsigned long tmp_energy = 0L; // for L=32768, max energy is 32768*16384*4=2^31, almost overflowing unsigned int
	for (unsigned int i=0; i<numBlocksE; i++) {
		tmp_energy += E_blocks[i];
	}
	energy = -((double)tmp_energy);

	// partially sum up the magnetization matrix in the GPU
	assert(numBlocksC%BLOCK_M==0);
	const unsigned int numBlocksM = DIV_CEIL(numBlocksC, BLOCK_M);
	dim3 dimBlockM(BLOCK_M);
	dim3 dimGridM(numBlocksM);
	assert(dimBlockM.x*dimBlockM.y<=THREADS_PER_BLOCK);
	assert(dimGridM.x<=BLOCKS_PER_GRID && dimGridM.y<=BLOCKS_PER_GRID);
	assert(BLOCK_M*Q*sizeof(unsigned int)<=SHARED_PER_BLOCK); // shared memory per block
	sumupMCUDA<<<dimGridM, dimBlockM>>>(M_matrix);
	CUT_CHECK_ERROR("Kernel sumupMCUDA execution failed");
	// device to host of magnetization
	static unsigned int M_blocks[numBlocksM*Q];
	CUDA_SAFE_CALL(cudaMemcpy(M_blocks, M_matrix, numBlocksM*Q*sizeof(unsigned int),
				  cudaMemcpyDeviceToHost));
	// CPU accumulating the remaining Q*numBlocksM magnetizations
	unsigned int M[Q] = {0};
	for (unsigned int i=0; i<numBlocksM*Q; i++) {
		M[i%Q] += M_blocks[i];
	}

	// compute maximum magnetization
	*M_max = 0;
	for (unsigned int i=0; i<Q; i++) {
		*M_max = MAX(*M_max, M[i]);
	}

	CUDA_SAFE_CALL(cudaFree(E_matrix));
	CUDA_SAFE_CALL(cudaFree(M_matrix));

	return energy;
}


static void cycle(byte* const white, byte* const black,
		  const double min, const double max,
		  const double step, const unsigned int calc_step,
		  struct statpoint stats[]) {
	unsigned int index = 0;
	int modifier = 0;
	double temp = 0.0;

	assert(white!=NULL && black!=NULL);
	assert((0<step && min<=max) || (step<0 && max<=min));
	modifier = (0<step)?1:-1;

	for (index=0, temp=min; modifier*temp<=modifier*max;
	     index++, temp+=step) {

		// equilibrium phase
		for (unsigned int j=0; j<TRAN; j++) {
			update(temp, white, black);
		}

		// measurement phase
		unsigned int measurements = 0;
		double e=0.0, e2=0.0, e4=0.0, m=0.0, m2=0.0, m4=0.0;
		for (unsigned int j=0; j<TMAX; j++) {
			update(temp, white, black);
			if (j%calc_step==0) {
				double energy = 0.0, mag = 0.0;
				unsigned int M_max = 0;
				energy = calculate(white, black, &M_max);
				mag = (Q*M_max/(1.0*N) - 1) / (double)(Q-1);
				e  += energy;
				e2 += energy*energy;
				e4 += energy*energy*energy*energy;
				m  += mag;
				m2 += mag*mag;
				m4 += mag*mag*mag*mag;
				measurements++;
			}
		}
		assert(index<NPOINTS);
		stats[index].t = temp;
		stats[index].e += e/measurements;
		stats[index].e2 += e2/measurements;
		stats[index].e4 += e4/measurements;
		stats[index].m += m/measurements;
		stats[index].m2 += m2/measurements;
		stats[index].m4 += m4/measurements;
	}
}


static void sample(byte* const white, byte* const black, struct statpoint stat[]) {
	assert(white!=NULL && black!=NULL);

	// set the device matrix to 0
	const size_t size = L*L/2*sizeof(byte);
	CUDA_SAFE_CALL(cudaMemset(white, 0, size));
	CUDA_SAFE_CALL(cudaMemset(black, 0, size));

	// temperature increasing cycle
	cycle(white, black,
	      TEMP_MIN, TEMP_MAX, DELTA_TEMP, DELTA_T,
	      stat);
}


static int ConfigureRandomNumbers(void) {
	// Allocate memory for RNG's
	unsigned long long h_x[NUM_THREADS];
	unsigned int h_a[NUM_THREADS];
	unsigned long long seed = (unsigned long long) SEED;

	// Init RNG's
	int error = init_RNG(h_x, h_a, NUM_THREADS, SAFE_PRIMES_FILENAME, seed);

	if (!error) {
		size_t size_x = NUM_THREADS * sizeof(unsigned long long);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_x, h_x, size_x));

		size_t size_a = NUM_THREADS * sizeof(unsigned int);
		assert(size_a<size_x);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_a, h_a, size_a));
	}

	return error;
}


int main(void)
{
	// the lattice
	byte *white = NULL, *black = NULL;
	// the stats
	struct statpoint stat[NPOINTS] = { {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} };

	double secs = 0.0;
	struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};

	// parameters checking
	assert(2<=Q); // at least Ising
	assert(Q<(1<<(sizeof(byte)*8))); // do not overflow the representation
	assert(TEMP_MIN<=TEMP_MAX);
	assert(0<DELTA_T && DELTA_T<TMAX); // at least one calculate()
	assert(TMAX%DELTA_T==0); // take equidistant calculate()
	assert(L%2==0); // we can halve height
	assert(512<=FRAME); // TODO: get rid of this
	assert(__builtin_popcount(FRAME)==1); // FRAME=2^k
	assert(L%FRAME==0); // we can frame the grid
	assert(FRAME%2==0); // frames could be halved
	assert((FRAME/2)%TILE==0); // half-frames could be tiled
	assert((L*L/2)*4L<UINT_MAX); // max energy, that is all spins are the same, fits into a ulong

	// set the GPGPU computing device
	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE));

	const size_t size = L * L/2 * sizeof(byte);
	CUDA_SAFE_CALL(cudaMalloc((void**) &white, size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &black, size));

	// print header
	printf("# Q: %i\n", Q);
	printf("# L: %i\n", L);
	printf("# Number of Samples: %i\n", SAMPLES);
	printf("# Minimum Temperature: %f\n", TEMP_MIN);
	printf("# Maximum Temperature: %f\n", TEMP_MAX);
	printf("# Temperature Step: %.12f\n", DELTA_TEMP);
	printf("# Equilibration Time: %i\n", TRAN);
	printf("# Measurement Time: %i\n", TMAX);
	printf("# Data Acquiring Step: %i\n", DELTA_T);
	printf("# Number of Points: %i\n", NPOINTS);

	// start timer
	assert(gettimeofday(&start, NULL)==0);

	if (ConfigureRandomNumbers()) {
		return 1;
	}

	// stop timer
	assert(gettimeofday(&end, NULL)==0);
	assert(timeval_subtract(&elapsed, &end, &start)==0);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Configure RNG Time (sec): %lf\n", secs);

	// start timer
	assert(gettimeofday(&start, NULL)==0);

	for (unsigned int i=0; i<SAMPLES; i++) {
		sample(white, black, stat);
	}

	// stop timer
	CUDA_SAFE_CALL(cudaThreadSynchronize()); // ensure all threads are done
	assert(gettimeofday(&end, NULL)==0);
	assert(timeval_subtract(&elapsed, &end, &start)==0);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Total Simulation Time (sec): %lf\n", secs);

	printf("# Temp\tE\tE^2\tE^4\tM\tM^2\tM^4\n");
	for (unsigned int i=0; i<NPOINTS; i++) {
		printf ("%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",
			stat[i].t,
			stat[i].e/((double)N*SAMPLES),
			stat[i].e2/((double)N*N*SAMPLES),
			stat[i].e4/((double)N*N*N*N*SAMPLES),
			stat[i].m/SAMPLES,
			stat[i].m2/SAMPLES,
			stat[i].m4/SAMPLES);
	}

	return 0;
}


/*
 * http://www.gnu.org/software/libtool/manual/libc/Elapsed-Time.html
 * Subtract the `struct timeval' values X and Y,
 * storing the result in RESULT.
 * return 1 if the difference is negative, otherwise 0.
 */
static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y) {
	/* Perform the carry for the later subtraction by updating y. */
	if (x->tv_usec < y->tv_usec) {
		int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
		y->tv_usec -= 1000000 * nsec;
		y->tv_sec += nsec;
	}
	if (x->tv_usec - y->tv_usec > 1000000) {
		int nsec = (x->tv_usec - y->tv_usec) / 1000000;
		y->tv_usec += 1000000 * nsec;
		y->tv_sec -= nsec;
	}

	/* Compute the time remaining to wait. tv_usec is certainly positive. */
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_usec = x->tv_usec - y->tv_usec;

	/* Return 1 if result is negative. */
	return x->tv_sec < y->tv_sec;
}
