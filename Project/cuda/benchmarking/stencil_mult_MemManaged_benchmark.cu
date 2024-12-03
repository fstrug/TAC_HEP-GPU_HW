#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <random>

using namespace std;

//const float A_val = 3.0f;
//const float B_val = 2.0f;

#define RADIUS 2
#define BLOCK_SIZE 32
const int DSIZE = 8192;
const int N = DSIZE-2*RADIUS;

// error checking macro
#define cudaCheckErrors()                                   \
do {                                                        \
	cudaError_t __err = cudaGetLastError();                 \
	if (__err != cudaSuccess) {                             \
		fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
			    cudaGetErrorString(__err),                  \
				__FILE__, __LINE__);                        \
		fprintf(stderr, "*** FAILED - ABORTING\n");         \
		exit(1);                                            \
	}                                                       \
} while (0)

// Stencil Matrix
__global__ void apply_stencil(const int *in, int *out, const int DSIZE, int N, int R){
	
    // Get thread index
	__shared__ int temp[(BLOCK_SIZE + 2 * RADIUS)*(BLOCK_SIZE + 2 * RADIUS)];

    int size_temp = (BLOCK_SIZE + 2 * RADIUS);
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

	int lindex_col = threadIdx.x + R;
	int lindex_row = threadIdx.y + R;

	// Only applying stencil on (DSIZE - R)**2 elements
    // Adjust index to match input matrix
	row += R;
    col += R;

	// Read input elements into shared memory
	temp[lindex_row*size_temp + lindex_col] = in[row*DSIZE + col];

	if (threadIdx.x < RADIUS) {
		temp[size_temp*lindex_row + (lindex_col - RADIUS)] = in[DSIZE*row + (col - RADIUS)];
        temp[size_temp*lindex_row + (lindex_col + BLOCK_SIZE)] = in[DSIZE*row + (col + BLOCK_SIZE)];
	}

	if (threadIdx.y < RADIUS ) {
		temp[size_temp*(lindex_row - RADIUS) + lindex_col] = in[DSIZE*(row - RADIUS) + col ];
        temp[size_temp*(lindex_row + BLOCK_SIZE) + lindex_col] = in[DSIZE*(row + BLOCK_SIZE) + col ];
	}
	__syncthreads();

	// Check that thread is in bounds
	// Will have some idle threads in blocks on edge of grid
	// Block size will probably not fit evenly into matrix of size (DSIZE - 2*R)**2
	if (col >= DSIZE - R || row >= DSIZE - R){ 
		return;
	}


    // Result will store stencil result
    int result = 0;
    for (int i = -R; i <= R; i++){
        if (i == 0){
            result += temp[lindex_row*size_temp + lindex_col];
        }
        else{
            result += temp[(lindex_row + i)*size_temp + lindex_col];
            result += temp[lindex_row*size_temp + (lindex_col+i)];
        }
    }
    out[row*DSIZE + col] = result;

}


// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const int *A, const int *B, int *C, int size) {
    // create thread x index
    // create thread y index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) {
        float temp = 0;
        for (int i = 0; i < size; i++){
            temp += A[idy*size + i] * B[i*size + idx];
        }
        C[idy*size+idx] = temp;                    
    }
}


int main(void) {
	clock_t tstart, tfinish;
	double ttotal = 0.0;
	tstart = clock();
	printf("\nRunning. . .\n\n");
	// Timing
    clock_t t0, t1, t2, t3;
    double t1sum=0.0;
    double t3sum=0.0;

	///////////////////////////////////
	// Host memory allocations
	int *A_in, *A_stenciled; // host copies of A, A_out, A_out_cpu
	int *B_in, *B_stenciled; // host copies of B, B_out, B_out_cpu
	int *C; // host copies of martix multiplication result

	// Alloc space for host copies and setup values
	int size = (DSIZE)*(DSIZE) * sizeof(int);

	cudaMallocManaged(&A_in, size);
    cudaMallocManaged(&A_stenciled, size);

	cudaMallocManaged(&B_in, size);
    cudaMallocManaged(&B_stenciled, size);

	cudaMallocManaged(&C, size);

	////////////////////////////////////
	// random number generator
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 10); // define the range

    // Fill the arrays with random values
    for (int i = 0; i < DSIZE*DSIZE; i++) {
		int random_val = distr(gen);
        A_in[i] = random_val;
        A_stenciled[i] = random_val;
    }

	for (int i = 0; i < DSIZE*DSIZE; i++) {
		int random_val = distr(gen);
        B_in[i] = random_val;
        B_stenciled[i] = random_val;
    }
	

	////////////////////////////////////
	// Launch apply_stencil() kernel on GPU
	int gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	
	t0 = clock();
	// Launch the kernel 
	apply_stencil<<<grid,block>>>(A_in, A_stenciled, DSIZE, N, RADIUS);
	apply_stencil<<<grid,block>>>(B_in, B_stenciled, DSIZE, N, RADIUS);
	cudaDeviceSynchronize();

	// Stencil Timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf ("Stenciling done. Compute took %f seconds\n", t1sum);

	////////////////////////////////////
	// Redefine grid/block size for matrix multiplication algorithm
	int gridSize_mult = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid_mult(gridSize_mult, gridSize_mult);
	dim3 block_mult(BLOCK_SIZE, BLOCK_SIZE);

	// Multiply matrices together
	t2 = clock();
	matrix_mul_gpu<<<grid_mult, block_mult>>>(A_stenciled, B_stenciled, C, DSIZE);
	cudaDeviceSynchronize();
	t3 = clock();
    t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
    printf ("Matrix multiplication done. Compute took %f seconds\n", t3sum);

	////////////////////////////////////
	// Memory Cleanup
	printf("Memory cleanup\n");
	// CPU memory
	cudaFree(A_in); cudaFree(A_stenciled);
    cudaFree(B_in); cudaFree(B_stenciled);
    cudaFree(C);

	printf("Exit\n\n");

	tfinish = clock();
	ttotal = ((double)(tfinish-tstart))/CLOCKS_PER_SEC;
	printf ("Total compute time took %f seconds\n", ttotal);

	return 0;
}