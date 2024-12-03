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
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	// Only applying stencil on (DSIZE - R)**2 elements
    // Adjust index to match input matrix
	row += R;
    col += R;

	// Check that thread is in bounds
	if (col >= DSIZE - R || row >= DSIZE - R){ 
		return;
	}


    // Result will store stencil result
    int result = 0;
    for (int i = -R; i <= R; i++){
        if (i == 0){
            result += in[row*DSIZE + col];
        }
        else{
            result += in[(row + i)*DSIZE + col];
            result += in[row*DSIZE + (col+i)];
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

	// Device memory allocations
	int *dA_in, *dA_stenciled; // device copies of A, A_stenciled
	int *dB_in, *dB_stenciled; // device copies of B, B_stenciled
	int *dC; // device copies of martix multiplication result

	// Alloc space for host copies and setup values
	int size = (DSIZE)*(DSIZE) * sizeof(int);

	A_in = (int *)malloc(size);
	A_stenciled = (int *)malloc(size);

	B_in = (int *)malloc(size);
	B_stenciled = (int *)malloc(size);

	C = (int *)malloc(size);

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
	// Alloc space for device copies
	cudaMalloc((void **)&dA_in, size);
    cudaMalloc((void **)&dA_stenciled, size);

	cudaMalloc((void **)&dB_in, size);
    cudaMalloc((void **)&dB_stenciled, size);

	cudaMalloc((void **)&dC, size);



	////////////////////////////////////
	// Copy to device
	cudaMemcpy(dA_in, A_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_stenciled, A_stenciled, size, cudaMemcpyHostToDevice);

	cudaMemcpy(dB_in, B_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB_stenciled, B_stenciled, size, cudaMemcpyHostToDevice);
    

	////////////////////////////////////
	// Launch apply_stencil() kernel on GPU
	int gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	
	t0 = clock();
	// Launch the kernel 
	apply_stencil<<<grid,block>>>(dA_in, dA_stenciled, DSIZE, N, RADIUS);
	apply_stencil<<<grid,block>>>(dB_in, dB_stenciled, DSIZE, N, RADIUS);
	cudaDeviceSynchronize();

	// Stencil Timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf ("Stenciling done. Compute took %f seconds\n", t1sum);

	// Copy result back to host
    cudaMemcpy(A_stenciled, dA_stenciled, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(B_stenciled, dB_stenciled, size, cudaMemcpyDeviceToHost);

	////////////////////////////////////
	// Redefine grid/block size for matrix multiplication algorithm
	int gridSize_mult = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid_mult(gridSize_mult, gridSize_mult);
	dim3 block_mult(BLOCK_SIZE, BLOCK_SIZE);

	// Multiply matrices together
	t2 = clock();
	matrix_mul_gpu<<<grid_mult, block_mult>>>(dA_stenciled, dB_stenciled, dC, DSIZE);
	cudaDeviceSynchronize();
	t3 = clock();
    t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
    printf ("Matrix multiplication done. Compute took %f seconds\n", t3sum);
	
	// Copy result Back
	cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

	////////////////////////////////////
	// Memory Cleanup
	printf("Memory cleanup\n");
	// CPU memory
	free(A_in); free(A_stenciled);
	free(B_in); free(B_stenciled);
	free(C);

	// GPU memory
	cudaFree(dA_in); cudaFree(dA_stenciled);
	cudaFree(dB_in); cudaFree(dB_stenciled);
	cudaFree(dC);

	printf("Exit\n\n");

	tfinish = clock();
	ttotal = ((double)(tfinish-tstart))/CLOCKS_PER_SEC;
	printf ("Total compute time took %f seconds\n", ttotal);

	return 0;
}