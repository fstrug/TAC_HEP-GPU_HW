#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <random>

using namespace std;

//const float A_val = 3.0f;
//const float B_val = 2.0f;

#define RADIUS 2
#define BLOCK_SIZE 32
const int DSIZE = 512;
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

void cpu_stencil_matrix(const int *A, int *out, int radius) {
    int result[DSIZE*DSIZE] = {};
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            if (i >= radius && i < DSIZE - radius && j >= radius && j < DSIZE - radius) {
                for (int k = -radius; k <= radius; k++) {
                    if (k == 0) {
                        result[i*DSIZE + j] += A[i*DSIZE + j];
                    } else {
                        result[i*DSIZE + j] += A[(i+k)*DSIZE + j];
                        result[i*DSIZE + j] += A[i*DSIZE + (j+k)];
                    }
                }
                out[i*DSIZE + j] = result[i*DSIZE + j];
            } 
            else {
                out[i*DSIZE + j] = A[i*DSIZE + j];
            }
        }
    }
}

__host__ void cpu_matrix_mult(const int *A, const int *B, int *C){
    for (int i=0; i < DSIZE; i++){
        for (int j=0; j < DSIZE; j++){
            for (int k=0; k < DSIZE; k++){
                C[i*DSIZE + j] += A[i*DSIZE + k] * B[k*DSIZE + j]; 
            }
        }
    } 
}

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


int compare_matrices(int *GPU, int *CPU){
	for (int i = 0; i < N + 2 * RADIUS; ++i) {
		for (int j = 0; j < N + 2 * RADIUS; ++j) {
			if (GPU[j+i*(N + 2 * RADIUS)] != CPU[j+i*(N + 2 * RADIUS)]){
				printf("Mismatch at index [%d,%d] \n", i, j);
				printf("GPU result: %d\nCPU result: %d\n",GPU[j+i*(N + 2 * RADIUS)], CPU[j+i*(N + 2 * RADIUS)]);
				return(-1);
			}

		}
	}
	printf("Matrices agree!\n");
	return(1);
}

__host__ void PRINTARRAY_2D(int *array, int size){
    for(int i = 0; i < size*size; i++) { 
        printf("%d ", array[i]);
        if ((i+1) % size == 0) printf("\n");
    }
	printf("\n");
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
	int *A_in, *A_stenciled, *A_stenciled_cpu; // host copies of A, A_out, A_out_cpu
	int *B_in, *B_stenciled, *B_stenciled_cpu; // host copies of B, B_out, B_out_cpu
	int *C, *C_cpu; // host copies of martix multiplication result

	// Device memory allocations
	int *dA_in, *dA_stenciled; // device copies of A, A_stenciled
	int *dB_in, *dB_stenciled; // device copies of B, B_stenciled
	int *dC; // device copies of martix multiplication result

	// Alloc space for host copies and setup values
	int size = (DSIZE)*(DSIZE) * sizeof(int);

	A_in = (int *)malloc(size);
	A_stenciled = (int *)malloc(size);
	A_stenciled_cpu = (int *)malloc(size);

	B_in = (int *)malloc(size);
	B_stenciled = (int *)malloc(size);
	B_stenciled_cpu = (int *)malloc(size);

	C = (int *)malloc(size);
	C_cpu = (int *)malloc(size);

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

	// Run on CPU
	cpu_stencil_matrix(A_in, A_stenciled_cpu, RADIUS);
	cpu_stencil_matrix(B_in, B_stenciled_cpu, RADIUS);


	////////////////////////////////////
	// Compare Matrices
	int codeA = compare_matrices(A_stenciled, A_stenciled_cpu);
	if (codeA == 1){
		printf("Stencil operation on matrix A agrees for GPU and CPU algorithms!\n");
	}
	else if (codeA == -1){
		printf("Stencil operation on matrix A does not agree for GPU and CPU algorithms!\n");
	}

	int codeB = compare_matrices(B_stenciled, B_stenciled_cpu);
	if (codeB == 1){
		printf("Stencil operation on matrix B agrees for GPU and CPU algorithms!\n");
	}
	else if (codeB == -1){
		printf("Stencil operation on matrix B does not agree for GPU and CPU algorithms!\n");
	}

	if (codeA == 1 && codeB == 1){
		printf("All stenciled matrices agree!\n\n");
	}

	////////////////////////////////////
	// Redefine grid/block size for matrix multiplication algorithm
	int gridSize_mult = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid_mult(gridSize_mult, gridSize_mult);
	dim3 block_mult(BLOCK_SIZE, BLOCK_SIZE);

	// Multiply matrices together
	t2 = clock();
	matrix_mul_gpu<<<grid_mult, block_mult>>>(dA_stenciled, dB_stenciled, dC, DSIZE);

	t3 = clock();
    t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
    printf ("Matrix multiplication done. Compute took %f seconds\n", t3sum);
	
	// Copy result Back
	cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

	// Perform on CPU
	cpu_matrix_mult(A_stenciled_cpu, B_stenciled_cpu, C_cpu);

	// Compare algorithms
	int codeC = compare_matrices(C, C_cpu);
	if (codeC == 1){
		printf("Matrix multiplication operation on matrices A and B agrees for GPU and CPU algorithms!\n\n");
	}
	else if (codeC == -1){
		printf("Matrix multiplication operation on matrices A and B does not agree for GPU and CPU algorithms!\n\n");
	}
	
	////////////////////////////////////
	// Memory Cleanup
	printf("Memory cleanup\n");
	// CPU memory
	free(A_in); free(A_stenciled); free(A_stenciled_cpu); 
	free(B_in); free(B_stenciled); free(B_stenciled_cpu);
	free(C); free(C_cpu);

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