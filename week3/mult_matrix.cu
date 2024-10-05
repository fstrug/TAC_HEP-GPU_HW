#include <stdio.h>
#include <time.h>

const int DSIZE = 3;
//const float A_val = 3.0f;
//const float B_val = 2.0f;

const int block_size_X = 32;
const int block_size_Y = 32;

const int gridDim_X = (DSIZE + block_size_X - 1)/block_size_X;
const int gridDim_Y = (DSIZE + block_size_Y - 1)/block_size_Y;

// error checking macro
#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0)

// Square matrix multiplication on CPU : C = A * B
void matrix_mul_cpu(const float *A, const float *B, float *C, int size) {
  //FIXME:
  //
  for (int i=0; i < size; i++){
    for (int j = 0; j < size; j++){
        float temp = 0;
        for (int k = 0; k < size; k++){
            temp += A[j*size + k] * B[k*size + i];
        }
        C[size*j + i] = temp;
    }   
  }
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int size) {

    //FIXME:
    // create thread x index
    // create thread y index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;;
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

int main() {

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // These are used for timing
    clock_t t0, t1, t2, t3;
    double t1sum=0.0;
    double t2sum=0.0;
    double t3sum=0.0;

    // start timing
    t0 = clock();

    // N*N matrices defined in 1 dimention
    // If you prefer to do this in 2-dimentions cupdate accordingly
    h_A = new float[DSIZE*DSIZE];
    h_B = new float[DSIZE*DSIZE];
    h_C = new float[DSIZE*DSIZE];
    for (int i = 0; i < DSIZE*DSIZE; i++){
        h_A[i] = i;
        h_B[i] = i;
        h_C[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory and copy input data from host to device
    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));

    //FIXME:Add all other allocations and copies from host to device
    cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));

    cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    cudaCheckErrors(err);

    // Launch kernel
    // Specify the block and grid dimentions 
    dim3 blockSize(block_size_X, block_size_Y);   //FIXME
    dim3 gridSize(gridDim_X,gridDim_Y);           //FIXME
    matrix_mul_gpu<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE);

    
    // Copy results back to host
    cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < DSIZE; i++) {
        printf("\n");
        for (int j = 0; j < DSIZE; j++) {
            int entry = DSIZE*i + j;
            printf("%f ",h_C[entry]); // avoids filling CL
        }
    }
    printf("\n");
    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    // FIXME
    // Excecute and time the cpu matrix multiplication function
    matrix_mul_cpu(h_A, h_B, h_C, DSIZE);
    // CPU timing
    t3 = clock();
    t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t3sum);




    // FIXME
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);     

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
