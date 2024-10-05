#include <stdio.h>
#include <assert.h>

const int DSIZE_X = 256;
const int DSIZE_Y = 256;

// Max block size is 1024
// Need to divide up grid
const int block_size_X = 32;
const int block_size_Y = 32;



const int gridDim_X = (DSIZE_X + block_size_X - 1)/block_size_X;
const int gridDim_Y = (DSIZE_Y + block_size_Y - 1)/block_size_Y;

__global__ void add_matrix(float *A, float *B, float *C, const int v_size_x, const int v_size_y)
    {
    //FIXME:
    // Express in terms of threads and blocks
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add the two matrices - make sure you are not out of range
    if (idx <  v_size_x && idy < v_size_y )
        C[v_size_x*idy + idx] =  A[v_size_x*idy + idx] + B[v_size_x*idy + idx];

}

int main()
{
    printf("Main started. \n");
    // Create and allocate memory for host and device pointers 
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE_X*DSIZE_Y];
    h_B = new float[DSIZE_X*DSIZE_Y];
    h_C = new float[DSIZE_X*DSIZE_Y];

    int N_entries = DSIZE_X * DSIZE_Y;
    cudaMalloc(&d_A, N_entries*sizeof(float));
    cudaMalloc(&d_B, N_entries*sizeof(float));
    cudaMalloc(&d_C, N_entries*sizeof(float));
    
    printf("Memory allocated!");
    // Fill in the matrices
    // FIXME
    for (int i = 0; i < DSIZE_Y; i++) {
        for (int j = 0; j < DSIZE_X; j++) {
            int entry = DSIZE_X*i + j;
            h_A[entry] = rand()/(float)RAND_MAX;
            h_B[entry] = rand()/(float)RAND_MAX;
        }
    }
    printf("Matrices filled. \n");

    // Copy from host to device
    cudaMemcpy(d_A, h_A, N_entries*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N_entries*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    // dim3 is a built in CUDA type that allows you to define the block 
    // size and grid size in more than 1 dimentions
    // Syntax : dim3(Nx,Ny,Nz)
    dim3 blockSize(block_size_X, block_size_Y); 
    dim3 gridSize(gridDim_X,gridDim_Y); 
    
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE_X, DSIZE_Y);
    cudaDeviceSynchronize();

    // Copy back to host 
    cudaMemcpy(h_C, d_C, N_entries*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make the addition was succesfull
    for (int i = 0; i < DSIZE_Y; i++) {
        printf("\n");
        printf("Entry [%d, 0]: ", i);
        printf("%f + %f = %f", h_A[i], h_B[i], h_C[i]);
        printf("\n");
        for (int j = 0; j < DSIZE_X; j++) {
            int entry = DSIZE_X*i + j;
            assert((h_A[entry] + h_B[entry]) == h_C[entry]); // avoids filling CL
                                                             // keeping print as per hw instruction
        }
    }
    printf("\n");

    // Free the memory
    free(h_A);
    free(h_B);
    free(h_C);     

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}