#include <stdio.h>


const int DSIZE = 40960;
const int block_size = 256;
const int grid_size = DSIZE/block_size;


__global__ void vector_swap(float *A, float *B,const int v_size){

    //FIXME:
    // Express the vector index in terms of threads and blocks
    int idx =  threadIdx.x + blockDim.x * blockIdx.x;
    
    // Swap the vector elements - make sure you are not out of range
    if (idx < v_size){
       float temp_A = A[idx];
       A[idx] = B[idx];
       B[idx] = temp_A;
    }
}


int main() {
    float *h_A, *h_B, *d_A, *d_B;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }


    // Allocate memory for host and device pointers 
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    
    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Check vector elements before swap operation
    printf("Vector elements before swap. \n");
    printf("Vector A Element 0: %f, Element 40959: %f \n", h_A[0], h_A[10]);
    printf("Vector B Element 0: %f, Element 40959: %f \n", h_B[0], h_B[10]);
    
    // Launch the kernel
    vector_swap<<<grid_size, block_size>>>(d_A, d_B, DSIZE);
    cudaDeviceSynchronize();
    
    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print and check some elements to make sure swapping was successfull
    printf("Vector elements after swap. \n");
    printf("Vector A Element 0: %f, Element 40959: %f \n", h_A[0], h_A[10]);
    printf("Vector B Element 0: %f, Element 40959: %f \n", h_B[0], h_B[10]);
    
    // Free the memory 
    free(h_A);
    free(h_B);

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
