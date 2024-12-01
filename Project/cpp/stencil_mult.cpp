#include <stdio.h>
#include <math.h>
#include <random>

const int DSIZE = 512;

#define PRINTARRAY_FLAT(array, length) \
for(int i = 0; i < length; i++) \
    printf("%d ", array[i]);

#define PRINTARRAY_2D(array, length) \
for(int i = 0; i < length; i++) { \
    printf("%d ", array[i]); \
    if ((i+1) % DSIZE == 0) printf("\n"); \
}

void stencil_matrix(const int *A, int *out, int radius) {
    int result = 0;
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            if (i >= radius && i < DSIZE - radius && j >= radius && j < DSIZE - radius) {
                for (int k = -radius; k <= radius; k++) {
                    if (k == 0) {
                        result += A[i*DSIZE + j];
                    } else {
                        result += A[(i+k)*DSIZE + j];
                        result += A[i*DSIZE + (j+k)];
                    }
                }
                out[i*DSIZE + j] = result;
            } 
            else {
                out[i*DSIZE + j] = A[i*DSIZE + j];
            }
        }
    }
}

void matrix_mult(const int *A, const int *B, int *C){
    for (int i=0; i < DSIZE; i++){
        for (int j=0; j < DSIZE; j++){
            for (int k=0; k < DSIZE; k++){
                C[i*DSIZE + j] += A[i*DSIZE + k] * B[k*DSIZE + j]; 
            }
        }
    } 
}

int main() {
    int A[DSIZE*DSIZE] = {};
    int A_stenciled[DSIZE*DSIZE] = {};
    int B[DSIZE*DSIZE] = {};
    int B_stenciled[DSIZE*DSIZE] = {};
    int C[DSIZE*DSIZE] = {};

    // random number generator
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 10); // define the range

    // Fill the arrays with random values
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        A[i] = distr(gen);
        B[i] = distr(gen);
    }

    stencil_matrix(A, A_stenciled, 1);
    stencil_matrix(B, B_stenciled, 1);
    // Print the arrays
    // printf("\n");
    // PRINTARRAY_2D(A, DSIZE*DSIZE);
    // printf("\n");
    // PRINTARRAY_2D(A_stenciled, DSIZE*DSIZE);
    // printf("\n");

    // //
    matrix_mult(A_stenciled, B_stenciled, C);

    // printf("\n");
    // PRINTARRAY_2D(A_stenciled, DSIZE*DSIZE);
    // printf("\n");
    // PRINTARRAY_2D(B_stenciled, DSIZE*DSIZE);
    // printf("\n");
    // PRINTARRAY_2D(C, DSIZE*DSIZE);
    // printf("\n");


    return 0;
}
