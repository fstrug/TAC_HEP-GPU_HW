#include <iostream>
#include <vector>
#include <cstdio>
#include <random>

#include <alpaka/alpaka.hpp>
#include "WorkDiv.hpp"

#include "config.h"


const int RADIUS = 1;
const int BLOCK_SIZE = 32;
const int DSIZE = 512;
const int N = DSIZE-2*RADIUS;

struct apply_stencil{
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  T const* __restrict__ in, 
                                  T* out, 
                                  int DSIZE,
                                  Vec2D size,
                                  int R) const{
    // Calculate indices for row and column
    for (auto ndindex : alpaka::uniformElementsND(acc, size)){
        auto col = ndindex[0] + R;
        auto row = ndindex[1] + R;
        auto index = row * DSIZE + col;
        
        // Bounds check
        if (col >= DSIZE - R || row >= DSIZE - R) {
            return;
        }

        // Apply stencil calculation
        int result = 0;
        for (int i = -R; i <= R; ++i) {
            if (i == 0) {
                result += in[row * DSIZE + col];
            } else {
                result += in[(row + i) * DSIZE + col];
                result += in[row * DSIZE + (col + i)];
            }
        }

        out[index] = result;
    }
    }
};

struct matrix_mult{
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  T const* __restrict__ A, 
                                  T const* __restrict__ B,
                                  T* C, 
                                  Vec2D size) const{

    for (auto ndindex : alpaka::uniformElementsND(acc, size)){
        auto idx = ndindex[0];
        auto idy = ndindex[1];
        auto index = idy * size[1] + idx;
        int temp = 0;
        for (int i = 0; i < DSIZE; i++){
            temp += A[idy*size[1] + i] * B[i*size[1] + idx];
        }
        C[index] = temp;
        }
    }
};

int main(void){
    // initialise the accelerator platform
    Platform platform;

    // require at least one device
    std::uint32_t n = alpaka::getDevCount(platform);
    if (n == 0) {
        exit(EXIT_FAILURE);
    }

    // use the single host device
    HostPlatform host_platform;
    Host host = alpaka::getDevByIdx(host_platform, 0u);
    std::cout << "Host:   " << alpaka::getName(host) << '\n';

    // use the first device
    Device device = alpaka::getDevByIdx(platform, 0u);
    std::cout << "Device: " << alpaka::getName(device) << '\n';

    // run the test the given device
    auto queue = Queue{device};

    // Alloc space for host copies and setup values
	uint32_t size = (DSIZE)*(DSIZE);

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto A = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
    auto A_stenciled_h = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
    auto B = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
    auto B_stenciled_h = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);

    auto C = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);

    
	////////////////////////////////////
	// random number generator
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 9); // define the range

    // fill the input buffers with random data
    for (uint32_t i = 0; i < size; ++i) {
        int random_val = distr(gen);
        A[i] = random_val;
        A_stenciled_h[i] = random_val;
    }

        for (uint32_t i = 0; i < size; ++i) {
        int random_val = distr(gen);
        B[i] = random_val;
        B_stenciled_h[i] = random_val;
    }

    // allocate input and output buffers on the device
    auto A_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
    auto B_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);

    auto A_stenciled_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
    auto B_stenciled_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);

    auto C_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::memcpy(queue, A_d, A);
    alpaka::memcpy(queue, B_d, B);
    alpaka::memcpy(queue, C_d, C);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::memcpy(queue, A_stenciled_d, A_stenciled_h);
    alpaka::memcpy(queue, B_stenciled_d, B_stenciled_h);

    ////////////////////////////
    // Stencil
    int gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
    auto div = makeWorkDiv<Acc2D>({gridSize, gridSize}, {BLOCK_SIZE, BLOCK_SIZE});
    std::cout << "Running apply_stencil with vector indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n\n";
    
    constexpr Vec2D stencil_size = {N, N};
    alpaka::exec<Acc2D>(
        queue, div, apply_stencil{}, A_d.data(), A_stenciled_d.data(), DSIZE, stencil_size, RADIUS);
    alpaka::exec<Acc2D>(
        queue, div, apply_stencil{}, B_d.data(), B_stenciled_d.data(), DSIZE, stencil_size, RADIUS);

    ////////////////////////////
    // Matrix multiplication
    int gridSize_mult = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
    auto div_mult = makeWorkDiv<Acc2D>({gridSize_mult, gridSize_mult}, {BLOCK_SIZE, BLOCK_SIZE});
    std::cout << "Running matrix_mult with vector indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div_mult) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div_mult) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div_mult) << " elements...\n\n";
    
    constexpr Vec2D matrix_mult_size = {DSIZE, DSIZE};
    alpaka::exec<Acc2D>(
        queue, div, matrix_mult{}, A_stenciled_d.data(), B_stenciled_d.data(), C_d.data(), matrix_mult_size);
    
    alpaka::memcpy(queue, A_stenciled_h, A_stenciled_d);
    alpaka::memcpy(queue, B_stenciled_h, B_stenciled_d);
    alpaka::memcpy(queue, C, C_d);
    alpaka::wait(queue);


    printf("Stencil and multiplication performed!\n");
    return(0);
}
