__global__ void matrix_sum(
    float *A,
    float *B,
    float *C,
    int rows,
    int cols
) {
    // blockDim is short for block dimension
    // blockDim.x == blockDim.y == 16 threads
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < cols && y < rows) {
        int ij = x + y*m; // row-major order
        C[ij] = A[ij] + B[ij];
    }
}

int main() {
    int rows = 32, cols = 48;
    float A[m][n], B[m][n], C[m][n];

    // initialization and cudaMemcpy
    // ...

    // dim3 is a 3d integer vector
    // dimensions omitted in the constructor
    // (e.g. z like here) are set to 1
    dim3 numBlocks(3, 2);
    dim3 numThreads(16, 16);
    matrix_sum<<blocks, threads>>(A, B, C, rows, cols);
}

