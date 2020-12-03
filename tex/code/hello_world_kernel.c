__global__ void matrix_sum(
    float *A,
    float *B,
    float *C,
    int m,
    int n
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < m && y < n) {
        int ij = x + y*m; // column-major order
        C[ij] = A[ij] + B[ij];
    }
}

int main() {
    int no_of_blocks = 10, threads_per_block = 16;
    int m = 100, n = 50;
    float A[m][n], B[m][n], C[m][n];
    matrix_sum<<m,n>>(A, B, C, m, n);
}

