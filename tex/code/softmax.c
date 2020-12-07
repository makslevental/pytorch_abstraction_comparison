template <typename dtype>
__global__ void softmax_loss_kernel(
    dtype *reduced_loss,
    dtype *predict,
    dtype *target,
    dtype *workspace,
    int batch_size,
    int num_outputs) {

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ dtype s_data[];
    dtype loss = 0.f;

    for (int c = 0; c < num_outputs; c++)
        loss -= (
            target[thread_id * num_outputs + c] *
            logf(predict[thread_id * num_outputs + c]
        );
    workspace[thread_id] = loss;
    // ...
}