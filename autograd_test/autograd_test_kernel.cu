#include <torch/extension.h>

__global__ void add_kernel(float *a, float *b, float *c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000000; j++) {
            c[index] = c[index] + 0.0001f;
        }
    }
}

void add(torch::Tensor a, torch::Tensor b, torch::Tensor c, cudaStream_t stream) {
    int size = a.numel();
    //cudaStream_t stream;
    //cudaStreamCreate(&stream);
    
    add_kernel<<<(size + 255) / 256, 256, 20000, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);
    //cudaStreamSynchronize(stream);
}

