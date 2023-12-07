#include <torch/extension.h>
#include <cuda_runtime.h>

void add(torch::Tensor a, torch::Tensor b, torch::Tensor c, cudaStream_t stream);

void add_interface(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    add(a, b, c, stream);
    cudaStreamSynchronize(stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_interface, "Add two tensors");
}

