#include "common.cuh"

#define CUDA_SNAKE_BLOCK_SIZE 256

void ggml_cuda_op_snake(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
