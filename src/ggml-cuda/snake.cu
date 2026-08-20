#include "snake.cuh"

// Snake activation (ACE-Step Oobleck VAE): y = x + sin(a*x)^2 * inv_b, with
// per-channel a and inv_b.  x / dst are [T, C] contiguous; a and inv_b hold one
// F32 value per channel.  One thread per element; channel c = idx / T.
static __global__ void snake_f32_kernel(
        const float * x, const float * a, const float * inv_b, float * dst,
        const int64_t ne, const int64_t T) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ne) {
        return;
    }

    const int64_t c  = idx / T;
    const float   xi = x[idx];
    const float   s  = sinf(a[c] * xi);
    dst[idx] = xi + s * s * inv_b[c];
}

void ggml_cuda_op_snake(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * a     = dst->src[1];
    const ggml_tensor * inv_b = dst->src[2];

    GGML_ASSERT(x->type     == GGML_TYPE_F32);
    GGML_ASSERT(a->type     == GGML_TYPE_F32);
    GGML_ASSERT(inv_b->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne = ggml_nelements(dst);
    const int64_t T  = x->ne[0];

    const int64_t num_blocks = (ne + CUDA_SNAKE_BLOCK_SIZE - 1) / CUDA_SNAKE_BLOCK_SIZE;
    snake_f32_kernel<<<num_blocks, CUDA_SNAKE_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) a->data, (const float *) inv_b->data,
        (float *) dst->data, ne, T);
}
