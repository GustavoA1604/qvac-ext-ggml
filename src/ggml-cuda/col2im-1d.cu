#include "col2im-1d.cuh"

// col2im_1d: scatter-add GEMM columns back to a 1D signal (GEMM-based
// conv_transpose_1d, ACE-Step Oobleck VAE).  Columns [K*OC, T_in] -> signal
// [T_out, OC], with T_out = (T_in - 1)*s0 + K - 2*p0.  Implemented as a gather:
// each output (t_out, oc) reads the (<= ceil(K/s0)) columns that land on it, so
// threads write disjoint outputs and no atomics are needed.  One thread per
// output element.  F32 only.
static __global__ void col2im_1d_f32_kernel(
        const float * col, float * dst,
        const int64_t T_out, const int64_t OC, const int64_t K, const int64_t T_in,
        const int64_t K_OC, const int s0, const int p0) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= T_out * OC) {
        return;
    }

    const int64_t t_out = gid % T_out;
    const int64_t oc    = gid / T_out;

    const int64_t t_abs = t_out + p0;   // position in the uncropped signal

    // Gather every (t_in, k) with t_in*s0 + k == t_abs, 0 <= k < K.
    int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;   // ceil((t_abs-K+1)/s0)
    if (t_in_min < 0) {
        t_in_min = 0;
    }
    int64_t t_in_max = t_abs / s0;
    if (t_in_max >= T_in) {
        t_in_max = T_in - 1;
    }

    float sum = 0.0f;
    for (int64_t t_in = t_in_min; t_in <= t_in_max; ++t_in) {
        const int64_t k = t_abs - t_in * s0;
        if (k >= 0 && k < K) {
            sum += col[(oc * K + k) + t_in * K_OC];
        }
    }

    dst[t_out + oc * T_out] = sum;
}

void ggml_cuda_op_col2im_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0]; // [K*OC, T_in]

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t OC = ggml_get_op_params_i32(dst, 1);
    const int32_t p0 = ggml_get_op_params_i32(dst, 2);

    const int64_t K_OC  = src->ne[0];
    const int64_t T_in  = src->ne[1];
    const int64_t K     = K_OC / OC;
    const int64_t T_out = dst->ne[0];

    const int64_t ne = T_out * OC;
    const int64_t num_blocks = (ne + CUDA_COL2IM_1D_BLOCK_SIZE - 1) / CUDA_COL2IM_1D_BLOCK_SIZE;
    col2im_1d_f32_kernel<<<num_blocks, CUDA_COL2IM_1D_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) src->data, (float *) dst->data,
        T_out, OC, K, T_in, K_OC, s0, p0);
}
