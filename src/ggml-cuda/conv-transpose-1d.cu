#include "conv-transpose-1d.cuh"

struct conv_transpose_1d_tap_range {
    int first;
    int last_inclusive;
};

static __device__ __forceinline__ conv_transpose_1d_tap_range conv_transpose_1d_taps(
        const int out_pos, const int kernel_size, const int s0, const int input_size) {
    const int first          = out_pos >= kernel_size ? (out_pos - kernel_size) / s0 + 1 : 0;
    const int last_inclusive = min(out_pos / s0, input_size - 1);
    return { first, last_inclusive };
}

static __device__ __forceinline__ float conv_transpose_1d_accumulate_taps(
        const float * kernel_channel, const float * input_channel, const int out_pos,
        const int s0, const conv_transpose_1d_tap_range taps, float accumulator) {
    for (int i = taps.first; i <= taps.last_inclusive; i++) {
        accumulator += kernel_channel[out_pos - i*s0] * input_channel[i];
    }
    return accumulator;
}

static __device__ __forceinline__ float conv_transpose_1d_accumulate_channels(
        const float * src0, const float * src1, const int channels, const int out_index,
        const int out_pos, const int s0, const int kernel_size, const int kernel_channel_stride,
        const int input_channel_stride, const conv_transpose_1d_tap_range taps,
        float accumulator) {
    for (int c = 0; c < channels; c++) {
        const float * kernel_channel = src0 + kernel_channel_stride * c + out_index * kernel_size;
        const float * input_channel  = src1 + input_channel_stride * c;

        accumulator = conv_transpose_1d_accumulate_taps(
            kernel_channel, input_channel, out_pos, s0, taps, accumulator);
    }
    return accumulator;
}

static  __global__ void conv_transpose_1d_kernel(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;
    int idx       = global_index % dst_ne0;

    const conv_transpose_1d_tap_range taps =
        conv_transpose_1d_taps(idx, src0_ne0, s0, src1_ne0);

    dst[global_index] = conv_transpose_1d_accumulate_channels(
        src0, src1, src0_ne2, out_index, idx, s0, src0_ne0,
        src0_ne0 * src0_ne1, src1_ne0, taps, /*accumulator=*/0.0f);
    GGML_UNUSED_VARS(p0, d0, src0_ne3, src1_ne3, dst_ne3, src1_ne1, dst_ne1, src1_ne2, dst_ne2);
}

static void conv_transpose_1d_f32_f32_cuda(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst,
        cudaStream_t stream) {

    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_kernel<<<num_blocks,CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
        s0,p0,d0,output_size,
        src0_ne0, src0_ne1,  src0_ne2, src0_ne3,
        src1_ne0, src1_ne1,  src1_ne2, src1_ne3,
        dst_ne0,  dst_ne1,   dst_ne2,  dst_ne3,
        src0,src1, dst);
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    const int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_cuda(s0, p0, d0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        src0_d, src1_d, dst_d, stream);
}
