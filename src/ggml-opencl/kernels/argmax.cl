#define ARGMAX_WG 256

// One workgroup reduces one row. Ties resolve to the largest index, matching
// ggml_vec_argmax_f32, which reassigns whenever the running max equals the element.
kernel void kernel_argmax_f32(
        global void * src0,
        ulong offset0,
        global void * dst,
        ulong offsetd,
        int ne00,
        ulong nb01
) {
    local float shmax[ARGMAX_WG];
    local int   shidx[ARGMAX_WG];

    const int row = get_group_id(0);
    const int lid = get_local_id(0);

    global const float * src = (global const float *)((global char *)src0 + offset0 + (ulong)row*nb01);
    global int * out = (global int *)((global char *)dst + offsetd);

    float best = -INFINITY;
    int   besti = 0;
    for (int i = lid; i < ne00; i += ARGMAX_WG) {
        const float v = src[i];
        if (v >= best) {
            best  = v;
            besti = i;
        }
    }

    shmax[lid] = best;
    shidx[lid] = besti;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = ARGMAX_WG/2; s > 0; s >>= 1) {
        if (lid < s) {
            const float o  = shmax[lid + s];
            const int   oi = shidx[lid + s];
            if (o > shmax[lid] || (o == shmax[lid] && oi > shidx[lid])) {
                shmax[lid] = o;
                shidx[lid] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[row] = shidx[0];
    }
}
