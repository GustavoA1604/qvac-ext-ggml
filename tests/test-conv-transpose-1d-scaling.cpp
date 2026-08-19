// Regression guard for the cost model of CONV_TRANSPOSE_1D.
//
// Each output element is reached by at most ceil(kernel/stride) input
// positions, so the cost per output element is a function of the kernel and
// the stride, not of how long the input is. An implementation that instead
// visits every input position for each output element still computes the
// right answer -- the extra positions contribute nothing -- so no correctness
// case can catch it, and a perf listing reports a number without failing.
//
// Two shapes differing only in input length pin this: identical kernel,
// stride and channels, input lengths a factor LENGTH_RATIO apart. Cost per
// output element stays flat when the taps bound the work and grows by
// LENGTH_RATIO when the whole input is walked, so the bound below sits an
// order of magnitude clear of both readings.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t KERNEL_SIZE   = 16;
constexpr int64_t STRIDE        = 4;
constexpr int64_t CHANNELS_IN   = 18;
constexpr int64_t CHANNELS_OUT  = 1;
// Both shapes have to saturate the device: at a few thousand outputs the
// launch overhead dominates the per-output cost and masks the difference.
constexpr int64_t SHORT_INPUT   = 16384;
constexpr int64_t LENGTH_RATIO  = 16;
constexpr int64_t LONG_INPUT    = SHORT_INPUT * LENGTH_RATIO;

// Flat is 1.0 and walking the whole input is LENGTH_RATIO (16), so this sits
// clear of both readings.
constexpr double MAX_COST_GROWTH = 4.0;

constexpr int TIMED_RUNS = 5;

struct run_result {
    double  nanoseconds_per_output;
    int64_t output_elements;
};

double time_one_run(ggml_backend_t backend, ggml_cgraph * graph) {
    const auto start = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count();
}

double fastest_run(ggml_backend_t backend, ggml_cgraph * graph) {
    ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);

    double fastest = time_one_run(backend, graph);
    for (int i = 1; i < TIMED_RUNS; i++) {
        const double elapsed = time_one_run(backend, graph);
        if (elapsed < fastest) {
            fastest = elapsed;
        }
    }
    return fastest;
}

run_result measure(ggml_backend_t backend, int64_t input_length) {
    ggml_init_params params = { size_t(32) * 1024 * 1024, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KERNEL_SIZE, CHANNELS_OUT, CHANNELS_IN);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_length, CHANNELS_IN);
    ggml_tensor * output = ggml_conv_transpose_1d(ctx, kernel, input, STRIDE, 0, 1);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> kernel_data(size_t(KERNEL_SIZE) * CHANNELS_OUT * CHANNELS_IN, 0.01f);
    std::vector<float> input_data(size_t(input_length) * CHANNELS_IN, 0.02f);
    ggml_backend_tensor_set(kernel, kernel_data.data(), 0, kernel_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    const double  nanoseconds = fastest_run(backend, graph);
    const int64_t elements    = ggml_nelements(output);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return { nanoseconds / double(elements), elements };
}

bool check_backend(ggml_backend_t backend) {
    const char * name = ggml_backend_name(backend);

    const run_result small = measure(backend, SHORT_INPUT);
    const run_result large = measure(backend, LONG_INPUT);
    const double     growth = large.nanoseconds_per_output / small.nanoseconds_per_output;

    printf("  %s: %lld outputs at %.4f ns each, %lld outputs at %.4f ns each -> growth %.2fx\n",
           name,
           (long long) small.output_elements, small.nanoseconds_per_output,
           (long long) large.output_elements, large.nanoseconds_per_output,
           growth);

    if (growth > MAX_COST_GROWTH) {
        printf("  %s: FAIL cost per output element grew %.2fx when the input grew %lldx; "
               "the work is not bounded by the contributing taps\n",
               name, growth, (long long) LENGTH_RATIO);
        return false;
    }
    return true;
}

} // namespace

int main() {
    ggml_backend_load_all();

    bool checked_any = false;
    bool all_passed  = true;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }

        checked_any = true;
        if (!check_backend(backend)) {
            all_passed = false;
        }
        ggml_backend_free(backend);
    }

    if (!checked_any) {
        printf("no non-CPU backend registered; nothing to check\n");
        return 0;
    }
    if (!all_passed) {
        return 1;
    }
    printf("OK\n");
    return 0;
}
