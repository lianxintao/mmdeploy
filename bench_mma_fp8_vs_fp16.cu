// Microbenchmark: raw tensor-core throughput of FP8 vs FP16 vs BF16 mma.sync.
//
//   FP8 : mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32  (2*16*8*32 = 8192 FLOP/inst)
//   FP16: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32    (2*16*8*16 = 4096 FLOP/inst)
//   BF16: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  (2*16*8*16 = 4096 FLOP/inst)
//
// Operands stay in registers; NACC independent accumulators give ILP so we
// measure issue/compute THROUGHPUT, not single-MMA latency. mma.sync is a
// warp-level op, so one asm = one MMA per warp (32 threads cooperate).
//
// The NACC sweep is the control: throughput rises with ILP until the pipeline
// is saturated, then plateaus. If the FP8/FP16 ratio holds ~2x at the plateau,
// the 2x is real (not a latency artifact at low ILP).
//
// NOTE: run on the local GPU. On SM120 this measures the legacy mma.sync path,
// which is NOT the architecture's tcgen05 peak; the FP8:FP16:BF16 *ratio* is the
// point, and it transfers to Ada (same K-ratio, same issue rate).
//
// Build:  nvcc -arch=native -O3 -o /tmp/bench_mma scripts/bench_mma_fp8_vs_fp16.cu
// Run:    /tmp/bench_mma

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while(0)

enum Kind { KFP8 = 0, KFP16 = 1, KBF16 = 2 };

__device__ __forceinline__ void mma_fp8(
        float &c0, float &c1, float &c2, float &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_fp16(
        float &c0, float &c1, float &c2, float &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(
        float &c0, float &c1, float &c2, float &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int ITERS, int NACC, int KIND>
__global__ void bench(const uint32_t* __restrict__ in, float* __restrict__ out) {
    uint32_t a0 = in[0], a1 = in[1], a2 = in[2], a3 = in[3], b0 = in[4], b1 = in[5];
    float c[NACC][4];
    #pragma unroll
    for (int j = 0; j < NACC; ++j) { c[j][0] = in[6]; c[j][1] = in[7]; c[j][2] = 0.f; c[j][3] = 0.f; }

    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int j = 0; j < NACC; ++j) {
            if (KIND == KFP8)       mma_fp8 (c[j][0], c[j][1], c[j][2], c[j][3], a0, a1, a2, a3, b0, b1);
            else if (KIND == KFP16) mma_fp16(c[j][0], c[j][1], c[j][2], c[j][3], a0, a1, a2, a3, b0, b1);
            else                    mma_bf16(c[j][0], c[j][1], c[j][2], c[j][3], a0, a1, a2, a3, b0, b1);
        }
    }

    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < NACC; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    if (s == -12345.0f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // keep live, never stores in practice
}

template <int KIND, int NACC>
double run(int blocks, int threads, int reps, const uint32_t* d_in, float* d_out) {
    constexpr int ITERS = 16384;
    const double flop_per_mma = (KIND == KFP8) ? (2.0 * 16 * 8 * 32) : (2.0 * 16 * 8 * 16);

    bench<ITERS, NACC, KIND><<<blocks, threads>>>(d_in, d_out);  // warmup
    cudaDeviceSynchronize();

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int r = 0; r < reps; ++r)
        bench<ITERS, NACC, KIND><<<blocks, threads>>>(d_in, d_out);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);

    double warps = double(blocks) * threads / 32.0;
    double mmas  = warps * ITERS * double(NACC) * reps;
    return mmas * flop_per_mma / (ms / 1e3) / 1e12;   // TFLOPS
}

template <int NACC>
void sweep_row(int blocks, int threads, int reps, const uint32_t* d_in, float* d_out) {
    double fp8  = run<KFP8 , NACC>(blocks, threads, reps, d_in, d_out);
    double fp16 = run<KFP16, NACC>(blocks, threads, reps, d_in, d_out);
    double bf16 = run<KBF16, NACC>(blocks, threads, reps, d_in, d_out);
    printf("  %4d   %9.1f   %9.1f   %9.1f      %5.2fx     %5.2fx\n",
           NACC, fp8, fp16, bf16, fp8 / fp16, fp8 / bf16);
}

int main() {
    int dev = 0; cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, dev));
    printf("GPU: %s  (sm_%d%d, %d SMs)\n\n", p.name, p.major, p.minor, p.multiProcessorCount);

    const int threads = 256;
    const int blocks  = p.multiProcessorCount * 8;   // saturate occupancy
    const int reps    = 10;

    uint32_t h_in[8];
    for (int i = 0; i < 6; ++i) h_in[i] = 0x3C003C00u;   // arbitrary nonzero operand bits
    h_in[6] = 0; h_in[7] = 0;
    uint32_t* d_in; float* d_out;
    CHECK(cudaMalloc(&d_in, sizeof(h_in)));
    CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));

    printf("ILP sweep (TFLOPS via mma.sync; ratio meaningful at the plateau):\n\n");
    printf("  NACC        FP8        FP16        BF16   FP8/FP16   FP8/BF16\n");
    printf("  ----   ---------   ---------   ---------   --------   --------\n");
    sweep_row< 1>(blocks, threads, reps, d_in, d_out);
    sweep_row< 2>(blocks, threads, reps, d_in, d_out);
    sweep_row< 4>(blocks, threads, reps, d_in, d_out);
    sweep_row< 8>(blocks, threads, reps, d_in, d_out);
    sweep_row<16>(blocks, threads, reps, d_in, d_out);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
