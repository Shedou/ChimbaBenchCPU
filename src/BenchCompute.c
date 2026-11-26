//MIT License
//Copyright (c) 2026 Андрей Цымбалов (Chimbal)

// FP32 (Scalar)
__attribute__((noinline))
static double worker_fp32_scalar(void) {
    float v0 = 1.0001f, v1 = 1.0002f, v2 = 1.0003f, v3 = 1.0004f;
    float v4 = 1.0005f, v5 = 1.0006f, v6 = 1.0007f, v7 = 1.0008f;
    float mul = 0.999999f, add = 0.000001f;
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = v0 * mul + add; v1 = v1 * mul + add;
            v2 = v2 * mul + add; v3 = v3 * mul + add;
            v4 = v4 * mul + add; v5 = v5 * mul + add;
            v6 = v6 * mul + add; v7 = v7 * mul + add;
            asm volatile("" : "+m"(mul) : "m"(v0), "m"(v1), "m"(v2), "m"(v3), "m"(v4), "m"(v5), "m"(v6), "m"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile float sink = v0 + v7; (void)sink;
    return ((double)iters * 16.0) / (elapsed * 1e9); // 8 скаляров * 2 ops = 16 FLOP
}

// Для совместимости с Сlang...
double bench_fp32_scalar(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_fp32_scalar(); }
    return total;
}

// FP32 (SSE)
__attribute__((target("sse")))
static double worker_fp32_sse(void) {
    __m128 v0 = _mm_set1_ps(1.0001f), v1 = _mm_set1_ps(1.0002f), v2 = _mm_set1_ps(1.0003f), v3 = _mm_set1_ps(1.0004f);
    __m128 v4 = _mm_set1_ps(1.0005f), v5 = _mm_set1_ps(1.0006f), v6 = _mm_set1_ps(1.0007f), v7 = _mm_set1_ps(1.0008f);
    __m128 mul = _mm_set1_ps(0.999999f), add = _mm_set1_ps(0.000001f);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm_add_ps(_mm_mul_ps(v0, mul), add); v1 = _mm_add_ps(_mm_mul_ps(v1, mul), add);
            v2 = _mm_add_ps(_mm_mul_ps(v2, mul), add); v3 = _mm_add_ps(_mm_mul_ps(v3, mul), add);
            v4 = _mm_add_ps(_mm_mul_ps(v4, mul), add); v5 = _mm_add_ps(_mm_mul_ps(v5, mul), add);
            v6 = _mm_add_ps(_mm_mul_ps(v6, mul), add); v7 = _mm_add_ps(_mm_mul_ps(v7, mul), add);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile float sink = _mm_cvtss_f32(v0) + _mm_cvtss_f32(v7); (void)sink;
    return ((double)iters * 64.0) / (elapsed * 1e9);
}

double bench_fp32_sse(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_fp32_sse(); }
    return total;
}

// FP32 (AVX)
__attribute__((target("avx")))
static double worker_fp32_avx(void) {
    __m256 v0 = _mm256_set1_ps(1.0001f), v1 = _mm256_set1_ps(1.0002f), v2 = _mm256_set1_ps(1.0003f), v3 = _mm256_set1_ps(1.0004f);
    __m256 v4 = _mm256_set1_ps(1.0005f), v5 = _mm256_set1_ps(1.0006f), v6 = _mm256_set1_ps(1.0007f), v7 = _mm256_set1_ps(1.0008f);
    __m256 mul = _mm256_set1_ps(0.999999f), add = _mm256_set1_ps(0.000001f);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm256_add_ps(_mm256_mul_ps(v0, mul), add); v1 = _mm256_add_ps(_mm256_mul_ps(v1, mul), add);
            v2 = _mm256_add_ps(_mm256_mul_ps(v2, mul), add); v3 = _mm256_add_ps(_mm256_mul_ps(v3, mul), add);
            v4 = _mm256_add_ps(_mm256_mul_ps(v4, mul), add); v5 = _mm256_add_ps(_mm256_mul_ps(v5, mul), add);
            v6 = _mm256_add_ps(_mm256_mul_ps(v6, mul), add); v7 = _mm256_add_ps(_mm256_mul_ps(v7, mul), add);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile float sink = _mm256_cvtss_f32(v0) + _mm256_cvtss_f32(v7); (void)sink;
    return ((double)iters * 128.0) / (elapsed * 1e9);
}

double bench_fp32_avx(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_fp32_avx(); }
    return total;
}

// FP32 (AVX+FMA)
__attribute__((target("avx,fma")))
static double worker_fp32_avx_fma(void) {
    __m256 v0 = _mm256_set1_ps(1.0001f), v1 = _mm256_set1_ps(1.0002f), v2 = _mm256_set1_ps(1.0003f), v3 = _mm256_set1_ps(1.0004f);
    __m256 v4 = _mm256_set1_ps(1.0005f), v5 = _mm256_set1_ps(1.0006f), v6 = _mm256_set1_ps(1.0007f), v7 = _mm256_set1_ps(1.0008f);
    __m256 mul = _mm256_set1_ps(0.999999f), add = _mm256_set1_ps(0.000001f);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm256_fmadd_ps(v0, mul, add); v1 = _mm256_fmadd_ps(v1, mul, add);
            v2 = _mm256_fmadd_ps(v2, mul, add); v3 = _mm256_fmadd_ps(v3, mul, add);
            v4 = _mm256_fmadd_ps(v4, mul, add); v5 = _mm256_fmadd_ps(v5, mul, add);
            v6 = _mm256_fmadd_ps(v6, mul, add); v7 = _mm256_fmadd_ps(v7, mul, add);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile float sink = _mm256_cvtss_f32(v0) + _mm256_cvtss_f32(v7); (void)sink;
    return ((double)iters * 128.0) / (elapsed * 1e9);
}

double bench_fp32_avx_fma(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_fp32_avx_fma(); }
    return total;
}

// FP32 (AVX512+FMA)
__attribute__((target("avx512f")))
static double worker_fp32_avx512(void) {
    __m512 v0 = _mm512_set1_ps(1.0001f), v1 = _mm512_set1_ps(1.0002f);
    __m512 v2 = _mm512_set1_ps(1.0003f), v3 = _mm512_set1_ps(1.0004f);
    __m512 v4 = _mm512_set1_ps(1.0005f), v5 = _mm512_set1_ps(1.0006f);
    __m512 v6 = _mm512_set1_ps(1.0007f), v7 = _mm512_set1_ps(1.0008f);
    __m512 mul = _mm512_set1_ps(0.999999f), add = _mm512_set1_ps(0.000001f);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm512_fmadd_ps(v0, mul, add); v1 = _mm512_fmadd_ps(v1, mul, add);
            v2 = _mm512_fmadd_ps(v2, mul, add); v3 = _mm512_fmadd_ps(v3, mul, add);
            v4 = _mm512_fmadd_ps(v4, mul, add); v5 = _mm512_fmadd_ps(v5, mul, add);
            v6 = _mm512_fmadd_ps(v6, mul, add); v7 = _mm512_fmadd_ps(v7, mul, add);
            asm volatile("" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3), "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile float sink = _mm512_cvtss_f32(v0) + _mm512_cvtss_f32(v7); (void)sink;
    return ((double)iters * 256.0) / (elapsed * 1e9); // 8 векторов * 16 float * 2 ops (FMA) = 256 FLOP
}

double bench_fp32_avx512(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_fp32_avx512(); }
    return total;
}

// FP64 (SSE2)
__attribute__((target("sse2")))
static double worker_fp64_sse2(void) {
    __m128d v0 = _mm_set1_pd(1.0001), v1 = _mm_set1_pd(1.0002), v2 = _mm_set1_pd(1.0003), v3 = _mm_set1_pd(1.0004);
    __m128d v4 = _mm_set1_pd(1.0005), v5 = _mm_set1_pd(1.0006), v6 = _mm_set1_pd(1.0007), v7 = _mm_set1_pd(1.0008);
    __m128d mul = _mm_set1_pd(0.999999), add = _mm_set1_pd(0.000001);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm_add_pd(_mm_mul_pd(v0, mul), add); v1 = _mm_add_pd(_mm_mul_pd(v1, mul), add);
            v2 = _mm_add_pd(_mm_mul_pd(v2, mul), add); v3 = _mm_add_pd(_mm_mul_pd(v3, mul), add);
            v4 = _mm_add_pd(_mm_mul_pd(v4, mul), add); v5 = _mm_add_pd(_mm_mul_pd(v5, mul), add);
            v6 = _mm_add_pd(_mm_mul_pd(v6, mul), add); v7 = _mm_add_pd(_mm_mul_pd(v7, mul), add);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile double sink = _mm_cvtsd_f64(v0) + _mm_cvtsd_f64(v7); (void)sink;
    return ((double)iters * 32.0) / (elapsed * 1e9);
}

double bench_fp64_sse2(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_fp64_sse2(); }
    return total;
}

// FP64 (AVX)
__attribute__((target("avx")))
static double worker_fp64_avx(void) {
    __m256d v0 = _mm256_set1_pd(1.0001), v1 = _mm256_set1_pd(1.0002), v2 = _mm256_set1_pd(1.0003), v3 = _mm256_set1_pd(1.0004);
    __m256d v4 = _mm256_set1_pd(1.0005), v5 = _mm256_set1_pd(1.0006), v6 = _mm256_set1_pd(1.0007), v7 = _mm256_set1_pd(1.0008);
    __m256d mul = _mm256_set1_pd(0.999999), add = _mm256_set1_pd(0.000001);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm256_add_pd(_mm256_mul_pd(v0, mul), add); v1 = _mm256_add_pd(_mm256_mul_pd(v1, mul), add);
            v2 = _mm256_add_pd(_mm256_mul_pd(v2, mul), add); v3 = _mm256_add_pd(_mm256_mul_pd(v3, mul), add);
            v4 = _mm256_add_pd(_mm256_mul_pd(v4, mul), add); v5 = _mm256_add_pd(_mm256_mul_pd(v5, mul), add);
            v6 = _mm256_add_pd(_mm256_mul_pd(v6, mul), add); v7 = _mm256_add_pd(_mm256_mul_pd(v7, mul), add);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile double sink = _mm256_cvtsd_f64(v0) + _mm256_cvtsd_f64(v7); (void)sink;
    return ((double)iters * 64.0) / (elapsed * 1e9); // 8 векторов * 4 double * 2 ops = 64 FLOP
}

double bench_fp64_avx(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_fp64_avx(); }
    return total;
}

// FP64 (AVX+FMA)
__attribute__((target("avx,fma")))
static double worker_fp64_avx_fma(void) {
    __m256d v0 = _mm256_set1_pd(1.0001), v1 = _mm256_set1_pd(1.0002), v2 = _mm256_set1_pd(1.0003), v3 = _mm256_set1_pd(1.0004);
    __m256d v4 = _mm256_set1_pd(1.0005), v5 = _mm256_set1_pd(1.0006), v6 = _mm256_set1_pd(1.0007), v7 = _mm256_set1_pd(1.0008);
    __m256d mul = _mm256_set1_pd(0.999999), add = _mm256_set1_pd(0.000001);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm256_fmadd_pd(v0, mul, add); v1 = _mm256_fmadd_pd(v1, mul, add);
            v2 = _mm256_fmadd_pd(v2, mul, add); v3 = _mm256_fmadd_pd(v3, mul, add);
            v4 = _mm256_fmadd_pd(v4, mul, add); v5 = _mm256_fmadd_pd(v5, mul, add);
            v6 = _mm256_fmadd_pd(v6, mul, add); v7 = _mm256_fmadd_pd(v7, mul, add);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
    iters += batch; elapsed = get_time_sec() - start;
    }
    volatile double sink = _mm256_cvtsd_f64(v0) + _mm256_cvtsd_f64(v7); (void)sink;
    return ((double)iters * 64.0) / (elapsed * 1e9);
}

double bench_fp64_avx_fma(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_fp64_avx_fma(); }
    return total;
}

// FP64 (AVX512+FMA)
__attribute__((target("avx512f")))
static double worker_fp64_avx512(void) {
    __m512d v0 = _mm512_set1_pd(1.0001), v1 = _mm512_set1_pd(1.0002);
    __m512d v2 = _mm512_set1_pd(1.0003), v3 = _mm512_set1_pd(1.0004);
    __m512d v4 = _mm512_set1_pd(1.0005), v5 = _mm512_set1_pd(1.0006);
    __m512d v6 = _mm512_set1_pd(1.0007), v7 = _mm512_set1_pd(1.0008);
    __m512d mul = _mm512_set1_pd(0.999999), add = _mm512_set1_pd(0.000001);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm512_fmadd_pd(v0, mul, add); v1 = _mm512_fmadd_pd(v1, mul, add);
            v2 = _mm512_fmadd_pd(v2, mul, add); v3 = _mm512_fmadd_pd(v3, mul, add);
            v4 = _mm512_fmadd_pd(v4, mul, add); v5 = _mm512_fmadd_pd(v5, mul, add);
            v6 = _mm512_fmadd_pd(v6, mul, add); v7 = _mm512_fmadd_pd(v7, mul, add);
            asm volatile("" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3), "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile double sink = _mm512_cvtsd_f64(v0) + _mm512_cvtsd_f64(v7); (void)sink;
    return ((double)iters * 128.0) / (elapsed * 1e9); // 8 векторов * 8 double * 2 ops (FMA) = 128 FLOP
}

double bench_fp64_avx512(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_fp64_avx512(); }
    return total;
}

// INT8/INT16 (SSSE3)
__attribute__((target("ssse3")))
static double worker_int8_ssse3(void) {
    __m128i a0 = _mm_set1_epi8(3),  a1 = _mm_set1_epi8(5),  a2 = _mm_set1_epi8(7),  a3 = _mm_set1_epi8(9);
    __m128i a4 = _mm_set1_epi8(11), a5 = _mm_set1_epi8(13), a6 = _mm_set1_epi8(15), a7 = _mm_set1_epi8(17);
    __m128i b  = _mm_set1_epi8(-2);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            a0 = _mm_maddubs_epi16(a0, b); a1 = _mm_maddubs_epi16(a1, b);
            a2 = _mm_maddubs_epi16(a2, b); a3 = _mm_maddubs_epi16(a3, b);
            a4 = _mm_maddubs_epi16(a4, b); a5 = _mm_maddubs_epi16(a5, b);
            a6 = _mm_maddubs_epi16(a6, b); a7 = _mm_maddubs_epi16(a7, b);
            asm volatile("" : "+x"(a0), "+x"(a1), "+x"(a2), "+x"(a3), "+x"(a4), "+x"(a5), "+x"(a6), "+x"(a7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm_cvtsi128_si32(a0) ^ _mm_cvtsi128_si32(a7); (void)sink;
    return ((double)iters * 192.0) / (elapsed * 1e9); // 8 векторов * 24 ops = 192 IOPS
}

double bench_int8_ssse3(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int8_ssse3(); }
    return total;
}

// INT8 (AVX2)
__attribute__((target("avx2")))
static double worker_int8_avx2(void) {
    __m256i v0 = _mm256_set1_epi8(1), v1 = _mm256_set1_epi8(2), v2 = _mm256_set1_epi8(3), v3 = _mm256_set1_epi8(4);
    __m256i v4 = _mm256_set1_epi8(5), v5 = _mm256_set1_epi8(6), v6 = _mm256_set1_epi8(7), v7 = _mm256_set1_epi8(8);
    __m256i inc = _mm256_set1_epi8(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm256_add_epi8(v0, inc); v0 = _mm256_xor_si256(v0, v1);
            v1 = _mm256_add_epi8(v1, inc); v1 = _mm256_xor_si256(v1, v2);
            v2 = _mm256_add_epi8(v2, inc); v2 = _mm256_xor_si256(v2, v3);
            v3 = _mm256_add_epi8(v3, inc); v3 = _mm256_xor_si256(v3, v4);
            v4 = _mm256_add_epi8(v4, inc); v4 = _mm256_xor_si256(v4, v5);
            v5 = _mm256_add_epi8(v5, inc); v5 = _mm256_xor_si256(v5, v6);
            v6 = _mm256_add_epi8(v6, inc); v6 = _mm256_xor_si256(v6, v7);
            v7 = _mm256_add_epi8(v7, inc); v7 = _mm256_xor_si256(v7, v0);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm256_cvtsi256_si32(v0) ^ _mm256_cvtsi256_si32(v7); (void)sink;
    return ((double)iters * 512.0) / (elapsed * 1e9); // 8 векторов * 32 int8 * 2 ops = 512 IOPS
}

double bench_int8_avx2(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int8_avx2(); }
    return total;
}

// INT8 (AVX512)
__attribute__((target("avx512bw")))
static double worker_int8_avx512(void) {
    __m512i v0 = _mm512_set1_epi8(1), v1 = _mm512_set1_epi8(2), v2 = _mm512_set1_epi8(3), v3 = _mm512_set1_epi8(4);
    __m512i v4 = _mm512_set1_epi8(5), v5 = _mm512_set1_epi8(6), v6 = _mm512_set1_epi8(7), v7 = _mm512_set1_epi8(8);
    __m512i inc = _mm512_set1_epi8(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm512_add_epi8(v0, inc); v0 = _mm512_xor_si512(v0, v1);
            v1 = _mm512_add_epi8(v1, inc); v1 = _mm512_xor_si512(v1, v2);
            v2 = _mm512_add_epi8(v2, inc); v2 = _mm512_xor_si512(v2, v3);
            v3 = _mm512_add_epi8(v3, inc); v3 = _mm512_xor_si512(v3, v4);
            v4 = _mm512_add_epi8(v4, inc); v4 = _mm512_xor_si512(v4, v5);
            v5 = _mm512_add_epi8(v5, inc); v5 = _mm512_xor_si512(v5, v6);
            v6 = _mm512_add_epi8(v6, inc); v6 = _mm512_xor_si512(v6, v7);
            v7 = _mm512_add_epi8(v7, inc); v7 = _mm512_xor_si512(v7, v0);
            asm volatile("" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3), "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm512_cvtsi512_si32(v0) ^ _mm512_cvtsi512_si32(v7); (void)sink;
    return ((double)iters * 1024.0) / (elapsed * 1e9); // 8 векторов * 64 int8 * 2 ops = 1024 IOPS
}

double bench_int8_avx512(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int8_avx512(); }
    return total;
}

// INT16 (SSE2)
__attribute__((target("sse2")))
static double worker_int16_sse2(void) {
    __m128i v0 = _mm_set1_epi16(1), v1 = _mm_set1_epi16(2), v2 = _mm_set1_epi16(3), v3 = _mm_set1_epi16(4);
    __m128i v4 = _mm_set1_epi16(5), v5 = _mm_set1_epi16(6), v6 = _mm_set1_epi16(7), v7 = _mm_set1_epi16(8);
    __m128i inc = _mm_set1_epi16(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm_add_epi16(v0, inc); v0 = _mm_xor_si128(v0, v1);
            v1 = _mm_add_epi16(v1, inc); v1 = _mm_xor_si128(v1, v2);
            v2 = _mm_add_epi16(v2, inc); v2 = _mm_xor_si128(v2, v3);
            v3 = _mm_add_epi16(v3, inc); v3 = _mm_xor_si128(v3, v4);
            v4 = _mm_add_epi16(v4, inc); v4 = _mm_xor_si128(v4, v5);
            v5 = _mm_add_epi16(v5, inc); v5 = _mm_xor_si128(v5, v6);
            v6 = _mm_add_epi16(v6, inc); v6 = _mm_xor_si128(v6, v7);
            v7 = _mm_add_epi16(v7, inc); v7 = _mm_xor_si128(v7, v0);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm_cvtsi128_si32(v0) ^ _mm_cvtsi128_si32(v7); (void)sink;
    return ((double)iters * 128.0) / (elapsed * 1e9); // 8 векторов * 8 int16 * 2 ops = 128 IOPS
}

double bench_int16_sse2(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int16_sse2(); }
    return total;
}

// INT16 (AVX2)
__attribute__((target("avx2")))
static double worker_int16_avx2(void) {
    __m256i v0 = _mm256_set1_epi16(1), v1 = _mm256_set1_epi16(2), v2 = _mm256_set1_epi16(3), v3 = _mm256_set1_epi16(4);
    __m256i v4 = _mm256_set1_epi16(5), v5 = _mm256_set1_epi16(6), v6 = _mm256_set1_epi16(7), v7 = _mm256_set1_epi16(8);
    __m256i inc = _mm256_set1_epi16(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm256_add_epi16(v0, inc); v0 = _mm256_xor_si256(v0, v1);
            v1 = _mm256_add_epi16(v1, inc); v1 = _mm256_xor_si256(v1, v2);
            v2 = _mm256_add_epi16(v2, inc); v2 = _mm256_xor_si256(v2, v3);
            v3 = _mm256_add_epi16(v3, inc); v3 = _mm256_xor_si256(v3, v4);
            v4 = _mm256_add_epi16(v4, inc); v4 = _mm256_xor_si256(v4, v5);
            v5 = _mm256_add_epi16(v5, inc); v5 = _mm256_xor_si256(v5, v6);
            v6 = _mm256_add_epi16(v6, inc); v6 = _mm256_xor_si256(v6, v7);
            v7 = _mm256_add_epi16(v7, inc); v7 = _mm256_xor_si256(v7, v0);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm256_cvtsi256_si32(v0) ^ _mm256_cvtsi256_si32(v7); (void)sink;
    return ((double)iters * 256.0) / (elapsed * 1e9); // 8 векторов * 16 int16 * 2 ops = 256 IOPS
}

double bench_int16_avx2(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int16_avx2(); }
    return total;
}

// INT16 (AVX512)
__attribute__((target("avx512bw")))
static double worker_int16_avx512(void) {
    __m512i v0 = _mm512_set1_epi16(1), v1 = _mm512_set1_epi16(2), v2 = _mm512_set1_epi16(3), v3 = _mm512_set1_epi16(4);
    __m512i v4 = _mm512_set1_epi16(5), v5 = _mm512_set1_epi16(6), v6 = _mm512_set1_epi16(7), v7 = _mm512_set1_epi16(8);
    __m512i inc = _mm512_set1_epi16(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm512_add_epi16(v0, inc); v0 = _mm512_xor_si512(v0, v1);
            v1 = _mm512_add_epi16(v1, inc); v1 = _mm512_xor_si512(v1, v2);
            v2 = _mm512_add_epi16(v2, inc); v2 = _mm512_xor_si512(v2, v3);
            v3 = _mm512_add_epi16(v3, inc); v3 = _mm512_xor_si512(v3, v4);
            v4 = _mm512_add_epi16(v4, inc); v4 = _mm512_xor_si512(v4, v5);
            v5 = _mm512_add_epi16(v5, inc); v5 = _mm512_xor_si512(v5, v6);
            v6 = _mm512_add_epi16(v6, inc); v6 = _mm512_xor_si512(v6, v7);
            v7 = _mm512_add_epi16(v7, inc); v7 = _mm512_xor_si512(v7, v0);
            asm volatile("" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3), "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm512_cvtsi512_si32(v0) ^ _mm512_cvtsi512_si32(v7); (void)sink;
    return ((double)iters * 512.0) / (elapsed * 1e9); // 8 векторов * 32 int16 * 2 ops = 512 IOPS
}

double bench_int16_avx512(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int16_avx512(); }
    return total;
}

// INT32 (Scalar)
static double worker_int32_scalar(void) {
    uint32_t v0 = 1, v1 = 2, v2 = 3, v3 = 4;
    uint32_t v4 = 5, v5 = 6, v6 = 7, v7 = 8;
    uint32_t inc = 1;
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = (v0 + inc) ^ v1; v1 = (v1 + inc) ^ v2;
            v2 = (v2 + inc) ^ v3; v3 = (v3 + inc) ^ v4;
            v4 = (v4 + inc) ^ v5; v5 = (v5 + inc) ^ v6;
            v6 = (v6 + inc) ^ v7; v7 = (v7 + inc) ^ v0;
            asm volatile("" : "+r"(v0), "+r"(v1), "+r"(v2), "+r"(v3), "+m"(inc) : "r"(v4), "r"(v5));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile uint32_t sink = v0 ^ v7; (void)sink;
    return ((double)iters * 16.0) / (elapsed * 1e9); // 8 скаляров * 2 ops = 16 IOPS
}

double bench_int32_scalar(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int32_scalar(); }
    return total;
}

// Изолированный пакет MMX без переменных float/double (не затирает стек x87)
__attribute__((target("mmx")))
static uint32_t run_int32_mmx_batch(uint64_t batch) {
    __m64 v0 = _mm_set1_pi32(1), v1 = _mm_set1_pi32(2), v2 = _mm_set1_pi32(3), v3 = _mm_set1_pi32(4);
    __m64 v4 = _mm_set1_pi32(5), v5 = _mm_set1_pi32(6), v6 = _mm_set1_pi32(7), v7 = _mm_set1_pi32(8);
    __m64 inc = _mm_set1_pi32(1);
    for (uint64_t i = 0; i < batch; ++i) {
        v0 = _mm_add_pi32(v0, inc); v0 = _mm_xor_si64(v0, v1);
        v1 = _mm_add_pi32(v1, inc); v1 = _mm_xor_si64(v1, v2);
        v2 = _mm_add_pi32(v2, inc); v2 = _mm_xor_si64(v2, v3);
        v3 = _mm_add_pi32(v3, inc); v3 = _mm_xor_si64(v3, v4);
        v4 = _mm_add_pi32(v4, inc); v4 = _mm_xor_si64(v4, v5);
        v5 = _mm_add_pi32(v5, inc); v5 = _mm_xor_si64(v5, v6);
        v6 = _mm_add_pi32(v6, inc); v6 = _mm_xor_si64(v6, v7);
        v7 = _mm_add_pi32(v7, inc); v7 = _mm_xor_si64(v7, v0);
        asm volatile("" : "+y"(v0), "+y"(v1), "+y"(v2), "+y"(v3), "+y"(v4), "+y"(v5), "+y"(v6), "+y"(v7));
    }
    uint32_t res = (uint32_t)(_mm_cvtsi64_si32(v0) ^ _mm_cvtsi64_si32(v7));
    _mm_empty(); // Сбрасываем MMX/x87 перед возвратом к вычислениям с double
    return res;
}

// INT32 (MMX, Pentium MMX / PII / PIII / K6 / Athlon XP)
static double worker_int32_mmx(void) {
    const uint64_t batch = 50000000ULL;
    uint64_t iters = 0;
    uint32_t acc = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        acc ^= run_int32_mmx_batch(batch);
        iters += batch;
        elapsed = get_time_sec() - start;
    }
    volatile uint32_t sink = acc; (void)sink;
    return ((double)iters * 32.0) / (elapsed * 1e9);
}

double bench_int32_mmx(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int32_mmx(); }
    return total;
}

// INT32 (SSE2)
__attribute__((target("sse2")))
static double worker_int32_sse2(void) {
    __m128i v0 = _mm_set1_epi32(1), v1 = _mm_set1_epi32(2), v2 = _mm_set1_epi32(3), v3 = _mm_set1_epi32(4);
    __m128i v4 = _mm_set1_epi32(5), v5 = _mm_set1_epi32(6), v6 = _mm_set1_epi32(7), v7 = _mm_set1_epi32(8);
    __m128i inc = _mm_set1_epi32(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm_add_epi32(v0, inc); v0 = _mm_xor_si128(v0, v1);
            v1 = _mm_add_epi32(v1, inc); v1 = _mm_xor_si128(v1, v2);
            v2 = _mm_add_epi32(v2, inc); v2 = _mm_xor_si128(v2, v3);
            v3 = _mm_add_epi32(v3, inc); v3 = _mm_xor_si128(v3, v4);
            v4 = _mm_add_epi32(v4, inc); v4 = _mm_xor_si128(v4, v5);
            v5 = _mm_add_epi32(v5, inc); v5 = _mm_xor_si128(v5, v6);
            v6 = _mm_add_epi32(v6, inc); v6 = _mm_xor_si128(v6, v7);
            v7 = _mm_add_epi32(v7, inc); v7 = _mm_xor_si128(v7, v0);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm_cvtsi128_si32(v0) ^ _mm_cvtsi128_si32(v7); (void)sink;
    return ((double)iters * 64.0) / (elapsed * 1e9);
}

double bench_int32_sse2(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int32_sse2(); }
    return total;
}

// INT32 (SSE4.1)
__attribute__((target("sse4.1")))
static double worker_int32_sse41(void) {
    __m128i v0 = _mm_set1_epi32(101), v1 = _mm_set1_epi32(103), v2 = _mm_set1_epi32(107), v3 = _mm_set1_epi32(109);
    __m128i v4 = _mm_set1_epi32(113), v5 = _mm_set1_epi32(127), v6 = _mm_set1_epi32(131), v7 = _mm_set1_epi32(137);
    __m128i mul = _mm_set1_epi32(1664525), inc = _mm_set1_epi32(1013904223);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm_add_epi32(_mm_mullo_epi32(v0, mul), inc);
            v1 = _mm_add_epi32(_mm_mullo_epi32(v1, mul), inc);
            v2 = _mm_add_epi32(_mm_mullo_epi32(v2, mul), inc);
            v3 = _mm_add_epi32(_mm_mullo_epi32(v3, mul), inc);
            v4 = _mm_add_epi32(_mm_mullo_epi32(v4, mul), inc);
            v5 = _mm_add_epi32(_mm_mullo_epi32(v5, mul), inc);
            v6 = _mm_add_epi32(_mm_mullo_epi32(v6, mul), inc);
            v7 = _mm_add_epi32(_mm_mullo_epi32(v7, mul), inc);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm_cvtsi128_si32(v0) ^ _mm_cvtsi128_si32(v7); (void)sink;
    return ((double)iters * 64.0) / (elapsed * 1e9); // 8 векторов * 4 int32 * 2 ops (mul+add) = 64 IOPS
}

double bench_int32_sse41(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int32_sse41(); }
    return total;
}

// INT32 (AVX2)
__attribute__((target("avx2")))
static double worker_int32_avx2(void) {
    __m256i v0 = _mm256_set1_epi32(1), v1 = _mm256_set1_epi32(2), v2 = _mm256_set1_epi32(3), v3 = _mm256_set1_epi32(4);
    __m256i v4 = _mm256_set1_epi32(5), v5 = _mm256_set1_epi32(6), v6 = _mm256_set1_epi32(7), v7 = _mm256_set1_epi32(8);
    __m256i inc = _mm256_set1_epi32(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm256_add_epi32(v0, inc); v0 = _mm256_xor_si256(v0, v1);
            v1 = _mm256_add_epi32(v1, inc); v1 = _mm256_xor_si256(v1, v2);
            v2 = _mm256_add_epi32(v2, inc); v2 = _mm256_xor_si256(v2, v3);
            v3 = _mm256_add_epi32(v3, inc); v3 = _mm256_xor_si256(v3, v4);
            v4 = _mm256_add_epi32(v4, inc); v4 = _mm256_xor_si256(v4, v5);
            v5 = _mm256_add_epi32(v5, inc); v5 = _mm256_xor_si256(v5, v6);
            v6 = _mm256_add_epi32(v6, inc); v6 = _mm256_xor_si256(v6, v7);
            v7 = _mm256_add_epi32(v7, inc); v7 = _mm256_xor_si256(v7, v0);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm256_cvtsi256_si32(v0) ^ _mm256_cvtsi256_si32(v7); (void)sink;
    return ((double)iters * 128.0) / (elapsed * 1e9);
}

double bench_int32_avx2(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int32_avx2(); }
    return total;
}

// INT32 (AVX512)
__attribute__((target("avx512f")))
static double worker_int32_avx512(void) {
    __m512i v0 = _mm512_set1_epi32(1), v1 = _mm512_set1_epi32(2), v2 = _mm512_set1_epi32(3), v3 = _mm512_set1_epi32(4);
    __m512i v4 = _mm512_set1_epi32(5), v5 = _mm512_set1_epi32(6), v6 = _mm512_set1_epi32(7), v7 = _mm512_set1_epi32(8);
    __m512i inc = _mm512_set1_epi32(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm512_add_epi32(v0, inc); v0 = _mm512_xor_si512(v0, v1);
            v1 = _mm512_add_epi32(v1, inc); v1 = _mm512_xor_si512(v1, v2);
            v2 = _mm512_add_epi32(v2, inc); v2 = _mm512_xor_si512(v2, v3);
            v3 = _mm512_add_epi32(v3, inc); v3 = _mm512_xor_si512(v3, v4);
            v4 = _mm512_add_epi32(v4, inc); v4 = _mm512_xor_si512(v4, v5);
            v5 = _mm512_add_epi32(v5, inc); v5 = _mm512_xor_si512(v5, v6);
            v6 = _mm512_add_epi32(v6, inc); v6 = _mm512_xor_si512(v6, v7);
            v7 = _mm512_add_epi32(v7, inc); v7 = _mm512_xor_si512(v7, v0);
            asm volatile("" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3), "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm512_cvtsi512_si32(v0) ^ _mm512_cvtsi512_si32(v7); (void)sink;
    return ((double)iters * 256.0
    ) / (elapsed * 1e9); // 8 векторов * 16 int32 * 2 ops = 256 IOPS
}

double bench_int32_avx512(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int32_avx512(); }
    return total;
}

// INT64 (SSE2)
__attribute__((target("sse2")))
static double worker_int64_sse2(void) {
    __m128i v0 = _mm_set1_epi64x(1), v1 = _mm_set1_epi64x(2), v2 = _mm_set1_epi64x(3), v3 = _mm_set1_epi64x(4);
    __m128i v4 = _mm_set1_epi64x(5), v5 = _mm_set1_epi64x(6), v6 = _mm_set1_epi64x(7), v7 = _mm_set1_epi64x(8);
    __m128i inc = _mm_set1_epi64x(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm_add_epi64(v0, inc); v0 = _mm_xor_si128(v0, v1);
            v1 = _mm_add_epi64(v1, inc); v1 = _mm_xor_si128(v1, v2);
            v2 = _mm_add_epi64(v2, inc); v2 = _mm_xor_si128(v2, v3);
            v3 = _mm_add_epi64(v3, inc); v3 = _mm_xor_si128(v3, v4);
            v4 = _mm_add_epi64(v4, inc); v4 = _mm_xor_si128(v4, v5);
            v5 = _mm_add_epi64(v5, inc); v5 = _mm_xor_si128(v5, v6);
            v6 = _mm_add_epi64(v6, inc); v6 = _mm_xor_si128(v6, v7);
            v7 = _mm_add_epi64(v7, inc); v7 = _mm_xor_si128(v7, v0);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm_cvtsi128_si32(v0) ^ _mm_cvtsi128_si32(v7); (void)sink;
    return ((double)iters * 32.0) / (elapsed * 1e9); // 8 векторов * 2 int64 * 2 ops = 32 IOPS
}

double bench_int64_sse2(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int64_sse2(); }
    return total;
}

// INT64 (AVX2)
__attribute__((target("avx2")))
static double worker_int64_avx2(void) {
    __m256i v0 = _mm256_set1_epi64x(1), v1 = _mm256_set1_epi64x(2), v2 = _mm256_set1_epi64x(3), v3 = _mm256_set1_epi64x(4);
    __m256i v4 = _mm256_set1_epi64x(5), v5 = _mm256_set1_epi64x(6), v6 = _mm256_set1_epi64x(7), v7 = _mm256_set1_epi64x(8);
    __m256i inc = _mm256_set1_epi64x(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm256_add_epi64(v0, inc); v0 = _mm256_xor_si256(v0, v1);
            v1 = _mm256_add_epi64(v1, inc); v1 = _mm256_xor_si256(v1, v2);
            v2 = _mm256_add_epi64(v2, inc); v2 = _mm256_xor_si256(v2, v3);
            v3 = _mm256_add_epi64(v3, inc); v3 = _mm256_xor_si256(v3, v4);
            v4 = _mm256_add_epi64(v4, inc); v4 = _mm256_xor_si256(v4, v5);
            v5 = _mm256_add_epi64(v5, inc); v5 = _mm256_xor_si256(v5, v6);
            v6 = _mm256_add_epi64(v6, inc); v6 = _mm256_xor_si256(v6, v7);
            v7 = _mm256_add_epi64(v7, inc); v7 = _mm256_xor_si256(v7, v0);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm256_cvtsi256_si32(v0) ^ _mm256_cvtsi256_si32(v7); (void)sink;
    return ((double)iters * 64.0) / (elapsed * 1e9); // 8 векторов * 4 int64 * 2 ops = 64 IOPS
}

double bench_int64_avx2(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int64_avx2(); }
    return total;
}

// INT64 (AVX512)
__attribute__((target("avx512f")))
static double worker_int64_avx512(void) {
    __m512i v0 = _mm512_set1_epi64(1), v1 = _mm512_set1_epi64(2), v2 = _mm512_set1_epi64(3), v3 = _mm512_set1_epi64(4);
    __m512i v4 = _mm512_set1_epi64(5), v5 = _mm512_set1_epi64(6), v6 = _mm512_set1_epi64(7), v7 = _mm512_set1_epi64(8);
    __m512i inc = _mm512_set1_epi64(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm512_add_epi64(v0, inc); v0 = _mm512_xor_si512(v0, v1);
            v1 = _mm512_add_epi64(v1, inc); v1 = _mm512_xor_si512(v1, v2);
            v2 = _mm512_add_epi64(v2, inc); v2 = _mm512_xor_si512(v2, v3);
            v3 = _mm512_add_epi64(v3, inc); v3 = _mm512_xor_si512(v3, v4);
            v4 = _mm512_add_epi64(v4, inc); v4 = _mm512_xor_si512(v4, v5);
            v5 = _mm512_add_epi64(v5, inc); v5 = _mm512_xor_si512(v5, v6);
            v6 = _mm512_add_epi64(v6, inc); v6 = _mm512_xor_si512(v6, v7);
            v7 = _mm512_add_epi64(v7, inc); v7 = _mm512_xor_si512(v7, v0);
            asm volatile("" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3), "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm512_cvtsi512_si32(v0) ^ _mm512_cvtsi512_si32(v7); (void)sink;
    return ((double)iters * 128.0) / (elapsed * 1e9); // 8 векторов * 8 int64 * 2 ops = 128 IOPS
}

double bench_int64_avx512(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int64_avx512(); }
    return total;
}

// INT8 (SSE2)
__attribute__((target("sse2")))
static double worker_int8_sse2(void) {
    __m128i v0 = _mm_set1_epi8(1), v1 = _mm_set1_epi8(2), v2 = _mm_set1_epi8(3), v3 = _mm_set1_epi8(4);
    __m128i v4 = _mm_set1_epi8(5), v5 = _mm_set1_epi8(6), v6 = _mm_set1_epi8(7), v7 = _mm_set1_epi8(8);
    __m128i inc = _mm_set1_epi8(1);
    const uint64_t batch = 50000000ULL; uint64_t iters = 0;
    #pragma omp barrier
    double start = get_time_sec(), elapsed = 0.0;
    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch; ++i) {
            v0 = _mm_add_epi8(v0, inc); v0 = _mm_xor_si128(v0, v1);
            v1 = _mm_add_epi8(v1, inc); v1 = _mm_xor_si128(v1, v2);
            v2 = _mm_add_epi8(v2, inc); v2 = _mm_xor_si128(v2, v3);
            v3 = _mm_add_epi8(v3, inc); v3 = _mm_xor_si128(v3, v4);
            v4 = _mm_add_epi8(v4, inc); v4 = _mm_xor_si128(v4, v5);
            v5 = _mm_add_epi8(v5, inc); v5 = _mm_xor_si128(v5, v6);
            v6 = _mm_add_epi8(v6, inc); v6 = _mm_xor_si128(v6, v7);
            v7 = _mm_add_epi8(v7, inc); v7 = _mm_xor_si128(v7, v0);
            asm volatile("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4), "+x"(v5), "+x"(v6), "+x"(v7));
        }
        iters += batch; elapsed = get_time_sec() - start;
    }
    volatile int sink = _mm_cvtsi128_si32(v0) ^ _mm_cvtsi128_si32(v7); (void)sink;
    return ((double)iters * 256.0) / (elapsed * 1e9); // 8 векторов * 16 int8 * 2 ops = 256 IOPS
}

double bench_int8_sse2(void) {
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    { total += worker_int8_sse2(); }
    return total;
}
