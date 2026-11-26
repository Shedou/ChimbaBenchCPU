//MIT License
//Copyright (c) 2026 Андрей Цымбалов (Chimbal)

__attribute__((target("avx2")))
static void read_chunk_avx2(const uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    __m256i s0 = _mm256_setzero_si256();
    __m256i s1 = _mm256_setzero_si256();
    for (size_t i = step_start; i < bytes; i += step_stride) {
        s0 = _mm256_add_epi32(s0, _mm256_load_si256((const __m256i*)(ptr + i)));
        s1 = _mm256_add_epi32(s1, _mm256_load_si256((const __m256i*)(ptr + i + 32)));
    }
    asm volatile("" : "+x"(s0), "+x"(s1));
    volatile int sink = _mm_cvtsi128_si32(_mm256_castsi256_si128(s0)) ^
                        _mm_cvtsi128_si32(_mm256_castsi256_si128(s1));
    (void)sink;
}

__attribute__((target("sse2")))
static void read_chunk_sse2(const uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    __m128i s0 = _mm_setzero_si128(), s1 = _mm_setzero_si128();
    __m128i s2 = _mm_setzero_si128(), s3 = _mm_setzero_si128();
    for (size_t i = step_start; i < bytes; i += step_stride) {
        s0 = _mm_add_epi32(s0, _mm_load_si128((const __m128i*)(ptr + i)));
        s1 = _mm_add_epi32(s1, _mm_load_si128((const __m128i*)(ptr + i + 16)));
        s2 = _mm_add_epi32(s2, _mm_load_si128((const __m128i*)(ptr + i + 32)));
        s3 = _mm_add_epi32(s3, _mm_load_si128((const __m128i*)(ptr + i + 48)));
    }
    asm volatile("" : "+x"(s0), "+x"(s1), "+x"(s2), "+x"(s3));
    volatile int sink = _mm_cvtsi128_si32(s0) ^ _mm_cvtsi128_si32(s3);
    (void)sink;
}

__attribute__((target("sse")))
static void read_chunk_sse(const uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
    for (size_t i = step_start; i < bytes; i += step_stride) {
        // Используем bitwise OR (_mm_or_ps) вместо сложения float (_mm_add_ps),
        // чтобы не словить замедление на денормализованных числах (denormals) от байтов 0x3C
        s0 = _mm_or_ps(s0, _mm_load_ps((const float*)(ptr + i)));
        s1 = _mm_or_ps(s1, _mm_load_ps((const float*)(ptr + i + 16)));
        s2 = _mm_or_ps(s2, _mm_load_ps((const float*)(ptr + i + 32)));
        s3 = _mm_or_ps(s3, _mm_load_ps((const float*)(ptr + i + 48)));
    }
    asm volatile("" : "+x"(s0), "+x"(s1), "+x"(s2), "+x"(s3));
    volatile float sink = _mm_cvtss_f32(s0) + _mm_cvtss_f32(s3);
    (void)sink;
}

__attribute__((target("mmx")))
static void read_chunk_mmx(const uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    __m64 s0 = _mm_setzero_si64(), s1 = _mm_setzero_si64();
    __m64 s2 = _mm_setzero_si64(), s3 = _mm_setzero_si64();
    for (size_t i = step_start; i < bytes; i += step_stride) {
        const __m64 *p = (const __m64*)(ptr + i);
        s0 = _mm_add_pi32(s0, p[0]);
        s1 = _mm_add_pi32(s1, p[1]);
        s2 = _mm_add_pi32(s2, p[2]);
        s3 = _mm_add_pi32(s3, p[3]);
        s0 = _mm_add_pi32(s0, p[4]);
        s1 = _mm_add_pi32(s1, p[5]);
        s2 = _mm_add_pi32(s2, p[6]);
        s3 = _mm_add_pi32(s3, p[7]);
    }
    asm volatile("" : "+y"(s0), "+y"(s1), "+y"(s2), "+y"(s3));
    volatile int sink = _mm_cvtsi64_si32(s0) ^ _mm_cvtsi64_si32(s3);
    (void)sink;
    _mm_empty(); // Обязательный сброс состояния FPU (EMMS)
}

static void read_chunk_scalar(const uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    uintptr_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (size_t i = step_start; i < bytes; i += step_stride) {
        const uintptr_t *p = (const uintptr_t*)(ptr + i);
        s0 += p[0]; s1 += p[1]; s2 += p[2]; s3 += p[3];
        #if UINTPTR_MAX == 0xFFFFFFFFULL // В 32-битном режиме 64 байта = 16 слов по 4 байта
        s0 += p[4]; s1 += p[5]; s2 += p[6]; s3 += p[7];
        s0 += p[8]; s1 += p[9]; s2 += p[10]; s3 += p[11];
        s0 += p[12]; s1 += p[13]; s2 += p[14]; s3 += p[15];
        #else // В 64-битном режиме 64 байта = 8 слов по 8 байт
        s0 += p[4]; s1 += p[5]; s2 += p[6]; s3 += p[7];
        #endif
    }
    asm volatile("" : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3));
    volatile uintptr_t sink = s0 ^ s3;
    (void)sink;
}

typedef void (*read_chunk_fn)(const uint8_t*, size_t, size_t, size_t);

// Замер пропускной способности чтения (per-thread буфер для L1/L2, общий для L3/RAM)
double measure_read_bandwidth(size_t bytes_per_thread, int is_shared) {
    if (bytes_per_thread == 0) return 0.0;

    int threads = omp_get_max_threads();
    size_t total_alloc = is_shared ? bytes_per_thread : (bytes_per_thread * (size_t)threads);
    uint8_t *buf = (uint8_t*)ALIGNED_ALLOC(64, total_alloc);
    if (!buf) return 0.0;
    memset(buf, 0x3C, total_alloc);

    // Выбираем самый быстрый доступный движок чтения от AVX2 до Scalar
    read_chunk_fn read_fn = read_chunk_scalar;
    if (__builtin_cpu_supports("avx2"))      read_fn = read_chunk_avx2;
    else if (__builtin_cpu_supports("sse2")) read_fn = read_chunk_sse2;
    else if (__builtin_cpu_supports("sse"))  read_fn = read_chunk_sse;
    else if (__builtin_cpu_supports("mmx"))  read_fn = read_chunk_mmx;

    uint64_t inner_repeats = (bytes_per_thread <= 2ULL * 1024ULL * 1024ULL)
                             ? (2ULL * 1024ULL * 1024ULL / bytes_per_thread) : 1ULL;
    if (inner_repeats < 1) inner_repeats = 1;

    uint64_t total_passes = 0;
    double start = get_time_sec(), elapsed = 0.0;

    while (elapsed < TARGET_SECONDS) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int num_t = omp_get_num_threads();

            if (!is_shared) {
                // Каждый поток читает свой собственный кусок L1/L2
                const uint8_t *my_buf = buf + (size_t)tid * bytes_per_thread;
                for (uint64_t r = 0; r < inner_repeats; ++r) {
                    read_fn(my_buf, bytes_per_thread, 0, 64);
                }
            } else {
                // Общий L3 / RAM делится между всеми потоками с шагом 64 * num_t
                size_t step_start = (size_t)tid * 64;
                size_t step_stride = (size_t)num_t * 64;
                for (uint64_t r = 0; r < inner_repeats; ++r) {
                    read_fn(buf, bytes_per_thread, step_start, step_stride);
                }
            }
        }
        total_passes += inner_repeats;
        elapsed = get_time_sec() - start;
    }

    double gb_read = ((double)total_passes * (double)total_alloc) / 1e9;
    ALIGNED_FREE(buf);
    return gb_read / elapsed;
}

double measure_random_latency(size_t bytes) {
    const size_t stride = 64;
    const size_t count = bytes / stride;
    if (count < 2) return 0.0;
    size_t *indices = (size_t*)malloc(count * sizeof(size_t));
    void **arena = (void**)ALIGNED_ALLOC(64, bytes);
    //if (!indices || !arena) return 0.0;
    if (!indices || !arena) {
        free(indices);
        ALIGNED_FREE(arena);
        return 0.0;
    }

    for (size_t i = 0; i < count; ++i) indices[i] = i;
    uint64_t rng = 0x123456789ABCDEFULL;
    for (size_t i = count - 1; i > 0; --i) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        size_t j = rng % i;
        size_t tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
    }
    for (size_t i = 0; i < count; ++i) {
        size_t curr = indices[i] * (stride / sizeof(void*));
        size_t next = indices[(i + 1) % count] * (stride / sizeof(void*));
        arena[curr] = (void*)&arena[next];
    }
    free(indices);

    void **p = (void**)arena;
    const uint64_t batch_steps = 10000000ULL;
    uint64_t total_steps = 0;
    double start = get_time_sec(), elapsed = 0.0;

    while (elapsed < TARGET_SECONDS) {
        for (uint64_t i = 0; i < batch_steps; ++i) {
            p = (void**)*p;
            asm volatile("" : "+r"(p));
        }
        total_steps += batch_steps;
        elapsed = get_time_sec() - start;
    }

    volatile void *sink = p; (void)sink;
    ALIGNED_FREE(arena);
    return (elapsed * 1e9) / (double)total_steps;
}

__attribute__((target("avx2")))
static void write_chunk_avx2(uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    __m256i v0 = _mm256_set1_epi32(0x5A5A5A5A);
    __m256i v1 = _mm256_set1_epi32(0xA5A5A5A5);
    for (size_t i = step_start; i < bytes; i += step_stride) {
        _mm256_store_si256((__m256i*)(ptr + i), v0);
        _mm256_store_si256((__m256i*)(ptr + i + 32), v1);
    }
    asm volatile("" : : "r"(ptr) : "memory");
}

__attribute__((target("sse2")))
static void write_chunk_sse2(uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    __m128i v0 = _mm_set1_epi32(0x5A5A5A5A);
    __m128i v1 = _mm_set1_epi32(0xA5A5A5A5);
    for (size_t i = step_start; i < bytes; i += step_stride) {
        _mm_store_si128((__m128i*)(ptr + i), v0);
        _mm_store_si128((__m128i*)(ptr + i + 16), v1);
        _mm_store_si128((__m128i*)(ptr + i + 32), v0);
        _mm_store_si128((__m128i*)(ptr + i + 48), v1);
    }
    asm volatile("" : : "r"(ptr) : "memory");
}

__attribute__((target("sse")))
static void write_chunk_sse(uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    __m128 v0 = _mm_set1_ps(1.5f);
    __m128 v1 = _mm_set1_ps(2.5f);
    for (size_t i = step_start; i < bytes; i += step_stride) {
        _mm_store_ps((float*)(ptr + i), v0);
        _mm_store_ps((float*)(ptr + i + 16), v1);
        _mm_store_ps((float*)(ptr + i + 32), v0);
        _mm_store_ps((float*)(ptr + i + 48), v1);
    }
    asm volatile("" : : "r"(ptr) : "memory");
}

__attribute__((target("mmx")))
static void write_chunk_mmx(uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    __m64 v0 = _mm_set1_pi32(0x5A5A5A5A);
    __m64 v1 = _mm_set1_pi32(0xA5A5A5A5);
    for (size_t i = step_start; i < bytes; i += step_stride) {
        __m64 *p = (__m64*)(ptr + i);
        p[0] = v0; p[1] = v1; p[2] = v0; p[3] = v1;
        p[4] = v0; p[5] = v1; p[6] = v0; p[7] = v1;
    }
    asm volatile("" : : "r"(ptr) : "memory");
    _mm_empty();
}

static void write_chunk_scalar(uint8_t *ptr, size_t bytes, size_t step_start, size_t step_stride) {
    const uintptr_t v0 = (uintptr_t)0x5A5A5A5A5A5A5A5AULL;
    const uintptr_t v1 = (uintptr_t)0xA5A5A5A5A5A5A5AULL;
    for (size_t i = step_start; i < bytes; i += step_stride) {
        uintptr_t *p = (uintptr_t*)(ptr + i);
        p[0] = v0; p[1] = v1; p[2] = v0; p[3] = v1;
        #if UINTPTR_MAX == 0xFFFFFFFFULL
        p[4] = v0; p[5] = v1; p[6] = v0; p[7] = v1;
        p[8] = v0; p[9] = v1; p[10] = v0; p[11] = v1;
        p[12] = v0; p[13] = v1; p[14] = v0; p[15] = v1;
        #else
        p[4] = v0; p[5] = v1; p[6] = v0; p[7] = v1;
        #endif
    }
    asm volatile("" : : "r"(ptr) : "memory");
}

typedef void (*write_chunk_fn)(uint8_t*, size_t, size_t, size_t);

// Замер пропускной способности записи (per-thread буфер для L1/L2, общий для L3/RAM)
double measure_write_bandwidth(size_t bytes_per_thread, int is_shared) {
    if (bytes_per_thread == 0) return 0.0;

    int threads = omp_get_max_threads();
    size_t total_alloc = is_shared ? bytes_per_thread : (bytes_per_thread * (size_t)threads);
    uint8_t *buf = (uint8_t*)ALIGNED_ALLOC(64, total_alloc);
    if (!buf) return 0.0;
    memset(buf, 0x00, total_alloc); // Прогреваем страницы (page fault) до начала замера

    write_chunk_fn write_fn = write_chunk_scalar;
    if (__builtin_cpu_supports("avx2"))      write_fn = write_chunk_avx2;
    else if (__builtin_cpu_supports("sse2")) write_fn = write_chunk_sse2;
    else if (__builtin_cpu_supports("sse"))  write_fn = write_chunk_sse;
    else if (__builtin_cpu_supports("mmx"))  write_fn = write_chunk_mmx;

    uint64_t inner_repeats = (bytes_per_thread <= 2ULL * 1024ULL * 1024ULL)
                             ? (2ULL * 1024ULL * 1024ULL / bytes_per_thread) : 1ULL;
    if (inner_repeats < 1) inner_repeats = 1;

    uint64_t total_passes = 0;
    double start = get_time_sec(), elapsed = 0.0;

    while (elapsed < TARGET_SECONDS) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int num_t = omp_get_num_threads();

            if (!is_shared) {
                uint8_t *my_buf = buf + (size_t)tid * bytes_per_thread;
                for (uint64_t r = 0; r < inner_repeats; ++r) {
                    write_fn(my_buf, bytes_per_thread, 0, 64);
                }
            } else {
                size_t step_start = (size_t)tid * 64;
                size_t step_stride = (size_t)num_t * 64;
                for (uint64_t r = 0; r < inner_repeats; ++r) {
                    write_fn(buf, bytes_per_thread, step_start, step_stride);
                }
            }
        }
        total_passes += inner_repeats;
        elapsed = get_time_sec() - start;
    }

    double gb_written = ((double)total_passes * (double)total_alloc) / 1e9;
    ALIGNED_FREE(buf);
    return gb_written / elapsed;
}


// Адаптивные размеры рабочих наборов:
// L1, L2 и L3 - половина от размера кэша ядра (гарантирует отсутствие вытеснения кодом/стеком). RAM - по умолчанию 8x от L3
const size_t MB = 1024ULL * 1024ULL;
const size_t MIN_RAM_TEST_SIZE = MINIMUM_RAM_TEST_SIZE * MB;
size_t get_l1_test_size(void)  { return g_cache.l1d_bytes / 2; }
size_t get_l2_test_size(void)  { return g_cache.l2_bytes / 2; }
size_t get_l3_test_size(void)  { return g_cache.l3_bytes / 2; }
size_t get_ram_test_size(void) {
    size_t largest_cache = g_cache.l3_bytes;
    if (largest_cache == 0) { largest_cache = g_cache.l2_bytes; }
    size_t target_size = largest_cache * TARGET_RAM_MULTIPLIER_SIZE;
    
    // Ограничение минимального размера для теста RAM
    if (target_size < MIN_RAM_TEST_SIZE) { return MIN_RAM_TEST_SIZE; }

    return target_size;
}

double bench_l1_bw(void)   { return measure_read_bandwidth(get_l1_test_size(),  0); }
double bench_l2_bw(void)   { return measure_read_bandwidth(get_l2_test_size(),  0); }
double bench_l3_bw(void)   { return measure_read_bandwidth(get_l3_test_size(),  1); }
double bench_ram_rd(void)  { return measure_read_bandwidth(get_ram_test_size(), 1); }

double bench_l1_wr(void)   { return measure_write_bandwidth(get_l1_test_size(),  0); }
double bench_l2_wr(void)   { return measure_write_bandwidth(get_l2_test_size(),  0); }
double bench_l3_wr(void)   { return measure_write_bandwidth(get_l3_test_size(),  1); }
double bench_ram_wr(void)  { return measure_write_bandwidth(get_ram_test_size(), 1); }

double bench_l1_lat(void)  { return measure_random_latency(get_l1_test_size()); }
double bench_l2_lat(void)  { return measure_random_latency(get_l2_test_size()); }
double bench_l3_lat(void)  { return measure_random_latency(get_l3_test_size()); }
double bench_ram_lat(void) { return measure_random_latency(get_ram_test_size()); }
