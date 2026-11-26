//MIT License
//
//Copyright (c) 2026 Андрей Цымбалов (Chimbal)
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

// =============================================================================
// Данный проект разрабатывался с помощью ИИ (не является полностью сгенерированным ИИ).
// Благодаря помощи ИИ скорость разработки и качество проекта значительно увеличились по сравнению с тем, если бы я писал вручную каждую строчку кода.
// В данном случае нельзя выделить долю ручного кода и сгенерированного, так как вручную внесённый код мог быть переработан ИИ и наоборот.
// =============================================================================
// Компиляция / Compile
// 
// Разработка проводилась в дистрибутиве Chimbalix (Debian 12) с компилятором GCC 12.2 (OpenMP 4.5).
// Более старые дистрибутивы не проверялись и не отлаживались в плане компиляции проекта.
// Основной целевой компилятор - это GCC. Код иногда дорабатывается для совместимости с компилятором Clang,
//  но этот компилятор не основной, так как не позволяет скомпилировать проект статически без зависимостей. 
// Для компиляции под Windows использовался w64devkit v2.5.0.
// 
// См. скрипты в каталоге Build-Scripts или следующими командами:
// 
// Windows: gcc -O3 -fopenmp -Wall -static ChimbaBenchCPU.c -o ChimbaBenchCPU.exe
// Linux/FreeBSD Portable: gcc -O3 -fopenmp -Wall -static ChimbaBenchCPU.c -o ChimbaBenchCPU-(Linux/FBSD)
// Предупреждение "warning: Using 'dlopen' in statically..." можно проигнорировать при сборке портативной версии бенчмарка в среде Linux.
// 
// Linux Dependency hell: gcc -O3 -fopenmp -Wall ChimbaBenchCPU.c -o ChimbaBenchCPU-Linux
// 
// Linux x86_64 Less dependency hell: gcc -O3 -fopenmp -Wall ChimbaBenchCPU.c -Wl,--wrap=__libc_start_main -o ChimbaBenchCPU-Linux
// 
// w64devkit (build tools for Windows):
// https://github.com/skeeto/w64devkit
// =============================================================================

// ==================================================================== //
// Костыль для Linux, понижающий требования к версии злополучного GLIBC //
// Сработает ТОЛЬКО если передан флаг -Wl,--wrap=__libc_start_main      //
#if defined(__linux__)
extern int __real___libc_start_main(int (*main)(int, char **, char **), int argc, char **ubp_av, void (*init)(void), void (*fini)(void), void (*rtld_fini)(void), void *stack_end) __attribute__((weak));
__asm__(".symver __real___libc_start_main,__libc_start_main@GLIBC_2.2.5");

int __wrap___libc_start_main(int (*main)(int, char **, char **), int argc, char **ubp_av, void (*init)(void), void (*fini)(void), void (*rtld_fini)(void), void *stack_end) {
    return __real___libc_start_main(main, argc, ubp_av, init, fini, rtld_fini, stack_end);
}
#endif
// ==================================================================== //

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <cpuid.h>
#include <immintrin.h>
#include <omp.h>

#ifdef _WIN32
    #include <windows.h>
    #include <direct.h>
    #include <malloc.h>
    #include <conio.h> // для функции ожидания нажатия
    #define MKDIR(dir) _mkdir(dir)
    #define ALIGNED_ALLOC(align, size) _aligned_malloc((size), (align))
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
    double get_time_sec(void) {
        static LARGE_INTEGER freq = {0};
        if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return (double)counter.QuadPart / (double)freq.QuadPart;
    }
#else
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <termios.h> // для функции ожидания нажатия
    #include <unistd.h> // для функции ожидания нажатия
#ifndef _WIN32
    #include <unistd.h>
    #include <sys/utsname.h>
    #ifdef __FreeBSD__
        #include <sys/sysctl.h>
    #endif
#endif
    #define MKDIR(dir) mkdir((dir), 0755)
    static inline void* cb_aligned_alloc(size_t align, size_t size) {
        void *ptr = NULL;
        return (posix_memalign(&ptr, align, size) == 0) ? ptr : NULL;
    }
    #define ALIGNED_ALLOC(align, size) cb_aligned_alloc((align), (size))
    #define ALIGNED_FREE(ptr) free(ptr)
    double get_time_sec(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif

// ==================================================================== //
const char *ProjectName = "ChimbaBenchCPU v1.0 (CPU&CacheMem, https://github.com/Shedou/ChimbaBenchCPU)\n";
const char *ProjectLicense = "MIT License - (c) 2026 Андрей Цымбалов (Chimbal)\n";

double TARGET_SECONDS = 2.0;                 // Время прохождения тестов по умолчанию в секундах
double TARGET_SECONDS_FAST = 0.5;            // Время прохождения тестов в секундах при ускоренном прохождении (аргумент -fast)
const size_t TARGET_RAM_MULTIPLIER_SIZE = 8; // Целевой множитель размера теста RAM относительно обьёма кэша. 64 MB L3 * 8 = 512 MB
const size_t MINIMUM_RAM_TEST_SIZE = 64;     // Минимальный размер для теста RAM в мегабайтах (1 MB = 1024 KB)
// ==================================================================== //

// ==================================================================== //

#include "PrintSystem.c"  // Функции вывода и записи результатов в файл
#include "CHeckSumMD5.c"  // MD5 Checksum
#include "SystemInfo.c"   // Функции сбора информации о системе
#include "BenchMemory.c"  // Модули тестов памяти
#include "BenchCompute.c" // Модули вычистительных тестов
#include "WaitKey.c"      // Функция паузы
#include "Help.c"         // Функция справки

// ==================================================================== //

void run_compute_suite(void) {
    int has_mmx   = __builtin_cpu_supports("mmx");
    int has_sse  = __builtin_cpu_supports("sse");
    int has_sse2 = __builtin_cpu_supports("sse2");
    int has_ssse3 = __builtin_cpu_supports("ssse3");
    int has_sse41 = __builtin_cpu_supports("sse4.1");
    int has_avx  = __builtin_cpu_supports("avx");
    int has_fma  = __builtin_cpu_supports("fma");
    int has_avx2 = __builtin_cpu_supports("avx2");
    int has_avx512f  = __builtin_cpu_supports("avx512f");
    int has_avx512bw = __builtin_cpu_supports("avx512bw");
    
    tee_printf("\n");
    tee_printf("-= Compute (%d thread(s)) =-\n", omp_get_max_threads());
    
    print_metric("FP32-Scalar    ", 1,                   bench_fp32_scalar,  "GFLOPS");
    print_metric("INT32-Scalar   ", 1,                   bench_int32_scalar, "GIOPS");
    print_metric("INT32-MMX      ", has_mmx,             bench_int32_mmx,    "GIOPS");
    print_metric("FP32-SSE       ", has_sse,             bench_fp32_sse,     "GFLOPS");
    print_metric("FP64-SSE2      ", has_sse2,            bench_fp64_sse2,    "GFLOPS");
    print_metric("INT8-SSE2      ", has_sse2,            bench_int8_sse2,    "GIOPS");
    print_metric("INT16-SSE2     ", has_sse2,            bench_int16_sse2,   "GIOPS");
    print_metric("INT32-SSE2     ", has_sse2,            bench_int32_sse2,   "GIOPS");
    print_metric("INT64-SSE2     ", has_sse2,            bench_int64_sse2,   "GIOPS");
    print_metric("INT8/16-SSSE3  ", has_ssse3,           bench_int8_ssse3,   "GIOPS");
    print_metric("INT32-SSE4.1   ", has_sse41,           bench_int32_sse41,  "GIOPS");
    print_metric("FP32-AVX       ", has_avx,             bench_fp32_avx,     "GFLOPS");
    print_metric("FP64-AVX       ", has_avx,             bench_fp64_avx,     "GFLOPS");
    print_metric("FP32-AVX+FMA   ", has_avx && has_fma,  bench_fp32_avx_fma, "GFLOPS");
    print_metric("FP64-AVX+FMA   ", has_avx && has_fma,  bench_fp64_avx_fma, "GFLOPS");
    print_metric("INT8-AVX2      ", has_avx2,            bench_int8_avx2,    "GIOPS");
    print_metric("INT16-AVX2     ", has_avx2,            bench_int16_avx2,   "GIOPS");
    print_metric("INT32-AVX2     ", has_avx2,            bench_int32_avx2,   "GIOPS");
    print_metric("INT64-AVX2     ", has_avx2,            bench_int64_avx2,   "GIOPS");
    print_metric("INT8-AVX512    ", has_avx512bw,        bench_int8_avx512,  "GIOPS");
    print_metric("INT16-AVX512   ", has_avx512bw,        bench_int16_avx512, "GIOPS");
    print_metric("INT32-AVX512   ", has_avx512f,         bench_int32_avx512, "GIOPS");
    print_metric("INT64-AVX512   ", has_avx512f,         bench_int64_avx512, "GIOPS");
    print_metric("FP32-AVX512+FMA", has_avx512f,         bench_fp32_avx512,  "GFLOPS");
    print_metric("FP64-AVX512+FMA", has_avx512f,         bench_fp64_avx512,  "GFLOPS");
}

char lbl[64];

void run_cachemem_read_test(void) {
    tee_printf("-= Cache & Memory Read (%d thread(s)) =-\n", omp_get_max_threads());
    snprintf(lbl, sizeof(lbl), "L1 Read (%zu KB/T)", get_l1_test_size() / 1024); print_metric(lbl, 1, bench_l1_bw, "GB/s");
    snprintf(lbl, sizeof(lbl), "L2 Read (%zu KB/T)", get_l2_test_size() / 1024); print_metric(lbl, 1, bench_l2_bw, "GB/s");
    if (g_cache.l3_bytes > 0) {
        snprintf(lbl, sizeof(lbl), "L3 Read (%.1f MB)", (double)get_l3_test_size() / (1024.0 * 1024.0));
        print_metric(lbl, 1, bench_l3_bw, "GB/s");
    } else {
        print_metric("L3 Read", 0, NULL, "GB/s");
    }
    snprintf(lbl, sizeof(lbl), "RAM Read (%zu MB)",   get_ram_test_size() / (1024 * 1024));
    print_metric(lbl, 1, bench_ram_rd, "GB/s");
}

void run_cachemem_write_test(void) {
    tee_printf("-= Cache & Memory Write (%d thread(s)) =-\n", omp_get_max_threads());
    snprintf(lbl, sizeof(lbl), "L1 Write (%zu KB/T)", get_l1_test_size() / 1024); print_metric(lbl, 1, bench_l1_wr, "GB/s");
    snprintf(lbl, sizeof(lbl), "L2 Write (%zu KB/T)", get_l2_test_size() / 1024); print_metric(lbl, 1, bench_l2_wr, "GB/s");
    if (g_cache.l3_bytes > 0) {
        snprintf(lbl, sizeof(lbl), "L3 Write (%.1f MB)", (double)get_l3_test_size() / (1024.0 * 1024.0));
        print_metric(lbl, 1, bench_l3_wr, "GB/s");
    } else {
        print_metric("L3 Write", 0, NULL, "GB/s");
    }
    snprintf(lbl, sizeof(lbl), "RAM Write (%zu MB)",   get_ram_test_size() / (1024 * 1024));
    print_metric(lbl, 1, bench_ram_wr, "GB/s");
}


void run_latency_test(void) {
    omp_set_num_threads(1);
    tee_printf("-= Latency =-\n");
    snprintf(lbl, sizeof(lbl), "L1 Latency (%zu KB)",   get_l1_test_size() / 1024); print_metric(lbl, 1, bench_l1_lat, "ns");
    snprintf(lbl, sizeof(lbl), "L2 Latency (%zu KB)",   get_l2_test_size() / 1024); print_metric(lbl, 1, bench_l2_lat, "ns");
    if (g_cache.l3_bytes > 0) {
        snprintf(lbl, sizeof(lbl), "L3 Latency (%.1f MB)", (double)get_l3_test_size() / (1024.0 * 1024.0));
        print_metric(lbl, 1, bench_l3_lat, "ns");
    } else {
        print_metric("L3 Latency", 0, NULL, "ns");
    }
    snprintf(lbl, sizeof(lbl), "RAM Latency (%zu MB)",  get_ram_test_size() / (1024 * 1024)); print_metric(lbl, 1, bench_ram_lat, "ns");
}

// ==================================================================== //

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001); // Переключаем вывод консоли Windows в UTF-8
#endif
    
    int max_threads = omp_get_max_threads(); // Первое получение и сохранение максимального количества потоков
    int skip_cahcemem = 0;
    int skip_compute = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0) { help_function(); } // Вызвав функцию справки назад дороги нет...
        if (strcmp(argv[i], "-fast") == 0 || strcmp(argv[i], "--fast") == 0) { TARGET_SECONDS = TARGET_SECONDS_FAST; }
        if (strcmp(argv[i], "-skip-cachemem") == 0 || strcmp(argv[i], "--skip-cachemem") == 0) { skip_cahcemem = 1; }
        if (strcmp(argv[i], "-skip-compute") == 0 || strcmp(argv[i], "--skip-compute") == 0) { skip_compute = 1; }
    }
    
    MKDIR("Results");
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char filename[256];
    snprintf(filename, sizeof(filename), "Results/benchmark_%04d-%02d-%02d_%02d-%02d-%02d.txt",
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);
    g_log_file = fopen(filename, "w");
    if (!g_log_file) { printf("Warning: Could not open %s for writing.\n", filename); }
    
    char exe_md5[33] = {0};
    get_binary_md5(argv[0], exe_md5);        // Получение MD5 хэша исполняемого файла
    tee_printf("%s", ProjectName); tee_printf("%s", ProjectLicense);
    tee_printf("Executable MD5: %s - %04d-%02d-%02d %02d:%02d:%02d \n", exe_md5,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    
    detect_cache_sizes();                    // Определение размера кэшей
    print_system_summary();                  // Получение и вывод информации о системе
    
    
    // Cache and Memory tests
    if (skip_cahcemem == 0) {
        omp_set_num_threads(1);
        run_cachemem_read_test();
        run_cachemem_write_test();
        
        if (max_threads > 1) {
            omp_set_num_threads(max_threads);
            run_cachemem_read_test();
            run_cachemem_write_test();
        }
        
        run_latency_test();
    } else {
        tee_printf("-= Cache & Memory Subsystem (Skipped)\n");
    }
    
    // Compute tests
    if (skip_compute == 0) {
        omp_set_num_threads(1);
        run_compute_suite();

        if (max_threads > 1) {
            omp_set_num_threads(max_threads);
            run_compute_suite();
        }
    } else {
        tee_printf("-= Compute (Skipped)\n");
    }
    
    if (g_log_file) fclose(g_log_file);
    wait_for_exit_key(); // Pause...
    return 0;
}
