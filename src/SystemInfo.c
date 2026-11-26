//MIT License
//Copyright (c) 2026 Андрей Цымбалов (Chimbal)

// --- CACHE TOPOLOGY DETECTION (CPUID) & ADAPTIVE BENCHMARKS ---
typedef struct {
    size_t l1d_bytes;
    size_t l1i_bytes;
    size_t l2_bytes;
    size_t l3_bytes;
} CacheInfo;

static CacheInfo g_cache = { 32ULL * 1024, 32ULL * 1024, 256ULL * 1024, 0 }; // L3 по умолчанию 0 (может отсутствовать)

typedef struct {
    double base_mhz;
    double max_mhz;
    double measured_mhz;
} CpuClocks;

static inline uint64_t read_tsc_serialized(void) {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(0, &eax, &ebx, &ecx, &edx);
    unsigned int lo, hi;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// Замер реальной текущей частоты ядра через цепочку из 256 зависимых инструкций ADD (1 такт = 1 ADD)
static double measure_active_boost_mhz(void) {
    uint32_t acc = 1;
    double t_warm = get_time_sec();
    // Прогрев 50 мс для выхода ядра в максимальный P-state (Turbo Boost / Precision Boost)
    while ((get_time_sec() - t_warm) < 0.05) {
        for (int i = 0; i < 1000; ++i) {
            asm volatile("addl $1, %0\n\t addl $1, %0\n\t addl $1, %0\n\t addl $1, %0\n\t" : "+r"(acc));
        }
    }

    const uint64_t block_cycles = 256ULL;
    const uint64_t batch = 200000ULL;
    uint64_t total_cycles = 0;

    double t1 = get_time_sec();
    double elapsed = 0.0;
    while (elapsed < 0.1) { // Замер за 100 мс
        for (uint64_t i = 0; i < batch; ++i) {
            asm volatile(
                ".rept 256\n\t"
                "addl $1, %0\n\t"
                ".endr\n\t"
                : "+r"(acc)
            );
        }
        total_cycles += batch * block_cycles;
        elapsed = get_time_sec() - t1;
    }
    volatile uint32_t sink = acc; (void)sink;

    return (elapsed > 0.0) ? ((double)total_cycles / (elapsed * 1e6)) : 0.0;
}

CpuClocks get_cpu_clocks(void) {
    CpuClocks clk = { 0.0, 0.0, 0 };
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned int max_leaf = __get_cpuid_max(0, NULL);

    // 1. Intel CPUID Leaf 0x16 (Skylake и новее: 6-е – 14-е поколения Core)
    if (max_leaf >= 0x16 && __get_cpuid(0x16, &eax, &ebx, &ecx, &edx)) {
        unsigned int base = eax & 0xFFFF;
        unsigned int max  = ebx & 0xFFFF;
        if (base > 0) clk.base_mhz = (double)base;
        if (max > 0)  clk.max_mhz  = (double)max;
    }

    // 2. Калибровка базовой частоты через RDTSC (если Leaf 0x16 не вернул base_mhz)
    if (clk.base_mhz <= 0.0 && __get_cpuid(1, &eax, &ebx, &ecx, &edx) && (edx & (1U << 4))) {
        double t_warm = get_time_sec();
        while ((get_time_sec() - t_warm) < 0.01) {}

        double t1 = get_time_sec();
        uint64_t tsc1 = read_tsc_serialized();
        while ((get_time_sec() - t1) < 0.1) {}
        uint64_t tsc2 = read_tsc_serialized();
        double t2 = get_time_sec();

        if (t2 > t1 && tsc2 > tsc1) {
            clk.base_mhz = (double)(tsc2 - tsc1) / ((t2 - t1) * 1e6);
        }
    }

    // 3. Паспортный Max Turbo может быть неизвестен (AMD Ryzen/Athlon, старые Intel) - замеряем реальный буст
    double measured_clock = measure_active_boost_mhz();
    clk.measured_mhz = measured_clock;

    return clk;
}

const char* get_openmp_version_str(void) {
#ifdef _OPENMP
    switch (_OPENMP) {
        case 200505: return "2.5";
        case 200805: return "3.0";
        case 201107: return "3.1";
        case 201307: return "4.0";
        case 201511: return "4.5";
        case 201811: return "5.0";
        case 202011: return "5.1";
        case 202111: return "5.2";
        case 202411: return "6.0";
        default: {
            static char buf[32];
            snprintf(buf, sizeof(buf), "%d", _OPENMP);
            return buf;
        }
    }
#else
    return "Disabled";
#endif
}

// Вспомогательная функция: определяет ядро/поколение, Family/Model/Stepping и наборы инструкций
static void get_cpu_details(char *arch_buf, size_t arch_size, char *isa_buf, size_t isa_size) {
    arch_buf[0] = '\0';
    isa_buf[0] = '\0';

    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned int max_leaf = __get_cpuid_max(0, NULL);
    unsigned int max_ext  = __get_cpuid_max(0x80000000, NULL);
    if (max_leaf < 1) return;

    unsigned int v_ebx = 0, v_ecx = 0, v_edx = 0;
    __get_cpuid(0, &eax, &v_ebx, &v_ecx, &v_edx);
    int is_amd       = (v_ebx == 0x68747541 && v_edx == 0x69746e65 && v_ecx == 0x444d4163); // AuthenticAMD
    int is_intel     = (v_ebx == 0x756e6547 && v_edx == 0x49656e69 && v_ecx == 0x6c65746e); // GenuineIntel
    int is_via       = (v_ebx == 0x746e6543 && v_edx == 0x48727561 && v_ecx == 0x736c7561) || // CentaurHauls
                       (v_ebx == 0x20414956 && v_edx == 0x20414956 && v_ecx == 0x20414956);   // VIA VIA VIA
    int is_cyrix     = (v_ebx == 0x69727943 && v_edx == 0x736e4978 && v_ecx == 0x64616574); // CyrixInstead
    int is_transmeta = (v_ebx == 0x756e6547 && v_edx == 0x54656e69 && v_ecx == 0x3638784d); // GenuineTMx86

    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    unsigned int stepping   = eax & 0xF;
    unsigned int base_model = (eax >> 4) & 0xF;
    unsigned int base_fam   = (eax >> 8) & 0xF;
    unsigned int ext_model  = (eax >> 16) & 0xF;
    unsigned int ext_fam    = (eax >> 20) & 0xFF;

    unsigned int family = base_fam + ((base_fam == 0xF) ? ext_fam : 0);
    unsigned int model  = base_model + ((base_fam == 0x6 || base_fam == 0xF) ? (ext_model << 4) : 0);

    const char *arch = NULL;

    // 1. Расшифровка микроархитектуры (ядра)
    if (is_amd) {
        switch (family) {
            case 5: arch = (model >= 8) ? "AMD K6-2/III" : "AMD K5/K6"; break;
            case 6:
                if (model <= 4)       arch = "K7 Pluto/Thunderbird";
                else if (model == 6)  arch = "K7 Palomino";
                else if (model == 8)  arch = "K7 Thoroughbred";
                else if (model == 10) arch = "K7 Barton/Thorton";
                else                  arch = "AMD K7";
                break;
            case 15: arch = "AMD K8 (Athlon 64/Sempron)"; break;
            case 16: arch = "AMD K10 (Phenom/Athlon II)"; break;
            case 18: arch = "AMD Llano (K10.5)"; break;
            case 20: arch = "AMD Bobcat"; break;
            case 21: arch = "AMD Bulldozer/Piledriver/Excavator"; break;
            case 22: arch = "AMD Jaguar/Puma"; break;
            case 23: arch = (model >= 0x30) ? "AMD Zen 2" : "AMD Zen/Zen+"; break;
            case 25: arch = (model >= 0x60) ? "AMD Zen 4" : "AMD Zen 3"; break;
            case 26: arch = "AMD Zen 5"; break;
        }
    } else if (is_intel) {
        if (family == 5) {
            arch = (edx & (1U << 23)) ? "Pentium MMX" : "Pentium (P5)";
        } else if (family == 6) {
            switch (model) {
                case 1:  arch = "Pentium Pro"; break;
                case 3: case 5: case 6: arch = "Pentium II / Celeron"; break;
                case 7: case 8: case 10: case 11: arch = "Pentium III"; break;
                case 9: case 13: arch = "Pentium M (Banias/Dothan)"; break;
                case 14: arch = "Core Solo/Duo (Yonah)"; break;
                case 15: case 22: arch = "Core 2 (65nm Conroe/Merom)"; break;
                case 23: case 29: arch = "Core 2 (45nm Penryn)"; break;
                case 26: case 30: case 31: case 46: arch = "Nehalem (1st Gen Core)"; break;
                case 37: case 44: case 47: arch = "Westmere (1st Gen Core 32nm)"; break;
                case 42: case 45: arch = "Sandy Bridge (2nd Gen Core)"; break;
                case 58: case 62: arch = "Ivy Bridge (3rd Gen Core)"; break;
                case 60: case 63: case 69: case 70: arch = "Haswell (4th Gen Core)"; break;
                case 61: case 71: case 79: case 86: arch = "Broadwell (5th Gen Core)"; break;
                case 78: case 94: case 85: arch = "Skylake (6th Gen Core)"; break;
                case 142: case 158: arch = "Kaby/Coffee Lake (7th-9th Gen)"; break;
                case 165: case 166: arch = "Comet Lake (10th Gen Core)"; break;
                case 125: case 126: arch = "Ice Lake (10th Gen 10nm)"; break;
                case 140: case 141: arch = "Tiger Lake (11th Gen Core)"; break;
                case 167: arch = "Rocket Lake (11th Gen Core)"; break;
                case 151: case 154: arch = "Alder Lake (12th Gen Core)"; break;
                case 183: case 186: case 191: arch = "Raptor Lake (13th-14th Gen)"; break;
                case 170: case 172: arch = "Meteor Lake (Core Ultra S1)"; break;
                case 189: case 198: arch = "Lunar/Arrow Lake (Core Ultra S2)"; break;
                case 28: case 38: case 39: case 53: case 54: arch = "Atom (Bonnell/Saltwell)"; break;
                case 55: case 74: case 76: arch = "Atom (Silvermont/Airmont)"; break;
                case 92: case 95: case 122: arch = "Atom (Goldmont/Plus)"; break;
                case 134: case 150: case 156: arch = "Atom (Tremont)"; break;
                case 190: arch = "Alder Lake-N (Gracemont)"; break;
            }
        } else if (family == 15) {
            if (model <= 2)      arch = "NetBurst (Willamette/Northwood)";
            else if (model <= 4) arch = "NetBurst (Prescott)";
            else                 arch = "NetBurst (Cedar Mill/Presler)";
        }
    } else if (is_via) {
        arch = (family == 6) ? "VIA C3 / C7 / Nano" : "IDT WinChip";
    } else if (is_cyrix) {
        arch = "Cyrix 6x86 / MII / MediaGX";
    } else if (is_transmeta) {
        arch = "Transmeta Crusoe / Efficeon";
    }

    if (arch) {
        snprintf(arch_buf, arch_size, " [%s | Fam %u, Mod %u, Step %u]", arch, family, model, stepping);
    } else {
        snprintf(arch_buf, arch_size, " [Fam %u, Mod %u, Step %u]", family, model, stepping);
    }

    // 2. Сбор поддерживаемых наборов инструкций (ISA)
    size_t pos = 0;
    #define APPEND_ISA(name) do { \
        if (pos < isa_size) \
            pos += snprintf(isa_buf + pos, isa_size - pos, "%s%s", (pos > 0 ? ", " : ""), name); \
    } while(0)

    // Базовые расширения (Leaf 1 EDX/ECX)
    if (edx & (1U << 0))  APPEND_ISA("x87");
    if (edx & (1U << 15)) APPEND_ISA("CMOV");
    if (edx & (1U << 23)) APPEND_ISA("MMX");

    // Расширенные инструкции AMD (Leaf 0x80000001: MMX+, 3DNow!, 3DNow!+, SSE4a, XOP, FMA4)
    unsigned int ext_eax = 0, ext_ebx = 0, ext_ecx = 0, ext_edx = 0;
    int has_ext1 = (max_ext >= 0x80000001) && __get_cpuid(0x80000001, &ext_eax, &ext_ebx, &ext_ecx, &ext_edx);

    if (has_ext1 && (ext_edx & (1U << 22))) APPEND_ISA("MMX+");
    if (edx & (1U << 25)) APPEND_ISA("SSE");
    if (edx & (1U << 26)) APPEND_ISA("SSE2");
    if (ecx & (1U << 0))  APPEND_ISA("SSE3");
    if (ecx & (1U << 9))  APPEND_ISA("SSSE3");
    if (has_ext1 && (ext_ecx & (1U << 6)))  APPEND_ISA("SSE4a");
    if (ecx & (1U << 19)) APPEND_ISA("SSE4.1");
    if (ecx & (1U << 20)) APPEND_ISA("SSE4.2");

    // 3DNow! и классические расширения AMD (Bulldozer/Piledriver)
    if (has_ext1) {
        if (ext_edx & (1U << 31)) APPEND_ISA("3DNow!");
        if (ext_edx & (1U << 30)) APPEND_ISA("3DNow!+");
        if (ext_ecx & (1U << 11)) APPEND_ISA("XOP");
        if (ext_ecx & (1U << 16)) APPEND_ISA("FMA4");
    }

    // Векторные, криптографические и битовые расширения
    if (ecx & (1U << 25)) APPEND_ISA("AES");
    if (ecx & (1U << 1))  APPEND_ISA("PCLMUL");
    if (ecx & (1U << 23)) APPEND_ISA("POPCNT");
    if (ecx & (1U << 29)) APPEND_ISA("F16C");
    if (ecx & (1U << 28)) APPEND_ISA("AVX");
    if (ecx & (1U << 12)) APPEND_ISA("FMA3");

    // Расширенные инструкции Leaf 7 (AVX2, BMI1/2, SHA, AVX-512)
    if (max_leaf >= 7) {
        unsigned int l7_eax = 0, l7_ebx = 0, l7_ecx = 0, l7_edx = 0;
        __cpuid_count(7, 0, l7_eax, l7_ebx, l7_ecx, l7_edx);
        if (l7_ebx & (1U << 5))  APPEND_ISA("AVX2");
        if (l7_ebx & (1U << 3))  APPEND_ISA("BMI1");
        if (l7_ebx & (1U << 8))  APPEND_ISA("BMI2");
        if (l7_ebx & (1U << 29)) APPEND_ISA("SHA");
        if (l7_ebx & (1U << 16)) APPEND_ISA("AVX-512F");
        if (l7_ebx & (1U << 30)) APPEND_ISA("AVX-512BW");
        if (l7_ebx & (1U << 31)) APPEND_ISA("AVX-512VL");
        if (l7_ecx & (1U << 11)) APPEND_ISA("AVX-512VNNI");
    }

    if (pos == 0) APPEND_ISA("x86 Base");
    #undef APPEND_ISA

}

// Определение поддержки и статуса аппаратной виртуализации (Intel VT-x / AMD-V)
static const char* get_virtualization_status(void) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned int max_leaf = __get_cpuid_max(0, NULL);
    unsigned int max_ext  = __get_cpuid_max(0x80000000, NULL);
    if (max_leaf < 1) return "Not Supported";

    unsigned int v_ebx = 0, v_ecx = 0, v_edx = 0;
    __get_cpuid(0, &eax, &v_ebx, &v_ecx, &v_edx);
    int is_amd   = (v_ebx == 0x68747541 && v_edx == 0x69746e65 && v_ecx == 0x444d4163);
    int is_intel = (v_ebx == 0x756e6547 && v_edx == 0x49656e69 && v_ecx == 0x6c65746e);

    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    int has_vtx = is_intel && (ecx & (1U << 5));
    int in_hypervisor = (ecx & (1U << 31)) != 0; // Запущено под гипервизором (VMware, VirtualBox, Hyper-V, WSL)

    int has_amdv = 0;
    if (is_amd && max_ext >= 0x80000001) {
        unsigned int ext_eax = 0, ext_ebx = 0, ext_ecx = 0, ext_edx = 0;
        if (__get_cpuid(0x80000001, &ext_eax, &ext_ebx, &ext_ecx, &ext_edx)) {
            has_amdv = (ext_ecx & (1U << 2)) != 0;
        }
    }

    if (!has_vtx && !has_amdv) {
        if (in_hypervisor) return "Running under Hypervisor (Nested VT off)";
        return "Not Supported";
    }

    const char *tech = has_vtx ? "Intel VT-x" : "AMD-V";
    int is_enabled = -1; // -1 = неизвестно, 0 = выключено в BIOS, 1 = включено

#ifdef _WIN32
    #ifndef PF_VIRT_FIRMWARE_ENABLED
    #define PF_VIRT_FIRMWARE_ENABLED 21
    #endif

    // Получаем версию Windows: в Windows 8 и новее версия >= 6.2
    DWORD ver = GetVersion();
    DWORD major = (DWORD)(LOBYTE(LOWORD(ver)));
    DWORD minor = (DWORD)(HIBYTE(LOWORD(ver)));
    int is_win8_or_newer = (major > 6) || (major == 6 && minor >= 2);

    if (IsProcessorFeaturePresent(PF_VIRT_FIRMWARE_ENABLED)) {
        is_enabled = 1;
    } else if (in_hypervisor) {
        is_enabled = 1;
    } else if (is_win8_or_newer) {
        // На Windows 8/10/11 возврат FALSE действительно означает, что виртуализация выключена в BIOS
        is_enabled = 0;
    } else {
        // На Windows XP / Vista / 7 API проверки BIOS отсутствует — оставляем -1 ("Supported")
        is_enabled = -1;
    }
#elif defined(__linux__)

    // Ядро Linux убирает vmx/svm из /proc/cpuinfo, если виртуализация заблокирована в BIOS
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        is_enabled = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "flags", 5) == 0) {
                if (strstr(line, " vmx") || strstr(line, " svm")) {
                    is_enabled = 1;
                }
                break;
            }
        }
        fclose(f);
    }
#endif

    static char virt_buf[64];
    if (is_enabled == 1) {
        snprintf(virt_buf, sizeof(virt_buf), "%s (Enabled%s)", tech, in_hypervisor ? ", Hypervisor Active" : "");
    } else if (is_enabled == 0) {
        snprintf(virt_buf, sizeof(virt_buf), "%s (Disabled in BIOS)", tech);
    } else {
        snprintf(virt_buf, sizeof(virt_buf), "%s (Supported)", tech);
    }
    return virt_buf;
}



void print_system_summary(void) {
    char brand[64] = "Unknown Processor";
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) && eax >= 0x80000004) {
        __get_cpuid(0x80000002, (unsigned int*)(brand + 0),  (unsigned int*)(brand + 4),
                                (unsigned int*)(brand + 8),  (unsigned int*)(brand + 12));
        __get_cpuid(0x80000003, (unsigned int*)(brand + 16), (unsigned int*)(brand + 20),
                                (unsigned int*)(brand + 24), (unsigned int*)(brand + 28));
        __get_cpuid(0x80000004, (unsigned int*)(brand + 32), (unsigned int*)(brand + 36),
                                (unsigned int*)(brand + 40), (unsigned int*)(brand + 44));
    }
    char *cpu_str = brand;
    while (*cpu_str == ' ') cpu_str++;

    double ram_gb = 0.0;
    char os_info[1024] = "Unknown OS";

#ifdef _WIN32
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        ram_gb = (double)mem.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
    }

    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);
    const char *arch_str = (si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64) ? "x86_64" : "x86";

    // Получаем реальную версию Windows напрямую из ntdll.dll без реестра и манифестов
    typedef LONG (WINAPI *RtlGetVersionPtr)(PRTL_OSVERSIONINFOEXW);
    RTL_OSVERSIONINFOEXW osvi;
    ZeroMemory(&osvi, sizeof(osvi));
    osvi.dwOSVersionInfoSize = sizeof(osvi);

    HMODULE hNtdll = GetModuleHandleA("ntdll.dll");
    RtlGetVersionPtr pRtlGetVersion = hNtdll ? (RtlGetVersionPtr)GetProcAddress(hNtdll, "RtlGetVersion") : NULL;

    if (pRtlGetVersion && pRtlGetVersion(&osvi) == 0) {
        int is_server = (osvi.wProductType != VER_NT_WORKSTATION);
        const char *win_name = "Windows";

        if (osvi.dwMajorVersion == 10 && osvi.dwMinorVersion == 0) {
            if (is_server) {
                if (osvi.dwBuildNumber >= 26100)      win_name = "Windows Server 2025";
                else if (osvi.dwBuildNumber >= 20348) win_name = "Windows Server 2022";
                else if (osvi.dwBuildNumber >= 17763) win_name = "Windows Server 2019";
                else                                  win_name = "Windows Server 2016";
            } else {
                // Windows 11 начинается с Build 22000
                win_name = (osvi.dwBuildNumber >= 22000) ? "Windows 11" : "Windows 10";
            }
        } else if (osvi.dwMajorVersion == 6) {
            if (osvi.dwMinorVersion == 3)      win_name = is_server ? "Windows Server 2012 R2" : "Windows 8.1";
            else if (osvi.dwMinorVersion == 2) win_name = is_server ? "Windows Server 2012"    : "Windows 8";
            else if (osvi.dwMinorVersion == 1) win_name = is_server ? "Windows Server 2008 R2" : "Windows 7";
            else if (osvi.dwMinorVersion == 0) win_name = is_server ? "Windows Server 2008"    : "Windows Vista";
        } else if (osvi.dwMajorVersion == 5) {
            if (osvi.dwMinorVersion == 2)      win_name = is_server ? "Windows Server 2003"    : "Windows XP x64";
            else if (osvi.dwMinorVersion == 1) win_name = "Windows XP";
            else if (osvi.dwMinorVersion == 0) win_name = "Windows 2000";
        }

        snprintf(os_info, sizeof(os_info), "%s (Build %lu, %s)",
                 win_name, (unsigned long)osvi.dwBuildNumber, arch_str);
    } else {
        snprintf(os_info, sizeof(os_info), "Windows (%s)", arch_str);
    }
#else
    struct utsname uts;
    if (uname(&uts) == 0) {
        snprintf(os_info, sizeof(os_info), "%s %s (%s)", uts.sysname, uts.release, uts.machine);
    }
    #ifdef __FreeBSD__
        uint64_t physmem = 0;
        size_t len = sizeof(physmem);
        if (sysctlbyname("hw.physmem", &physmem, &len, NULL, 0) == 0) {
            ram_gb = (double)physmem / (1024.0 * 1024.0 * 1024.0);
        }
    #else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            ram_gb = ((double)pages * (double)page_size) / (1024.0 * 1024.0 * 1024.0);
        }
    #endif
#endif

    CpuClocks clk = get_cpu_clocks();
    tee_printf("-=====================- System Summary -=====================-\n");
    tee_printf(" CPU:      %s\n", cpu_str);
    char cpu_arch[128];
    char cpu_isa[512];
    get_cpu_details(cpu_arch, sizeof(cpu_arch), cpu_isa, sizeof(cpu_isa));
    tee_printf(" Arch:    %s\n", cpu_arch);
    tee_printf(" ISA:      %s\n", cpu_isa); // Instruction Set Architecture
    tee_printf(" Virt:     %s\n", get_virtualization_status());
    
    if (clk.base_mhz > 0.0) {
        if (clk.max_mhz > clk.base_mhz * 1.02) {
            tee_printf(" Clock:    %.0f MHz Base | %.0f MHz Boost | %.0f MHz Measured\n", clk.base_mhz, clk.max_mhz, clk.measured_mhz);
        } else {
            tee_printf(" Clock:    %.0f MHz Base | %.0f MHz Measured \n", clk.base_mhz, clk.measured_mhz);
        }
    } else {
        tee_printf(" Clock:    %.0f MHz Measured \n", clk.measured_mhz);
    }
    
    if (g_cache.l3_bytes > 0) {
        if (g_cache.l1i_bytes > 0) {
            tee_printf(" Cache:    L1D %zu KB | L1I %zu KB | L2 %zu KB | L3 %.1f MB\n",
                       g_cache.l1d_bytes / 1024, g_cache.l1i_bytes / 1024,
                       g_cache.l2_bytes / 1024, (double)g_cache.l3_bytes / (1024.0 * 1024.0));
        } else {
            tee_printf(" Cache:    L1D %zu KB | L2 %zu KB | L3 %.1f MB\n",
                       g_cache.l1d_bytes / 1024, g_cache.l2_bytes / 1024,
                       (double)g_cache.l3_bytes / (1024.0 * 1024.0));
        }
    } else {
        if (g_cache.l1i_bytes > 0) {
            tee_printf(" Cache:    L1D %zu KB | L1I %zu KB | L2 %zu KB | L3 None\n",
                       g_cache.l1d_bytes / 1024, g_cache.l1i_bytes / 1024,
                       g_cache.l2_bytes / 1024);
        } else {
            tee_printf(" Cache:    L1D %zu KB | L2 %zu KB | L3 None\n",
                       g_cache.l1d_bytes / 1024, g_cache.l2_bytes / 1024);
        }
    }

    

    tee_printf(" Threads:  %d logical processor(s)\n", omp_get_max_threads());
    tee_printf(" Memory:   %.2f GB RAM\n", ram_gb);
    tee_printf(" OS:       %s\n", os_info);
    tee_printf(" Compiler: GCC %d.%d.%d (OpenMP %s) | Test duration: %.1f s\n\n",
               __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__, get_openmp_version_str(), TARGET_SECONDS);
}

// Декодер однобайтовых дескрипторов Intel (Leaf 0x02) для Pentium II / III / 4 / M
static void parse_intel_leaf2_desc(uint8_t desc) {
    switch (desc) {
        // L1 Instruction Cache (и Trace Cache у Pentium 4)
        case 0x06: g_cache.l1i_bytes = 8ULL * 1024; break;
        case 0x08: case 0x71: g_cache.l1i_bytes = 16ULL * 1024; break; // 0x71 = 16K uOps Trace Cache (P4)
        case 0x09: case 0x30: case 0x72: g_cache.l1i_bytes = 32ULL * 1024; break;
        case 0x70: g_cache.l1i_bytes = 12ULL * 1024; break; // 12K uOps Trace Cache (P4 Willamette/Northwood)

        // L1 Data Cache
        case 0x0A: case 0x66: g_cache.l1d_bytes = 8ULL * 1024; break;
        case 0x0C: case 0x60: case 0x67: g_cache.l1d_bytes = 16ULL * 1024; break;
        case 0x0D: case 0x68: case 0x2C: g_cache.l1d_bytes = 32ULL * 1024; break;
        case 0x0E: g_cache.l1d_bytes = 24ULL * 1024; break;

        // L2 Unified Cache
        case 0x41: case 0x79: case 0x39: case 0x3B: g_cache.l2_bytes = 128ULL * 1024; break;
        case 0x3A: g_cache.l2_bytes = 192ULL * 1024; break;
        case 0x42: case 0x7A: case 0x3C: case 0x82: g_cache.l2_bytes = 256ULL * 1024; break;
        case 0x3D: g_cache.l2_bytes = 384ULL * 1024; break;
        case 0x43: case 0x7B: case 0x3E: case 0x7F: case 0x80: case 0x83: g_cache.l2_bytes = 512ULL * 1024; break;
        case 0x44: case 0x7C: case 0x78: case 0x84: g_cache.l2_bytes = 1024ULL * 1024; break;
        case 0x45: case 0x7D: case 0x85: g_cache.l2_bytes = 2048ULL * 1024; break;
        case 0x48: g_cache.l2_bytes = 3072ULL * 1024; break;
        case 0x49: case 0x4E: g_cache.l2_bytes = 4096ULL * 1024; break;
        case 0x86: g_cache.l2_bytes = 512ULL * 1024; break;
        case 0x87: g_cache.l2_bytes = 1024ULL * 1024; break;

        // L3 Unified Cache (Pentium 4 Extreme Edition, ранние Xeon)
        case 0x22: g_cache.l3_bytes = 512ULL * 1024; break;
        case 0x23: case 0xD0: g_cache.l3_bytes = 1024ULL * 1024; break;
        case 0xD1: case 0xD6: g_cache.l3_bytes = 2048ULL * 1024; break;
        case 0x25: g_cache.l3_bytes = 2048ULL * 1024; break;
        case 0xD2: case 0xD7: case 0xDC: g_cache.l3_bytes = 4096ULL * 1024; break;
        case 0x29: g_cache.l3_bytes = 4096ULL * 1024; break;
        case 0xD8: case 0xDD: case 0xE2: g_cache.l3_bytes = 8192ULL * 1024; break;
        case 0x46: g_cache.l3_bytes = 4096ULL * 1024; break;
        case 0x47: g_cache.l3_bytes = 8192ULL * 1024; break;
        case 0x4A: g_cache.l3_bytes = 6144ULL * 1024; break;
        case 0x4B: g_cache.l3_bytes = 8192ULL * 1024; break;
        case 0x4C: g_cache.l3_bytes = 12288ULL * 1024; break;
        case 0x4D: g_cache.l3_bytes = 16384ULL * 1024; break;
        default: break;
    }
}

void detect_cache_sizes(void) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned int max_leaf = __get_cpuid_max(0, NULL);
    unsigned int max_ext  = __get_cpuid_max(0x80000000, NULL);

    unsigned int vendor_ebx = 0, vendor_ecx = 0, vendor_edx = 0;
    __get_cpuid(0, &eax, &vendor_ebx, &vendor_ecx, &vendor_edx);
    int is_amd   = (vendor_ebx == 0x68747541 && vendor_edx == 0x69746e65 && vendor_ecx == 0x444d4163); // "AuthenticAMD"
    int is_intel = (vendor_ebx == 0x756e6547 && vendor_edx == 0x49656e69 && vendor_ecx == 0x6c65746e); // "GenuineIntel"

    // 1. Legacy Intel Cache Descriptors (Leaf 0x02) — Pentium II / III / 4 / M
    if (is_intel && max_leaf >= 0x02) {
        __cpuid_count(0x02, 0, eax, ebx, ecx, edx);
        unsigned int count = eax & 0xFF;
        for (unsigned int iter = 0; iter < count; ++iter) {
            if (iter > 0) __cpuid_count(0x02, 0, eax, ebx, ecx, edx);
            unsigned int regs[4] = { eax & ~0xFFU, ebx, ecx, edx };
            for (int r = 0; r < 4; ++r) {
                // Старший бит == 0 означает валидность регистра
                if ((regs[r] & 0x80000000U) == 0) {
                    for (int b = 0; b < 4; ++b) {
                        uint8_t desc = (regs[r] >> (b * 8)) & 0xFF;
                        if (desc != 0x00 && desc != 0xFF) {
                            parse_intel_leaf2_desc(desc);
                        }
                    }
                }
            }
        }
    }

    // 2. Legacy AMD L1D & L1I (Leaf 0x80000005) — K6-2+ / K7 / K8 / K10 (Athlon XP / 64 / 7550)
    if (is_amd && max_ext >= 0x80000005) {
        __get_cpuid(0x80000005, &eax, &ebx, &ecx, &edx);
        size_t l1d_kb = (ecx >> 24) & 0xFF;
        size_t l1i_kb = (edx >> 24) & 0xFF;
        if (l1d_kb > 0) g_cache.l1d_bytes = l1d_kb * 1024;
        if (l1i_kb > 0) g_cache.l1i_bytes = l1i_kb * 1024;
    }

    // 3. Legacy Extended L2 & AMD L3 (Leaf 0x80000006) — K7 / K8 / K10 / старые Intel / VIA
    if (max_ext >= 0x80000006) {
        __get_cpuid(0x80000006, &eax, &ebx, &ecx, &edx);
        size_t l2_kb = (ecx >> 16) & 0xFFFF;
        if (l2_kb > 0) g_cache.l2_bytes = l2_kb * 1024;

        if (is_amd) {
            size_t l3_512kb_blocks = (edx >> 18) & 0x3FFF;
            if (l3_512kb_blocks > 0) {
                g_cache.l3_bytes = l3_512kb_blocks * 512ULL * 1024;
            }
        }
    }

    // 4. Deterministic Cache Parameters (Leaf 0x04) — Pentium 4 (Prescott+), Core, Core 2, современные Intel
    if (max_leaf >= 0x04) {
        for (unsigned int i = 0; i < 16; ++i) {
            __cpuid_count(0x04, i, eax, ebx, ecx, edx);
            unsigned int cache_type  = eax & 0x1F;
            unsigned int cache_level = (eax >> 5) & 0x07;
            if (cache_type == 0) break;
            if (cache_type == 1 || cache_type == 2 || cache_type == 3) {
                size_t line_size  = (ebx & 0xFFF) + 1;
                size_t partitions = ((ebx >> 12) & 0x3FF) + 1;
                size_t ways       = ((ebx >> 22) & 0x3FF) + 1;
                size_t sets       = ecx + 1;
                size_t total_size = ways * partitions * line_size * sets;

                if (cache_level == 1 && cache_type == 1)      g_cache.l1d_bytes = total_size;
                else if (cache_level == 1 && cache_type == 2) g_cache.l1i_bytes = total_size;
                else if (cache_level == 2)                    g_cache.l2_bytes  = total_size;
                else if (cache_level == 3)                    g_cache.l3_bytes  = total_size;
            }
        }
    }

    // 5. AMD Extended Cache Topology (Leaf 0x8000001D) — Bulldozer / FX / Ryzen / EPYC
    if (max_ext >= 0x8000001D) {
        for (unsigned int i = 0; i < 16; ++i) {
            __cpuid_count(0x8000001D, i, eax, ebx, ecx, edx);
            unsigned int cache_type  = eax & 0x1F;
            unsigned int cache_level = (eax >> 5) & 0x07;
            if (cache_type == 0) break;
            if (cache_type == 1 || cache_type == 2 || cache_type == 3) {
                size_t line_size  = (ebx & 0xFFF) + 1;
                size_t partitions = ((ebx >> 12) & 0x3FF) + 1;
                size_t ways       = ((ebx >> 22) & 0x3FF) + 1;
                size_t sets       = ecx + 1;
                size_t total_size = ways * partitions * line_size * sets;

                if (cache_level == 1 && cache_type == 1)      g_cache.l1d_bytes = total_size;
                else if (cache_level == 1 && cache_type == 2) g_cache.l1i_bytes = total_size;
                else if (cache_level == 2)                    g_cache.l2_bytes  = total_size;
                else if (cache_level == 3)                    g_cache.l3_bytes  = total_size;
            }
        }
    }
}
