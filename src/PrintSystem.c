//MIT License
//Copyright (c) 2026 Андрей Цымбалов (Chimbal)

static FILE *g_log_file = NULL;

void tee_printf(const char *fmt, ...) {
    va_list args1, args2;
    va_start(args1, fmt);
    va_copy(args2, args1);
    vprintf(fmt, args1);
    fflush(stdout);
    if (g_log_file) {
        vfprintf(g_log_file, fmt, args2);
        fflush(g_log_file);
    }
    va_end(args1);
    va_end(args2);
}

void print_metric(const char *name, int supported, double (*fn)(void), const char *unit) {
    tee_printf(" %-24s", name);
    if (supported) {
        double val = fn();
        tee_printf("%8.2f %s\n", val, unit);
    } else {
        tee_printf("not supported\n");
    }
}
