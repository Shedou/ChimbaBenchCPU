//MIT License
//Copyright (c) 2026 Андрей Цымбалов (Chimbal)

void help_function() {
    printf("%s", ProjectName);
    printf("%s", ProjectLicense);
    printf("\n-help | --help - Displays help (current page)");
    printf("\n-fast | --fast - Speeds up benchmark execution at the cost of result accuracy");
    printf("\n-skip-cachemem - Skip Cache & Memory test");
    printf("\n-skip-compute  - Skip Compute test\n");
    wait_for_exit_key(); // Pause...
    exit(0); // Закрывает программу без возврата к функции main()...
}
