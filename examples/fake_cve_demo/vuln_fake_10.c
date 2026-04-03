/* 样例：while+strcpy 链（勿用于生产） */
#include <stdio.h>
#include <string.h>

void vuln_fake_10(const char *input) {
    char buf[15];
    const char *p = input;
    size_t k = 0;
    while (*p != '\0' && k < 100) {
        buf[k] = *p;
        p++;
        k++;
    }
    buf[k] = '\0';
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_10(s);
    return 0;
}
