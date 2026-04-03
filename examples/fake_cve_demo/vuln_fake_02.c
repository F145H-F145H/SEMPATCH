/* 样例：无界循环写入（勿用于生产） */
#include <stdio.h>

void vuln_fake_02(const char *input) {
    char buf[10];
    size_t i;
    for (i = 0; input[i] != '\0'; i++) {
        buf[i] = input[i];
    }
    buf[i] = '\0';
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_02(s);
    return 0;
}
