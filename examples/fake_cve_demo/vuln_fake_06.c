/* 样例：逐字节写入无上限（勿用于生产） */
#include <stdio.h>

void vuln_fake_06(const char *input) {
    char buf[9];
    int i = 0;
    while (input[i] != '\0') {
        buf[i] = input[i];
        i++;
    }
    buf[i] = '\0';
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_06(s);
    return 0;
}
