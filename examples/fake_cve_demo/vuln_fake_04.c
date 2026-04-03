/* 样例：memcpy 长度误用（勿用于生产） */
#include <stdio.h>
#include <string.h>

void vuln_fake_04(const char *input) {
    char buf[8];
    size_t n = strlen(input);
    memcpy(buf, input, n);
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_04(s);
    return 0;
}
