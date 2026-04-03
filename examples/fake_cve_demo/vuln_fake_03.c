/* 样例：sprintf 至小缓冲（勿用于生产） */
#include <stdio.h>

void vuln_fake_03(const char *input) {
    char buf[16];
    (void)sprintf(buf, "%s", input);
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_03(s);
    return 0;
}
