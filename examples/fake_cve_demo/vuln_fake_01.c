/* 样例：栈缓冲 strcpy 溢出模式（勿用于生产） */
#include <stdio.h>
#include <string.h>

void vuln_fake_01(const char *input) {
    char buf[12];
    strcpy(buf, input);
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_01(s);
    return 0;
}
