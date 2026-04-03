/* 样例：strcat 至栈缓冲（勿用于生产）；亦作 query.elf 同源（Makefile 复制） */
#include <stdio.h>
#include <string.h>

void vuln_fake_05(const char *input) {
    char buf[14];
    buf[0] = '\0';
    strcat(buf, input);
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_05(s);
    return 0;
}
