/* 样例：双阶段 strcpy（勿用于生产） */
#include <stdio.h>
#include <string.h>

void vuln_fake_07(const char *input) {
    char tmp[6];
    char buf[10];
    strcpy(tmp, input);
    strcpy(buf, tmp);
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_07(s);
    return 0;
}
