/* 样例： strncpy 未保证终止 + 再打印（勿用于生产） */
#include <stdio.h>
#include <string.h>

void vuln_fake_08(const char *input) {
    char buf[11];
    strncpy(buf, input, sizeof(buf));
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_08(s);
    return 0;
}
