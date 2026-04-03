/* 样例 1：strcpy 栈缓冲区溢出（勿用于生产） */
#include <stdio.h>
#include <string.h>

static int helper_add(int a, int b) {
    int r = a + b;
    if (r < 0) return 0;
    return r;
}

void vuln_buf_overflow(const char *input) {
    char buf[16];
    int val = helper_add(1, 2);
    if (input == NULL) {
        (void)printf("null\n");
        return;
    }
    if (val > 0) {
        strcpy(buf, input);
    }
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "test";
    vuln_buf_overflow(s);
    return 0;
}
