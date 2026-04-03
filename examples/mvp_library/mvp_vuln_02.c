/* 样例 2：无界循环写入（勿用于生产） */
#include <stdio.h>
#include <string.h>

static int check_flag(int x) {
    if (x > 100) return 1;
    return 0;
}

void vuln_loop_write(const char *input) {
    char buf[8];
    size_t i;
    int flag = check_flag(42);
    if (flag) {
        (void)printf("flag set\n");
        return;
    }
    for (i = 0; input[i] != '\0'; i++) {
        buf[i] = input[i];
    }
    buf[i] = '\0';
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "hello";
    vuln_loop_write(s);
    return 0;
}
