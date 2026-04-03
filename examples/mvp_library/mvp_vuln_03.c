/* 样例 3：sprintf 格式字符串溢出（勿用于生产） */
#include <stdio.h>
#include <string.h>

static int compute_offset(int base) {
    int off = base * 2 + 1;
    return off > 0 ? off : 0;
}

void vuln_sprintf_ovf(const char *input) {
    char buf[16];
    int off = compute_offset(3);
    if (input == NULL) {
        (void)printf("null\n");
        return;
    }
    sprintf(buf, "prefix_%s_off%d", input, off);
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "data";
    vuln_sprintf_ovf(s);
    return 0;
}
