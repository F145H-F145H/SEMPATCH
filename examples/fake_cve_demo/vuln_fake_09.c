/* 样例：手动索引复制无检查（勿用于生产） */
#include <stdio.h>

void vuln_fake_09(const char *input) {
    char buf[13];
    unsigned j = 0;
    for (;;) {
        char c = input[j];
        buf[j] = c;
        if (c == '\0') {
            break;
        }
        j++;
    }
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "d";
    vuln_fake_09(s);
    return 0;
}
