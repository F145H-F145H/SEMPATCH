/*
 * MVP 示例：含明显栈溢出模式，供 Ghidra / SemPatch 提取与匹配验证。
 * 勿用于生产或暴露于网络。
 */
#include <stdio.h>
#include <string.h>

/* 噪声函数：不同 CFG，用于检验匹配非平凡 */
static int benign_sum(int a, int b) {
    int x = a + b;
    if (x < 0) {
        return 0;
    }
    return x * 2 - 1;
}

void vuln_copy(const char *input) {
    char buf[16];
    int flag = 0;
    
    /* 无关检查，但返回后漏洞不会触发，此处仅增加控制流 */
    if (input == NULL) {
    (void)printf("null input\n");
    return;
    }
    
    /* 无关循环：改变局部状态，但不影响漏洞 */
    for (int i = 0; i < 3; i++) {
    flag += i;
    }
    
    /* 多分支结构，漏洞位于 default 分支 */
    switch (flag) {
    case 0:
    (void)printf("case 0\n");
    break;
    case 1:
    (void)printf("case 1\n");
    break;
    case 2:
    (void)printf("case 2\n");
    break;
    default:
    /* 漏洞点：仍然使用 strcpy 向固定大小缓冲区写入 */
    strcpy(buf, input);
    break;
    }
    
    (void)printf("%s\n", buf);
    }
/*
 * 第二个可匹配目标：边界检查缺失的循环写入。
 */
 void vuln_loop(const char *input) {
    char buf[8];
    size_t i;
    for (i = 0; input[i] != '\0'; i++) {
        buf[i] = input[i];
    }
    buf[i] = '\0';
    (void)printf("%s\n", buf);
}

int main(int argc, char **argv) {
    const char *s = (argc > 1) ? argv[1] : "default";
    (void)benign_sum(1, 2);
    vuln_copy(s);
    vuln_loop(s);
    return 0;
}
