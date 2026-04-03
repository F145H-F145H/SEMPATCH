# 非 PIE 可执行文件，便于按符号名过滤漏洞库（build_vuln_db.py --filter）
CC := gcc
CFLAGS := -Wall -Wextra -O0 -g -fno-pie -no-pie

.PHONY: all clean

all: vulnerable

vulnerable: vulnerable.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f vulnerable
