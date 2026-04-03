# 非 PIE ELF，供 Ghidra / SemPatch 伪 CVE 库与匹配 demo 使用。
CC := gcc
CFLAGS := -Wall -Wextra -O0 -g -fno-pie -no-pie
OUT := build

VULN_SRCS := $(wildcard vuln_fake_*.c)
VULN_ELFS := $(patsubst vuln_fake_%.c,$(OUT)/vuln_fake_%.elf,$(VULN_SRCS))

.PHONY: all clean query

all: $(VULN_ELFS) query

$(OUT):
	mkdir -p $(OUT)

$(OUT)/vuln_fake_%.elf: vuln_fake_%.c | $(OUT)
	$(CC) $(CFLAGS) -o $@ $<

# 查询样例：与库中 vuln_fake_05 二进制一致，便于验收 Top-K 出现 FAKE-CVE-0005
query: $(OUT)/vuln_fake_05.elf
	cp -f $< $(OUT)/query.elf

clean:
	rm -rf $(OUT)
