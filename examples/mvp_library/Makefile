# 非 PIE 可执行文件，供 Ghidra / SemPatch 特征提取
CC := gcc
CFLAGS := -Wall -Wextra -O0 -g -fno-pie -no-pie
SRCS := mvp_vuln_01.c mvp_vuln_02.c mvp_vuln_03.c
ELFS := $(patsubst %.c,build/%.elf,$(SRCS))

.PHONY: all clean

all: $(ELFS)

build/%.elf: %.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf build
