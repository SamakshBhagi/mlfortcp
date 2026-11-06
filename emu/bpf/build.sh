#!/bin/bash
# Builds the eBPF executor (stcc.bpf.o), its skeleton and the user-space agent (stcc_link).
# Needs clang >= 15, libelf, zlib, and bpftool/libbpf (https://github.com/libbpf/bpftool).
#   BPFTOOL_SRC=/path/to/bpftool/src ./build.sh
set -e
cd "$(dirname "$0")"
B=${BPFTOOL_SRC:-$HOME/emu/bpftool/src}
L=$B/libbpf
[ -f vmlinux.h ] || "$B/bpftool" btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
clang -O2 -g -target bpf -D__TARGET_ARCH_x86 -I. -I"$L/include" -c stcc.bpf.c -o stcc.bpf.o
"$B/bpftool" gen skeleton stcc.bpf.o name stcc > stcc.skel.h
gcc -O2 -g -I. -I"$L/include" stcc_link.c "$L/libbpf.a" -lelf -lz -lm -o stcc_link
echo "built stcc.bpf.o and stcc_link"
