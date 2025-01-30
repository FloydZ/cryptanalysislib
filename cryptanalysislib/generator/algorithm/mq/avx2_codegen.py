import sys

L = 8

def ffs(i):
    if i == 0:
        return -1
    k = 0
    while i & 0x0001 == 0:
        k += 1
        i >>= 1
    return k

def idxq(i, j):
    assert i < j
    return i + j * (j - 1) // 2

# unrolling factor
if len(sys.argv) >= 2:
    L = int(sys.argv[1])

Fq_memref = None
Fl = {}
Fl[0] = "ymm0"   # 1
Fl[1] = "ymm1"   # 1/2
Fl[2] = "ymm2"   # 1/4
Fl[3] = "ymm3"   # 1/8
Fl[4] = "ymm4"   # 1/16
Fl[5] = "ymm5"   # 1/32
Fl[6] = "ymm6"   # 1/64

Fq = {}
Fq[idxq(0, 1)] = "ymm7"  # 1/4
Fq[idxq(0, 2)] = "ymm8"  # 1/8
Fq[idxq(1, 2)] = "ymm9"  # 1/8
Fq[idxq(0, 3)] = "ymm10" # 1/16
Fq[idxq(1, 3)] = "ymm11" # 1/16
Fq[idxq(2, 3)] = "ymm12" # 1/16
Fq[idxq(0, 4)] = "ymm13" # 1/32


def output_comparison(i, between_cmp_msk=None, between_msk_test=None, between_test_jmp=None):
    # before the XORs, the comparison
    # print('vpcmpeqw %ymm0, %ymm15, %ymm15'.format())
    print("ymm15 =_mm256_cmpeq_epi16(ymm0, ymm15);")
    if between_cmp_msk:
        print(between_cmp_msk)
    # print('vpmovmskb %ymm15, %r11d')
    print('mask = _mm256_movemask_epi8(ymm15);')
    if between_msk_test:
        print(between_msk_test)
    #print('test %r11d, %r11d')
    if between_test_jmp:
        print(between_test_jmp)
    #print('jne ._report_solution_{0}'.format(i))
    #print('._step_{0}_end:'.format(i))
    print('if (mask != 0) { goto _report_solution_{0}; }'.format(i))
    print('step_{0}_end:'.format(i))


def compute_update(i, a, b):
    # There are 3 possible cases :
    # 1a. Fq in register, Fl in register
    # 1b. Fq in memory,   Fl in register
    #  2. Fq in memory,   Fl in memory
    if a in Fl:
        if b in Fq: # reg / reg
            #xor1 = "vpxor {src}, {dst}, {dst}".format(src=Fq[b], dst=Fl[a])
            xor1 = "_mm256_xor_si256({dst}, ${src}, ${dst});".format(src=Fq[b], dst=Fl[a])
        elif Fq_memref is None: # mem / reg
            #xor1 = "vpxor {offset}(%rdi), {dst}, {dst}".format(offset=32*b, dst=Fl[a])
            xor1 = "_mm256_xor_si256({dst}, (%rdi) + {offset}, {dst});".format(offset=32*b, dst=Fl[a])
        else: # mem(alpha) / reg
            # xor1 = "vpxor {src}, {dst}, {dst}".format(src=Fq_memref, dst=Fl[a])
            xor1 = "_mm256_xor_si256(${dst}, {src}, {dst});".format(src=Fq_memref, dst=Fl[a])
        #xor2 = "vpxor {src}, %ymm0, %ymm0".format(src=Fl[a])
        xor2 = "_mm256_xor_si256(ymm, {src}, ymm0);".format(src=Fl[a])
        return (xor1, xor2)

    else:             # (a not in Fl)
        assert b not in Fq
        # xor1a = "vmovdqa {offset}(%rsi), %ymm14".format(offset=32*a) # load Fl[a]
        xor1a = "ymm14 = _mm256_load_si256(rsi + {offset});".format(offset=32*a) # load Fl[a]

        if Fq_memref is None: 
            #xor1b = "vpxor {offset}(%rdi), %ymm14, %ymm14".format(offset=32*b)
            xor1b = "__mm256_xor_si256(ymm14, (rdi) + {offset}, ymm14);".format(offset=32*b)
        else:
            #xor1b = "vpxor {src}, %ymm14, %ymm14".format(src=Fq_memref)
            xor1b = "_mm256_xor_si256(ymm14, {src}, ymm14);".format(src=Fq_memref)
        # xor1c = "vmovdqa %ymm14, {offset}(%rsi)".format(offset=32*a) # store Fl[a]
        xor1c = "_mm256_store_si256(rsi + {offset}, ymm14);".format(offset=32*a) # store Fl[a]
        # xor2 = "vpxor %ymm14, %ymm0, %ymm0"
        xor2 = "_mm256_xor_si256(ymm0, ymm14, ymm0);"
        return ("\n".join([xor1a, xor1b, xor1c]), xor2)
