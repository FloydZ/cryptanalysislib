import sys

L = 2

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
    print("\tymm15 =_mm256_cmpeq_epi16(ymm0, ymm15);")
    if between_cmp_msk:
        print(between_cmp_msk)
    # print('vpmovmskb %ymm15, %r11d')
    print('\tmask = _mm256_movemask_epi8(ymm15);')
    if between_msk_test:
        print(between_msk_test)
    #print('test %r11d, %r11d')
    if between_test_jmp:
        print(between_test_jmp)
    #print('jne ._report_solution_{0}'.format(i))
    #print('._step_{0}_end:'.format(i))
    print('\tif (mask != 0) {{ goto _report_solution_{0}; }}'.format(i))
    print('\t_step_{0}_end:'.format(i))


def compute_update(i, a, b):
    # There are 3 possible cases :
    # 1a. Fq in register, Fl in register
    # 1b. Fq in memory,   Fl in register
    #  2. Fq in memory,   Fl in memory
    if a in Fl:
        if b in Fq: # reg / reg
            #xor1 = "vpxor {src}, {dst}, {dst}".format(src=Fq[b], dst=Fl[a])
            xor1 = "\t{dst} = _mm256_xor_si256({dst}, {src});".format(src=Fq[b], dst=Fl[a])
        elif Fq_memref is None: # mem / reg
            #xor1 = "vpxor {offset}(%rdi), {dst}, {dst}".format(offset=32*b, dst=Fl[a])
            xor1 = "\t{dst} = _mm256_xor_si256({dst}, *(__m256i *)(rdi + {offset}));".format(offset=32*b, dst=Fl[a])
        else: # mem(alpha) / reg
            # xor1 = "vpxor {src}, {dst}, {dst}".format(src=Fq_memref, dst=Fl[a])
            xor1 = "\t{dst} = _mm256_xor_si256({dst}, *(__m256i *)({src}));".format(src=Fq_memref, dst=Fl[a])
        #xor2 = "vpxor {src}, %ymm0, %ymm0".format(src=Fl[a])
        xor2 = "\tymm0 = _mm256_xor_si256(ymm0, {src});".format(src=Fl[a])
        return (xor1, xor2)

    else:             # (a not in Fl)
        assert b not in Fq
        # xor1a = "vmovdqa {offset}(%rsi), %ymm14".format(offset=32*a) # load Fl[a]
        xor1a = "ymm14 = _mm256_load_si256((__m256i *)(rsi + {offset}));".format(offset=32*a) # load Fl[a]

        if Fq_memref is None: 
            #xor1b = "vpxor {offset}(%rdi), %ymm14, %ymm14".format(offset=32*b)
            xor1b = "ymm14 = __mm256_xor_si256(ymm14, *(__m256 *)(rdi + {offset}));".format(offset=32*b)
        else:
            #xor1b = "vpxor {src}, %ymm14, %ymm14".format(src=Fq_memref)
            xor1b = "ymm14 = _mm256_xor_si256(ymm14, *(__m256 *){src});".format(src=Fq_memref)
        # xor1c = "vmovdqa %ymm14, {offset}(%rsi)".format(offset=32*a) # store Fl[a]
        xor1c = "_mm256_store_si256((__m256i *)(rsi + {offset}), ymm14);".format(offset=32*a) # store Fl[a]
        # xor2 = "vpxor %ymm14, %ymm0, %ymm0"
        xor2 = "\t_mm256_xor_si256(ymm0, ymm14, ymm0);"
        return ("\n\t".join([xor1a, xor1b, xor1c]), xor2)

print("#include <stdint.h>")
print("#include <immintrin.h>")
#print("""struct solution_t {
#	uint32_t x;
#	uint32_t mask;
#};""")
print( "struct solution_t* solver(uint16_t *rdi, uint16_t *rsi,const uint32_t alpha, const uint32_t beta, const uint32_t gamma, struct solution_t *buffer) {" )
print("\tuint32_t mask = 0;")
print( "\t// load the most-frequently used values into vector registers" )
for i, reg in Fl.items():
    # print("vmovdqa {offset}(%rsi), {reg}   ## {reg} = Fl[{i}]".format(offset=i*32, reg=reg, i=i))
    print("\t__m256i {reg} = _mm256_load_si256((__m256i *)(rsi + {offset}));".format(offset=i*32, reg=reg))
print()
for x, reg in Fq.items():
    # print("vmovdqa {offset}(%rdi), {reg}   ## {reg} = Fq[{idx}]".format(offset=x*32, reg=reg, idx=x))
    print("\t__m256i {reg} = _mm256_load_si256((__m256i *)(rdi + {offset}));".format(offset=x*32, reg=reg))

print("\t__m256i ymm14;")
print("\t__m256i ymm15 = _mm256_set1_epi8(0);")
print()

alpha = 0
for i in range((1 << L) - 1):
    ########################## UNROLLED LOOP #######################################
    idx1 = ffs(i + 1)                       
    idx2 = ffs((i + 1) ^ (1 << idx1))
    a = idx1 + 1                              # offset dans Fl
    Fq_memref = None
    if idx2 == -1:
        # Fq_memref = "{offset}(%rdi, %rdx)".format(offset=32*alpha)
        Fq_memref = "rdi + alpha + {offset}".format(offset=32*alpha)
        b = "alpha + {}".format(alpha)
        alpha += 1
    else:
        assert idx1 < idx2
        b = idxq(idx1, idx2)                  # offset dans Fq

    print()
    print('\t// step {:3d} : Fl[0] ^= (Fl[{}] ^= Fq[{}])'.format(i, a, b))
    print()
    xor1, xor2 = compute_update(i, a, b)
    output_comparison(i)
    print(xor1)
    print(xor2)
    print()

print('\t// end of the unrolled chunk #')
print()
print("\t// Save the Fl[1:] back to memory")
for i, reg in Fl.items():
    if i == 0:
        continue
    # print("vmovdqa {reg}, {offset:2d}(%rsi)     #Fl[{i}] <-- {reg}".format(offset=i*32, reg=reg, i=i))
    print("\t_mm256_store_si256((__m256i *)(rsi + {offset}), {reg});".format(offset=i*32, reg=reg))

print()
print('\t// special last step {:3d} : Fl[0] ^= (Fl[beta] ^= Fq[gamma])'.format((1 << L) - 1))
print()
output_comparison((1 << L) - 1)
#print("vmovdqa (%rsi, %rcx), %ymm14")                  # load Fl[beta]
print("\tymm14 = _mm256_load_si256((__m256i *)(rsi + beta));")          # load Fl[beta]
#print("vpxor (%rdi, %r8), %ymm14, %ymm14")             # xor Fq[gamma]
print("\tymm14 = _mm256_xor_si256(*(__m256i *)(rdi + gamma), ymm14); ") # xor Fq[gamma]
#print("vmovdqa %ymm14, (%rsi, %rcx)")     # store Fl[beta]
print("\t_mm256_store_si256((__m256i *)(rsi + beta), ymm14);")          # store Fl[beta]
#print("vpxor %ymm14, %ymm0, %ymm0")
print("\tymm0 = _mm256_xor_si256(ymm0, ymm14);")
print()
print("\t// Save Fl[0] back to memory")
#print("vmovdqa %ymm0, (%rsi)     #Fl[0] <-- %ymm0")
print("\t_mm256_store_si256((__m256i *)rsi, ymm0);")                 # Fl[0] <-- %ymm0
print()
#print('ret')
print('\treturn buffer;')
print()
print()

########################## WHEN SOLUTION FOUND #######################################

print('\t// now the code that reports solutions')
print()

for i in range(1<<L):
    #print('._report_solution_{i}:          # GrayCode(i + {i}) is a solution'.format(i=i))
    print('\t_report_solution_{i}:                  // GrayCode(i + {i}) is a solution'.format(i=i))
    #print('vpxor %ymm15, %ymm15, %ymm15    # reset %ymm15 to zero')
    print('\tymm15 = _mm256_xor_si256(ymm15, ymm15);// reset %ymm15 to zero')
    #print('movl ${i},  0(%rax)             # buffer.x = {i}'.format(i=i))
    #print('movl %r11d, 4(%rax)             # buffer.mask = %r11')
    #print('addq $8, %rax                   # buffer++'); 
    #print('jmp ._step_{i}_end'.format(i=i))  # return to the enumeration 
    print('\tbuffer->x = {i};'.format(i=i))
    print('\tbuffer->mask = mask;')
    print('\tbuffer++;'); 
    print('\tgoto _step_{i}_end;'.format(i=i))  # return to the enumeration 
    print()

print('}')
