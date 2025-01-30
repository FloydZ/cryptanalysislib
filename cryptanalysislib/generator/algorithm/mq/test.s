	.file	"test.c"
	.text
	.p2align 4
	.globl	solver
	.type	solver, @function
solver:
.LFB6433:
	.cfi_startproc
	vmovdqa	(%rsi), %ymm1
	vpxor	%xmm0, %xmm0, %xmm0
	vmovdqa	32(%rsi), %ymm8
	movq	%rsi, %rax
	vmovdqa	64(%rsi), %ymm2
	vmovdqa	96(%rsi), %ymm7
	vpcmpeqw	%ymm0, %ymm1, %ymm0
	vmovdqa	128(%rsi), %ymm6
	vmovdqa	(%rdi), %ymm3
	vmovdqa	160(%rsi), %ymm5
	vmovdqa	192(%rsi), %ymm4
	vpmovmskb	%ymm0, %esi
	testl	%esi, %esi
	je	.L2
	movl	$0, (%rdx)
	vpxor	%xmm0, %xmm0, %xmm0
	addq	$8, %rdx
	movl	%esi, -4(%rdx)
.L2:
	movl	%ecx, %ecx
	vpxor	(%rdi,%rcx), %ymm8, %ymm8
	vpxor	%ymm8, %ymm1, %ymm1
	vpcmpeqw	%ymm0, %ymm1, %ymm0
	vpmovmskb	%ymm0, %esi
	testl	%esi, %esi
	je	.L3
	movl	$1, (%rdx)
	vpxor	%xmm0, %xmm0, %xmm0
	addq	$8, %rdx
	movl	%esi, -4(%rdx)
.L3:
	vpxor	32(%rdi,%rcx), %ymm2, %ymm2
	vpxor	%ymm2, %ymm1, %ymm1
	vpcmpeqw	%ymm0, %ymm1, %ymm0
	vpmovmskb	%ymm0, %ecx
	testl	%ecx, %ecx
	je	.L4
	movl	$2, (%rdx)
	vpxor	%xmm0, %xmm0, %xmm0
	addq	$8, %rdx
	movl	%ecx, -4(%rdx)
.L4:
	vpxor	%ymm8, %ymm3, %ymm3
	vmovdqa	%ymm2, 64(%rax)
	vpxor	%ymm3, %ymm1, %ymm1
	vmovdqa	%ymm3, 32(%rax)
	vpcmpeqw	%ymm0, %ymm1, %ymm0
	vmovdqa	%ymm7, 96(%rax)
	vmovdqa	%ymm6, 128(%rax)
	vmovdqa	%ymm5, 160(%rax)
	vpmovmskb	%ymm0, %ecx
	vmovdqa	%ymm4, 192(%rax)
	testl	%ecx, %ecx
	je	.L5
	movl	$3, (%rdx)
	movl	%ecx, 4(%rdx)
.L5:
	movl	%r8d, %r8d
	movl	%r9d, %r9d
	addq	%rax, %r8
	vmovdqa	(%rdi,%r9), %ymm0
	vpxor	(%r8), %ymm0, %ymm0
	vpxor	%ymm0, %ymm1, %ymm1
	vmovdqa	%ymm0, (%r8)
	vmovdqa	%ymm1, (%rax)
	vzeroupper
	xorl	%eax, %eax
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	xorl	%esi, %esi
	xorl	%edi, %edi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	ret
	.cfi_endproc
.LFE6433:
	.size	solver, .-solver
	.ident	"GCC: (GNU) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
