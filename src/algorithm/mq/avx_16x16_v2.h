#include <stdint.h>
#include <immintrin.h>
struct solution_t* solver(uint16_t *_rdi, uint16_t *_rsi, uint32_t alpha, uint32_t beta, uint32_t gamma, struct solution_t *buffer) {
	uint8_t *rdi = (uint8_t *)_rdi;
	uint8_t *rsi = (uint8_t *)_rsi;
	alpha <<= 5;
	beta <<= 5;
	gamma <<= 5;
	uint32_t mask = 0;
	// load the most-frequently used values into vector registers
	__m256i ymm0 = _mm256_load_si256((__m256i *)(rsi + 0));
	__m256i ymm1 = _mm256_load_si256((__m256i *)(rsi + 32));
	__m256i ymm2 = _mm256_load_si256((__m256i *)(rsi + 64));
	__m256i ymm3 = _mm256_load_si256((__m256i *)(rsi + 96));
	__m256i ymm4 = _mm256_load_si256((__m256i *)(rsi + 128));
	__m256i ymm5 = _mm256_load_si256((__m256i *)(rsi + 160));
	__m256i ymm6 = _mm256_load_si256((__m256i *)(rsi + 192));

	__m256i ymm7 = _mm256_load_si256((__m256i *)(rdi + 0));
	__m256i ymm8 = _mm256_load_si256((__m256i *)(rdi + 32));
	__m256i ymm9 = _mm256_load_si256((__m256i *)(rdi + 64));
	__m256i ymm10 = _mm256_load_si256((__m256i *)(rdi + 96));
	__m256i ymm11 = _mm256_load_si256((__m256i *)(rdi + 128));
	__m256i ymm12 = _mm256_load_si256((__m256i *)(rdi + 160));
	__m256i ymm13 = _mm256_load_si256((__m256i *)(rdi + 192));
	__m256i ymm14 = _mm256_set1_epi8(0);
	__m256i ymm15 = _mm256_set1_epi8(0);


	// step   0 : Fl[0] ^= (Fl[1] ^= Fq[alpha + 0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_0; }
	_step_0_end:
	ymm1 = _mm256_xor_si256(ymm1, *(__m256i *)(rdi + alpha + 0));
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step   1 : Fl[0] ^= (Fl[2] ^= Fq[alpha + 1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_1; }
	_step_1_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + alpha + 32));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step   2 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_2; }
	_step_2_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step   3 : Fl[0] ^= (Fl[3] ^= Fq[alpha + 2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_3; }
	_step_3_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + alpha + 64));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step   4 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_4; }
	_step_4_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step   5 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_5; }
	_step_5_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step   6 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_6; }
	_step_6_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step   7 : Fl[0] ^= (Fl[4] ^= Fq[alpha + 3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_7; }
	_step_7_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + alpha + 96));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step   8 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_8; }
	_step_8_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step   9 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_9; }
	_step_9_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  10 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_10; }
	_step_10_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  11 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_11; }
	_step_11_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  12 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_12; }
	_step_12_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  13 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_13; }
	_step_13_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  14 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_14; }
	_step_14_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  15 : Fl[0] ^= (Fl[5] ^= Fq[alpha + 4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_15; }
	_step_15_end:
	ymm5 = _mm256_xor_si256(ymm5, *(__m256i *)(rdi + alpha + 128));
	ymm0 = _mm256_xor_si256(ymm0, ymm5);


	// step  16 : Fl[0] ^= (Fl[1] ^= Fq[6])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_16; }
	_step_16_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm13);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  17 : Fl[0] ^= (Fl[2] ^= Fq[7])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_17; }
	_step_17_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 224));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  18 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_18; }
	_step_18_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  19 : Fl[0] ^= (Fl[3] ^= Fq[8])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_19; }
	_step_19_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 256));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  20 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_20; }
	_step_20_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  21 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_21; }
	_step_21_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  22 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_22; }
	_step_22_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  23 : Fl[0] ^= (Fl[4] ^= Fq[9])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_23; }
	_step_23_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 288));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step  24 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_24; }
	_step_24_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  25 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_25; }
	_step_25_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  26 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_26; }
	_step_26_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  27 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_27; }
	_step_27_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  28 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_28; }
	_step_28_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  29 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_29; }
	_step_29_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  30 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_30; }
	_step_30_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  31 : Fl[0] ^= (Fl[6] ^= Fq[alpha + 5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_31; }
	_step_31_end:
	ymm6 = _mm256_xor_si256(ymm6, *(__m256i *)(rdi + alpha + 160));
	ymm0 = _mm256_xor_si256(ymm0, ymm6);


	// step  32 : Fl[0] ^= (Fl[1] ^= Fq[10])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_32; }
	_step_32_end:
	ymm1 = _mm256_xor_si256(ymm1, *(__m256i *)(rdi + 320));
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  33 : Fl[0] ^= (Fl[2] ^= Fq[11])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_33; }
	_step_33_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 352));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  34 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_34; }
	_step_34_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  35 : Fl[0] ^= (Fl[3] ^= Fq[12])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_35; }
	_step_35_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 384));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  36 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_36; }
	_step_36_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  37 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_37; }
	_step_37_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  38 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_38; }
	_step_38_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  39 : Fl[0] ^= (Fl[4] ^= Fq[13])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_39; }
	_step_39_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 416));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step  40 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_40; }
	_step_40_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  41 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_41; }
	_step_41_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  42 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_42; }
	_step_42_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  43 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_43; }
	_step_43_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  44 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_44; }
	_step_44_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  45 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_45; }
	_step_45_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  46 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_46; }
	_step_46_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  47 : Fl[0] ^= (Fl[5] ^= Fq[14])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_47; }
	_step_47_end:
	ymm5 = _mm256_xor_si256(ymm5, *(__m256i *)(rdi + 448));
	ymm0 = _mm256_xor_si256(ymm0, ymm5);


	// step  48 : Fl[0] ^= (Fl[1] ^= Fq[6])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_48; }
	_step_48_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm13);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  49 : Fl[0] ^= (Fl[2] ^= Fq[7])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_49; }
	_step_49_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 224));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  50 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_50; }
	_step_50_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  51 : Fl[0] ^= (Fl[3] ^= Fq[8])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_51; }
	_step_51_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 256));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  52 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_52; }
	_step_52_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  53 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_53; }
	_step_53_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  54 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_54; }
	_step_54_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  55 : Fl[0] ^= (Fl[4] ^= Fq[9])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_55; }
	_step_55_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 288));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step  56 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_56; }
	_step_56_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  57 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_57; }
	_step_57_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  58 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_58; }
	_step_58_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  59 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_59; }
	_step_59_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  60 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_60; }
	_step_60_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  61 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_61; }
	_step_61_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  62 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_62; }
	_step_62_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  63 : Fl[0] ^= (Fl[7] ^= Fq[alpha + 6])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_63; }
	_step_63_end:
	ymm14 = _mm256_load_si256((__m256i *)(rsi + 224));
	ymm14 = _mm256_xor_si256(ymm14, *(__m256i *)(rdi + alpha + 192));
	_mm256_store_si256((__m256i *)(rsi + 224), ymm14);
	ymm0 = _mm256_xor_si256(ymm0, ymm14);


	// step  64 : Fl[0] ^= (Fl[1] ^= Fq[15])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_64; }
	_step_64_end:
	ymm1 = _mm256_xor_si256(ymm1, *(__m256i *)(rdi + 480));
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  65 : Fl[0] ^= (Fl[2] ^= Fq[16])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_65; }
	_step_65_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 512));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  66 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_66; }
	_step_66_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  67 : Fl[0] ^= (Fl[3] ^= Fq[17])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_67; }
	_step_67_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 544));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  68 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_68; }
	_step_68_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  69 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_69; }
	_step_69_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  70 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_70; }
	_step_70_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  71 : Fl[0] ^= (Fl[4] ^= Fq[18])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_71; }
	_step_71_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 576));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step  72 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_72; }
	_step_72_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  73 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_73; }
	_step_73_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  74 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_74; }
	_step_74_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  75 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_75; }
	_step_75_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  76 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_76; }
	_step_76_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  77 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_77; }
	_step_77_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  78 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_78; }
	_step_78_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  79 : Fl[0] ^= (Fl[5] ^= Fq[19])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_79; }
	_step_79_end:
	ymm5 = _mm256_xor_si256(ymm5, *(__m256i *)(rdi + 608));
	ymm0 = _mm256_xor_si256(ymm0, ymm5);


	// step  80 : Fl[0] ^= (Fl[1] ^= Fq[6])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_80; }
	_step_80_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm13);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  81 : Fl[0] ^= (Fl[2] ^= Fq[7])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_81; }
	_step_81_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 224));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  82 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_82; }
	_step_82_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  83 : Fl[0] ^= (Fl[3] ^= Fq[8])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_83; }
	_step_83_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 256));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  84 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_84; }
	_step_84_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  85 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_85; }
	_step_85_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  86 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_86; }
	_step_86_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  87 : Fl[0] ^= (Fl[4] ^= Fq[9])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_87; }
	_step_87_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 288));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step  88 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_88; }
	_step_88_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  89 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_89; }
	_step_89_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  90 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_90; }
	_step_90_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  91 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_91; }
	_step_91_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step  92 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_92; }
	_step_92_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  93 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_93; }
	_step_93_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  94 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_94; }
	_step_94_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  95 : Fl[0] ^= (Fl[6] ^= Fq[20])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_95; }
	_step_95_end:
	ymm6 = _mm256_xor_si256(ymm6, *(__m256i *)(rdi + 640));
	ymm0 = _mm256_xor_si256(ymm0, ymm6);


	// step  96 : Fl[0] ^= (Fl[1] ^= Fq[10])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_96; }
	_step_96_end:
	ymm1 = _mm256_xor_si256(ymm1, *(__m256i *)(rdi + 320));
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  97 : Fl[0] ^= (Fl[2] ^= Fq[11])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_97; }
	_step_97_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 352));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step  98 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_98; }
	_step_98_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step  99 : Fl[0] ^= (Fl[3] ^= Fq[12])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_99; }
	_step_99_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 384));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 100 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_100; }
	_step_100_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 101 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_101; }
	_step_101_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 102 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_102; }
	_step_102_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 103 : Fl[0] ^= (Fl[4] ^= Fq[13])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_103; }
	_step_103_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 416));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 104 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_104; }
	_step_104_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 105 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_105; }
	_step_105_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 106 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_106; }
	_step_106_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 107 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_107; }
	_step_107_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 108 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_108; }
	_step_108_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 109 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_109; }
	_step_109_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 110 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_110; }
	_step_110_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 111 : Fl[0] ^= (Fl[5] ^= Fq[14])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_111; }
	_step_111_end:
	ymm5 = _mm256_xor_si256(ymm5, *(__m256i *)(rdi + 448));
	ymm0 = _mm256_xor_si256(ymm0, ymm5);


	// step 112 : Fl[0] ^= (Fl[1] ^= Fq[6])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_112; }
	_step_112_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm13);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 113 : Fl[0] ^= (Fl[2] ^= Fq[7])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_113; }
	_step_113_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 224));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 114 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_114; }
	_step_114_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 115 : Fl[0] ^= (Fl[3] ^= Fq[8])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_115; }
	_step_115_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 256));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 116 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_116; }
	_step_116_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 117 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_117; }
	_step_117_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 118 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_118; }
	_step_118_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 119 : Fl[0] ^= (Fl[4] ^= Fq[9])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_119; }
	_step_119_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 288));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 120 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_120; }
	_step_120_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 121 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_121; }
	_step_121_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 122 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_122; }
	_step_122_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 123 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_123; }
	_step_123_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 124 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_124; }
	_step_124_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 125 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_125; }
	_step_125_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 126 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_126; }
	_step_126_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 127 : Fl[0] ^= (Fl[8] ^= Fq[alpha + 7])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_127; }
	_step_127_end:
	ymm14 = _mm256_load_si256((__m256i *)(rsi + 256));
	ymm14 = _mm256_xor_si256(ymm14, *(__m256i *)(rdi + alpha + 224));
	_mm256_store_si256((__m256i *)(rsi + 256), ymm14);
	ymm0 = _mm256_xor_si256(ymm0, ymm14);


	// step 128 : Fl[0] ^= (Fl[1] ^= Fq[21])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_128; }
	_step_128_end:
	ymm1 = _mm256_xor_si256(ymm1, *(__m256i *)(rdi + 672));
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 129 : Fl[0] ^= (Fl[2] ^= Fq[22])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_129; }
	_step_129_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 704));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 130 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_130; }
	_step_130_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 131 : Fl[0] ^= (Fl[3] ^= Fq[23])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_131; }
	_step_131_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 736));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 132 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_132; }
	_step_132_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 133 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_133; }
	_step_133_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 134 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_134; }
	_step_134_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 135 : Fl[0] ^= (Fl[4] ^= Fq[24])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_135; }
	_step_135_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 768));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 136 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_136; }
	_step_136_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 137 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_137; }
	_step_137_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 138 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_138; }
	_step_138_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 139 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_139; }
	_step_139_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 140 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_140; }
	_step_140_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 141 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_141; }
	_step_141_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 142 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_142; }
	_step_142_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 143 : Fl[0] ^= (Fl[5] ^= Fq[25])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_143; }
	_step_143_end:
	ymm5 = _mm256_xor_si256(ymm5, *(__m256i *)(rdi + 800));
	ymm0 = _mm256_xor_si256(ymm0, ymm5);


	// step 144 : Fl[0] ^= (Fl[1] ^= Fq[6])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_144; }
	_step_144_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm13);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 145 : Fl[0] ^= (Fl[2] ^= Fq[7])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_145; }
	_step_145_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 224));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 146 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_146; }
	_step_146_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 147 : Fl[0] ^= (Fl[3] ^= Fq[8])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_147; }
	_step_147_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 256));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 148 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_148; }
	_step_148_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 149 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_149; }
	_step_149_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 150 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_150; }
	_step_150_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 151 : Fl[0] ^= (Fl[4] ^= Fq[9])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_151; }
	_step_151_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 288));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 152 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_152; }
	_step_152_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 153 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_153; }
	_step_153_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 154 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_154; }
	_step_154_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 155 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_155; }
	_step_155_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 156 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_156; }
	_step_156_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 157 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_157; }
	_step_157_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 158 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_158; }
	_step_158_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 159 : Fl[0] ^= (Fl[6] ^= Fq[26])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_159; }
	_step_159_end:
	ymm6 = _mm256_xor_si256(ymm6, *(__m256i *)(rdi + 832));
	ymm0 = _mm256_xor_si256(ymm0, ymm6);


	// step 160 : Fl[0] ^= (Fl[1] ^= Fq[10])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_160; }
	_step_160_end:
	ymm1 = _mm256_xor_si256(ymm1, *(__m256i *)(rdi + 320));
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 161 : Fl[0] ^= (Fl[2] ^= Fq[11])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_161; }
	_step_161_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 352));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 162 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_162; }
	_step_162_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 163 : Fl[0] ^= (Fl[3] ^= Fq[12])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_163; }
	_step_163_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 384));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 164 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_164; }
	_step_164_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 165 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_165; }
	_step_165_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 166 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_166; }
	_step_166_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 167 : Fl[0] ^= (Fl[4] ^= Fq[13])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_167; }
	_step_167_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 416));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 168 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_168; }
	_step_168_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 169 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_169; }
	_step_169_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 170 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_170; }
	_step_170_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 171 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_171; }
	_step_171_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 172 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_172; }
	_step_172_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 173 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_173; }
	_step_173_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 174 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_174; }
	_step_174_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 175 : Fl[0] ^= (Fl[5] ^= Fq[14])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_175; }
	_step_175_end:
	ymm5 = _mm256_xor_si256(ymm5, *(__m256i *)(rdi + 448));
	ymm0 = _mm256_xor_si256(ymm0, ymm5);


	// step 176 : Fl[0] ^= (Fl[1] ^= Fq[6])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_176; }
	_step_176_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm13);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 177 : Fl[0] ^= (Fl[2] ^= Fq[7])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_177; }
	_step_177_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 224));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 178 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_178; }
	_step_178_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 179 : Fl[0] ^= (Fl[3] ^= Fq[8])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_179; }
	_step_179_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 256));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 180 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_180; }
	_step_180_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 181 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_181; }
	_step_181_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 182 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_182; }
	_step_182_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 183 : Fl[0] ^= (Fl[4] ^= Fq[9])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_183; }
	_step_183_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 288));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 184 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_184; }
	_step_184_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 185 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_185; }
	_step_185_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 186 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_186; }
	_step_186_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 187 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_187; }
	_step_187_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 188 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_188; }
	_step_188_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 189 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_189; }
	_step_189_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 190 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_190; }
	_step_190_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 191 : Fl[0] ^= (Fl[7] ^= Fq[27])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_191; }
	_step_191_end:
	ymm14 = _mm256_load_si256((__m256i *)(rsi + 224));
	ymm14 = _mm256_xor_si256(ymm14, *(__m256i *)(rdi + 864));
	_mm256_store_si256((__m256i *)(rsi + 224), ymm14);
	ymm0 = _mm256_xor_si256(ymm0, ymm14);


	// step 192 : Fl[0] ^= (Fl[1] ^= Fq[15])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_192; }
	_step_192_end:
	ymm1 = _mm256_xor_si256(ymm1, *(__m256i *)(rdi + 480));
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 193 : Fl[0] ^= (Fl[2] ^= Fq[16])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_193; }
	_step_193_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 512));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 194 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_194; }
	_step_194_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 195 : Fl[0] ^= (Fl[3] ^= Fq[17])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_195; }
	_step_195_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 544));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 196 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_196; }
	_step_196_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 197 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_197; }
	_step_197_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 198 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_198; }
	_step_198_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 199 : Fl[0] ^= (Fl[4] ^= Fq[18])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_199; }
	_step_199_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 576));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 200 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_200; }
	_step_200_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 201 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_201; }
	_step_201_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 202 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_202; }
	_step_202_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 203 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_203; }
	_step_203_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 204 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_204; }
	_step_204_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 205 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_205; }
	_step_205_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 206 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_206; }
	_step_206_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 207 : Fl[0] ^= (Fl[5] ^= Fq[19])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_207; }
	_step_207_end:
	ymm5 = _mm256_xor_si256(ymm5, *(__m256i *)(rdi + 608));
	ymm0 = _mm256_xor_si256(ymm0, ymm5);


	// step 208 : Fl[0] ^= (Fl[1] ^= Fq[6])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_208; }
	_step_208_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm13);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 209 : Fl[0] ^= (Fl[2] ^= Fq[7])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_209; }
	_step_209_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 224));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 210 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_210; }
	_step_210_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 211 : Fl[0] ^= (Fl[3] ^= Fq[8])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_211; }
	_step_211_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 256));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 212 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_212; }
	_step_212_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 213 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_213; }
	_step_213_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 214 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_214; }
	_step_214_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 215 : Fl[0] ^= (Fl[4] ^= Fq[9])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_215; }
	_step_215_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 288));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 216 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_216; }
	_step_216_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 217 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_217; }
	_step_217_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 218 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_218; }
	_step_218_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 219 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_219; }
	_step_219_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 220 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_220; }
	_step_220_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 221 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_221; }
	_step_221_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 222 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_222; }
	_step_222_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 223 : Fl[0] ^= (Fl[6] ^= Fq[20])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_223; }
	_step_223_end:
	ymm6 = _mm256_xor_si256(ymm6, *(__m256i *)(rdi + 640));
	ymm0 = _mm256_xor_si256(ymm0, ymm6);


	// step 224 : Fl[0] ^= (Fl[1] ^= Fq[10])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_224; }
	_step_224_end:
	ymm1 = _mm256_xor_si256(ymm1, *(__m256i *)(rdi + 320));
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 225 : Fl[0] ^= (Fl[2] ^= Fq[11])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_225; }
	_step_225_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 352));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 226 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_226; }
	_step_226_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 227 : Fl[0] ^= (Fl[3] ^= Fq[12])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_227; }
	_step_227_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 384));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 228 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_228; }
	_step_228_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 229 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_229; }
	_step_229_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 230 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_230; }
	_step_230_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 231 : Fl[0] ^= (Fl[4] ^= Fq[13])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_231; }
	_step_231_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 416));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 232 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_232; }
	_step_232_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 233 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_233; }
	_step_233_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 234 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_234; }
	_step_234_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 235 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_235; }
	_step_235_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 236 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_236; }
	_step_236_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 237 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_237; }
	_step_237_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 238 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_238; }
	_step_238_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 239 : Fl[0] ^= (Fl[5] ^= Fq[14])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_239; }
	_step_239_end:
	ymm5 = _mm256_xor_si256(ymm5, *(__m256i *)(rdi + 448));
	ymm0 = _mm256_xor_si256(ymm0, ymm5);


	// step 240 : Fl[0] ^= (Fl[1] ^= Fq[6])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_240; }
	_step_240_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm13);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 241 : Fl[0] ^= (Fl[2] ^= Fq[7])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_241; }
	_step_241_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + 224));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 242 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_242; }
	_step_242_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 243 : Fl[0] ^= (Fl[3] ^= Fq[8])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_243; }
	_step_243_end:
	ymm3 = _mm256_xor_si256(ymm3, *(__m256i *)(rdi + 256));
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 244 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_244; }
	_step_244_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 245 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_245; }
	_step_245_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 246 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_246; }
	_step_246_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 247 : Fl[0] ^= (Fl[4] ^= Fq[9])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_247; }
	_step_247_end:
	ymm4 = _mm256_xor_si256(ymm4, *(__m256i *)(rdi + 288));
	ymm0 = _mm256_xor_si256(ymm0, ymm4);


	// step 248 : Fl[0] ^= (Fl[1] ^= Fq[3])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_248; }
	_step_248_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm10);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 249 : Fl[0] ^= (Fl[2] ^= Fq[4])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_249; }
	_step_249_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm11);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 250 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_250; }
	_step_250_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 251 : Fl[0] ^= (Fl[3] ^= Fq[5])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_251; }
	_step_251_end:
	ymm3 = _mm256_xor_si256(ymm3, ymm12);
	ymm0 = _mm256_xor_si256(ymm0, ymm3);


	// step 252 : Fl[0] ^= (Fl[1] ^= Fq[1])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_252; }
	_step_252_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm8);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step 253 : Fl[0] ^= (Fl[2] ^= Fq[2])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_253; }
	_step_253_end:
	ymm2 = _mm256_xor_si256(ymm2, ymm9);
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step 254 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_254; }
	_step_254_end:
	ymm1 = _mm256_xor_si256(ymm1, ymm7);
	ymm0 = _mm256_xor_si256(ymm0, ymm1);

	// end of the unrolled chunk #

	// Save the Fl[1:] back to memory
	_mm256_store_si256((__m256i *)(rsi + 32), ymm1);
	_mm256_store_si256((__m256i *)(rsi + 64), ymm2);
	_mm256_store_si256((__m256i *)(rsi + 96), ymm3);
	_mm256_store_si256((__m256i *)(rsi + 128), ymm4);
	_mm256_store_si256((__m256i *)(rsi + 160), ymm5);
	_mm256_store_si256((__m256i *)(rsi + 192), ymm6);

	// special last step 255 : Fl[0] ^= (Fl[beta] ^= Fq[gamma])

	ymm14 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm14);
	if (mask != 0) { goto _report_solution_255; }
	_step_255_end:
	ymm14 = _mm256_load_si256((__m256i *)(rsi + beta));
	ymm14 = _mm256_xor_si256(*(__m256i *)(rdi + gamma), ymm14); 
	_mm256_store_si256((__m256i *)(rsi + beta), ymm14);
	ymm0 = _mm256_xor_si256(ymm0, ymm14);

	// Save Fl[0] back to memory
	_mm256_store_si256((__m256i *)rsi, ymm0);

	return buffer;


	// now the code that reports solutions

	_report_solution_0:                  // GrayCode(i + 0) is a solution
	buffer->x = 0;
	buffer->mask = mask;
	buffer++;
	goto _step_0_end;

	_report_solution_1:                  // GrayCode(i + 1) is a solution
	buffer->x = 1;
	buffer->mask = mask;
	buffer++;
	goto _step_1_end;

	_report_solution_2:                  // GrayCode(i + 2) is a solution
	buffer->x = 2;
	buffer->mask = mask;
	buffer++;
	goto _step_2_end;

	_report_solution_3:                  // GrayCode(i + 3) is a solution
	buffer->x = 3;
	buffer->mask = mask;
	buffer++;
	goto _step_3_end;

	_report_solution_4:                  // GrayCode(i + 4) is a solution
	buffer->x = 4;
	buffer->mask = mask;
	buffer++;
	goto _step_4_end;

	_report_solution_5:                  // GrayCode(i + 5) is a solution
	buffer->x = 5;
	buffer->mask = mask;
	buffer++;
	goto _step_5_end;

	_report_solution_6:                  // GrayCode(i + 6) is a solution
	buffer->x = 6;
	buffer->mask = mask;
	buffer++;
	goto _step_6_end;

	_report_solution_7:                  // GrayCode(i + 7) is a solution
	buffer->x = 7;
	buffer->mask = mask;
	buffer++;
	goto _step_7_end;

	_report_solution_8:                  // GrayCode(i + 8) is a solution
	buffer->x = 8;
	buffer->mask = mask;
	buffer++;
	goto _step_8_end;

	_report_solution_9:                  // GrayCode(i + 9) is a solution
	buffer->x = 9;
	buffer->mask = mask;
	buffer++;
	goto _step_9_end;

	_report_solution_10:                  // GrayCode(i + 10) is a solution
	buffer->x = 10;
	buffer->mask = mask;
	buffer++;
	goto _step_10_end;

	_report_solution_11:                  // GrayCode(i + 11) is a solution
	buffer->x = 11;
	buffer->mask = mask;
	buffer++;
	goto _step_11_end;

	_report_solution_12:                  // GrayCode(i + 12) is a solution
	buffer->x = 12;
	buffer->mask = mask;
	buffer++;
	goto _step_12_end;

	_report_solution_13:                  // GrayCode(i + 13) is a solution
	buffer->x = 13;
	buffer->mask = mask;
	buffer++;
	goto _step_13_end;

	_report_solution_14:                  // GrayCode(i + 14) is a solution
	buffer->x = 14;
	buffer->mask = mask;
	buffer++;
	goto _step_14_end;

	_report_solution_15:                  // GrayCode(i + 15) is a solution
	buffer->x = 15;
	buffer->mask = mask;
	buffer++;
	goto _step_15_end;

	_report_solution_16:                  // GrayCode(i + 16) is a solution
	buffer->x = 16;
	buffer->mask = mask;
	buffer++;
	goto _step_16_end;

	_report_solution_17:                  // GrayCode(i + 17) is a solution
	buffer->x = 17;
	buffer->mask = mask;
	buffer++;
	goto _step_17_end;

	_report_solution_18:                  // GrayCode(i + 18) is a solution
	buffer->x = 18;
	buffer->mask = mask;
	buffer++;
	goto _step_18_end;

	_report_solution_19:                  // GrayCode(i + 19) is a solution
	buffer->x = 19;
	buffer->mask = mask;
	buffer++;
	goto _step_19_end;

	_report_solution_20:                  // GrayCode(i + 20) is a solution
	buffer->x = 20;
	buffer->mask = mask;
	buffer++;
	goto _step_20_end;

	_report_solution_21:                  // GrayCode(i + 21) is a solution
	buffer->x = 21;
	buffer->mask = mask;
	buffer++;
	goto _step_21_end;

	_report_solution_22:                  // GrayCode(i + 22) is a solution
	buffer->x = 22;
	buffer->mask = mask;
	buffer++;
	goto _step_22_end;

	_report_solution_23:                  // GrayCode(i + 23) is a solution
	buffer->x = 23;
	buffer->mask = mask;
	buffer++;
	goto _step_23_end;

	_report_solution_24:                  // GrayCode(i + 24) is a solution
	buffer->x = 24;
	buffer->mask = mask;
	buffer++;
	goto _step_24_end;

	_report_solution_25:                  // GrayCode(i + 25) is a solution
	buffer->x = 25;
	buffer->mask = mask;
	buffer++;
	goto _step_25_end;

	_report_solution_26:                  // GrayCode(i + 26) is a solution
	buffer->x = 26;
	buffer->mask = mask;
	buffer++;
	goto _step_26_end;

	_report_solution_27:                  // GrayCode(i + 27) is a solution
	buffer->x = 27;
	buffer->mask = mask;
	buffer++;
	goto _step_27_end;

	_report_solution_28:                  // GrayCode(i + 28) is a solution
	buffer->x = 28;
	buffer->mask = mask;
	buffer++;
	goto _step_28_end;

	_report_solution_29:                  // GrayCode(i + 29) is a solution
	buffer->x = 29;
	buffer->mask = mask;
	buffer++;
	goto _step_29_end;

	_report_solution_30:                  // GrayCode(i + 30) is a solution
	buffer->x = 30;
	buffer->mask = mask;
	buffer++;
	goto _step_30_end;

	_report_solution_31:                  // GrayCode(i + 31) is a solution
	buffer->x = 31;
	buffer->mask = mask;
	buffer++;
	goto _step_31_end;

	_report_solution_32:                  // GrayCode(i + 32) is a solution
	buffer->x = 32;
	buffer->mask = mask;
	buffer++;
	goto _step_32_end;

	_report_solution_33:                  // GrayCode(i + 33) is a solution
	buffer->x = 33;
	buffer->mask = mask;
	buffer++;
	goto _step_33_end;

	_report_solution_34:                  // GrayCode(i + 34) is a solution
	buffer->x = 34;
	buffer->mask = mask;
	buffer++;
	goto _step_34_end;

	_report_solution_35:                  // GrayCode(i + 35) is a solution
	buffer->x = 35;
	buffer->mask = mask;
	buffer++;
	goto _step_35_end;

	_report_solution_36:                  // GrayCode(i + 36) is a solution
	buffer->x = 36;
	buffer->mask = mask;
	buffer++;
	goto _step_36_end;

	_report_solution_37:                  // GrayCode(i + 37) is a solution
	buffer->x = 37;
	buffer->mask = mask;
	buffer++;
	goto _step_37_end;

	_report_solution_38:                  // GrayCode(i + 38) is a solution
	buffer->x = 38;
	buffer->mask = mask;
	buffer++;
	goto _step_38_end;

	_report_solution_39:                  // GrayCode(i + 39) is a solution
	buffer->x = 39;
	buffer->mask = mask;
	buffer++;
	goto _step_39_end;

	_report_solution_40:                  // GrayCode(i + 40) is a solution
	buffer->x = 40;
	buffer->mask = mask;
	buffer++;
	goto _step_40_end;

	_report_solution_41:                  // GrayCode(i + 41) is a solution
	buffer->x = 41;
	buffer->mask = mask;
	buffer++;
	goto _step_41_end;

	_report_solution_42:                  // GrayCode(i + 42) is a solution
	buffer->x = 42;
	buffer->mask = mask;
	buffer++;
	goto _step_42_end;

	_report_solution_43:                  // GrayCode(i + 43) is a solution
	buffer->x = 43;
	buffer->mask = mask;
	buffer++;
	goto _step_43_end;

	_report_solution_44:                  // GrayCode(i + 44) is a solution
	buffer->x = 44;
	buffer->mask = mask;
	buffer++;
	goto _step_44_end;

	_report_solution_45:                  // GrayCode(i + 45) is a solution
	buffer->x = 45;
	buffer->mask = mask;
	buffer++;
	goto _step_45_end;

	_report_solution_46:                  // GrayCode(i + 46) is a solution
	buffer->x = 46;
	buffer->mask = mask;
	buffer++;
	goto _step_46_end;

	_report_solution_47:                  // GrayCode(i + 47) is a solution
	buffer->x = 47;
	buffer->mask = mask;
	buffer++;
	goto _step_47_end;

	_report_solution_48:                  // GrayCode(i + 48) is a solution
	buffer->x = 48;
	buffer->mask = mask;
	buffer++;
	goto _step_48_end;

	_report_solution_49:                  // GrayCode(i + 49) is a solution
	buffer->x = 49;
	buffer->mask = mask;
	buffer++;
	goto _step_49_end;

	_report_solution_50:                  // GrayCode(i + 50) is a solution
	buffer->x = 50;
	buffer->mask = mask;
	buffer++;
	goto _step_50_end;

	_report_solution_51:                  // GrayCode(i + 51) is a solution
	buffer->x = 51;
	buffer->mask = mask;
	buffer++;
	goto _step_51_end;

	_report_solution_52:                  // GrayCode(i + 52) is a solution
	buffer->x = 52;
	buffer->mask = mask;
	buffer++;
	goto _step_52_end;

	_report_solution_53:                  // GrayCode(i + 53) is a solution
	buffer->x = 53;
	buffer->mask = mask;
	buffer++;
	goto _step_53_end;

	_report_solution_54:                  // GrayCode(i + 54) is a solution
	buffer->x = 54;
	buffer->mask = mask;
	buffer++;
	goto _step_54_end;

	_report_solution_55:                  // GrayCode(i + 55) is a solution
	buffer->x = 55;
	buffer->mask = mask;
	buffer++;
	goto _step_55_end;

	_report_solution_56:                  // GrayCode(i + 56) is a solution
	buffer->x = 56;
	buffer->mask = mask;
	buffer++;
	goto _step_56_end;

	_report_solution_57:                  // GrayCode(i + 57) is a solution
	buffer->x = 57;
	buffer->mask = mask;
	buffer++;
	goto _step_57_end;

	_report_solution_58:                  // GrayCode(i + 58) is a solution
	buffer->x = 58;
	buffer->mask = mask;
	buffer++;
	goto _step_58_end;

	_report_solution_59:                  // GrayCode(i + 59) is a solution
	buffer->x = 59;
	buffer->mask = mask;
	buffer++;
	goto _step_59_end;

	_report_solution_60:                  // GrayCode(i + 60) is a solution
	buffer->x = 60;
	buffer->mask = mask;
	buffer++;
	goto _step_60_end;

	_report_solution_61:                  // GrayCode(i + 61) is a solution
	buffer->x = 61;
	buffer->mask = mask;
	buffer++;
	goto _step_61_end;

	_report_solution_62:                  // GrayCode(i + 62) is a solution
	buffer->x = 62;
	buffer->mask = mask;
	buffer++;
	goto _step_62_end;

	_report_solution_63:                  // GrayCode(i + 63) is a solution
	buffer->x = 63;
	buffer->mask = mask;
	buffer++;
	goto _step_63_end;

	_report_solution_64:                  // GrayCode(i + 64) is a solution
	buffer->x = 64;
	buffer->mask = mask;
	buffer++;
	goto _step_64_end;

	_report_solution_65:                  // GrayCode(i + 65) is a solution
	buffer->x = 65;
	buffer->mask = mask;
	buffer++;
	goto _step_65_end;

	_report_solution_66:                  // GrayCode(i + 66) is a solution
	buffer->x = 66;
	buffer->mask = mask;
	buffer++;
	goto _step_66_end;

	_report_solution_67:                  // GrayCode(i + 67) is a solution
	buffer->x = 67;
	buffer->mask = mask;
	buffer++;
	goto _step_67_end;

	_report_solution_68:                  // GrayCode(i + 68) is a solution
	buffer->x = 68;
	buffer->mask = mask;
	buffer++;
	goto _step_68_end;

	_report_solution_69:                  // GrayCode(i + 69) is a solution
	buffer->x = 69;
	buffer->mask = mask;
	buffer++;
	goto _step_69_end;

	_report_solution_70:                  // GrayCode(i + 70) is a solution
	buffer->x = 70;
	buffer->mask = mask;
	buffer++;
	goto _step_70_end;

	_report_solution_71:                  // GrayCode(i + 71) is a solution
	buffer->x = 71;
	buffer->mask = mask;
	buffer++;
	goto _step_71_end;

	_report_solution_72:                  // GrayCode(i + 72) is a solution
	buffer->x = 72;
	buffer->mask = mask;
	buffer++;
	goto _step_72_end;

	_report_solution_73:                  // GrayCode(i + 73) is a solution
	buffer->x = 73;
	buffer->mask = mask;
	buffer++;
	goto _step_73_end;

	_report_solution_74:                  // GrayCode(i + 74) is a solution
	buffer->x = 74;
	buffer->mask = mask;
	buffer++;
	goto _step_74_end;

	_report_solution_75:                  // GrayCode(i + 75) is a solution
	buffer->x = 75;
	buffer->mask = mask;
	buffer++;
	goto _step_75_end;

	_report_solution_76:                  // GrayCode(i + 76) is a solution
	buffer->x = 76;
	buffer->mask = mask;
	buffer++;
	goto _step_76_end;

	_report_solution_77:                  // GrayCode(i + 77) is a solution
	buffer->x = 77;
	buffer->mask = mask;
	buffer++;
	goto _step_77_end;

	_report_solution_78:                  // GrayCode(i + 78) is a solution
	buffer->x = 78;
	buffer->mask = mask;
	buffer++;
	goto _step_78_end;

	_report_solution_79:                  // GrayCode(i + 79) is a solution
	buffer->x = 79;
	buffer->mask = mask;
	buffer++;
	goto _step_79_end;

	_report_solution_80:                  // GrayCode(i + 80) is a solution
	buffer->x = 80;
	buffer->mask = mask;
	buffer++;
	goto _step_80_end;

	_report_solution_81:                  // GrayCode(i + 81) is a solution
	buffer->x = 81;
	buffer->mask = mask;
	buffer++;
	goto _step_81_end;

	_report_solution_82:                  // GrayCode(i + 82) is a solution
	buffer->x = 82;
	buffer->mask = mask;
	buffer++;
	goto _step_82_end;

	_report_solution_83:                  // GrayCode(i + 83) is a solution
	buffer->x = 83;
	buffer->mask = mask;
	buffer++;
	goto _step_83_end;

	_report_solution_84:                  // GrayCode(i + 84) is a solution
	buffer->x = 84;
	buffer->mask = mask;
	buffer++;
	goto _step_84_end;

	_report_solution_85:                  // GrayCode(i + 85) is a solution
	buffer->x = 85;
	buffer->mask = mask;
	buffer++;
	goto _step_85_end;

	_report_solution_86:                  // GrayCode(i + 86) is a solution
	buffer->x = 86;
	buffer->mask = mask;
	buffer++;
	goto _step_86_end;

	_report_solution_87:                  // GrayCode(i + 87) is a solution
	buffer->x = 87;
	buffer->mask = mask;
	buffer++;
	goto _step_87_end;

	_report_solution_88:                  // GrayCode(i + 88) is a solution
	buffer->x = 88;
	buffer->mask = mask;
	buffer++;
	goto _step_88_end;

	_report_solution_89:                  // GrayCode(i + 89) is a solution
	buffer->x = 89;
	buffer->mask = mask;
	buffer++;
	goto _step_89_end;

	_report_solution_90:                  // GrayCode(i + 90) is a solution
	buffer->x = 90;
	buffer->mask = mask;
	buffer++;
	goto _step_90_end;

	_report_solution_91:                  // GrayCode(i + 91) is a solution
	buffer->x = 91;
	buffer->mask = mask;
	buffer++;
	goto _step_91_end;

	_report_solution_92:                  // GrayCode(i + 92) is a solution
	buffer->x = 92;
	buffer->mask = mask;
	buffer++;
	goto _step_92_end;

	_report_solution_93:                  // GrayCode(i + 93) is a solution
	buffer->x = 93;
	buffer->mask = mask;
	buffer++;
	goto _step_93_end;

	_report_solution_94:                  // GrayCode(i + 94) is a solution
	buffer->x = 94;
	buffer->mask = mask;
	buffer++;
	goto _step_94_end;

	_report_solution_95:                  // GrayCode(i + 95) is a solution
	buffer->x = 95;
	buffer->mask = mask;
	buffer++;
	goto _step_95_end;

	_report_solution_96:                  // GrayCode(i + 96) is a solution
	buffer->x = 96;
	buffer->mask = mask;
	buffer++;
	goto _step_96_end;

	_report_solution_97:                  // GrayCode(i + 97) is a solution
	buffer->x = 97;
	buffer->mask = mask;
	buffer++;
	goto _step_97_end;

	_report_solution_98:                  // GrayCode(i + 98) is a solution
	buffer->x = 98;
	buffer->mask = mask;
	buffer++;
	goto _step_98_end;

	_report_solution_99:                  // GrayCode(i + 99) is a solution
	buffer->x = 99;
	buffer->mask = mask;
	buffer++;
	goto _step_99_end;

	_report_solution_100:                  // GrayCode(i + 100) is a solution
	buffer->x = 100;
	buffer->mask = mask;
	buffer++;
	goto _step_100_end;

	_report_solution_101:                  // GrayCode(i + 101) is a solution
	buffer->x = 101;
	buffer->mask = mask;
	buffer++;
	goto _step_101_end;

	_report_solution_102:                  // GrayCode(i + 102) is a solution
	buffer->x = 102;
	buffer->mask = mask;
	buffer++;
	goto _step_102_end;

	_report_solution_103:                  // GrayCode(i + 103) is a solution
	buffer->x = 103;
	buffer->mask = mask;
	buffer++;
	goto _step_103_end;

	_report_solution_104:                  // GrayCode(i + 104) is a solution
	buffer->x = 104;
	buffer->mask = mask;
	buffer++;
	goto _step_104_end;

	_report_solution_105:                  // GrayCode(i + 105) is a solution
	buffer->x = 105;
	buffer->mask = mask;
	buffer++;
	goto _step_105_end;

	_report_solution_106:                  // GrayCode(i + 106) is a solution
	buffer->x = 106;
	buffer->mask = mask;
	buffer++;
	goto _step_106_end;

	_report_solution_107:                  // GrayCode(i + 107) is a solution
	buffer->x = 107;
	buffer->mask = mask;
	buffer++;
	goto _step_107_end;

	_report_solution_108:                  // GrayCode(i + 108) is a solution
	buffer->x = 108;
	buffer->mask = mask;
	buffer++;
	goto _step_108_end;

	_report_solution_109:                  // GrayCode(i + 109) is a solution
	buffer->x = 109;
	buffer->mask = mask;
	buffer++;
	goto _step_109_end;

	_report_solution_110:                  // GrayCode(i + 110) is a solution
	buffer->x = 110;
	buffer->mask = mask;
	buffer++;
	goto _step_110_end;

	_report_solution_111:                  // GrayCode(i + 111) is a solution
	buffer->x = 111;
	buffer->mask = mask;
	buffer++;
	goto _step_111_end;

	_report_solution_112:                  // GrayCode(i + 112) is a solution
	buffer->x = 112;
	buffer->mask = mask;
	buffer++;
	goto _step_112_end;

	_report_solution_113:                  // GrayCode(i + 113) is a solution
	buffer->x = 113;
	buffer->mask = mask;
	buffer++;
	goto _step_113_end;

	_report_solution_114:                  // GrayCode(i + 114) is a solution
	buffer->x = 114;
	buffer->mask = mask;
	buffer++;
	goto _step_114_end;

	_report_solution_115:                  // GrayCode(i + 115) is a solution
	buffer->x = 115;
	buffer->mask = mask;
	buffer++;
	goto _step_115_end;

	_report_solution_116:                  // GrayCode(i + 116) is a solution
	buffer->x = 116;
	buffer->mask = mask;
	buffer++;
	goto _step_116_end;

	_report_solution_117:                  // GrayCode(i + 117) is a solution
	buffer->x = 117;
	buffer->mask = mask;
	buffer++;
	goto _step_117_end;

	_report_solution_118:                  // GrayCode(i + 118) is a solution
	buffer->x = 118;
	buffer->mask = mask;
	buffer++;
	goto _step_118_end;

	_report_solution_119:                  // GrayCode(i + 119) is a solution
	buffer->x = 119;
	buffer->mask = mask;
	buffer++;
	goto _step_119_end;

	_report_solution_120:                  // GrayCode(i + 120) is a solution
	buffer->x = 120;
	buffer->mask = mask;
	buffer++;
	goto _step_120_end;

	_report_solution_121:                  // GrayCode(i + 121) is a solution
	buffer->x = 121;
	buffer->mask = mask;
	buffer++;
	goto _step_121_end;

	_report_solution_122:                  // GrayCode(i + 122) is a solution
	buffer->x = 122;
	buffer->mask = mask;
	buffer++;
	goto _step_122_end;

	_report_solution_123:                  // GrayCode(i + 123) is a solution
	buffer->x = 123;
	buffer->mask = mask;
	buffer++;
	goto _step_123_end;

	_report_solution_124:                  // GrayCode(i + 124) is a solution
	buffer->x = 124;
	buffer->mask = mask;
	buffer++;
	goto _step_124_end;

	_report_solution_125:                  // GrayCode(i + 125) is a solution
	buffer->x = 125;
	buffer->mask = mask;
	buffer++;
	goto _step_125_end;

	_report_solution_126:                  // GrayCode(i + 126) is a solution
	buffer->x = 126;
	buffer->mask = mask;
	buffer++;
	goto _step_126_end;

	_report_solution_127:                  // GrayCode(i + 127) is a solution
	buffer->x = 127;
	buffer->mask = mask;
	buffer++;
	goto _step_127_end;

	_report_solution_128:                  // GrayCode(i + 128) is a solution
	buffer->x = 128;
	buffer->mask = mask;
	buffer++;
	goto _step_128_end;

	_report_solution_129:                  // GrayCode(i + 129) is a solution
	buffer->x = 129;
	buffer->mask = mask;
	buffer++;
	goto _step_129_end;

	_report_solution_130:                  // GrayCode(i + 130) is a solution
	buffer->x = 130;
	buffer->mask = mask;
	buffer++;
	goto _step_130_end;

	_report_solution_131:                  // GrayCode(i + 131) is a solution
	buffer->x = 131;
	buffer->mask = mask;
	buffer++;
	goto _step_131_end;

	_report_solution_132:                  // GrayCode(i + 132) is a solution
	buffer->x = 132;
	buffer->mask = mask;
	buffer++;
	goto _step_132_end;

	_report_solution_133:                  // GrayCode(i + 133) is a solution
	buffer->x = 133;
	buffer->mask = mask;
	buffer++;
	goto _step_133_end;

	_report_solution_134:                  // GrayCode(i + 134) is a solution
	buffer->x = 134;
	buffer->mask = mask;
	buffer++;
	goto _step_134_end;

	_report_solution_135:                  // GrayCode(i + 135) is a solution
	buffer->x = 135;
	buffer->mask = mask;
	buffer++;
	goto _step_135_end;

	_report_solution_136:                  // GrayCode(i + 136) is a solution
	buffer->x = 136;
	buffer->mask = mask;
	buffer++;
	goto _step_136_end;

	_report_solution_137:                  // GrayCode(i + 137) is a solution
	buffer->x = 137;
	buffer->mask = mask;
	buffer++;
	goto _step_137_end;

	_report_solution_138:                  // GrayCode(i + 138) is a solution
	buffer->x = 138;
	buffer->mask = mask;
	buffer++;
	goto _step_138_end;

	_report_solution_139:                  // GrayCode(i + 139) is a solution
	buffer->x = 139;
	buffer->mask = mask;
	buffer++;
	goto _step_139_end;

	_report_solution_140:                  // GrayCode(i + 140) is a solution
	buffer->x = 140;
	buffer->mask = mask;
	buffer++;
	goto _step_140_end;

	_report_solution_141:                  // GrayCode(i + 141) is a solution
	buffer->x = 141;
	buffer->mask = mask;
	buffer++;
	goto _step_141_end;

	_report_solution_142:                  // GrayCode(i + 142) is a solution
	buffer->x = 142;
	buffer->mask = mask;
	buffer++;
	goto _step_142_end;

	_report_solution_143:                  // GrayCode(i + 143) is a solution
	buffer->x = 143;
	buffer->mask = mask;
	buffer++;
	goto _step_143_end;

	_report_solution_144:                  // GrayCode(i + 144) is a solution
	buffer->x = 144;
	buffer->mask = mask;
	buffer++;
	goto _step_144_end;

	_report_solution_145:                  // GrayCode(i + 145) is a solution
	buffer->x = 145;
	buffer->mask = mask;
	buffer++;
	goto _step_145_end;

	_report_solution_146:                  // GrayCode(i + 146) is a solution
	buffer->x = 146;
	buffer->mask = mask;
	buffer++;
	goto _step_146_end;

	_report_solution_147:                  // GrayCode(i + 147) is a solution
	buffer->x = 147;
	buffer->mask = mask;
	buffer++;
	goto _step_147_end;

	_report_solution_148:                  // GrayCode(i + 148) is a solution
	buffer->x = 148;
	buffer->mask = mask;
	buffer++;
	goto _step_148_end;

	_report_solution_149:                  // GrayCode(i + 149) is a solution
	buffer->x = 149;
	buffer->mask = mask;
	buffer++;
	goto _step_149_end;

	_report_solution_150:                  // GrayCode(i + 150) is a solution
	buffer->x = 150;
	buffer->mask = mask;
	buffer++;
	goto _step_150_end;

	_report_solution_151:                  // GrayCode(i + 151) is a solution
	buffer->x = 151;
	buffer->mask = mask;
	buffer++;
	goto _step_151_end;

	_report_solution_152:                  // GrayCode(i + 152) is a solution
	buffer->x = 152;
	buffer->mask = mask;
	buffer++;
	goto _step_152_end;

	_report_solution_153:                  // GrayCode(i + 153) is a solution
	buffer->x = 153;
	buffer->mask = mask;
	buffer++;
	goto _step_153_end;

	_report_solution_154:                  // GrayCode(i + 154) is a solution
	buffer->x = 154;
	buffer->mask = mask;
	buffer++;
	goto _step_154_end;

	_report_solution_155:                  // GrayCode(i + 155) is a solution
	buffer->x = 155;
	buffer->mask = mask;
	buffer++;
	goto _step_155_end;

	_report_solution_156:                  // GrayCode(i + 156) is a solution
	buffer->x = 156;
	buffer->mask = mask;
	buffer++;
	goto _step_156_end;

	_report_solution_157:                  // GrayCode(i + 157) is a solution
	buffer->x = 157;
	buffer->mask = mask;
	buffer++;
	goto _step_157_end;

	_report_solution_158:                  // GrayCode(i + 158) is a solution
	buffer->x = 158;
	buffer->mask = mask;
	buffer++;
	goto _step_158_end;

	_report_solution_159:                  // GrayCode(i + 159) is a solution
	buffer->x = 159;
	buffer->mask = mask;
	buffer++;
	goto _step_159_end;

	_report_solution_160:                  // GrayCode(i + 160) is a solution
	buffer->x = 160;
	buffer->mask = mask;
	buffer++;
	goto _step_160_end;

	_report_solution_161:                  // GrayCode(i + 161) is a solution
	buffer->x = 161;
	buffer->mask = mask;
	buffer++;
	goto _step_161_end;

	_report_solution_162:                  // GrayCode(i + 162) is a solution
	buffer->x = 162;
	buffer->mask = mask;
	buffer++;
	goto _step_162_end;

	_report_solution_163:                  // GrayCode(i + 163) is a solution
	buffer->x = 163;
	buffer->mask = mask;
	buffer++;
	goto _step_163_end;

	_report_solution_164:                  // GrayCode(i + 164) is a solution
	buffer->x = 164;
	buffer->mask = mask;
	buffer++;
	goto _step_164_end;

	_report_solution_165:                  // GrayCode(i + 165) is a solution
	buffer->x = 165;
	buffer->mask = mask;
	buffer++;
	goto _step_165_end;

	_report_solution_166:                  // GrayCode(i + 166) is a solution
	buffer->x = 166;
	buffer->mask = mask;
	buffer++;
	goto _step_166_end;

	_report_solution_167:                  // GrayCode(i + 167) is a solution
	buffer->x = 167;
	buffer->mask = mask;
	buffer++;
	goto _step_167_end;

	_report_solution_168:                  // GrayCode(i + 168) is a solution
	buffer->x = 168;
	buffer->mask = mask;
	buffer++;
	goto _step_168_end;

	_report_solution_169:                  // GrayCode(i + 169) is a solution
	buffer->x = 169;
	buffer->mask = mask;
	buffer++;
	goto _step_169_end;

	_report_solution_170:                  // GrayCode(i + 170) is a solution
	buffer->x = 170;
	buffer->mask = mask;
	buffer++;
	goto _step_170_end;

	_report_solution_171:                  // GrayCode(i + 171) is a solution
	buffer->x = 171;
	buffer->mask = mask;
	buffer++;
	goto _step_171_end;

	_report_solution_172:                  // GrayCode(i + 172) is a solution
	buffer->x = 172;
	buffer->mask = mask;
	buffer++;
	goto _step_172_end;

	_report_solution_173:                  // GrayCode(i + 173) is a solution
	buffer->x = 173;
	buffer->mask = mask;
	buffer++;
	goto _step_173_end;

	_report_solution_174:                  // GrayCode(i + 174) is a solution
	buffer->x = 174;
	buffer->mask = mask;
	buffer++;
	goto _step_174_end;

	_report_solution_175:                  // GrayCode(i + 175) is a solution
	buffer->x = 175;
	buffer->mask = mask;
	buffer++;
	goto _step_175_end;

	_report_solution_176:                  // GrayCode(i + 176) is a solution
	buffer->x = 176;
	buffer->mask = mask;
	buffer++;
	goto _step_176_end;

	_report_solution_177:                  // GrayCode(i + 177) is a solution
	buffer->x = 177;
	buffer->mask = mask;
	buffer++;
	goto _step_177_end;

	_report_solution_178:                  // GrayCode(i + 178) is a solution
	buffer->x = 178;
	buffer->mask = mask;
	buffer++;
	goto _step_178_end;

	_report_solution_179:                  // GrayCode(i + 179) is a solution
	buffer->x = 179;
	buffer->mask = mask;
	buffer++;
	goto _step_179_end;

	_report_solution_180:                  // GrayCode(i + 180) is a solution
	buffer->x = 180;
	buffer->mask = mask;
	buffer++;
	goto _step_180_end;

	_report_solution_181:                  // GrayCode(i + 181) is a solution
	buffer->x = 181;
	buffer->mask = mask;
	buffer++;
	goto _step_181_end;

	_report_solution_182:                  // GrayCode(i + 182) is a solution
	buffer->x = 182;
	buffer->mask = mask;
	buffer++;
	goto _step_182_end;

	_report_solution_183:                  // GrayCode(i + 183) is a solution
	buffer->x = 183;
	buffer->mask = mask;
	buffer++;
	goto _step_183_end;

	_report_solution_184:                  // GrayCode(i + 184) is a solution
	buffer->x = 184;
	buffer->mask = mask;
	buffer++;
	goto _step_184_end;

	_report_solution_185:                  // GrayCode(i + 185) is a solution
	buffer->x = 185;
	buffer->mask = mask;
	buffer++;
	goto _step_185_end;

	_report_solution_186:                  // GrayCode(i + 186) is a solution
	buffer->x = 186;
	buffer->mask = mask;
	buffer++;
	goto _step_186_end;

	_report_solution_187:                  // GrayCode(i + 187) is a solution
	buffer->x = 187;
	buffer->mask = mask;
	buffer++;
	goto _step_187_end;

	_report_solution_188:                  // GrayCode(i + 188) is a solution
	buffer->x = 188;
	buffer->mask = mask;
	buffer++;
	goto _step_188_end;

	_report_solution_189:                  // GrayCode(i + 189) is a solution
	buffer->x = 189;
	buffer->mask = mask;
	buffer++;
	goto _step_189_end;

	_report_solution_190:                  // GrayCode(i + 190) is a solution
	buffer->x = 190;
	buffer->mask = mask;
	buffer++;
	goto _step_190_end;

	_report_solution_191:                  // GrayCode(i + 191) is a solution
	buffer->x = 191;
	buffer->mask = mask;
	buffer++;
	goto _step_191_end;

	_report_solution_192:                  // GrayCode(i + 192) is a solution
	buffer->x = 192;
	buffer->mask = mask;
	buffer++;
	goto _step_192_end;

	_report_solution_193:                  // GrayCode(i + 193) is a solution
	buffer->x = 193;
	buffer->mask = mask;
	buffer++;
	goto _step_193_end;

	_report_solution_194:                  // GrayCode(i + 194) is a solution
	buffer->x = 194;
	buffer->mask = mask;
	buffer++;
	goto _step_194_end;

	_report_solution_195:                  // GrayCode(i + 195) is a solution
	buffer->x = 195;
	buffer->mask = mask;
	buffer++;
	goto _step_195_end;

	_report_solution_196:                  // GrayCode(i + 196) is a solution
	buffer->x = 196;
	buffer->mask = mask;
	buffer++;
	goto _step_196_end;

	_report_solution_197:                  // GrayCode(i + 197) is a solution
	buffer->x = 197;
	buffer->mask = mask;
	buffer++;
	goto _step_197_end;

	_report_solution_198:                  // GrayCode(i + 198) is a solution
	buffer->x = 198;
	buffer->mask = mask;
	buffer++;
	goto _step_198_end;

	_report_solution_199:                  // GrayCode(i + 199) is a solution
	buffer->x = 199;
	buffer->mask = mask;
	buffer++;
	goto _step_199_end;

	_report_solution_200:                  // GrayCode(i + 200) is a solution
	buffer->x = 200;
	buffer->mask = mask;
	buffer++;
	goto _step_200_end;

	_report_solution_201:                  // GrayCode(i + 201) is a solution
	buffer->x = 201;
	buffer->mask = mask;
	buffer++;
	goto _step_201_end;

	_report_solution_202:                  // GrayCode(i + 202) is a solution
	buffer->x = 202;
	buffer->mask = mask;
	buffer++;
	goto _step_202_end;

	_report_solution_203:                  // GrayCode(i + 203) is a solution
	buffer->x = 203;
	buffer->mask = mask;
	buffer++;
	goto _step_203_end;

	_report_solution_204:                  // GrayCode(i + 204) is a solution
	buffer->x = 204;
	buffer->mask = mask;
	buffer++;
	goto _step_204_end;

	_report_solution_205:                  // GrayCode(i + 205) is a solution
	buffer->x = 205;
	buffer->mask = mask;
	buffer++;
	goto _step_205_end;

	_report_solution_206:                  // GrayCode(i + 206) is a solution
	buffer->x = 206;
	buffer->mask = mask;
	buffer++;
	goto _step_206_end;

	_report_solution_207:                  // GrayCode(i + 207) is a solution
	buffer->x = 207;
	buffer->mask = mask;
	buffer++;
	goto _step_207_end;

	_report_solution_208:                  // GrayCode(i + 208) is a solution
	buffer->x = 208;
	buffer->mask = mask;
	buffer++;
	goto _step_208_end;

	_report_solution_209:                  // GrayCode(i + 209) is a solution
	buffer->x = 209;
	buffer->mask = mask;
	buffer++;
	goto _step_209_end;

	_report_solution_210:                  // GrayCode(i + 210) is a solution
	buffer->x = 210;
	buffer->mask = mask;
	buffer++;
	goto _step_210_end;

	_report_solution_211:                  // GrayCode(i + 211) is a solution
	buffer->x = 211;
	buffer->mask = mask;
	buffer++;
	goto _step_211_end;

	_report_solution_212:                  // GrayCode(i + 212) is a solution
	buffer->x = 212;
	buffer->mask = mask;
	buffer++;
	goto _step_212_end;

	_report_solution_213:                  // GrayCode(i + 213) is a solution
	buffer->x = 213;
	buffer->mask = mask;
	buffer++;
	goto _step_213_end;

	_report_solution_214:                  // GrayCode(i + 214) is a solution
	buffer->x = 214;
	buffer->mask = mask;
	buffer++;
	goto _step_214_end;

	_report_solution_215:                  // GrayCode(i + 215) is a solution
	buffer->x = 215;
	buffer->mask = mask;
	buffer++;
	goto _step_215_end;

	_report_solution_216:                  // GrayCode(i + 216) is a solution
	buffer->x = 216;
	buffer->mask = mask;
	buffer++;
	goto _step_216_end;

	_report_solution_217:                  // GrayCode(i + 217) is a solution
	buffer->x = 217;
	buffer->mask = mask;
	buffer++;
	goto _step_217_end;

	_report_solution_218:                  // GrayCode(i + 218) is a solution
	buffer->x = 218;
	buffer->mask = mask;
	buffer++;
	goto _step_218_end;

	_report_solution_219:                  // GrayCode(i + 219) is a solution
	buffer->x = 219;
	buffer->mask = mask;
	buffer++;
	goto _step_219_end;

	_report_solution_220:                  // GrayCode(i + 220) is a solution
	buffer->x = 220;
	buffer->mask = mask;
	buffer++;
	goto _step_220_end;

	_report_solution_221:                  // GrayCode(i + 221) is a solution
	buffer->x = 221;
	buffer->mask = mask;
	buffer++;
	goto _step_221_end;

	_report_solution_222:                  // GrayCode(i + 222) is a solution
	buffer->x = 222;
	buffer->mask = mask;
	buffer++;
	goto _step_222_end;

	_report_solution_223:                  // GrayCode(i + 223) is a solution
	buffer->x = 223;
	buffer->mask = mask;
	buffer++;
	goto _step_223_end;

	_report_solution_224:                  // GrayCode(i + 224) is a solution
	buffer->x = 224;
	buffer->mask = mask;
	buffer++;
	goto _step_224_end;

	_report_solution_225:                  // GrayCode(i + 225) is a solution
	buffer->x = 225;
	buffer->mask = mask;
	buffer++;
	goto _step_225_end;

	_report_solution_226:                  // GrayCode(i + 226) is a solution
	buffer->x = 226;
	buffer->mask = mask;
	buffer++;
	goto _step_226_end;

	_report_solution_227:                  // GrayCode(i + 227) is a solution
	buffer->x = 227;
	buffer->mask = mask;
	buffer++;
	goto _step_227_end;

	_report_solution_228:                  // GrayCode(i + 228) is a solution
	buffer->x = 228;
	buffer->mask = mask;
	buffer++;
	goto _step_228_end;

	_report_solution_229:                  // GrayCode(i + 229) is a solution
	buffer->x = 229;
	buffer->mask = mask;
	buffer++;
	goto _step_229_end;

	_report_solution_230:                  // GrayCode(i + 230) is a solution
	buffer->x = 230;
	buffer->mask = mask;
	buffer++;
	goto _step_230_end;

	_report_solution_231:                  // GrayCode(i + 231) is a solution
	buffer->x = 231;
	buffer->mask = mask;
	buffer++;
	goto _step_231_end;

	_report_solution_232:                  // GrayCode(i + 232) is a solution
	buffer->x = 232;
	buffer->mask = mask;
	buffer++;
	goto _step_232_end;

	_report_solution_233:                  // GrayCode(i + 233) is a solution
	buffer->x = 233;
	buffer->mask = mask;
	buffer++;
	goto _step_233_end;

	_report_solution_234:                  // GrayCode(i + 234) is a solution
	buffer->x = 234;
	buffer->mask = mask;
	buffer++;
	goto _step_234_end;

	_report_solution_235:                  // GrayCode(i + 235) is a solution
	buffer->x = 235;
	buffer->mask = mask;
	buffer++;
	goto _step_235_end;

	_report_solution_236:                  // GrayCode(i + 236) is a solution
	buffer->x = 236;
	buffer->mask = mask;
	buffer++;
	goto _step_236_end;

	_report_solution_237:                  // GrayCode(i + 237) is a solution
	buffer->x = 237;
	buffer->mask = mask;
	buffer++;
	goto _step_237_end;

	_report_solution_238:                  // GrayCode(i + 238) is a solution
	buffer->x = 238;
	buffer->mask = mask;
	buffer++;
	goto _step_238_end;

	_report_solution_239:                  // GrayCode(i + 239) is a solution
	buffer->x = 239;
	buffer->mask = mask;
	buffer++;
	goto _step_239_end;

	_report_solution_240:                  // GrayCode(i + 240) is a solution
	buffer->x = 240;
	buffer->mask = mask;
	buffer++;
	goto _step_240_end;

	_report_solution_241:                  // GrayCode(i + 241) is a solution
	buffer->x = 241;
	buffer->mask = mask;
	buffer++;
	goto _step_241_end;

	_report_solution_242:                  // GrayCode(i + 242) is a solution
	buffer->x = 242;
	buffer->mask = mask;
	buffer++;
	goto _step_242_end;

	_report_solution_243:                  // GrayCode(i + 243) is a solution
	buffer->x = 243;
	buffer->mask = mask;
	buffer++;
	goto _step_243_end;

	_report_solution_244:                  // GrayCode(i + 244) is a solution
	buffer->x = 244;
	buffer->mask = mask;
	buffer++;
	goto _step_244_end;

	_report_solution_245:                  // GrayCode(i + 245) is a solution
	buffer->x = 245;
	buffer->mask = mask;
	buffer++;
	goto _step_245_end;

	_report_solution_246:                  // GrayCode(i + 246) is a solution
	buffer->x = 246;
	buffer->mask = mask;
	buffer++;
	goto _step_246_end;

	_report_solution_247:                  // GrayCode(i + 247) is a solution
	buffer->x = 247;
	buffer->mask = mask;
	buffer++;
	goto _step_247_end;

	_report_solution_248:                  // GrayCode(i + 248) is a solution
	buffer->x = 248;
	buffer->mask = mask;
	buffer++;
	goto _step_248_end;

	_report_solution_249:                  // GrayCode(i + 249) is a solution
	buffer->x = 249;
	buffer->mask = mask;
	buffer++;
	goto _step_249_end;

	_report_solution_250:                  // GrayCode(i + 250) is a solution
	buffer->x = 250;
	buffer->mask = mask;
	buffer++;
	goto _step_250_end;

	_report_solution_251:                  // GrayCode(i + 251) is a solution
	buffer->x = 251;
	buffer->mask = mask;
	buffer++;
	goto _step_251_end;

	_report_solution_252:                  // GrayCode(i + 252) is a solution
	buffer->x = 252;
	buffer->mask = mask;
	buffer++;
	goto _step_252_end;

	_report_solution_253:                  // GrayCode(i + 253) is a solution
	buffer->x = 253;
	buffer->mask = mask;
	buffer++;
	goto _step_253_end;

	_report_solution_254:                  // GrayCode(i + 254) is a solution
	buffer->x = 254;
	buffer->mask = mask;
	buffer++;
	goto _step_254_end;

	_report_solution_255:                  // GrayCode(i + 255) is a solution
	buffer->x = 255;
	buffer->mask = mask;
	buffer++;
	goto _step_255_end;

}
