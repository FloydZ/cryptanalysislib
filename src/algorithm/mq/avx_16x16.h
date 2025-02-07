	#include <stdint.h>
	#include <immintrin.h>
//struct solution_t {
//	uint32_t x;
//	uint32_t mask;
//};
struct solution_t* solver(uint16_t *rdi, uint16_t *rsi, const uint32_t alpha, const uint32_t beta, const uint32_t gamma, struct solution_t *buffer) {
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
	__m256i ymm14;
	__m256i ymm15 = _mm256_set1_epi8(0);


	// step   0 : Fl[0] ^= (Fl[1] ^= Fq[alpha + 0])

	ymm15 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm15);
	if (mask != 0) { goto _report_solution_0; }
	_step_0_end:
	ymm1 = _mm256_xor_si256(ymm1, *(__m256i *)(rdi + alpha + 0));
	ymm0 = _mm256_xor_si256(ymm0, ymm1);


	// step   1 : Fl[0] ^= (Fl[2] ^= Fq[alpha + 1])

	ymm15 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm15);
	if (mask != 0) { goto _report_solution_1; }
	_step_1_end:
	ymm2 = _mm256_xor_si256(ymm2, *(__m256i *)(rdi + alpha + 32));
	ymm0 = _mm256_xor_si256(ymm0, ymm2);


	// step   2 : Fl[0] ^= (Fl[1] ^= Fq[0])

	ymm15 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm15);
	if (mask != 0) { goto _report_solution_2; }
	_step_2_end:
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

	// special last step   3 : Fl[0] ^= (Fl[beta] ^= Fq[gamma])

	ymm15 =_mm256_cmpeq_epi16(ymm0, ymm15);
	mask = _mm256_movemask_epi8(ymm15);
	if (mask != 0) { goto _report_solution_3; }
	_step_3_end:
	ymm14 = _mm256_load_si256((__m256i *)(rsi + beta));
	ymm14 = _mm256_xor_si256(*(__m256i *)(rdi + gamma), ymm14); 
	_mm256_store_si256((__m256i *)(rsi + beta), ymm14);
	ymm0 = _mm256_xor_si256(ymm0, ymm14);

	// Save Fl[0] back to memory
	_mm256_store_si256((__m256i *)rsi, ymm0);

	return buffer;


	// now the code that reports solutions

	_report_solution_0:                  // GrayCode(i + 0) is a solution
	ymm15 = _mm256_xor_si256(ymm15, ymm15);// reset %ymm15 to zero
	buffer->x = 0;
	buffer->mask = mask;
	buffer++;
	goto _step_0_end;

	_report_solution_1:                  // GrayCode(i + 1) is a solution
	ymm15 = _mm256_xor_si256(ymm15, ymm15);// reset %ymm15 to zero
	buffer->x = 1;
	buffer->mask = mask;
	buffer++;
	goto _step_1_end;

	_report_solution_2:                  // GrayCode(i + 2) is a solution
	ymm15 = _mm256_xor_si256(ymm15, ymm15);// reset %ymm15 to zero
	buffer->x = 2;
	buffer->mask = mask;
	buffer++;
	goto _step_2_end;

	_report_solution_3:                  // GrayCode(i + 3) is a solution
	ymm15 = _mm256_xor_si256(ymm15, ymm15);// reset %ymm15 to zero
	buffer->x = 3;
	buffer->mask = mask;
	buffer++;
	goto _step_3_end;

}
