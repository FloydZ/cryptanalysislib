#ifndef CRYPTANALYSISLIB_COMBINATION_LEXICOGRAPHIC_H
#define CRYPTANALYSISLIB_COMBINATION_LEXICOGRAPHIC_H
#include <cstdint>

class Combinations_Fq_Lexicographic {
	/// length to enumerate
	const uint32_t n;

	/// max value to enumerate
	const uint32_t q;


	Combinations_Fq_Lexicographic(const uint32_t n, const uint32_t q) :
	n(n), q(q) {}

	/// \param stack array of size
	/// \param sp current stack pointer. Do not think about this
	/// \param k max weight to enumerate
	/// \return the number of subsequent numbers which have hamming weight < k.
	uint64_t restricted_zeta_update_stack(uint32_t *stack, uint32_t *sp, const uint64_t k) {
		if (stack[*sp] == k) {
			uint32_t i = *sp + 1;

			// walk up
			while (stack[i] == k)
				i += 1;

			// update
			stack[i] += 1;
			const uint32_t val = stack[i];
			const uint32_t altered_sp = i;

			// walk down
			while (i > 0) {
				stack[i - 1] = val;
				i -= 1;
			}

			// fix up stack pointer
			*sp = 0;
			return (1ull << (altered_sp + 1)) - 1;
		} else {
			stack[*sp] += 1;
			return 1ull;
		}
	}
public:
};

#endif//CRYPTANALYSISLIB_COMBINATION_LEXICOGRAPHIC_H
