#ifndef CRYPTANALYSISLIB_COMBINATION_CHASE2_H
#define CRYPTANALYSISLIB_COMBINATION_CHASE2_H

#include <cstdint>

#include "helper.h"


// TODO write specialisation for p=2,3
// enumerate all cp=1,...,p
template<const uint32_t n, 
		 const uint32_t p>
class chase_full {
	static_assert(p > 0);
	static_assert(n > p);

	using T = uint16_t;
	// not storing the walking pointer
	uint32_t stack[p-1];
	uint32_t sp = 0; // stack pointer
private:
	/// \param start inclusive
	/// \param end exclusive
	/// \param direction true count up
	///					 false count down
	template<typename F>
	constexpr void enumerate1(F &&f,
							  const uint32_t start=0,
							  const uint32_t end=n,
						      const bool direction = true) {
		ASSERT(end > start);
		ASSERT(end <= n);

		if (!direction) {
			// count backwards
			for (uint32_t i = end - 1; i > start; i--) {
				f(i-1, i);
			}

			return;
		}

		for (uint32_t i = start + 1; i < end; i++) {
			f(i-1, i);
		}
	}
	
	/// \return true if the exit condition is met.
	constexpr bool check_exit(const uint32_t cp = p) {
		for (uint32_t i = 0; i < cp-1; i++) {
			if (stack[i] != n-i-2) { return false; }
		}
		return true;
	}
public:
	template<typename F>
	constexpr void enumerate(F &&f,
							 const uint32_t cp = 1,
							 const bool start_direction=false) {
		if constexpr (p == 1) {
			enumerate1(f);
			return;
		}

		if (cp == 1) { 
			enumerate1(f,0,n);
			return enumerate(f, 2, false);
		}
		
		// exit condition
		if (cp > p) { return; }
		

		// reset stack 
		for (uint32_t i = 0; i < cp-1; i++) { stack[i] = i; }
		sp = cp-2; // set it to the largest value
		uint32_t direction = start_direction;

		// add new weight
		f(0, n+1);

		while(true) {
			enumerate1(f, stack[sp] + 1 + (1 - direction), n, direction);
			// update direction, either walk up or down
			direction = (direction + 1) & 1;

			// update stack 
			stack[sp] += 1;
			f(stack[sp], stack[sp] - 1);
			if (check_exit(cp)) { break; }
			if (stack[sp] == (n - sp - 2)) {
				
			}
		}

		return enumerate(f, cp+1, false);
	}
};

#endif
