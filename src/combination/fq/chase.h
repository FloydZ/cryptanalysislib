#ifndef CRYPTANALYSISLIB_COMBINATION_CHASE_H
#define CRYPTANALYSISLIB_COMBINATION_CHASE_H

#include <cstdint>
#include <vector>
#include "math/ipow.h"


class Combinations_Fq_Chase {
//private:
public:
	/// length to enumerate
	const uint32_t n;

	/// max value to enumerate
	const uint32_t q, qm1;

	/// max hamming weight to enumerate
	const uint32_t w;

	size_t chase_size;
	size_t gray_size;

	/// TODO this is the output container
	std::vector<uint32_t> a;

	/// stuff for the mixed radix enumeration
	std::vector<uint32_t> f;
	std::vector<uint32_t> s; // sentinel

	/// stuff for the chase sequence
	std::vector<int> chase_w;
	std::vector<int> chase_a;

	/// \return  number of elements in the gray code
	constexpr size_t compute_gray_size() noexcept{
		uint64_t sum = 1;
		//for (uint64_t i = 1; i <= w; ++i) {
		//	uint64_t tmp = bc(w, i);
		//	for (uint64_t j = 0; j < i; ++j) {
		//		tmp *= q;
		//	}
		//	sum += tmp;
		//}

		for (uint64_t i = 0; i < w; i++) {
			sum *= qm1;
		}
		// just make sure that we do not return zero.
		return std::max(sum, uint64_t(1));
	}
	/// just for debugging
	/// \param two_changes
	void print_state(int two_changes=false) {
		for (uint32_t i = 0; i < n; i++) {
			printf("%u", a[i]);
		}
		if (two_changes)
			printf(" t");
		printf("\n");
	}

	///
	/// \param print_j
	void print_state(uint32_t print_j) {
		for (uint32_t i = 0; i < n; i++) {
			printf("%u", a[i]);
		}
		printf(" j:%d\n", print_j);
	}

	void print_chase_state(int r, int j) {
		for (int i = w-1; i >= 0; i--) {
			printf("%u", n-chase_a[i]-1);
		}
		printf(" r:%d j:%d\n", r, j);
	}

	///
	/// \return 1 if the sequence is still valid
	/// 	    0 if the sequence is finidhed
	int mixed_radix_grey() {
		while (1) {
			print_state(0);
			uint32_t j = f[0];
			f[0] = 0;
			if (j == n)
				return 0;

			a[j] = (a[j] + 1) % q;

			if (a[j] == s[j]) {
				s[j] = (s[j]-1 + q) % q;
				f[j] = f[j+1];
				f[j+1] = j + 1;
			}
		}

		return 1;
	}

	/// NOTE: this enumerates on a length w NOT on length n
	/// NOTE: you have to free this stuff yourself
	/// NOTE: the input must be of size sum_bc(w, w)*q**w
	/// NOTE: returns 0 on success
	/// NOTE: only enumerates to q-1
	int changelist_mixed_radix_grey(uint16_t *ret) {
		uint32_t j = 0;
		size_t ctr = 0;

		while (true) {
			// print_state(j);
			j = f[0];
			f[0] = 0;
			ret[ctr++] = j;
			if (j == w)
				return 0;

			a[j] = (a[j] + 1) % (qm1);

			if (a[j] == s[j]) {
				s[j] = (s[j]-1 + qm1) % qm1;
				f[j] = f[j+1];
				f[j+1] = j + 1;
			}
		}
	}

	/// \param r helper value, init with 0
	/// \param jj returns the change position
	void chase(int *r, int *jj) {
		bool found_r = false;
		int j;
		for (j = *r; !chase_w[j]; j++) {
			int b = chase_a[j] + 1;
			int n = chase_a[j + 1];
			if (b < (chase_w[j + 1] ? n - (2 - (n & 1u)) : n)) {
				if ((b & 1) == 0 && b + 1 < n) {
					b++;
				}

				chase_a[j] = b;
				if (!found_r) {
					*r = (int)(j > 1 ? j - 1 : 0);
				}

				*jj = j;
				return;
			}

			chase_w[j] = chase_a[j] - 1 >= j;
			if (chase_w[j] && !found_r) {
				*r = (int)j;
				found_r = true;
			}
		}

		int b = (int)chase_a[j] - 1;
		if ((b & 1) != 0 && b - 1 >= j) {
			b--;
		}

		chase_a[j] = b;
		chase_w[j] = b - 1 >= j;
		if (!found_r) {
			*r = j;
		}

		*jj = j;
	}

	/// NOTE: this function inverts the output of the chase sequence
	/// \param ret
	void changelist_chase(std::pair<uint16_t,uint16_t> *ret) {
		int r = 0, j = 0;

		uint32_t tmp[w+1];
		for (uint32_t i = 0; i < w; ++i) {
			tmp[i] = n-chase_a[i]-1;
		}

		for (uint32_t i = 0; i < chase_size; ++i) {
			//print_chase_state(r, j);
			chase(&r, &j);
			ASSERT(j < w);

			ret[i].first = j;
			ret[i].second = n-chase_a[j]-1;

			tmp[j] = n-chase_a[j]-1;
		}
	}

	/// ignore this function
	/// \param gray_cl
	/// \param chase_cl
	void build_list(const uint16_t *gray_cl,
	                const std::pair<uint16_t,uint16_t> *chase_cl){
		/// init
		std::vector<uint32_t > vec(n, 0);
		for (uint32_t i = 0; i < w; ++i) {
			vec[i] = 1u;
		}

		std::vector<uint32_t> current_set(w, 0);
		for (uint32_t i = 0; i < w; ++i) {
			current_set[i] = i;
		}

		auto print_state_ = [&]() {
		  for (int k = 0; k < n; ++k) {
			  printf("%d", vec[k]);
		  }
		  //printf(" gcl: %d\n", gray_cl[j]);
		  printf("\n");
		};
		/// iterate over all
		for (uint32_t i = 0; i < chase_size; ++i) {
			for (uint32_t j = 0; j < gray_size-1; ++j) {
				print_state_();

				/// TODO change this is to the normal vector addition
				vec[current_set[gray_cl[j]]] = (vec[current_set[gray_cl[j]]] + 1u) % q;
				vec[current_set[gray_cl[j]]] = std::max(1u, vec[current_set[gray_cl[j]]]);
		    }

			print_state_();

			/// advance the current set by one
			const uint32_t j = chase_cl[i].first;
			const uint32_t a = current_set[j];
			const uint32_t b = chase_cl[i].second;
			current_set[j] = b;

			/// TODO replace to the normal vector addition
			vec[a] = 0;//(vec[a] + 1u) % q;
			vec[b] = 1;
		}
	}

public:
	///
	/// \param n length
	/// \param q field size
	/// \param w max hamming weight to enumerate
	constexpr Combinations_Fq_Chase(const uint32_t n,
	                                const uint32_t q,
	                                const uint32_t w) :
	    n(n), q(q), qm1(q-1), w(w) {
		chase_size = bc(n, w) - 1;
		gray_size = compute_gray_size();

		/// init the restricted gray code
		a.resize(n);
		f.resize(n + 1);
		s.resize(n);

		for (uint32_t i = 0; i < n; ++i) {
			a[i] = 0;
			f[i] = i;
			s[i] = qm1-1;
		}

		f[n] = n;

		/// init the chase sequence
		chase_a.resize(w+1);
		chase_w.resize(w+1);
		for (uint32_t i = 0; i < w + 1; ++i) {
			chase_a[i] = n - (w - i);
			chase_w[i] = true;
		}
	}

	/// return
	constexpr uint32_t step() {
		return 1;
	}

};
#endif//CRYPTANALYSISLIB_CHASE_H
