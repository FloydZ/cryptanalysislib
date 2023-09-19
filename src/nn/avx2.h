


	alignas(32) const __m256i avx_nn_k_mask = k < 32 ? _mm256_set1_epi32((uint32_t) (1ull << (k % 32)) - 1ull) : k < 64 ? _mm256_set1_epi64x((1ull << (k % 64)) - 1ull)
																														: _mm256_set1_epi32(0);
	///
	/// NOTE: upper bound `d` is inclusive
	/// \tparam exact:
	/// \param in
	/// \return
	template<const bool exact=false>
	int compare_256_32(const __m256i in1, const __m256i in2) const noexcept {
		if constexpr(EXACT) {
			const __m256i tmp2 = _mm256_cmpeq_epi32(in1, in2);
			return _mm256_movemask_ps((__m256) tmp2);
		}

		const __m256i tmp1 = _mm256_xor_si256(in1, in2);
#ifdef USE_AVX512
		const __m256i pop = _mm256_popcnt_epi32(tmp1);
#else
		const __m256i pop = popcount_avx2_32(tmp1);
#endif

		if constexpr(dk_bruteforce_weight > 0) {
			if constexpr (EXACT) {
				const __m256i avx_exact_weight32 = _mm256_set1_epi32(dk_bruteforce_weight);
				const __m256i tmp2 = _mm256_cmpeq_epi32(avx_exact_weight32, pop);
				return _mm256_movemask_ps((__m256) tmp2);
			} else  {
				const __m256i avx_weight32 = _mm256_set1_epi32(dk_bruteforce_weight+1);
				const __m256i tmp2 = _mm256_cmpgt_epi32(avx_weight32, pop);
				return _mm256_movemask_ps((__m256) tmp2);
			}

			// just to make sure that the compiler will not compiler the
			// following code
			return 0;
		}

		if constexpr (EXACT) {
			const __m256i avx_exact_weight32 = _mm256_set1_epi32(d);
			const __m256i tmp2 = _mm256_cmpeq_epi32(avx_exact_weight32, pop);
			return _mm256_movemask_ps((__m256) tmp2);
		} else  {
			const __m256i avx_weight32 = _mm256_set1_epi32(d+1);
			const __m256i tmp2 = _mm256_cmpgt_epi32(avx_weight32, pop);
			return _mm256_movemask_ps((__m256) tmp2);
		}
	}

	///
	/// NOTE: upper bound `d` is inclusive
	/// \param in
	/// \return
	template<bool exact=false>
	int compare_256_64(const __m256i in1, const __m256i in2) const noexcept {
		if constexpr(EXACT) {
			const __m256i tmp2 = _mm256_cmpeq_epi32(in1, in2);
			return _mm256_movemask_ps((__m256) tmp2);
		}

		const __m256i tmp1 = _mm256_xor_si256(in1, in2);
#ifdef USE_AVX512
		const __m256i pop = _mm256_popcnt_epi64(tmp1);
#else
		const __m256i pop = popcount_avx2_64(tmp1);
#endif

		/// in the special case where we want to match on a different weight to speed up
		/// the computation. This makes only sense if `dk_bruteforce_weight` < dk.
		if constexpr(dk_bruteforce_weight > 0) {
			if constexpr (EXACT) {
				const __m256i avx_exact_weight64 = _mm256_set1_epi64x(dk_bruteforce_weight);
#ifdef USE_AVX512
				return _mm256_cmp_epi64_mask(avx_exact_weight64, pop, 0);
#else
				const __m256i tmp2 = _mm256_cmpeq_epi64(avx_exact_weight64, pop);
				return _mm256_movemask_ps((__m256) tmp2);
#endif
			} else  {
				const __m256i avx_weight64 = _mm256_set1_epi64x(dk_bruteforce_weight+1);
#ifdef USE_AVX512
				return _mm256_cmp_epi64_mask(avx_weight64, pop, 6);
#else
				const __m256i tmp2 = _mm256_cmpgt_epi64(avx_weight64, pop);
				return _mm256_movemask_pd((__m256d) tmp2);
#endif
			}

			// just to make sure that the compiler will not compiler the
			// following code
			return 0;
		}

		if constexpr (EXACT) {
			const __m256i avx_exact_weight64 = _mm256_set1_epi64x(d);
#ifdef USE_AVX512
			return _mm256_cmp_epi64_mask(avx_exact_weight64, pop, 0);
#else
			const __m256i tmp2 = _mm256_cmpeq_epi64(avx_exact_weight64, pop);
			return _mm256_movemask_pd((__m256d) tmp2);
#endif
		} else {
			const __m256i avx_weight64 = _mm256_set1_epi64x(d+1);
#ifdef USE_AVX512
			return _mm256_cmp_epi64_mask(avx_weight64, pop, 5);
#else
			const __m256i tmp2 = _mm256_cmpgt_epi64(avx_weight64, pop);
			return _mm256_movemask_pd((__m256d) tmp2);
#endif
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: only compares a single 32 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 23...43) is impossible.
	/// NOTE: uses avx2
	/// NOTE: only a single 32bit element is compared.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_32(const size_t e1,
							const size_t e2) noexcept {
		ASSERT(n <= 32);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(d < 16);

		/// difference of the memory location in the right list
		const __m256i loadr = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

		for (size_t i = s1; i < e1; ++i) {
			// NOTE: implicit typecast because T = uint64
			const __m256i li = _mm256_set1_epi32(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			const uint32_t *ptr_r = (uint32_t *)L2;

			for (size_t j = s2; j < s2+(e2+7)/8; ++j, ptr_r += 16) {
				const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 8);
				const int m = compare_256_32(li, ri);

				if (m) {
					const size_t jprime = j*8 + __builtin_ctz(m);
					if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
						//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
						found_solution(i, jprime);
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only compares a single 64 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_64(const size_t e1,
	                        const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(n <= 64);
		ASSERT(n > 32);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// strive is the number of bytes to the next element
		constexpr uint32_t stride = 8;

		/// difference of the memory location in the right list
		const __m128i loadr = {(1ull << 32u),  (2ul) | (3ull << 32u)};

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li = _mm256_set1_epi64x(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 4) {
				const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr, stride);
				const int m = compare_256_64<true>(li, ri);

				if (m) {
					const size_t jprime = j*4+__builtin_ctz(m);

					if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
						//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
						found_solution(i, jprime);
					}
				}
			}
		}
	}

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_64_1x1(const size_t e1,
	                            const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n > 32);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);


		for (size_t i = s1; i < e1; ++i) {
			const __m256i li = _mm256_set1_epi64x(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 1) {
				const __m256i ri = LOAD256(ptr_r);
				const int m = compare_256_64<true>(li, ri);

				if (m) {
					const size_t jprime = j*4 + __builtin_ctz(m);

					if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
						found_solution(i, jprime);
					}
				}
			}
		}
	}

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// NOTE: compared to `bruteforce_avx2_64_1x1` this unrolls `u` elementes in the left
	///			list and `v` elements on the right.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_64_uxv(const size_t e1,
	                            const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		__m256i lii[u], rii[v];

		for (size_t i = s1; i < e1; i += u) {

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				lii[j] = _mm256_set1_epi64x(L1[i + j][0]);
			}

			/// NOTE: only possible because L2 is a continuous memory block
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; j += v, ptr_r += v) {

				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii[s] = LOAD256(ptr_r + s);
				}

				#pragma unroll
				for (uint32_t a1 = 0; a1 < u; ++a1) {
					const __m256i tmp1 = lii[a1];
					#pragma unroll
					for (uint32_t a2 = 0; a2 < v; ++a2) {
						const __m256i tmp2 = rii[a2];
						const int m = compare_256_64<true>(tmp1, tmp2);

						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m);
							const size_t iprime = i + a1;

							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								found_solution(iprime, jprime);
							}
						} // if
					} // for v
				} // for u
			} // for right list
		} // for left list
	}

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// NOTE: compared to `bruteforce_avx2_64_1x1` this unrolls `u` elements in the left
	///			list and `v` elements on the right.
	/// NOTE: compared to `bruteforce_avx2_64_uxv` this function is not only comparing 1
	///			element of the left list with u elements from the right. Side
	///			Internally the loop is unrolled to compare u*4 elements to v on the right
	/// NOTE: assumes the input list to of length multiple of 16
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_64_uxv_shuffle(const size_t e1,
	                                    const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		__m256i lii[u], rii[v];
		__m256i *ptr_l = (__m256i *)L1;

		for (size_t i = s1; i < s1 + (e1+3)/4; i += u, ptr_l += u) {

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				lii[j] = LOAD256(ptr_l + j);
			}

			/// NOTE: only possible because L2 is a continuous memory block
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; j += v, ptr_r += v) {

				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii[s] = LOAD256(ptr_r + s);
				}

				#pragma unroll
				for (uint32_t a1 = 0; a1 < u; ++a1) {
					const __m256i tmp1 = lii[a1];
					#pragma unroll
					for (uint32_t a2 = 0; a2 < v; ++a2) {
						__m256i tmp2 = rii[a2];
						int m = compare_256_64<true>(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m);
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						m = compare_256_64<true>(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 3;
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						m = compare_256_64<true>(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 2;
							const size_t iprime = i*4 + a1*4+ __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						m = compare_256_64<true>(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 1;
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_128(const size_t e1,
	                         const size_t e2) noexcept {

		ASSERT(n <= 128);
		ASSERT(n > 64);
		ASSERT(2 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (2ull << 32u), (4ul) | (6ull << 32u)};
		const __m128i loadr2 = {1ull | (3ull << 32u), (5ul) | (7ull << 32u)};

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li1 = _mm256_set1_epi64x(L1[i][0]);
			const __m256i li2 = _mm256_set1_epi64x(L1[i][1]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 8) {
				const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr1, 8);
				const int m1 = compare_256_64(li1, ri);

				if (m1) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr2, 8);
					const int m1 = compare_256_64(li2, ri);

					if (m1) {
						const size_t jprime = j*4;

						if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
							//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(i, jprime + __builtin_ctz(m1));
						} // if
					}
				}
			}
		}
	}

	/// TODO explain
	/// \tparam u
	/// \tparam v
	/// \param mask
	/// \param m1
	/// \param round
	/// \param i
	/// \param j
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_128_32_2_uxv_helper(uint32_t mask,
											 const uint8_t *__restrict__ m1,
											 const uint32_t round,
											 const size_t i,
											 const size_t j) noexcept {
		while (mask > 0) {
			const uint32_t ctz = __builtin_ctz(mask);

			const uint32_t test_i = ctz / v;
			const uint32_t test_j = ctz % u;

			const uint32_t inner_i2 = __builtin_ctz(m1[ctz]);
			const uint32_t inner_j2 = inner_i2;

			const int32_t off_l = test_i*8 + inner_i2;
			const int32_t off_r = test_j*8 + ((8+inner_j2-round)%8);


			const T *test_tl = ((T *)L1) + i*2 + off_l*2;
			const T *test_tr = ((T *)L2) + j*2 + off_r*2;
			if (compare_u64_ptr<0>(test_tl, test_tr)) {
				found_solution(i + off_l, j + off_r);
			}

			mask ^= 1u << ctz;
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: checks weight d on the first 2 limbs. Then direct checking.
	/// NOTE: unrolls the left loop by u and the right by v
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_128_32_2_uxv(const size_t e1,
									  const size_t e2) noexcept {
		static_assert(u <= 8);
		static_assert(v <= 8);
		ASSERT(n <= 128);
		ASSERT(n > 64);
		ASSERT(2 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// some constants
		const __m256i zero = _mm256_set1_epi32(0);
		const __m256i shuffl = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
		const __m256i loadr = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
		constexpr size_t ptr_ctr_l = u*8, ptr_ctr_r = v*8;
		constexpr size_t ptr_inner_ctr_l = 8*4, ptr_inner_ctr_r = 8*4;

		/// container for the unrolling
		__m256i lii_1[u], rii_1[v], lii_2[u], rii_2[v];

		/// container for the solutions masks
		constexpr uint32_t size_m1 = std::max(u*v, 32u);
		alignas(32) uint8_t m1[size_m1] = {0}; /// NOTE: init with 0 is important
		__m256i *m1_256 = (__m256i *)m1;

		uint32_t *ptr_l = (uint32_t *)L1;
		for (size_t i = s1; i < s1 + e1; i += ptr_ctr_l, ptr_l += ptr_ctr_l*4) {

			#pragma unroll
			for (uint32_t s = 0; s < u; ++s) {
				lii_1[s] = _mm256_i32gather_epi32((const int *)(ptr_l + s*ptr_inner_ctr_l + 0), loadr, 4);
				lii_2[s] = _mm256_i32gather_epi32((const int *)(ptr_l + s*ptr_inner_ctr_l + 1), loadr, 4);
			}

			uint32_t *ptr_r = (uint32_t *)L2;
			for (size_t j = s2; j < s2 + e2; j += ptr_ctr_r, ptr_r += ptr_ctr_r*4) {

				// load the fi
				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii_1[s] = _mm256_i32gather_epi32((const int *)(ptr_r + s*ptr_inner_ctr_r + 0), loadr, 4);
					rii_2[s] = _mm256_i32gather_epi32((const int *)(ptr_r + s*ptr_inner_ctr_r + 1), loadr, 4);
				}

				/// Do the 8x8 shuffle
				#pragma unroll
				for (uint32_t l = 0; l < 8; ++l) {
					if (l > 0) {
						// shuffle the right side
						#pragma unroll
						for (uint32_t s2s = 0; s2s < v; ++s2s) {
							rii_1[s2s] = _mm256_permutevar8x32_epi32(rii_1[s2s], shuffl);
							rii_2[s2s] = _mm256_permutevar8x32_epi32(rii_2[s2s], shuffl);
						}
					}

					// compare the first limb
					#pragma unroll
					for (uint32_t f1 = 0; f1 < u; ++f1) {
						#pragma  unroll
						for (uint32_t f2 = 0; f2 < v; ++f2) {
							m1[f1*u + f2] = compare_256_32(lii_1[f1], rii_1[f2]);
						}
					}

					// early exit
					uint32_t mask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)m1), zero));
					if (unlikely(mask == 0)) {
						continue;
					}

					//// second limb
					#pragma unroll
					for (uint32_t f1 = 0; f1 < u; ++f1) {
						#pragma unroll
						for (uint32_t f2 = 0; f2 < v; ++f2) {
							//if (m1[f1*u + f2]) {
								m1[f1*u + f2] &= compare_256_32(lii_2[f1], rii_2[f2]);
							//}
						}
					}


					// early exit from the second limb computations
					mask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)m1), zero));
					if (likely(mask == 0)) {
						continue;
					}

					// maybe write back a solution
					bruteforce_avx2_128_32_2_uxv_helper<u,v>(mask, m1, l, i, j);
				} // 8x8 shuffle
			} // j: enumerate right side
		} // i: enumerate left side
	} // end func

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: this check every element on the left against 4 on the right
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_256(const size_t e1,
	                         const size_t e2) noexcept {

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (4ull << 32u), ( 8ul) | (12ull << 32u)};
		const __m128i loadr2 = {1ull | (5ull << 32u), ( 9ul) | (13ull << 32u)};
		const __m128i loadr3 = {2ull | (6ull << 32u), (10ul) | (14ull << 32u)};
		const __m128i loadr4 = {3ull | (7ull << 32u), (11ul) | (15ull << 32u)};

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li1 = _mm256_set1_epi64x(L1[i][0]);
			const __m256i li2 = _mm256_set1_epi64x(L1[i][1]);
			const __m256i li3 = _mm256_set1_epi64x(L1[i][2]);
			const __m256i li4 = _mm256_set1_epi64x(L1[i][3]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 16) {
				const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr1, 8);
				const int m1 = compare_256_64(li1, ri);

				if (m1) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr2, 8);
					const int m1 = compare_256_64(li2, ri);

					if (m1) {
						const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr3, 8);
						const int m1 = compare_256_64(li3, ri);

						if (m1) {
							const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr4, 8);
							const int m1 = compare_256_64(li4, ri);
							if (m1) {
								const size_t jprime = j*4 + __builtin_ctz(m1);
								if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
									//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
									found_solution(i, jprime);
								}
							}
						}
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: additional to 'bruteforce_avx2_256' this functions unrolls `u` elements of
	///		the left list.
	/// NOTE: i think this approach is stupid. It checks each limb if wt < d.
	///			This is only good to find false ones, but not correct ones
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<uint32_t u>
	void bruteforce_avx2_256_ux4(const size_t e1,
	                             const size_t e2) noexcept {
		static_assert(u > 0, "");
		static_assert(u <= 8, "");

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (4ull << 32u), ( 8ul) | (12ull << 32u)};
		const __m128i loadr2 = {1ull | (5ull << 32u), ( 9ul) | (13ull << 32u)};
		const __m128i loadr3 = {2ull | (6ull << 32u), (10ul) | (14ull << 32u)};
		const __m128i loadr4 = {3ull | (7ull << 32u), (11ul) | (15ull << 32u)};

		alignas(32) __m256i li[u*4u];
		alignas(32) uint32_t m1s[8] = {0}; // this clearing is important

		/// allowed weight to match on
		uint32_t m1s_tmp = 0;
		const uint32_t m1s_mask = 1u << 31;

		for (size_t i = s1; i < s1 + e1; i += u) {
			/// Example u = 2
			/// li[0] = L[0][0]
			/// li[1] = L[1][0]
			/// li[2] = L[0][1]
			/// ...
			//#pragma unroll
			for (uint32_t ui = 0; ui < u; ui++) {
				li[ui + 0*u] = _mm256_set1_epi64x(L1[i + ui][0]);
				li[ui + 1*u] = _mm256_set1_epi64x(L1[i + ui][1]);
				li[ui + 2*u] = _mm256_set1_epi64x(L1[i + ui][2]);
				li[ui + 3*u] = _mm256_set1_epi64x(L1[i + ui][3]);
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 16) {
				//#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr1, 8);
					const uint32_t tmp  = compare_256_64(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr2, 8);
					const uint32_t tmp  = compare_256_64(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr3, 8);
					const uint32_t tmp  = compare_256_64(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr4, 8);
					const uint32_t tmp  = compare_256_64(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp) {
					ASSERT(__builtin_popcount(m1s_tmp) == 1);
					const uint32_t m1s_ctz = __builtin_ctz(m1s_tmp);
					const uint32_t bla = __builtin_ctz(m1s[m1s_ctz]);
					const size_t iprime = i + m1s_ctz;
					const size_t jprime = j*4 + bla;

					if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
						found_solution(iprime, jprime);
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 8.
	/// NOTE: additional to 'bruteforce_avx2_256' this functions unrolls `u` elements of
	///		the left list.
	///	NOTE: compared to `bruteforce_avx2_256_ux4` this function compares on 32 bit
	/// NOTE only made for extremely low weight.
	/// BUG: can only find one solution at the time.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<uint32_t u>
	void bruteforce_avx2_256_32_ux8(const size_t e1,
	                                const size_t e2) noexcept {
		static_assert(u > 0, "");
		static_assert(u <= 8, "");

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// NOTE: limit arbitrary but needed for correctness
		ASSERT(d < 7);

		/// difference of the memory location in the right list
		const __m256i loadr1 = _mm256_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56);
		const __m256i loadr_add = _mm256_set1_epi32(1);
		__m256i loadr;

		alignas(32) __m256i li[u*8u];
		alignas(32) uint32_t m1s[8] = {0}; // this clearing is important

		/// allowed weight to match on
		uint32_t m1s_tmp = 0;
		const uint32_t m1s_mask = 1u << 31;

		for (size_t i = s1; i < s1 + e1; i += u) {
			/// Example u = 2
			/// li[0] = L[0][0]
			/// li[1] = L[1][0]
			/// li[2] = L[0][1]
			/// ...
			#pragma unroll
			for (uint32_t ui = 0; ui < u; ui++) {
				#pragma unroll
				for (uint32_t uii = 0; uii < 8; uii++) {
					const uint32_t tmp = ((uint32_t *)L1[i + ui])[uii];
					li[ui + uii*u] = _mm256_set1_epi32(tmp);
				}
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			uint32_t *ptr_r = (uint32_t *)L2;

			for (size_t j = s2; j < s2+(e2+7)/8; ++j, ptr_r += 64) {
				loadr = loadr1;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[4 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[5 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[6 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[7 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp) {
					// TODO limitation.
					ASSERT(__builtin_popcount(m1s_tmp) == 1);
					const uint32_t m1s_ctz = __builtin_ctz(m1s_tmp);
					const uint32_t bla = __builtin_ctz(m1s[m1s_ctz]);
					const size_t iprime = i + m1s_ctz;
					const size_t jprime = j*8 + bla;
					//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
					if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
						found_solution(iprime, jprime);
					}
				}
			}
		}
	}

	/// TODO explain
	/// \tparam off
	/// \tparam rotation
	/// \param m1sx
	/// \param m1s
	/// \param ptr_l
	/// \param ptr_r
	/// \param i
	/// \param j
	template<const uint32_t off, const uint32_t rotation>
	void bruteforce_avx2_256_32_8x8_helper(uint32_t m1sx,
	                                       const uint8_t *m1s,
	                                       const uint32_t *ptr_l,
	                                       const uint32_t *ptr_r,
	                                       const size_t i, const size_t j) noexcept {
		static_assert(rotation < 8);
		static_assert(off%32 == 0);

		while (m1sx > 0) {
			const uint32_t ctz1 = __builtin_ctz(m1sx);
			const uint32_t ctz = off + ctz1;
			const uint32_t m1sc = m1s[ctz];
			const uint32_t m1sc_ctz = __builtin_ctz(m1sc);

			const uint32_t test_j = ctz % 8;
			const uint32_t test_i = ctz / 8;

			// NOTE: the signed is important
			const int32_t off_l = test_i * 8 + m1sc_ctz;
			const int32_t off_r = test_j * 8 + (8-rotation +m1sc_ctz)%8;

			const uint64_t *test_tl = (uint64_t *) (ptr_l + off_l*8);
			const uint64_t *test_tr = (uint64_t *) (ptr_r + off_r*8);
			if (compare_u64_ptr(test_tl, test_tr)) {
				found_solution(i + off_l, j + off_r);
			}

			m1sx ^= 1u << ctz1;
		}
	}

	/// NOTE: this is hyper optimized for the case if there is only one solution with extremely low weight.
	/// \param e1 size of the list L1
	/// \param e2 size of the list L2
	void bruteforce_avx2_256_32_8x8(const size_t e1,
	                                const size_t e2) noexcept {
		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		//ASSERT(e1 >= 64);
		//ASSERT(e2 >= 64);

		ASSERT(d < 16);

		uint32_t *ptr_l = (uint32_t *)L1;

		/// difference of the memory location in the right list
		const __m256i loadr1 = _mm256_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56);
		const __m256i shuffl = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);

		alignas(32) uint8_t m1s[64];

		/// helper to detect zeros
		const __m256i zero = _mm256_set1_epi32(0);

		for (size_t i = s1; i < s1 + e1; i += 64, ptr_l += 512) {
			const __m256i l1 = _mm256_i32gather_epi32((const int *)(ptr_l +   0), loadr1, 4);
			const __m256i l2 = _mm256_i32gather_epi32((const int *)(ptr_l +  64), loadr1, 4);
			const __m256i l3 = _mm256_i32gather_epi32((const int *)(ptr_l + 128), loadr1, 4);
			const __m256i l4 = _mm256_i32gather_epi32((const int *)(ptr_l + 192), loadr1, 4);
			const __m256i l5 = _mm256_i32gather_epi32((const int *)(ptr_l + 256), loadr1, 4);
			const __m256i l6 = _mm256_i32gather_epi32((const int *)(ptr_l + 320), loadr1, 4);
			const __m256i l7 = _mm256_i32gather_epi32((const int *)(ptr_l + 384), loadr1, 4);
			const __m256i l8 = _mm256_i32gather_epi32((const int *)(ptr_l + 448), loadr1, 4);

			uint32_t *ptr_r = (uint32_t *)L2;
			for (size_t j = s1; j < s2 + e2; j += 64, ptr_r += 512) {
				__m256i r1 = _mm256_i32gather_epi32((const int *)(ptr_r +   0), loadr1, 4);
				__m256i r2 = _mm256_i32gather_epi32((const int *)(ptr_r +  64), loadr1, 4);
				__m256i r3 = _mm256_i32gather_epi32((const int *)(ptr_r + 128), loadr1, 4);
				__m256i r4 = _mm256_i32gather_epi32((const int *)(ptr_r + 192), loadr1, 4);
				__m256i r5 = _mm256_i32gather_epi32((const int *)(ptr_r + 256), loadr1, 4);
				__m256i r6 = _mm256_i32gather_epi32((const int *)(ptr_r + 320), loadr1, 4);
				__m256i r7 = _mm256_i32gather_epi32((const int *)(ptr_r + 384), loadr1, 4);
				__m256i r8 = _mm256_i32gather_epi32((const int *)(ptr_r + 448), loadr1, 4);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				uint32_t m1s1 = _mm256_movemask_epi8(~(_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero)));
				uint32_t m1s2 = _mm256_movemask_epi8(~(_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero)));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 0>(m1s2, m1s, ptr_l, ptr_r, i, j); }


				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 1>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 1>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 2>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 2>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 3>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 3>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 4>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 4>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 5>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 5>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 6>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 6>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 7>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 7>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}
	
	/// TODO explain
	/// \tparam off
	/// \param m1sx
	/// \param m1s
	/// \param ptr_l
	/// \param ptr_r
	/// \param i
	/// \param j
	template<const uint32_t off>
	void bruteforce_avx2_256_64_4x4_helper(uint32_t m1sx,
	                                       const uint8_t *__restrict__ m1s,
	                                       const uint64_t *__restrict__ ptr_l,
	                                       const uint64_t *__restrict__ ptr_r,
	                                       const size_t i, const size_t j) noexcept {
		while (m1sx > 0) {
			const uint32_t ctz1 = __builtin_ctz(m1sx);
			const uint32_t ctz = off + ctz1;
			const uint32_t m1sc = m1s[ctz];
			const uint32_t inner_ctz = __builtin_ctz(m1sc);

			const uint32_t test_j = ctz % 4;
			const uint32_t test_inner = (inner_ctz + (ctz / 16)) % 4;
			const uint32_t test_i = (ctz % 16)/4;

			const uint32_t off_l = test_i * 4 + inner_ctz;
			const uint32_t off_r = test_j * 4 + test_inner;

			const T *test_tl = ptr_l + off_l*4;
			const T *test_tr = ptr_r + off_r*4;
			if (compare_u64_ptr(test_tl, test_tr)) {
				found_solution(i + off_l, j + off_r);
			}

			m1sx ^= 1u << ctz1;
		}
	}

	/// TODO explain
	/// \param stack
	/// \param a0
	/// \param a1
	/// \param a2
	/// \param a3
	/// \param b0
	/// \param b1
	/// \param b2
	/// \param b3
	void BRUTEFORCE256_64_4x4_STEP2(uint8_t* stack,
                                    __m256i a0,__m256i a1,__m256i a2,__m256i a3,
                                    __m256i b0,__m256i b1,__m256i b2,__m256i b3) noexcept {
		stack[0] = (uint8_t) compare_256_64(a0, b0);
		stack[1] = (uint8_t) compare_256_64(a0, b1);
		stack[2] = (uint8_t) compare_256_64(a0, b2);
		stack[3] = (uint8_t) compare_256_64(a0, b3);
		stack[4] = (uint8_t) compare_256_64(a1, b0);
		stack[5] = (uint8_t) compare_256_64(a1, b1);
		stack[6] = (uint8_t) compare_256_64(a1, b2);
		stack[7] = (uint8_t) compare_256_64(a1, b3);
		stack[8] = (uint8_t) compare_256_64(a2, b0);
		stack[9] = (uint8_t) compare_256_64(a2, b1);
		stack[10] = (uint8_t) compare_256_64(a2, b2);
		stack[11] = (uint8_t) compare_256_64(a2, b3);
		stack[12] = (uint8_t) compare_256_64(a3, b0);
		stack[13] = (uint8_t) compare_256_64(a3, b1);
		stack[14] = (uint8_t) compare_256_64(a3, b2);
		stack[15] = (uint8_t) compare_256_64(a3, b3);
		b0 = _mm256_permute4x64_epi64(b0, 0b00111001);
		b1 = _mm256_permute4x64_epi64(b1, 0b00111001);
		b2 = _mm256_permute4x64_epi64(b2, 0b00111001);
		b3 = _mm256_permute4x64_epi64(b3, 0b00111001);
		stack[16] = (uint8_t) compare_256_64(a0, b0);
		stack[17] = (uint8_t) compare_256_64(a0, b1);
		stack[18] = (uint8_t) compare_256_64(a0, b2);
		stack[19] = (uint8_t) compare_256_64(a0, b3);
		stack[20] = (uint8_t) compare_256_64(a1, b0);
		stack[21] = (uint8_t) compare_256_64(a1, b1);
		stack[22] = (uint8_t) compare_256_64(a1, b2);
		stack[23] = (uint8_t) compare_256_64(a1, b3);
		stack[24] = (uint8_t) compare_256_64(a2, b0);
		stack[25] = (uint8_t) compare_256_64(a2, b1);
		stack[26] = (uint8_t) compare_256_64(a2, b2);
		stack[27] = (uint8_t) compare_256_64(a2, b3);
		stack[28] = (uint8_t) compare_256_64(a3, b0);
		stack[29] = (uint8_t) compare_256_64(a3, b1);
		stack[30] = (uint8_t) compare_256_64(a3, b2);
		stack[31] = (uint8_t) compare_256_64(a3, b3);
		b0 = _mm256_permute4x64_epi64(b0, 0b00111001);
		b1 = _mm256_permute4x64_epi64(b1, 0b00111001);
		b2 = _mm256_permute4x64_epi64(b2, 0b00111001);
		b3 = _mm256_permute4x64_epi64(b3, 0b00111001);
		stack[32] = (uint8_t) compare_256_64(a0, b0);
		stack[33] = (uint8_t) compare_256_64(a0, b1);
		stack[34] = (uint8_t) compare_256_64(a0, b2);
		stack[35] = (uint8_t) compare_256_64(a0, b3);
		stack[36] = (uint8_t) compare_256_64(a1, b0);
		stack[37] = (uint8_t) compare_256_64(a1, b1);
		stack[38] = (uint8_t) compare_256_64(a1, b2);
		stack[39] = (uint8_t) compare_256_64(a1, b3);
		stack[40] = (uint8_t) compare_256_64(a2, b0);
		stack[41] = (uint8_t) compare_256_64(a2, b1);
		stack[42] = (uint8_t) compare_256_64(a2, b2);
		stack[43] = (uint8_t) compare_256_64(a2, b3);
		stack[44] = (uint8_t) compare_256_64(a3, b0);
		stack[45] = (uint8_t) compare_256_64(a3, b1);
		stack[46] = (uint8_t) compare_256_64(a3, b2);
		stack[47] = (uint8_t) compare_256_64(a3, b3);
		b0 = _mm256_permute4x64_epi64(b0, 0b00111001);
		b1 = _mm256_permute4x64_epi64(b1, 0b00111001);
		b2 = _mm256_permute4x64_epi64(b2, 0b00111001);
		b3 = _mm256_permute4x64_epi64(b3, 0b00111001);
		stack[48] = (uint8_t) compare_256_64(a0, b0);
		stack[49] = (uint8_t) compare_256_64(a0, b1);
		stack[50] = (uint8_t) compare_256_64(a0, b2);
		stack[51] = (uint8_t) compare_256_64(a0, b3);
		stack[52] = (uint8_t) compare_256_64(a1, b0);
		stack[53] = (uint8_t) compare_256_64(a1, b1);
		stack[54] = (uint8_t) compare_256_64(a1, b2);
		stack[55] = (uint8_t) compare_256_64(a1, b3);
		stack[56] = (uint8_t) compare_256_64(a2, b0);
		stack[57] = (uint8_t) compare_256_64(a2, b1);
		stack[58] = (uint8_t) compare_256_64(a2, b2);
		stack[59] = (uint8_t) compare_256_64(a2, b3);
		stack[60] = (uint8_t) compare_256_64(a3, b0);
		stack[61] = (uint8_t) compare_256_64(a3, b1);
		stack[62] = (uint8_t) compare_256_64(a3, b2);
		stack[63] = (uint8_t) compare_256_64(a3, b3);
	}

	/// NOTE: this is hyper optimized for the case if there is only one solution.
	/// NOTE: uses avx2
	/// NOTE: hardcoded unroll parameter of 4
	/// NOTE: only checks the first limb. If this passes the weight check all others are checked
	/// 		within `check_solution`
	/// \param e1 end index of list 1
	/// \param e2 end index of list 2
	void bruteforce_avx2_256_64_4x4(const size_t e1,
	                                const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;

		ASSERT(n <= 256);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(dk < 32);

		/// NOTE is already aligned
		T *ptr_l = (T *)L1;

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (4ull << 32u), ( 8ul) | (12ull << 32u)};
		alignas(32) uint8_t m1s[64];

		/// allowed weight to match on
		const __m256i zero = _mm256_set1_epi32(0);

		for (size_t i = s1; i < s1 + e1; i += 16, ptr_l += 64) {
			const __m256i l1 = _mm256_i32gather_epi64((const long long int *)(ptr_l +  0), loadr1, 8);
			const __m256i l2 = _mm256_i32gather_epi64((const long long int *)(ptr_l + 16), loadr1, 8);
			const __m256i l3 = _mm256_i32gather_epi64((const long long int *)(ptr_l + 32), loadr1, 8);
			const __m256i l4 = _mm256_i32gather_epi64((const long long int *)(ptr_l + 48), loadr1, 8);

			/// reset right list pointer
			T *ptr_r = (T *)L2;

			#pragma unroll 4
			for (size_t j = s1; j < s2 + e2; j += 16, ptr_r += 64) {
				__m256i r1 = _mm256_i32gather_epi64((const long long int *)(ptr_r +  0), loadr1, 8);
				__m256i r2 = _mm256_i32gather_epi64((const long long int *)(ptr_r + 16), loadr1, 8);
				__m256i r3 = _mm256_i32gather_epi64((const long long int *)(ptr_r + 32), loadr1, 8);
				__m256i r4 = _mm256_i32gather_epi64((const long long int *)(ptr_r + 48), loadr1, 8);

				BRUTEFORCE256_64_4x4_STEP2(m1s, l1, l2, l3, l4, r1, r2, r3, r4);
				uint32_t m1s1 = _mm256_movemask_epi8((_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero)));
				uint32_t m1s2 = _mm256_movemask_epi8((_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero)));

				if (m1s1 != 0) { bruteforce_avx2_256_64_4x4_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_64_4x4_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}



	template<const uint32_t off, const uint32_t bucket_size>
	void bruteforce_avx2_256_64_4x4_rearrange_helper(uint32_t m1sx,
										             const uint8_t *__restrict__ m1s,
										             const uint64_t *__restrict__ ptr_l,
										             const uint64_t *__restrict__ ptr_r,
										             const size_t i, const size_t j) noexcept {
		while (m1sx > 0) {
			const uint32_t ctz1 = __builtin_ctz(m1sx);
			const uint32_t ctz = off + ctz1;
			const uint32_t m1sc = m1s[ctz];
			const uint32_t inner_ctz = __builtin_ctz(m1sc);

			const uint32_t test_j = ctz % 4;
			const uint32_t test_inner = (inner_ctz + (ctz / 16)) % 4;
			const uint32_t test_i = (ctz % 16)/4;

			const uint32_t off_l = test_i * 4 + inner_ctz;
			const uint32_t off_r = test_j * 4 + test_inner;
			ASSERT(off_l < 16);
			ASSERT(off_r < 16);

			uint32_t wt = 0;
			for (uint32_t s = 0; s < ELEMENT_NR_LIMBS; s++) {
				const T t1 = ptr_l[off_l + s*bucket_size];
				const T t2 = ptr_r[off_r + s*bucket_size];
				wt += __builtin_popcountll(t1 ^ t2);
			}

			ASSERT(wt);
			if (wt <= d) {
				/// TODO tell the thing it needs to get the solutions from the buckets
				solutions.resize(solutions_nr + 1);
				solutions[solutions_nr++] = std::pair<size_t, size_t>{i + off_l, j + off_r};
				//found_solution(i + off_l, j + off_r);
			}

			m1sx ^= 1u << ctz1;
		}
	}
	/// NOTE: this is hyper optimized for the case if there is only one solution.
	///
	/// \param e1 end index of list 1
	/// \param e2 end index of list 2
	template<const uint32_t bucket_size>
	void bruteforce_avx2_256_64_4x4_rearrange(const size_t e1,
									          const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 <= bucket_size);
		ASSERT(e2 <= bucket_size);
		ASSERT(n <= 256);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(dk < 32);

		/// NOTE is already aligned
		T *ptr_l = (T *)LB;

		/// difference of the memory location in the right list
		alignas(32) uint8_t m1s[64];

		/// allowed weight to match on
		const __m256i zero = _mm256_set1_epi32(0);

		size_t i = s1;
        #pragma unroll 2
		for (; i < s1 + e1; i += 16, ptr_l += 16) {
			const __m256i l1 = _mm256_load_si256((const __m256i *)(ptr_l +  0));
			const __m256i l2 = _mm256_load_si256((const __m256i *)(ptr_l +  4));
			const __m256i l3 = _mm256_load_si256((const __m256i *)(ptr_l +  8));
			const __m256i l4 = _mm256_load_si256((const __m256i *)(ptr_l + 12));

			/// reset right list pointer
			T *ptr_r = (T *)RB;

			#pragma unroll 4
			for (size_t j = s1; j < s2 + e2; j += 16, ptr_r += 16) {
				__m256i r1 = _mm256_load_si256((const __m256i *)(ptr_r +  0));
				__m256i r2 = _mm256_load_si256((const __m256i *)(ptr_r +  4));
				__m256i r3 = _mm256_load_si256((const __m256i *)(ptr_r +  8));
				__m256i r4 = _mm256_load_si256((const __m256i *)(ptr_r + 12));

				BRUTEFORCE256_64_4x4_STEP2(m1s, l1, l2, l3, l4, r1, r2, r3, r4);
				uint32_t m1s1 = _mm256_movemask_epi8((_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero)));
				uint32_t m1s2 = _mm256_movemask_epi8((_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero)));

				if (m1s1 != 0) { bruteforce_avx2_256_64_4x4_rearrange_helper< 0, bucket_size>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_64_4x4_rearrange_helper<32, bucket_size>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: in comparison to `bruteforce_avx_256` this implementation used no `gather`
	///		instruction, but rather direct (aligned) loads.
	/// NOTE: only works if exact matching is active
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_256_v2(const size_t e1,
	                            const size_t e2) noexcept {
		ASSERT(EXACT);
		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);
		__m256i *ptr_l = (__m256i *)L1;

		for (size_t i = s1; i < e1; ++i, ptr_l += 1) {
			const __m256i li1 = LOAD256(ptr_l);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+e2; ++j, ptr_r += 1) {
				const __m256i ri = LOAD256(ptr_r);
				const __m256i tmp1 = _mm256_xor_si256(li1, ri);
				const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
				const int m1 = _mm256_movemask_pd((__m256d) tmp2);

				if (m1) {
					//std::cout << L1[i][0] << " " << L2[j][0] << " " << L2[j+1][0] << " " << L2[j-1][0] << "\n";
					found_solution(i, j);
				}
			}
		}
	}



	/// mother of all bruteforce algorithms. This is a selector function,
	/// which tries to select heuristically the best subroutine
	/// \param e1
	/// \param e2
	void bruteforce(const size_t e1,
	                const size_t e2) noexcept {
		if constexpr (32 < n and n <= 64) {
			bruteforce_avx2_64_uxv<4, 4>(e1, e2);
		} else if constexpr (64 < n and n <= 128){
			//bruteforce_128(e1, e2);
			bruteforce_avx2_128_32_2_uxv<4, 4>(e1, e2);
			//bruteforce_avx2_128(e1, e2);

			// actualy also work
			// bruteforce_avx2_256_32_8x8(e1, e2);
		} else if constexpr (128 < n and n <= 256) {
			// TODO optimal value
			if (e1 < 10 && e2 < 10) {
				bruteforce_256(e1, e2);
				//bruteforce_avx2_256(e1, e2);
				//bruteforce_avx2_256_64_4x4(e1, e2);
				return;
			}
			
			// in the low weight case with have better implementations
			if constexpr (d < 16) {
				// bruteforce_avx2_256_32_ux8<8>(e1, e2);
				//bruteforce_avx2_256_32_8x8(e1, e2);
				bruteforce_avx2_256_64_4x4(e1, e2);
				return;
			}

			// generic best implementations for every weight

			//bruteforce_256(e1, e2);
			bruteforce_avx2_256_64_4x4(e1, e2);
		} else {
			ASSERT(false);
		}
	}

	/// NOTE: uses avx2 instructions
	/// NOTE: currently not correct
	/// \param gt_mask comparison mask. From this mask we can deduct the TODO
	/// \param ctr
	/// \return
	size_t swap_avx_64(const __m256i gt_mask, const size_t ctr,
					   Element *__restrict__ ptr, Element *__restrict__ L) const noexcept {
		ASSERT(n <= 64);
		ASSERT(n > 32);

		// extract the int bit mask
		const int wt = _mm256_movemask_ps((__m256) gt_mask);

		// make sure only sane inputs make it.
		ASSERT(wt < (1u<<8u));

		/// see the description of this magic in `shuffle_down_64`
		const uint64_t expanded_mask_org = _pdep_u64(wt, 0b0101010101010101);
		const uint64_t expanded_mask = expanded_mask_org * 0b11;
		constexpr uint64_t identity_indices = 0b11100100;
		constexpr uint64_t identity_mask = (1u << 8u) - 1u;

		if (wt & 0b1111) {
			const uint64_t indices_down = _pext_u64(identity_indices, expanded_mask & identity_mask);
			const uint64_t indices_up   = _pdep_u64(identity_indices, expanded_mask & identity_mask);

			//const __m256i lower_down = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(uint16_t(wanted_indices_down)));
			//const __m256i tmp_down   = _mm256_i64gather_epi64(((uint64_t *) ptr), lower_down, 8);

			// store up
			const __m256i tmp_up1    = _mm256_lddqu_si256((__m256i *) (L + ctr));
			// TODO probably this needs to be done with a loop up table.
			//const __m256i tmp_up     = _mm256_permute4x64_epi64(tmp_up1, indices_up);

			// gt mask still wrong
			//_mm256_maskstore_epi64((long long *)ptr, gt_mask, tmp_up);
		}

		if (wt > 0b0001111) {
			const uint64_t wanted_indices_down2 = _pext_u64(identity_indices, expanded_mask >> 8u);
			const uint64_t wanted_indices_up2 = _pdep_u64(identity_indices, expanded_mask >> 8u);
		}

		return ctr + __builtin_popcount(wt);
	}

	/// copies `from` to `to` if the bit in `gt_mask` is set.
	/// \param wt int mask selecting the upper elements to shift down
	/// \param to
	/// \param from
	/// \return
	template<const uint32_t limit>
	inline size_t swap(int wt, Element *__restrict__ to, 
				Element *__restrict__ from) const noexcept {
		ASSERT(wt < (1u<<limit));

		uint32_t nctr = 0;

		#pragma unroll
		for (uint32_t i = 0; i < limit; ++i) {
			if (wt & 1u) {
				std::swap(to[nctr++], from[i]);
			}

			wt >>= 1u;
		}

		return nctr;
	}

	/// same as `swap` with the difference that this functions is swapping
	/// based on the count trailing bits functions.
	/// \param wt bit mask selecting the `from` elements
	/// \param to swp to
	/// \param from
	/// \return number of elements swapped
	inline size_t swap_ctz(uint32_t wt,
	                       Element *__restrict__ to,
	                       Element *__restrict__ from) const noexcept {
		uint32_t nctr = 0;
		const uint32_t bit_limit = __builtin_popcount(wt);
		for (uint32_t i = 0; i < bit_limit; ++i) {
			const uint32_t pos = __builtin_ctz(wt);
			std::swap(to[i], from[pos]);

			// clear the set bit.
			wt ^= 1u << pos;
		}

		return bit_limit;
	}

	/// TODO explain
	/// \tparam bucket_size
	/// \param wt
	/// \param to
	/// \param from
	/// \return
	template<const uint32_t bucket_size>
	inline size_t swap_ctz_rearrange(uint32_t wt,
						   			T *__restrict__ to,
						   			Element *__restrict__ from) const noexcept {
		if constexpr (!USE_REARRANGE) {
			ASSERT(false);
			return 0;
		}

		const uint32_t bit_limit = __builtin_popcount(wt);
		for (uint32_t i = 0; i < bit_limit; ++i) {
			const uint32_t pos = __builtin_ctz(wt);

			#pragma unroll
			for(uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
				ASSERT(i + j*bucket_size < (ELEMENT_NR_LIMBS * bucket_size));
				to[i + j*bucket_size] = from[pos][j];
			}

			// clear the set bit.
			wt ^= 1u << pos;
		}

		return bit_limit;
	}

	/// executes the comparison operator in the NN subroutine on 32 bit limbs
	/// \param tmp input
	/// \return a mask // TODO
	inline uint32_t compare_nn_on32(const __m256i tmp) const noexcept {
		static_assert(NN_EQUAL + NN_LOWER + NN_BOUNDS == 1);
		if constexpr (NN_EQUAL) {
			static_assert(epsilon == 0);
			const __m256i avx_nn_weight32 = _mm256_set1_epi32 (dk+NN_LOWER);
			const __m256i gt_mask = _mm256_cmpeq_epi32(avx_nn_weight32, tmp);
			return _mm256_movemask_ps((__m256) gt_mask);
		}

		if constexpr (NN_LOWER) {
			static_assert(epsilon == 0);
			const __m256i avx_nn_weight32 = _mm256_set1_epi32 (dk+NN_LOWER);
			const __m256i gt_mask = _mm256_cmpgt_epi32(avx_nn_weight32, tmp);
			return _mm256_movemask_ps((__m256) gt_mask);
		}

		if constexpr (NN_BOUNDS) {
			static_assert(epsilon > 0);
			static_assert(epsilon < dk);

			const __m256i avx_nn_weight32 = _mm256_set1_epi32 (dk+NN_LOWER+epsilon);
			const __m256i avx_nn_weight_lower32 = _mm256_set1_epi32(dk-epsilon);

			const __m256i lt_mask = _mm256_cmpgt_epi32(avx_nn_weight_lower32, tmp);
			const __m256i gt_mask = _mm256_cmpgt_epi32(avx_nn_weight32, tmp);
			return _mm256_movemask_ps((__m256) _mm256_and_si256(lt_mask, gt_mask));
		}
	}

	/// executes the comparison operator in the NN subroutine
	/// \param tmp
	/// \return
	inline uint32_t compare_nn_on64(const __m256i tmp) const noexcept {
		static_assert(NN_EQUAL + NN_LOWER + NN_BOUNDS == 1);
		if constexpr (NN_EQUAL) {
			static_assert(epsilon == 0);
			const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);
			const __m256i gt_mask = _mm256_cmpeq_epi64(avx_nn_weight64, tmp);
			return _mm256_movemask_pd((__m256d) gt_mask);
		}

		if constexpr (NN_LOWER) {
			static_assert(epsilon == 0);
			const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);
			const __m256i gt_mask = _mm256_cmpgt_epi64(avx_nn_weight64, tmp);
			return _mm256_movemask_pd((__m256d) gt_mask);
		}

		if constexpr (NN_BOUNDS) {
			static_assert(epsilon > 0);
			static_assert(epsilon < dk);
			const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);
			const __m256i avx_nn_weight_lower64 = _mm256_set1_epi64x(dk-epsilon);

			const __m256i lt_mask = _mm256_cmpgt_epi64(avx_nn_weight_lower64, tmp);
			const __m256i gt_mask = _mm256_cmpgt_epi64(avx_nn_weight64, tmp);
			return _mm256_movemask_pd((__m256d) _mm256_and_si256(lt_mask, gt_mask));
		}
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on32_simple(const size_t e1,
									const uint32_t z,
							        Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		const size_t s1 = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi32(z);
		alignas(32) const __m256i offset = _mm256_setr_epi32(0*enl, 1*enl, 2*enl, 3*enl,
		                                                     4*enl, 5*enl, 6*enl, 7*enl);

		// NR of partial solutions found
		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*4);
		Element *org_ptr = L;
	
		for (size_t i = s1; i < (e1+7)/8; i++, ptr += 8, org_ptr += 8) {
			const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 32) {
				tmp = _mm256_and_si256(tmp, avx_nn_k_mask);
			}
			const __m256i tmp_pop = popcount_avx2_32(tmp);
			const uint32_t wt = compare_nn_on32(tmp_pop);
			ASSERT(wt < (1ul << 8));

			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<8>(wt, L + ctr, org_ptr);
			}
		}
	
		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32_t
	/// NOTE: fixxed unrolling factor by 4
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on32(const size_t e1,
	                         const uint32_t z,
	                         Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 32);


		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi32(z);
		alignas(32) const __m256i offset = _mm256_setr_epi32(0*enl, 1*enl, 2*enl, 3*enl,
		                                                     4*enl, 5*enl, 6*enl, 7*enl);

		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*4);
		Element *org_ptr = L;

		constexpr uint32_t u = 4;
		constexpr uint32_t off = u*8;
		for (; i+off <= e1; i += off, ptr += off, org_ptr += off) {
			__m256i ptr_tmp0 = _mm256_i32gather_epi32(ptr +  0, offset, 8);
			ptr_tmp0 = _mm256_xor_si256(ptr_tmp0, z256);
			if constexpr (k < 32) { ptr_tmp0 = _mm256_and_si256(ptr_tmp0, avx_nn_k_mask); }
			__m256i ptr_tmp1 = _mm256_i32gather_epi32(ptr +  8, offset, 8);
			ptr_tmp1 = _mm256_xor_si256(ptr_tmp1, z256);
			if constexpr (k < 32) { ptr_tmp1 = _mm256_and_si256(ptr_tmp1, avx_nn_k_mask); }
			__m256i ptr_tmp2 = _mm256_i32gather_epi32(ptr + 16, offset, 8);
			ptr_tmp2 = _mm256_xor_si256(ptr_tmp2, z256);
			if constexpr (k < 32) { ptr_tmp2 = _mm256_and_si256(ptr_tmp2, avx_nn_k_mask); }
			__m256i ptr_tmp3 = _mm256_i32gather_epi32(ptr + 24, offset, 8);
			ptr_tmp3 = _mm256_xor_si256(ptr_tmp3, z256);
			if constexpr (k < 32) { ptr_tmp3 = _mm256_and_si256(ptr_tmp3, avx_nn_k_mask); }

			uint32_t wt = 0;
			__m256i tmp_pop = popcount_avx2_32(ptr_tmp0);
			wt = compare_nn_on32(tmp_pop);

			tmp_pop = popcount_avx2_32(ptr_tmp1);
			wt ^= compare_nn_on32(tmp_pop) << 8u;

			tmp_pop = popcount_avx2_32(ptr_tmp2);
			wt ^= compare_nn_on32(tmp_pop) << 16u;

			tmp_pop = popcount_avx2_32(ptr_tmp3);
			wt ^= compare_nn_on32(tmp_pop) << 24u;

			if (wt) {
				ctr += swap_ctz(wt, L + ctr, org_ptr);
			}
		}

		// tail work
		// #pragma unroll 4
		for (; i+8 < e1+7; i+=8, ptr += 8, org_ptr += 8) {
			const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 32) {
				tmp = _mm256_and_si256(tmp, avx_nn_k_mask);
			}
			const __m256i tmp_pop = popcount_avx2_32(tmp);
			const int wt = compare_nn_on32(tmp_pop);

			ASSERT(wt < 1u << 8u);
			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<8>(wt, L + ctr, org_ptr);
			}
		}

		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32_t
	/// NOTE: fixxed unrolling factor by 4
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb, const uint32_t bucket_size>
	size_t avx2_sort_nn_on32_rearrange(const size_t e1,
							 		   const uint32_t z,
							 		   Element *__restrict__ L,
	                                   T *__restrict__ B) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi32(z);
		alignas(32) const __m256i offset = _mm256_setr_epi32(0*enl, 1*enl, 2*enl, 3*enl,
															 4*enl, 5*enl, 6*enl, 7*enl);

		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*4);
		Element *org_ptr = L;

		constexpr uint32_t u = 4;
		constexpr uint32_t off = u*8;
		for (; i+off <= e1; i += off, ptr += off, org_ptr += off) {
			__m256i ptr_tmp0 = _mm256_i32gather_epi32(ptr +  0, offset, 8);
			ptr_tmp0 = _mm256_xor_si256(ptr_tmp0, z256);
			if constexpr (k < 32) { ptr_tmp0 = _mm256_and_si256(ptr_tmp0, avx_nn_k_mask); }
			__m256i ptr_tmp1 = _mm256_i32gather_epi32(ptr +  8, offset, 8);
			ptr_tmp1 = _mm256_xor_si256(ptr_tmp1, z256);
			if constexpr (k < 32) { ptr_tmp1 = _mm256_and_si256(ptr_tmp1, avx_nn_k_mask); }
			__m256i ptr_tmp2 = _mm256_i32gather_epi32(ptr + 16, offset, 8);
			ptr_tmp2 = _mm256_xor_si256(ptr_tmp2, z256);
			if constexpr (k < 32) { ptr_tmp2 = _mm256_and_si256(ptr_tmp2, avx_nn_k_mask); }
			__m256i ptr_tmp3 = _mm256_i32gather_epi32(ptr + 24, offset, 8);
			ptr_tmp3 = _mm256_xor_si256(ptr_tmp3, z256);
			if constexpr (k < 32) { ptr_tmp3 = _mm256_and_si256(ptr_tmp3, avx_nn_k_mask); }

			uint32_t wt = 0;
			__m256i tmp_pop = popcount_avx2_32(ptr_tmp0);
			wt = compare_nn_on32(tmp_pop);

			tmp_pop = popcount_avx2_32(ptr_tmp1);
			wt ^= compare_nn_on32(tmp_pop) << 8u;

			tmp_pop = popcount_avx2_32(ptr_tmp2);
			wt ^= compare_nn_on32(tmp_pop) << 16u;

			tmp_pop = popcount_avx2_32(ptr_tmp3);
			wt ^= compare_nn_on32(tmp_pop) << 24u;

			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		// tail work
		// #pragma unroll 4
		for (; i+8 < e1+7; i+=8, ptr += 8, org_ptr += 8) {
			const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 32) {
				tmp = _mm256_and_si256(tmp, avx_nn_k_mask);
			}
			const __m256i tmp_pop = popcount_avx2_32(tmp);
			const int wt = compare_nn_on32(tmp_pop);

			ASSERT(wt < 1u << 8u);
			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		return ctr;
	}
	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on64_simple(const size_t e1,
							 	   	const uint64_t z,
							 	   	Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);


		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi64x(z);
		alignas(32) const __m256i offset = _mm256_setr_epi64x(0*enl, 1*enl, 2*enl, 3*enl);

		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*8);
		Element *org_ptr = L;

		// #pragma unroll 4
		for (; i < (e1+3)/4; i++, ptr += 4, org_ptr += 4) {
			const __m256i ptr_tmp = _mm256_i64gather_epi64((const long long int *)ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 64) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
			const __m256i tmp_pop = popcount_avx2_64(tmp);
			const uint32_t wt = compare_nn_on64(tmp_pop);
			ASSERT(wt < 1u << 4u);

			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<4>(wt, L + ctr, org_ptr);
			}
		}


		return ctr;
	}


	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// NOTE: hardcoded unroll parameter u=8
	/// NOTE: this version rearranges the elements in the list L
	///       in the following manner:
	///				TODO
	/// \param: e1 end index
	/// \param: random value z
	/// \param: L: input list
	/// \param: B: output bucket (rearranged)
	template<const uint32_t limb, const uint32_t bucket_size>
	size_t avx2_sort_nn_on64_rearrange(const size_t e1,
							 		   const uint64_t z,
	                                   Element *__restrict__ L,
	                                   T *B) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		__m256i z256 = _mm256_set1_epi64x(z);
		__m256i offset = _mm256_setr_epi64x(0*enl, 1*enl, 2*enl, 3*enl);
		const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);

		size_t ctr = 0;

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*8);
		Element *org_ptr = L;

		constexpr uint32_t u = 8;
		constexpr uint32_t off = u*4;
		for (; i+off <= e1; i += off, ptr += off, org_ptr += off) {
			__m256i ptr_tmp0 = _mm256_i64gather_epi64(ptr +  0, offset, 8);
			ptr_tmp0 = _mm256_xor_si256(ptr_tmp0, z256);
			if constexpr (k < 64) { ptr_tmp0 = _mm256_and_si256(ptr_tmp0, avx_nn_k_mask); }
			__m256i ptr_tmp1 = _mm256_i64gather_epi64(ptr +  4, offset, 8);
			ptr_tmp1 = _mm256_xor_si256(ptr_tmp1, z256);
			if constexpr (k < 64) { ptr_tmp1 = _mm256_and_si256(ptr_tmp1, avx_nn_k_mask); }
			__m256i ptr_tmp2 = _mm256_i64gather_epi64(ptr +  8, offset, 8);
			ptr_tmp2 = _mm256_xor_si256(ptr_tmp2, z256);
			if constexpr (k < 64) { ptr_tmp2 = _mm256_and_si256(ptr_tmp2, avx_nn_k_mask); }
			__m256i ptr_tmp3 = _mm256_i64gather_epi64(ptr + 12, offset, 8);
			ptr_tmp3 = _mm256_xor_si256(ptr_tmp3, z256);
			if constexpr (k < 64) { ptr_tmp3 = _mm256_and_si256(ptr_tmp3, avx_nn_k_mask); }
			__m256i ptr_tmp4 = _mm256_i64gather_epi64(ptr + 16, offset, 8);
			ptr_tmp4 = _mm256_xor_si256(ptr_tmp4, z256);
			if constexpr (k < 64) { ptr_tmp4 = _mm256_and_si256(ptr_tmp4, avx_nn_k_mask); }
			__m256i ptr_tmp5 = _mm256_i64gather_epi64(ptr + 20, offset, 8);
			ptr_tmp5 = _mm256_xor_si256(ptr_tmp5, z256);
			if constexpr (k < 64) { ptr_tmp5 = _mm256_and_si256(ptr_tmp5, avx_nn_k_mask); }
			__m256i ptr_tmp6 = _mm256_i64gather_epi64(ptr + 24, offset, 8);
			ptr_tmp6 = _mm256_xor_si256(ptr_tmp6, z256);
			if constexpr (k < 64) { ptr_tmp6 = _mm256_and_si256(ptr_tmp6, avx_nn_k_mask); }
			__m256i ptr_tmp7 = _mm256_i64gather_epi64(ptr + 28, offset, 8);
			ptr_tmp7 = _mm256_xor_si256(ptr_tmp7, z256);
			if constexpr (k < 64) { ptr_tmp7 = _mm256_and_si256(ptr_tmp7, avx_nn_k_mask); }

			__m256i tmp_pop = popcount_avx2_64(ptr_tmp0);
			uint32_t wt = compare_nn_on64(tmp_pop);

			tmp_pop = popcount_avx2_64(ptr_tmp1);
			wt ^= compare_nn_on64(tmp_pop) << 4u;

			tmp_pop = popcount_avx2_64(ptr_tmp2);
			wt ^= compare_nn_on64(tmp_pop) << 8u;

			tmp_pop = popcount_avx2_64(ptr_tmp3);
			wt ^= compare_nn_on64(tmp_pop) << 12u;

			tmp_pop = popcount_avx2_64(ptr_tmp4);
			wt ^= compare_nn_on64(tmp_pop) << 16u;

			tmp_pop = popcount_avx2_64(ptr_tmp5);
			wt ^= compare_nn_on64(tmp_pop) << 20u;

			tmp_pop = popcount_avx2_64(ptr_tmp6);
			wt ^= compare_nn_on64(tmp_pop) << 24u;

			tmp_pop = popcount_avx2_64(ptr_tmp7);
			wt ^= compare_nn_on64(tmp_pop) << 28u;
			//ASSERT(uint64_t(wt) < (1ull << 32ull));

			ASSERT(ctr <= LIST_SIZE);
			ASSERT(ctr <= e1);

			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		// #pragma unroll 4
		for (; i < (e1+3)/4; i++, ptr += 4, org_ptr += 4) {
			const __m256i ptr_tmp = _mm256_i64gather_epi64(ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 64) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
			const __m256i tmp_pop = popcount_avx2_64(tmp);

			//const __m256i gt_mask = _mm256_cmpgt_epi64(mask, tmp_pop);
			const __m256i gt_mask = _mm256_cmpeq_epi64(avx_nn_weight64, tmp_pop);
			const int wt = _mm256_movemask_pd((__m256d) gt_mask);
			ASSERT(wt < 1u << 4u);
			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// NOTE: hardcoded unroll parameter u=8
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on64(const size_t e1,
					    	 const uint64_t z,
							 Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi64x(z);
		alignas(32) const __m256i offset = _mm256_setr_epi64x(0*enl, 1*enl, 2*enl, 3*enl);
		const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);

		size_t ctr = 0;

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*8);
		Element *org_ptr = L;

		constexpr uint32_t u = 8;
		constexpr uint32_t off = u*4;
		for (; i+off <= e1; i += off, ptr += off, org_ptr += off) {
			__m256i ptr_tmp0 = _mm256_i64gather_epi64((const uint64_t *)(ptr +  0), offset, 8);
			ptr_tmp0 = _mm256_xor_si256(ptr_tmp0, z256);
			if constexpr (k < 64) { ptr_tmp0 = _mm256_and_si256(ptr_tmp0, avx_nn_k_mask); }
			__m256i ptr_tmp1 = _mm256_i64gather_epi64((const uint64_t *)(ptr +  4), offset, 8);
			ptr_tmp1 = _mm256_xor_si256(ptr_tmp1, z256);
			if constexpr (k < 64) { ptr_tmp1 = _mm256_and_si256(ptr_tmp1, avx_nn_k_mask); }
			__m256i ptr_tmp2 = _mm256_i64gather_epi64((const uint64_t *)(ptr +  8), offset, 8);
			ptr_tmp2 = _mm256_xor_si256(ptr_tmp2, z256);
			if constexpr (k < 64) { ptr_tmp2 = _mm256_and_si256(ptr_tmp2, avx_nn_k_mask); }
			__m256i ptr_tmp3 = _mm256_i64gather_epi64((const uint64_t *)(ptr + 12), offset, 8);
			ptr_tmp3 = _mm256_xor_si256(ptr_tmp3, z256);
			if constexpr (k < 64) { ptr_tmp3 = _mm256_and_si256(ptr_tmp3, avx_nn_k_mask); }
			__m256i ptr_tmp4 = _mm256_i64gather_epi64((const uint64_t *)(ptr + 16), offset, 8);
			ptr_tmp4 = _mm256_xor_si256(ptr_tmp4, z256);
			if constexpr (k < 64) { ptr_tmp4 = _mm256_and_si256(ptr_tmp4, avx_nn_k_mask); }
			__m256i ptr_tmp5 = _mm256_i64gather_epi64((const uint64_t *)(ptr + 20), offset, 8);
			ptr_tmp5 = _mm256_xor_si256(ptr_tmp5, z256);
			if constexpr (k < 64) { ptr_tmp5 = _mm256_and_si256(ptr_tmp5, avx_nn_k_mask); }
			__m256i ptr_tmp6 = _mm256_i64gather_epi64((const uint64_t *)(ptr + 24), offset, 8);
			ptr_tmp6 = _mm256_xor_si256(ptr_tmp6, z256);
			if constexpr (k < 64) { ptr_tmp6 = _mm256_and_si256(ptr_tmp6, avx_nn_k_mask); }
			__m256i ptr_tmp7 = _mm256_i64gather_epi64(((const uint64_t *)ptr + 28), offset, 8);
			ptr_tmp7 = _mm256_xor_si256(ptr_tmp7, z256);
			if constexpr (k < 64) { ptr_tmp7 = _mm256_and_si256(ptr_tmp7, avx_nn_k_mask); }

			__m256i tmp_pop = popcount_avx2_64(ptr_tmp0);
			uint32_t wt = compare_nn_on64(tmp_pop);

			tmp_pop = popcount_avx2_64(ptr_tmp1);
			wt ^= compare_nn_on64(tmp_pop) << 4u;

			tmp_pop = popcount_avx2_64(ptr_tmp2);
			wt ^= compare_nn_on64(tmp_pop) << 8u;

			tmp_pop = popcount_avx2_64(ptr_tmp3);
			wt ^= compare_nn_on64(tmp_pop) << 12u;

			tmp_pop = popcount_avx2_64(ptr_tmp4);
			wt ^= compare_nn_on64(tmp_pop) << 16u;

			tmp_pop = popcount_avx2_64(ptr_tmp5);
			wt ^= compare_nn_on64(tmp_pop) << 20u;

			tmp_pop = popcount_avx2_64(ptr_tmp6);
			wt ^= compare_nn_on64(tmp_pop) << 24u;

			tmp_pop = popcount_avx2_64(ptr_tmp7);
			wt ^= compare_nn_on64(tmp_pop) << 28u;
			//ASSERT(uint64_t(wt) < (1ull << 32ull));

			ASSERT(ctr <= LIST_SIZE);
			ASSERT(ctr <= e1);

			if (wt) {
				ctr += swap_ctz(wt, L + ctr, org_ptr);
			}
		}

		// #pragma unroll 4
		for (; i < (e1+3)/4; i++, ptr += 4, org_ptr += 4) {
			const __m256i ptr_tmp = _mm256_i64gather_epi64((const uint64_t *)ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 64) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
			const __m256i tmp_pop = popcount_avx2_64(tmp);
	
			//const __m256i gt_mask = _mm256_cmpgt_epi64(mask, tmp_pop);
			const __m256i gt_mask = _mm256_cmpeq_epi64(avx_nn_weight64, tmp_pop);
			const int wt = _mm256_movemask_pd((__m256d) gt_mask);
			ASSERT(wt < 1u << 4u);
			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<4>(wt, L + ctr, org_ptr);
			}
		}
	
	
		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32_t
	/// NOTE: dont call this function at first.
	/// NOTE: the current implementation will overflow the given e1, e2 in multiples of u*4
	/// NOTE: make sure that `new_e1` and `new_e2` are zero
	/// \tparam limb current limb
	/// \tparam u number of checks to unroll
	/// \param e1 end of List L1
	/// \param e2 end of List L2
	/// \param new_e1 new end of List L1
	/// \param new_e2 new end of List L2
	/// \param z random element to match on
	template<const uint32_t limb, const uint32_t u>
	void avx2_sort_nn_on_double32(const size_t e1,
								  const size_t e2,
								  size_t &new_e1,
								  size_t &new_e2,
								  const uint32_t z) noexcept {
		static_assert(u <= 8);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(e2 <= LIST_SIZE);
		ASSERT(k <= 32);
		ASSERT(new_e1 == 0);
		ASSERT(new_e2 == 0);
		ASSERT(dk <= 16);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi32(z);
		alignas(32) const __m256i offset = _mm256_setr_epi32(0*enl, 1*enl, 2*enl, 3*enl,
		                                                     4*enl, 5*enl, 6*enl, 7*enl);

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr_L1 = (Element *)(((uint8_t *)L1) + limb*4);
		Element *org_ptr_L1 = L1;
		Element *ptr_L2 = (Element *)(((uint8_t *)L2) + limb*4);
		Element *org_ptr_L2 = L2;

		constexpr uint32_t off = 8*u;
		const size_t min_e = (std::min(e1, e2) + off - 1);
		for (; i+off <= min_e; i += off, ptr_L1 += off, org_ptr_L1 += off,
		                                 ptr_L2 += off, org_ptr_L2 += off) {
			uint64_t wt_L1 = 0, wt_L2 = 0;

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				/// load the left list
				__m256i ptr_tmp_L1 = _mm256_i32gather_epi32(ptr_L1 +  8*j, offset, 8);
				ptr_tmp_L1 = _mm256_xor_si256(ptr_tmp_L1, z256);
				if constexpr (k < 32) { ptr_tmp_L1 = _mm256_and_si256(ptr_tmp_L1, avx_nn_k_mask); }
				ptr_tmp_L1 = popcount_avx2_32(ptr_tmp_L1);
				wt_L1 ^= compare_nn_on32(ptr_tmp_L1) << (8u * j);

				/// load the right list
				__m256i ptr_tmp_L2 = _mm256_i32gather_epi32(ptr_L2 +  8*j, offset, 8);
				ptr_tmp_L2 = _mm256_xor_si256(ptr_tmp_L2, z256);
				if constexpr (k < 32) { ptr_tmp_L2 = _mm256_and_si256(ptr_tmp_L2, avx_nn_k_mask); }
				ptr_tmp_L2 = popcount_avx2_32(ptr_tmp_L2);
				wt_L2 ^= compare_nn_on32(ptr_tmp_L2) << (8u * j);
			}

			if (wt_L1) {
				new_e1 += swap_ctz(wt_L1, L1 + new_e1, org_ptr_L1);
			}

			if (wt_L2) {
				new_e2 += swap_ctz(wt_L2, L2 + new_e2, org_ptr_L2);
			}

			ASSERT(new_e1 <= LIST_SIZE);
			ASSERT(new_e2 <= LIST_SIZE);
		}

		// tail work
		// #pragma unroll 4
		//size_t i2 = i;
		//for (; i < e1; i += 8, ptr_L1 += 8, org_ptr_L1 += 8) {
		//	const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr_L1, offset, 8);
		//	__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
		//	if constexpr (k < 32) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
		//	const __m256i tmp_pop = popcount_avx2_32(tmp);
		//	const int wt = compare_nn_on32(tmp_pop);
		//	if (wt) {
		//		new_e1 += swap<8>(wt, L1 + new_e1, org_ptr_L1);
		//	}
		//}
		//for (; i2 < e2; i2 += 8, ptr_L2 += 8, org_ptr_L2 += 8) {
		//	const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr_L2, offset, 8);
		//	__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
		//	if constexpr (k < 32) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
		//	const __m256i tmp_pop = popcount_avx2_32(tmp);
		//	const int wt = compare_nn_on32(tmp_pop);
		//	if (wt) {
		//		new_e2 += swap<8>(wt, L2 + new_e2, org_ptr_L1);
		//	}
		//}
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// NOTE: dont call this function at first.
	/// NOTE: the current implementation will overflow the given e1, e2 in multiples of u*4
	/// NOTE: make sure that `new_e1` and `new_e2` are zero
	/// \tparam limb current limb
	/// \tparam u number of checks to unroll
	/// \param e1 end of List L1
	/// \param e2 end of List L2
	/// \param new_e1 new end of List L1
	/// \param new_e2 new end of List L2
	/// \param z random element to match on
	template<const uint32_t limb, const uint32_t u>
	void avx2_sort_nn_on_double64(const size_t e1,
							      const size_t e2,
								  size_t &new_e1,
	                              size_t &new_e2,
								  const uint64_t z) noexcept {
		static_assert(u <= 16);
		static_assert(u > 0);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);
		ASSERT(new_e1 == 0);
		ASSERT(new_e2 == 0);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi64x(z);
		alignas(32) const __m256i offset = _mm256_setr_epi64x(0*enl, 1*enl, 2*enl, 3*enl);

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr_L1 = (Element *)(((uint8_t *)L1) + limb*8);
		Element *org_ptr_L1 = L1;
		Element *ptr_L2 = (Element *)(((uint8_t *)L2) + limb*8);
		Element *org_ptr_L2 = L2;

		const size_t min_e = (std::min(e1, e2) + 4*u - 1);
		for (; i+(4*u) <= min_e; i += (4*u), ptr_L1 += (4*u), org_ptr_L1 += (4*u),
											 ptr_L2 += (4*u), org_ptr_L2 += (4*u)) {
			uint32_t wt_L1 = 0, wt_L2 = 0;

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				__m256i ptr_tmp_L1 = _mm256_i64gather_epi64(ptr_L1 +  4*j, offset, 8);
				ptr_tmp_L1 = _mm256_xor_si256(ptr_tmp_L1, z256);
				if constexpr (k < 64) { ptr_tmp_L1 = _mm256_and_si256(ptr_tmp_L1, avx_nn_k_mask); }
				ptr_tmp_L1 = popcount_avx2_64(ptr_tmp_L1);
				wt_L1 ^= compare_nn_on64(ptr_tmp_L1) << (4u * j);

				__m256i ptr_tmp_L2 = _mm256_i64gather_epi64(ptr_L2 +  4*j, offset, 8);
				ptr_tmp_L2 = _mm256_xor_si256(ptr_tmp_L2, z256);
				if constexpr (k < 64) { ptr_tmp_L2 = _mm256_and_si256(ptr_tmp_L2, avx_nn_k_mask); }
				ptr_tmp_L2 = popcount_avx2_64(ptr_tmp_L2);
				wt_L2 ^= compare_nn_on64(ptr_tmp_L2) << (4u * j);
			}

			if (wt_L1) {
				new_e1 += swap_ctz(wt_L1, L1 + new_e1, org_ptr_L1);
			}

			if (wt_L2) {
				new_e2 += swap_ctz(wt_L2, L2 + new_e2, org_ptr_L2);
			}
		}

	}


	/// runs the Esser, Kübler, Zweydinger NN on a the two lists
	/// dont call ths function normally.
	/// \tparam level current level of the
	/// \param e1 end of list L1
	/// \param e2 end of list L2
	template<const uint32_t level>
	void avx2_nn_internal(const size_t e1,
						  const size_t e2) noexcept {
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(e2 <= LIST_SIZE);

		/// NOTE: is this really the only wat to get around the restriction
		/// of partly specialized template functions?
		if constexpr (level < 1) {
			bruteforce(e1, e2);
		} else {
			size_t new_e1 = 0,
					new_e2 = 0;

			if constexpr (k <= 32) {
				const uint32_t z =  fastrandombytes_uint64();
				// avx2_sort_nn_on_double32<r-level, 4>(e1, e2, new_e1, new_e2, z);
				new_e1 = avx2_sort_nn_on32<r - level>(e1, z, L1);
				new_e2 = avx2_sort_nn_on32<r - level>(e2, z, L2);
			} else if constexpr (k <= 64) {
				const uint64_t z = fastrandombytes_uint64();
				//avx2_sort_nn_on_double64<r-level, 4>(e1, e2, new_e1, new_e2, z);
				new_e1 = avx2_sort_nn_on64<r - level>(e1, z, L1);
				new_e2 = avx2_sort_nn_on64<r - level>(e2, z, L2);
			} else  {
				ASSERT(false);
			}

			ASSERT(new_e1 <= LIST_SIZE);
			ASSERT(new_e2 <= LIST_SIZE);
			ASSERT(new_e1 <= e1);
			ASSERT(new_e2 <= e2);



			if (unlikely(new_e1 == 0 or new_e2 == 0)) {
				return;
			}

			if ((new_e1 < BRUTEFORCE_THRESHHOLD) || (new_e2 < BRUTEFORCE_THRESHHOLD)) {
#ifdef DEBUG
				std::cout << level << " " << new_e1 << " " << e1 << " " << new_e2 << " " << e2 << "\n";
#endif
				bruteforce(new_e1, new_e2);
				return;
			}

			for (uint32_t i = 0; i < N; i++) {
#ifdef DEBUG
				if (i == 0) {
					std::cout << level << " " << i << " " << new_e1 << " " << e1 << " " << new_e2 << " " << e2 << "\n";
				}
#endif

				/// predict the future:
				if constexpr (USE_REARRANGE) {
					const uint32_t next_size = uint32_t(survive_prob * double(new_e1));
					//printf("%d\n", next_size);
					if (next_size < BUCKET_SIZE) {
						size_t new_new_e1, new_new_e2;
						if constexpr (k <= 32) {
							const uint32_t z =  fastrandombytes_uint64();
							new_new_e1 = avx2_sort_nn_on32_rearrange<r - level + 1, BUCKET_SIZE>(new_e1, z, L1, LB);
							new_new_e2 = avx2_sort_nn_on32_rearrange<r - level + 1, BUCKET_SIZE>(new_e2, z, L2, RB);
						} else if constexpr (k <= 64) {
							const uint64_t z = fastrandombytes_uint64();
							new_new_e1 = avx2_sort_nn_on64_rearrange<r - level + 1, BUCKET_SIZE>(new_e1, z, L1, LB);
							new_new_e2 = avx2_sort_nn_on64_rearrange<r - level + 1, BUCKET_SIZE>(new_e2, z, L2, RB);
						} else {
							ASSERT(false);
						}

						/// Now bruteforce the (rearranges) buckets
						bruteforce_avx2_256_64_4x4_rearrange<BUCKET_SIZE>(new_new_e1, new_new_e2);
					}
				} else {
					/// normal code path
					avx2_nn_internal<level - 1>(new_e1, new_e2);
				}

				if (unlikely(solutions_nr)){
#if DEBUG
					std::cout << "sol: " << level << " " << i << " " << new_e1 << " " << new_e2 << "\n";
#endif
					break;
				}
			}
		}
	}

#ifdef __AVX512F__
	alignas(64) const __m512i avx512_weight32 = _mm512_set1_epi32(d+1);
	alignas(64) const __m512i avx512_weight64 = _mm512_set1_epi64(d+1);
	alignas(64) const __m512i avx512_exact_weight32 = _mm512_set1_epi32(d);
	alignas(64) const __m512i avx512_exact_weight64 = _mm512_set1_epi64(d);

	///
	/// \param in
	/// \return
	__m512i	popcount_avx512_32(const __m512i in) {
		return _mm512_popcnt_epi32(in);
	}

	///
	/// \param in
	/// \return
	__m512i	popcount_avx512_64(const __m512i in) {
		return _mm512_popcnt_epi64(in);
	}

	///
	/// NOTE: upper bound `d` is inclusive
	/// \param in
	/// \return
	template<bool exact=false>
	int compare_512_32(const __m512i in1, const __m512i in2) {
		if constexpr(EXACT) {
			return _mm512_cmpeq_epi32_mask(in1, in2);
		} else {
			const __m512i tmp1 = _mm512_xor_si512(in1, in2);
			const __m512i pop = popcount_avx512_32(tmp1);

			if constexpr (exact) {
				return _mm512_cmpeq_epi32_mask(avx512_exact_weight32, pop);
			} else  {
				return _mm512_cmpgt_epi32_mask(avx512_weight32, pop);
			}
		}
	}

	///
	/// NOTE: upper bound `d` is inclusive
	/// \param in
	/// \return
	template<bool exact=false>
	int compare_512_64(const __m512i in1, const __m512i in2) {
		if constexpr(EXACT) {
			return _mm512_cmpeq_epi64_mask(in1, in2);
		} else {
			const __m512i tmp1 = _mm512_xor_si512(in1, in2);
			const __m512i pop = popcount_avx512_32(tmp1);

			if constexpr (exact) {
				return _mm512_cmpeq_epi64_mask(avx512_exact_weight64, pop);
			} else  {
				return _mm512_cmpgt_epi64_mask(avx512_weight64, pop);
			}
		}
	}

	///
	/// \param e1
	/// \param e2
	void bruteforce_avx512_32_16x16(const size_t e1,
	                               const size_t e2) noexcept {
		ASSERT(n <= 32);
		ASSERT(ELEMENT_NR_LIMBS == 1);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		ASSERT(e1 >= 64);
		ASSERT(e2 >= 64);

		ASSERT(d < 16);

		__m512i *ptr_l = (__m512i *)L1;
		const __m512i perm = _mm512_setr_epi32(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);

		for (size_t i = 0; i < (e1+15)/16; i++) {
			const __m512i li = _mm512_load_si512(ptr_l);
			__m512i *ptr_r = (__m512i *)L2;

			for (size_t j = 0; j < (e2+15)/16; j++) {
				__m512i ri = _mm512_load_si512(ptr_r);
				const int m = compare_512_32(li, ri);

				if (m) {
					const size_t jprime = j*16;
					const size_t iprime = i*16;

					if (compare_u64_ptr((T *) (L1 + iprime), (T *)(L2 + jprime))) {
						// TODO
						printf("k\n");
					}
				}

				// shuffle 
				ri = _mm512_permutexvar_epi32(perm, ri);
			}
 		}
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32_t
	/// NOTE: dont call this function at first.
	/// NOTE: the current implementation will overflow the given e1, e2 in multiples of u*4
	/// NOTE: make sure that `new_e1` and `new_e2` are zero
	/// \tparam limb current limb
	/// \tparam u number of checks to unroll
	/// \param e1 end of List L1
	/// \param e2 end of List L2
	/// \param new_e1 new end of List L1
	/// \param new_e2 new end of List L2
	/// \param z random element to match on
	template<const uint32_t limb, const uint32_t u>
	void avx512_sort_nn_on_double32(const size_t e1,
								    const size_t e2,
								    size_t &new_e1,
								    size_t &new_e2,
								    const uint64_t z) noexcept {
		static_assert(u <= 8);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(e2 <= LIST_SIZE);
		ASSERT(k <= 32);
		ASSERT(new_e1 == 0);
		ASSERT(new_e2 == 0);
		ASSERT(dk <= 16);


	}
#endif
