#ifndef CRYPTANALYSISLIB_MEMORY_MEMSET_H
#define CRYPTANALYSISLIB_MEMORY_MEMSET_H

#ifndef CRYPTANALYSISLIB_MEMORY_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/memory/memory.h>`"
#endif

#include <cstddef>
#include <cstdint>
#include "simd/simd.h"

namespace cryptanalysislib {
	namespace internal {

		template<typename T>
#if __cplusplus > 201709L
			requires std::is_integral_v<T>
#endif
		void memset_u256_u8(T *out,
		                    const T in,
		                    const size_t nr_elements) {

			if constexpr (sizeof(T) == 1) {
				const size_t bytes = nr_elements;
				if (bytes <= 16) {
#ifdef __clang__
					// ach clang, why you no support for programmable jump tables
					for (size_t i = 0; i < bytes; ++i) {
						out[i] = in;
					}
					return;
#else
					uint32_t in2 = in * 0x1010101;// broadcast in into all bytes
					void *MemsetJTab[] = {
					        &&M00, &&M01, &&M02, &&M03, &&M04, &&M05, &&M06, &&M07,
					        &&M08, &&M09, &&M10, &&M11, &&M12, &&M13, &&M14, &&M15, &&M16};

					goto *MemsetJTab[bytes];

				M16:
					(*(uint32_t *) (out + 12)) = in2;
				M12:
					(*(uint32_t *) (out + 8)) = in2;
				M08:
					(*(uint32_t *) (out + 4)) = in2;
				M04:
					(*(uint32_t *) (out + 0)) = in2;
				M00:
					return;

				M15:
					(*(uint32_t *) (out + 11)) = in2;
				M11:
					(*(uint32_t *) (out + 7)) = in2;
				M07:
					(*(uint32_t *) (out + 3)) = in2;
				M03:
					(*(uint16_t *) (out + 1)) = (uint16_t) in2;
				M01:
					(*(uint8_t *) (out + 0)) = (uint8_t) in2;
					return;

				M14:
					(*(uint32_t *) (out + 10)) = in2;
				M10:
					(*(uint32_t *) (out + 6)) = in2;
				M06:
					(*(uint32_t *) (out + 2)) = in2;
				M02:
					(*(uint16_t *) (out + 0)) = (uint8_t) in2;
					return;

				M13:
					(*(uint32_t *) (out + 9)) = in2;
				M09:
					(*(uint32_t *) (out + 5)) = in2;
				M05:
					(*(uint32_t *) (out + 1)) = in2;
					(*(uint8_t *)  (out + 0)) = (uint8_t) in2;
					return;
#endif
				}

				const _uint8x16_t in3 = _uint8x16_t::set1(in);
				uint8_t *end = out + bytes;
				if (bytes > 32) {
					_uint8x16_t::unaligned_store(out, in3);
					out += 16;
					out = (uint8_t *)((uintptr_t)(out) & -16);
					_uint8x16_t::unaligned_store(out, in3);
					out += 16;
					out = (uint8_t *)((uintptr_t)(out) & -32);

					const uint8x32_t in4 = uint8x32_t::set1(in);
					// not really correct
					for (size_t i = 0; i < (bytes-31)/32; ++i) {
						uint8x32_t::aligned_store(out, in4);
						out += 32;
					}

					uint8x32_t::unaligned_store(end - 32, in4);
					return;
				}

				// case 16 < bytes <= 32;
				_uint8x16_t::unaligned_store(out, in3);
				_uint8x16_t::unaligned_store(end - 16, in3);
			} else {
				// no idea how to implement the goto
				if (nr_elements < 16) {
					// all other types
					for (size_t i = 0; i < nr_elements; ++i) {
						out[i] = in;
					}
					return;
				}

				// NOTE: this is chosen for avx2
				constexpr size_t alignment = 32;
				constexpr size_t N = alignment / sizeof(T);
				using S = TxN_t<T, N>;

				const size_t bytes = sizeof(T) * nr_elements;

				S in1 = S::set1(in);
				T *end = out + nr_elements;
				if (bytes > alignment) {
					S::unaligned_store(out, in1);
					out += N;

					out = (T *)((uintptr_t)(out) & -alignment);
					const size_t limit = (bytes - alignment + 1u)/alignment;
					for (size_t i = 0; i < limit; ++i) {
						S::aligned_store(out, in1);
						out += N;
					}
					S::unaligned_store(end - N, in1);
					return;
				}

				// case 16 < bytes <= 32;

				using S_half = TxN_t<T, N/2>;
				S_half in_half = S_half ::set1(in);
				S_half::unaligned_store(out, in_half);
				S_half::unaligned_store(end - N, in_half);
			}
		}


#ifdef USE_AVX512BW

		template<typename T>
#if __cplusplus > 201709L
		requires std::is_integral_v<T>
#endif
		void memsetU512BW(T *out,
						  const T in,
						  const size_t nr_elements) noexcept {

			const size_t bytes = nr_elements * sizeof(T);
			if (bytes >= 128) {
				if constexpr (sizeof(T) == 1) {
					uint8x64_t t = uint8x64_t::set1(in);
					uint8x64_t::unaligned_store(out, t);

					uint8_t *out2 = out + bytes;
					uint8x64_t::unaligned_store(out2 - 0x40, t);
					out2 = (uint8_t *) (((uintptr_t) (out2)) & -0x40);

					out += 0x40;
					out = (uint8_t *) (((uintptr_t) out) & -0x40);
					out = (uint8_t *) ((uintptr_t) out - (uintptr_t) out2);

					while (out != nullptr) {
						uint8x64_t::aligned_store((uint8_t *) ((uintptr_t) out + (uintptr_t) out2), t);
						out += 0x40;
					}

					return;
				} else {
					constexpr size_t alignment = 64;
					constexpr size_t N = alignment / sizeof(T);
					using S = TxN_t<T, N>;

					S t = S::set1(in);
					S::unaligned_store(out, t);

					T *out2 = out + nr_elements;
					S::unaligned_store(out2 - N, t);
					out2 = (T *)(((uintptr_t)(out2)) & -0x40);

					out += N;
					out = (T *)(((uintptr_t)out) & -0x40);
					out = (T *)((uintptr_t)out - (uintptr_t)out2);

					while (out != nullptr) {
						S::aligned_store((T *)((uintptr_t)out + (uintptr_t)out2), t);
						out	+= N;
					}

					return;

				}
			}

			memset_u256_u8(out, in, nr_elements);
		}
#endif

		/// \param out output array
		/// \param in input symbol
		/// \param nr_elements number of arrays NOT bytes
		template<typename T>
		inline void memset_bytes(T *out,
		                  		 const T in,
		                  		 const size_t nr_elements) noexcept {
#ifdef USE_AVX512BW
			memsetU512BW(out, in, nr_elements);
			return;
#endif
			memset_u256_u8(out, in, nr_elements);
		}
	} // end internal

	/// \tparam T type
	/// \param out output array
	/// \param in input symbol
	/// \param len number of elements NOT byts
	template<typename T>
	constexpr void memset(T *out,
	                      const T in,
	                      const size_t len) noexcept {
		if (std::is_constant_evaluated()) {
			for (size_t j = 0; j < len; ++j) {
				out[j] = in;
			}

			return;
		}

		// fast version
		if constexpr ((sizeof(T) <= 8) && std::is_integral_v<T>) {
			cryptanalysislib::internal::memset_bytes(out, in, len);
			return;
		}

		// backup, for anything else
		for (size_t j = 0; j < len; ++j) {
			out[j] = in;
		}
	}
}
#endif
