#ifndef CRYPTANALYSISLIB_MEMORY_MEMMOVE_H
#define CRYPTANALYSISLIB_MEMORY_MEMMOVE_H

#ifndef CRYPTANALYSISLIB_MEMORY_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/memory/memory.h>`"
#endif

#include <cstddef>
#include <cstdint>

#include "simd/simd.h"

namespace cryptanalysislib {
	namespace internal {
		/// moves at most 64 bytes. Every byte is loaded before the first
		/// store, so the buffers may overlap in any direction.
		/// \param out[out]: destination
		/// \param in[in]: source
		/// \param bytes[in]: number of bytes, <= 64
		constexpr inline void memmove_small_u8(uint8_t *out,
		                                       const uint8_t *in,
		                                       const size_t bytes) noexcept {
			if (bytes >= 32) {
				// [0, 32) and [bytes-32, bytes) cover everything
				const uint8x32_t a = uint8x32_t::unaligned_load(in);
				const uint8x32_t b = uint8x32_t::unaligned_load(in + bytes - 32);
				uint8x32_t::unaligned_store(out, a);
				uint8x32_t::unaligned_store(out + bytes - 32, b);
			} else if (bytes >= 16) {
				const _uint8x16_t a = _uint8x16_t::unaligned_load(in);
				const _uint8x16_t b = _uint8x16_t::unaligned_load(in + bytes - 16);
				_uint8x16_t::unaligned_store(out, a);
				_uint8x16_t::unaligned_store(out + bytes - 16, b);
			} else if (bytes >= 8) {
				uint64_t a, b;
				__builtin_memcpy(&a, in, 8);
				__builtin_memcpy(&b, in + bytes - 8, 8);
				__builtin_memcpy(out, &a, 8);
				__builtin_memcpy(out + bytes - 8, &b, 8);
			} else if (bytes >= 4) {
				uint32_t a, b;
				__builtin_memcpy(&a, in, 4);
				__builtin_memcpy(&b, in + bytes - 4, 4);
				__builtin_memcpy(out, &a, 4);
				__builtin_memcpy(out + bytes - 4, &b, 4);
			} else if (bytes >= 2) {
				uint16_t a, b;
				__builtin_memcpy(&a, in, 2);
				__builtin_memcpy(&b, in + bytes - 2, 2);
				__builtin_memcpy(out, &a, 2);
				__builtin_memcpy(out + bytes - 2, &b, 2);
			} else if (bytes == 1) {
				*out = *in;
			}
		}

		/// \param out[out]: destination
		/// \param in[in]: source, may overlap with `out`
		/// \param bytes[in]: number of bytes to move
		constexpr inline void memmove_u8(uint8_t *out,
		                                 const uint8_t *in,
		                                 size_t bytes) noexcept {
			if ((out == in) || (bytes == 0)) {
				return;
			}

			if (bytes <= 64) {
				memmove_small_u8(out, in, bytes);
				return;
			}

			// NOTE: the first and the last 64 bytes are loaded up front and
			// stored at the very end, so the main loop can use 64-byte aligned
			// stores. In each iteration all 64 bytes are loaded before they are
			// stored. Copying front to back is correct if `out` is below `in`
			// (or the buffers do not overlap), as the stores never reach source
			// bytes which are not loaded yet. Otherwise back to front.
			using S = uint8x32_t;
			const S h0 = S::unaligned_load(in);
			const S h1 = S::unaligned_load(in + 32);
			const S t0 = S::unaligned_load(in + bytes - 64);
			const S t1 = S::unaligned_load(in + bytes - 32);

			if (((uintptr_t)out < (uintptr_t)in) ||
			    ((uintptr_t)out >= (uintptr_t)in + bytes)) {
				// first position with a 64-byte aligned destination, in [1, 64]
				size_t i = 64u - ((uintptr_t)out & 63u);
				while (bytes - i > 64) {
					const S a = S::unaligned_load(in + i);
					const S b = S::unaligned_load(in + i + 32);
					S::aligned_store(out + i, a);
					S::aligned_store(out + i + 32, b);
					i += 64;
				}
			} else {
				// end of the last 64-byte aligned destination block
				size_t i = bytes - ((((uintptr_t)out + bytes) & 63u) ? (((uintptr_t)out + bytes) & 63u) : 64u);
				while (i > 64) {
					i -= 64;
					const S a = S::unaligned_load(in + i);
					const S b = S::unaligned_load(in + i + 32);
					S::aligned_store(out + i, a);
					S::aligned_store(out + i + 32, b);
				}
			}

			S::unaligned_store(out + bytes - 64, t0);
			S::unaligned_store(out + bytes - 32, t1);
			S::unaligned_store(out, h0);
			S::unaligned_store(out + 32, h1);
		}
	} // end namespace internal

	/// same as the C `memmove`: the buffers may overlap
	/// VERY IMPORTANT NOTE: as for `cryptanalysislib::memcpy` the last
	///		argument is not the number of bytes but the number of elements
	/// \tparam T element type
	/// \param out[out]: destination
	/// \param in[in]: source
	/// \param len[in]: number of elements
	template<typename T>
	constexpr inline void memmove(T *out,
	                              const T *in,
	                              const size_t len) noexcept {
		if consteval {
			// NOTE: the pointer comparison is only valid within the same
			// 	array, which is always the case for overlapping buffers
			if (out < in) {
				for (size_t j = 0; j < len; j++) { out[j] = in[j]; }
			} else {
				for (size_t j = len; j > 0; j--) { out[j - 1] = in[j - 1]; }
			}
		} else {
			internal::memmove_u8((uint8_t *)out, (const uint8_t *)in, len * sizeof(T));
		}
	}
} // end namespace cryptanalysislib
#endif//CRYPTANALYSISLIB_MEMORY_MEMMOVE_H
