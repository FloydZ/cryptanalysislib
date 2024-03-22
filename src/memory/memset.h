#ifndef CRYPTANALYSISLIB_MEMORY_MEMSET_H
#define CRYPTANALYSISLIB_MEMORY_MEMSET_H

#ifndef CRYPTANALYSISLIB_MEMORY_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/memory.h>`"
#endif

#include <cstddef>
#include <cstdint>

namespace cryptanalysislib {
	namespace internal {

		void memset256(uint8_t *out, uint8_t in, size_t bytes) {
			if (bytes <= 16) {
#ifdef __clang__
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

			// TODO there are better ways todo this
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
			return;
		}


#ifdef USE_AVX512BW
		constexpr void memsetU512BW(uint8_t *out,
									const uint8_t in,
									const size_t bytes) noexcept {
			uint8x64_t t = uint8x64_t::set1(in);
			if (bytes >= 128) { 
				uint8x64_t::unaligned_store(out, t);
				
				uint8_t *out2 = out + bytes;
				uint8x64_t::unaligned_store(out2 - 0x40, t);
				out2 = (uint8_t *)(((uintptr_t)(out2)) & -0x40);

				out += 0x40;
				out = (uint8_t *)(((uintptr_t)out) & -0x40);
				out = (uint8_t *)((uintptr_t)out - (uintptr_t)out2);

				while (out != 0) {
					uint8x64_t::aligned_store((uint8_t *)((uintptr_t)out + (uintptr_t)out2), t);
					out	+= 0x40;
				}

				return;
			}

			memset256(out, in, bytes);
		}
#endif

		///
		/// \tparam T
		/// \param out
		/// \param in
		/// \param len
		/// \param pos
		/// \return
		template<typename T>
		constexpr void memset(T *out,
		                      const T in,
		                      const size_t bytes) {
			/// TODO case T != uint8
#ifdef USE_AVX512BW
			memsetU512BW(out, in, bytes);
			return;
#endif
			memset256((uint8_t *)out, (uint8_t)in, bytes);
		}
	} // end internal

	///
	/// \tparam T
	/// \param out
	/// \param in
	/// \param len
	/// \return
	template<typename T>
	constexpr void memset(T *out, const T in, size_t len) {
		cryptanalysislib::internal::memset(out, in, sizeof(T) * len);
	}
}
#endif
