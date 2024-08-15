#ifndef CRYPTANALYSISLIB_MEMORY_MEMCPY_H
#define CRYPTANALYSISLIB_MEMORY_MEMCPY_H

#ifndef CRYPTANALYSISLIB_MEMORY_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/memory/memory.h>`"
#endif

#include <cstddef>
#include "simd/simd.h"
#include "common.h"

namespace cryptanalysislib {
	namespace internal {

		/// alters the pointers
		/// code original from agner fogs asmlib
		/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIMwAcpK4AMngMmAByPgBGmMQSZlykAA6oCoRODB7evv5BaRmOAmER0SxxCVxJtpj2xQxCBEzEBDk%2BfoG19VlNLQSlUbHxickKza3teV3j/YPllaMAlLaoXsTI7BzmAMzhyN5YANQmO27j%2BIIAdAin2CYaAIK7%2B4eYJ2cXtHgxN3cPzzMewYBy8x1ObjwLBY4QIxHCfx22BOTwBzDYChSTE2R1h8WYtBOAHYrKiNABOAD01KOCDqKXiRyoXhBDSuRwAKgg8Aojox%2BBtMLyCHSjmgWCk6IyiHyYQQjkwTgA2DQANxYqFV6AAjl4TCrcQxxsQvA4sgCHlSaSYAKxuLHEVhHFJw52oPHEI4y9BMZpXC0U6mUy0himWyKoBXfYAMNiCI4RTBYdABq2UlHk0PkmJeeU7MxHGIATwIQqOCgQa1o6EL72ImCYPpi9FTQZOdvrBHWDCOLLEeBjSeVGgA%2BiOWGYbUq8PqNKm0EbS6oUp7Zo5kIbvhEjmOJ1O8COvL2CQOIugR7RUI2RwB3R0pBnECC7yfTw9igTjI4AKhdxCWxKks8YYUp23Y/n%2BpxASGRIACIBsBaa0vSjLMqyWTslyPJ8gwAqbMKoripK9CejKmBygqx79oONbhMapoNK21p2g6Tp/m6HpepGPrNEx6a2vaLROjxioiVx5ZEPWfHtm4YHED2DCoCK4TAPOn5LiuEm%2BngG7hFu7yqu6NZ9tGZ4juMJCYLe96Ps%2B46vgeR6/nCpA7vZ%2B7viJAEmCSqaWs5nqnLBRwiVBflwQhamLpgy6rs066buE7y7tUASOVRplJheV7nneTAPvEI6pXZLCpelC5fgF3m%2BWSmagZgXbyRBcJhbVPnwWSiEVRpcXabpDD6Uchl4MZJ40eZklWXlBXEEVgQlWV74Ba5KWBOlXmAX5FIBR8wWhTs0FhhFqJwTS6JCliOIegSCH0Q4O5eLCSqSKoZgjvKPlAUc30ftFsVaQlj2CPm71HCEACSACyFhCLtRxmK1jw/b2GRGEc3wVO9RYMnDQMEM972IwCyMslkm1PMjJOwgEoOqgEtoWFwSq2vBB3E5T3140ztNMwz9M2qzX0c72sIg/Kqr5gzkgs4jwsi4IBPi89DMIwLsvC6taVDal6vfe1RMUz9Qbs99bYCQQrEsFGp5JibRxmyxQlW5Bhum8xskNd2dulhKBilhC3WFqgnjWzRcNUGISj/K77ZWDaFiKfgCiiMQKZqwLv3jDFmlrjpiXboeT0vW98qXo2ECB8NNZVQmqAxZsLrk0jwt4FQmc9UcEDUWe1VC3L31yT23dZWX6AQH%2BSy65T%2BudXLg8ZTb56j%2BPcKT2zMcz8Bzdu8GMcO4JjrOy1e/u4PdsMwzic8inafwRnuf9YNhcK8XoPD0vOUV%2BpQ1Gc1/61/XTAjdPp22RkwXMkYAqpThgtNaP4AKQXXtvSmz98av3lGsAgU9kaYKuKqaBQU/46yQXPT2TVMFT03ghZGxsT78UdofZ0x9kH7zPhveOl8MDXxaLfFmAM856SSg9Iur1QYmUXtlculdf410UoA4BNVkHI01uVb%2BUDAgwJUe%2BQO34EEtRIcLQOWijwEAlHDa6YgQAgHEeNUe1l8qPjmgEFexBUprz7igvGz1REYNzNgn6uD8EaMIaYlI/iB5kJ7BQgxetjpbxodSO2%2B8LZO1DmeJJ7tLZMOIBk%2BhB8nThG9uRFIftMAB2/jEYOhJ37h0jmUpEdsH75wMr/Cy9YIBVz/q5IxXj0GGgAnI1QDcPqKLlq3du2dPRd0yugXuoDhbvwmpZFxrlwjuPmRzM%2BMTp5xI2QvcabTMArP6ZQ3ZdDckySyS7FhmTUmFJjk0wR25OmLMOR0mRf5unf1Qd4ku/SAFDKASMjxYCIFEOCTsYKsCta6OyeE8FARzEMDwcQjxVDzl0Muak65CS8lZPuUox5A0hGdJsWZN5nTlrt2ES/Hx/zBnDKbnLcBMp1GIsIdC9KsLEEgp%2BhY2gViyVZUOfYmaTiXGpVWcioJAR1kb12RFJBAJOlsBYMgFIRY5CvggHjGm8pvyYNIFtckctdWg2/OEI1bUKRy0DhkAAXlZeUxZSwKAAkyvlbcIAurLBCI4z05kxwSR%2BFkH0zj%2BskOySGmp3j5gALRMzjQEONkg41mATYWEsQo9k0jQBqpkxBUBWxFO8VwezRZ/LQKGuGcaIAVveksH1Ch4WYLhq26wmbXXwvCEik4lhO3Zu2cjcZX9q3/EhUcON%2BZA1KI1r0nxgqxrksmlASlSwIDtv7VW%2BMHajgaBWDStBC7rFLpHp/d5I14F1p7B27dH1%2B17qWE%2B%2BFyj50l0Xa8ldF7q7rs3RYENO7%2B1MwPT89BH6ZmSLHt%2Bq9Pbb1rEA/%2B4Dz6h2GPg/eqwE7JYodiR1Wd30R13o%2BPcCdCalQzv7j9UDx6hXnjedB3RG7cx9v/XekDb73rgYkcvej67YNbrQ0%2BuVeHKaEesIQ3m2HiS4bGV60TDSSOyo9RR78Oqi7moAox9DAGCDeQnSp3p6nO58ZYwJl9P1ROWEIfTSTm8ZOdzk8R4KKbyMUZ/Kp4GfzYWaeY9p3TwV9P1v1Rp4zvmzPfQsxh4K0sbNxLs6Ond8mnNmBc8p9zBBuZBc7n%2B3zcMAuCAy/AozN7%2BOhqE657TzHCEIxi9Jlusm0NEcIQmlL/c8sED1YV7zcHSu5bSx1rzIXWPwts8LLZaKFXBQ4CsAVHAbS8D8BwLQpBUCcDcGJ/tCg1iCj7TsHgpACCaCmysAA1iAG0Gh9CcEkPNw7y3OC8AUCAC7B3FtTdIHAWASAiJSjIBQL%2BEofsoAMEYLgRINDJBoLQUsxBHvetuzEcILQiycD2wj5gxAiwAHkYjaEwA4FHvBxRxgIJjgayPXukCwDmYAbgxC0Ee9wXgWAWCGGAOICn%2BB6xmlVEKW7MU8e5i2HtvEM2lvfBiI6DHHgsC3bhFCAnKwqAGGAAoAAangTAN5McMgW3t/gggRBiHYFIGQghFAqHUBT3QyRgfGHWzYcXj3IArFQC6LIDO42Y52JOlnGxbiQuYI4HnvAY3EHhFgJ3EAVh2Dxw0FwDB3CeA6BIZIoQkoLBGFwC7hRMgCCmH4LgyQc8NHmMMKoF2Y9mgEH0SYSe8iF%2B6LH3oEwBjp7LxICvLf88p9sC30vFRM/7tWOsTYEhpucDm6QBbS2VscCOKoAISo43PTFLbo4oOrgaCuFwTuuBCAkB21wJYvAXtaCfaQU753LscGu1P27s%2BHtPf24d8/M2zA3Ypw/5/r3z885h1kEASQIAA%3D
		/// \param out
		/// \param in
		/// \param bytes
		/// \return
		constexpr void memcpyU256(uint8_t *out,
		                          const uint8_t *in,
		                          const size_t bytes) noexcept {
			if (bytes < 64) {
				// count < 64. Move 32-16-8-4-2-1 bytes
				// copy from the end
				int32_t count = -(int32_t)bytes;
				out = out + bytes;
				in = in + bytes;
				if (count <= -32) {
					_uint64x2_t::unaligned_store((void *)(out + count +  0), _uint64x2_t::unaligned_load((void *)(in + count +  0)));
					_uint64x2_t::unaligned_store((void *)(out + count + 16), _uint64x2_t::unaligned_load((void *)(in + count + 16)));
					count += 32;
				}
				if (count <= -16) {
					_uint64x2_t::unaligned_store((void *)(out + count), _uint64x2_t::unaligned_load((void *)(in + count)));
					count += 16;
				}
				if (count <= -8) {
					*(uint64_t *) (out + count) = *(uint64_t *) (in + count);
					count += 8;
				}
				if (count <= -4) {
					*(uint32_t *) (out + count) = *(uint32_t *) (in + count);
					count += 4;
				}
				if (count <= -2) {
					*(uint16_t *) (out + count) = *(uint16_t *) (in + count);
					count += 2;
				}
				if (count <= -1) {
					*(uint8_t *) (out + count) = *(uint8_t *) (in + count);
				}
				return;
			}

			const uintptr_t t = ((uintptr_t )out) & 0b11111;
			size_t bytes2 = bytes;
			if (t) {
				if (t & 1u) {
					*out = *in;
					out += 1; in += 1; bytes2 -= 1;
				}
				if (t & 2u) {
					*(uint16_t *) out = *(uint16_t *) in;
					out += 2; in += 2; bytes2 -= 2;
				}
				if (t & 3u) {
					*(uint32_t *) out = *(uint32_t *) in;
					out += 4; in += 4; bytes2 -= 4;
				}
				if (t & 8u) {
					*(uint64_t *) out = *(uint64_t *) in;
					out += 8; in += 8; bytes2 -= 8;
				}
				if (t & 16u) {
					_uint64x2_t::aligned_store((void *)out,
							_uint64x2_t::unaligned_load((void *)in));
					out += 16; in += 16; bytes2 -= 16;
				}
			}

			// TODO check if partly overlap
			// TODO for very big loops not temporary hints

			// now dest is aligned by 32
			size_t ctr = 0;
			for (size_t i = 0; i < bytes2/32; ++i) {
				uint64x4_t::aligned_store((void *)(out + ctr),
				                          uint64x4_t::unaligned_load((uint64_t *)(in + ctr)));
				ctr += 32;
			}
			out += bytes2;
			in += bytes2;
			int32_t count = -int32_t((bytes - ctr + t) % 32u);

			// tail mng
			if (count <= -16) {
				_uint64x2_t::unaligned_store((void *)(out + count), _uint64x2_t::unaligned_load((void *)(in + count)));
				count += 16;
			}
			if (count <= -8) {
				*(uint64_t *)(out + count) = *(uint64_t *)(in + count);
				count += 8;
			}
			if (count <= -4) {
				*(uint32_t *)(out + count) = *(uint32_t *)(in + count);
				count += 4;
			}
			if (count <= -2) {
				*(uint16_t *)(out + count) = *(uint16_t *)(in + count);
				count += 2;
			}
			if (count <= -1) {
				*(uint8_t *) (out + count)  = *(uint8_t *)(in + count);
			}
		}

#ifdef USE_AVX512BW
		constexpr void memcpyU512BW(uint8_t *out,
									uint8_t *in,
									const size_t bytes) noexcept {

			if (bytes > 0x80) {
				const uint64x8_t t1 = uint64x8_t::unaligned_load((uint64_t *)(in));
				const uint64x8_t t2 = uint64x8_t::unaligned_load((uint64_t *)(in + bytes - 64u));

				uint8_t *out2 = out;
				out += bytes;
				out = (uint8_t *)(((uintptr_t)out) & (-0x40));

				int64_t ctr = out - out2;
				in += ctr;

				ctr =  ctr & -0x40;
				ctr = -ctr;
				while (ctr != 0) {
					uint64x8_t::unaligned_store(out + ctr, uint64x8_t::unaligned_load((uint64_t *)(in + ctr)));
					ctr += 64;
				}

				uint64x8_t::unaligned_store(out2, t1);
				uint64x8_t::unaligned_store(out2 + bytes - 0x40, t2);
				return;
			}

			memcpyU256(out, in, bytes);
		}
#endif

		/// \tparam T
		/// \param out
		/// \param in
		/// \param bytes
		/// \return
		template<typename T>
 		constexpr void memcpy(T *out, 
				const T *in, 
				const size_t bytes) {
			auto *in2 = (uint8_t *)in;
#ifdef USE_AVX512BW
			memcpyU512BW((uint8_t *)out, in2, bytes);
#else
			memcpyU256((uint8_t *)out, in2, bytes);
#endif
		}
 	}

	/// VERY IMPORTANT NOTE: in comparison to the original the last arguments is
	/// is not the number of bytes but the number of elements
	/// \tparam T
	/// \param out
	/// \param in
	/// \param len number of elements
	/// \return nothing
	template<typename T>
	constexpr void memcpy(T *out, const T *in, size_t len) {
		internal::memcpy(out, in, len*sizeof(T));
	}
}
#endif
