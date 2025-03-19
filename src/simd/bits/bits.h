#ifndef CRYPTANALYSISLIB_SIMD_BITS_H
#define CRYPTANALYSISLIB_SIMD_BITS_H

#include <cstdint>

/// Deposit contiguous low bits from unsigned 64-bit integer a to `dst` at the
/// corresponding bit locations specified by `mask`;
/// all other bits in `dst` are set to zero.
template<typename T>
constexpr inline T pdep(const T a,
					    const T mask) noexcept {
#ifdef USE_AVX2 
	if constexpr (sizeof(T) == 4) {
		return _pdep_u32(a, mask);
	} else if constexpr (sizeof(T) == 8) {
		return _pdep_u64(a, mask);
	}
#endif

	T tmp = a, dst = 0;
	uint32_t m = 0, k = 0;

	while (m < 64) {
		if (mask & (1u << k)) {
			dst ^= tmp & (1u << k);
			k += 1;
		}

		m += 1;
	}

	return dst;
}

// using _pdep_u32_ = pdep<uint32_t>;
// using _pdep_u64_ = pdep<uint64_t>;
//

/// TODO
//unsigned int pext32_emu(unsigned int v, unsigned int m) {
//	unsigned int ret = 0, pc = __popcnt(m);
//	switch (pc) {
//		case 0:
//			ret = 0;
//			break;
//		case 1:
//			ret = (v & m) != 0;
//			break;
//		case 2: {
//				unsigned int msb = _bextr_u32(v, (31 - _lzcnt_u32(m)), 1);
//				unsigned int lsb = _bextr_u32(v, _tzcnt_u32(m), 1);
//				ret = (msb << 1) | lsb;
//			}
//		   break;
//		case 3: {
//				const unsigned int lz = 31 - _lzcnt_u32(m);
//				const unsigned int tz = _tzcnt_u32(m);
//				unsigned int msb = _bextr_u32(v, lz, 1);
//				unsigned int lsb = _bextr_u32(v, tz, 1);
//				m = _blsr_u32(m);
//				unsigned int csb = _bextr_u32(v, _tzcnt_u32(m), 1);
//				ret = (msb << 2) | (csb << 1) | lsb;
//			}
//			break;
//		case 4: {
//				const unsigned int lz = 31 - _lzcnt_u32(m);
//				const unsigned int tz = _tzcnt_u32(m);
//				unsigned int msb1 = _bextr_u32(v, lz, 1);
//				unsigned int lsb1 = _bextr_u32(v, tz, 1);
//				m &= ~((1 << lz) | (1 << tz));
//				unsigned int msb0 = _bextr_u32(v, 31 - _lzcnt_u32(m), 1);
//				unsigned int lsb0 = _bextr_u32(v, _tzcnt_u32(m), 1);
//				ret = (msb1 << 3) | (msb0 << 2) | (lsb0 << 1) | lsb1;
//			break;
//		}
//		//case 5:
//		//case 6: {
//		//		unsigned int lsb = 0, msb = 0;
//		//		do {
//		//		const unsigned int lz = 31 - _lzcnt_u32(m);
//		//		const unsigned int tz = _tzcnt_u32(m);
//		//			lsb = _rotr(lsb | _bextr_u32(v, tz, 0x1), 1);
//		//			msb = ((msb << 1) | _bextr_u32(v, lz, 0x1));
//		//			m &= ~((1 << lz) | (1 << tz));
//		//		} while (m);
//		//		ret = _rotl((lsb << (pc & 1)) | msb, (pc >> 1));
//		//	} break;
//		default: {
//			__m128i mm		= _mm_cvtsi32_si128(~m);
//			__m128i mtwo	= _mm_set1_epi64x((~0ULL) - 1);
//			__m128i clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
//	unsigned int	bit0	= _mm_cvtsi128_si32(clmul);
//	unsigned int	a		= v & m;
//					a		= (~bit0 & a) | ((bit0 & a) >> 1);
//					mm		= _mm_and_si128(mm, clmul);
//					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
//	unsigned int	bit1	= _mm_cvtsi128_si32(clmul);
//					a		= (~bit1 & a) | ((bit1 & a) >> 2);
//					mm		= _mm_and_si128(mm, clmul);
//					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
//	unsigned int	bit2	= _mm_cvtsi128_si32(clmul);
//					a		= (~bit2 & a) | ((bit2 & a) >> 4);
//					mm		= _mm_and_si128(mm, clmul);
//					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
//	unsigned int	bit3	= _mm_cvtsi128_si32(clmul);
//					a		= (~bit3 & a) | ((bit3 & a) >> 8);
//					mm		= _mm_and_si128(mm, clmul);
//					clmul	= _mm_sub_epi64(_mm_setzero_si128(), mm);
//	unsigned int	bit4	= _mm_cvtsi128_si32(clmul);
//					bit4	+= bit4;
//					ret		= (unsigned int)((~bit4 & a) | ((bit4 & a) >> 16));
//			break;
//		}
//	}
//	return ret;
//};
//
//unsigned int pdep32_emu(unsigned int v, unsigned int m) {
//	unsigned int ret = 0, pc = __popcnt(m);
//	switch (pc) {
//		case 0:
//			ret = 0;
//			break;
//		case 1:
//			ret = (v & 1) << _tzcnt_u32(m);
//			break;
//		case 2:
//			ret = (((v << (32 - pc)) & 0x80000000) >> _lzcnt_u32(m)) | ((v & 1) << _tzcnt_u32(m));
//			break;
//		case 3:
//		case 4:
//		case 5:
//		case 6:
//		case 7:
//		case 8: 
//		case 9:
//		case 10: 
//		case 11:
//		case 12:
//		case 13: {
//			unsigned int lsb = 0, msb = 0;
//			unsigned int v1 = v << (32 - pc);
//			for (unsigned int i = 0; i < pc / 2  ; i++) {
//				const unsigned int tz = _tzcnt_u32(m);
//				const unsigned int lz = _lzcnt_u32(m);
//				m &= ~((0x80000000 >> lz) | (1 << tz));
//				msb = (v1 & 0x80000000) >> lz;
//				lsb = (v & 1) << tz;
//				ret |= (msb | lsb);
//				v >>= 1;
//				v1 <<= 1;
//			}
//			ret |= ((pc & 1) & v) << _tzcnt_u32(m);
//			break;
//		}
//		default: {
//			__m128i mtwo	= _mm_set1_epi64x((~0ULL) - 1);
//			__m128i mm		= _mm_cvtsi32_si128(~m);
//			__m128i bit0	= _mm_clmulepi64_si128(mm, mtwo, 0);
//					mm		= _mm_and_si128(mm, bit0);
//			__m128i bit1	= _mm_clmulepi64_si128(mm, mtwo, 0);
//					mm		= _mm_and_si128(mm, bit1);
//			__m128i bit2	= _mm_clmulepi64_si128(mm, mtwo, 0);
//					mm		= _mm_and_si128(mm, bit2);
//			__m128i bit3	= _mm_clmulepi64_si128(mm, mtwo, 0);
//					mm		= _mm_and_si128(mm, bit3);
//			__m128i bit4	= _mm_sub_epi64(_mm_setzero_si128(), mm);
//					bit4	= _mm_add_epi64(bit4, bit4);
//			__m128i a		= _mm_cvtsi32_si128(_bzhi_u32(v, pc));
//
//					bit4	= _mm_srli_epi64(bit4, 16);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit4, a),_mm_slli_epi64(_mm_and_si128(bit4, a), 16));
//					bit3	= _mm_srli_epi64(bit3, 8);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit3, a),_mm_slli_epi64(_mm_and_si128(bit3, a), 8));
//					bit2	= _mm_srli_epi64(bit2, 4);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit2, a),_mm_slli_epi64(_mm_and_si128(bit2, a), 4));
//					bit1	= _mm_srli_epi64(bit1, 2);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit1, a),_mm_slli_epi64(_mm_and_si128(bit1, a), 2));
//					bit0	= _mm_srli_epi64(bit0, 1);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit0, a),_mm_slli_epi64(_mm_and_si128(bit0, a), 1));
//			ret = _mm_cvtsi128_si32(a);
//		}
//		break;
//	}
//	return ret;
//};
//
//#if defined (_M_X64)
//unsigned __int64 pext64_emu(unsigned __int64 v, unsigned __int64 m) {
//	unsigned __int64 ret = 0;
//	unsigned int pc = (unsigned int)__popcnt64(m);
//	switch (pc) {
//		case 0:
//			ret = 0;
//			break;
//		case 1:
//			ret = (v & m) != 0;
//			break;
//		case 2: {
//				unsigned __int64 msb = _bextr_u64(v, (unsigned int)(63 - _lzcnt_u64(m)), 1);
//				unsigned __int64 lsb = _bextr_u64(v, (unsigned int)_tzcnt_u64(m), 1);
//				ret = (msb << 1) | lsb;
//			}
//			break;
//		case 3: {
//				unsigned __int64 msb = _bextr_u64(v, (unsigned int)(63 - _lzcnt_u64(m)), 1);
//				unsigned __int64 lsb = _bextr_u64(v, (unsigned int)_tzcnt_u64(m), 1);
//				m = _blsr_u64(m);
//				unsigned __int64 csb = _bextr_u64(v, (unsigned int)_tzcnt_u64(m), 1);
//				ret = (msb << 2) | (csb << 1) | lsb;
//			}
//			break;
//		case 4: {
//				const unsigned int lz1 = (unsigned int)(63 - _lzcnt_u64(m));
//				const unsigned int tz1 = (unsigned int)_tzcnt_u64(m);
//				m &= ~((1ULL << lz1) | (1ULL << tz1));
//				unsigned __int64 msb1 = _bextr_u64(v, lz1, 1);
//				unsigned __int64 lsb1 = _bextr_u64(v, tz1, 1);
//				ret = (msb1 << 3) | lsb1;
//				unsigned __int64 msb0 = _bextr_u64(v, (unsigned int)(63 - _lzcnt_u64(m)), 1);
//				unsigned __int64 lsb0 = _bextr_u64(v, (unsigned int)_tzcnt_u64(m), 1);
//				ret |= (msb0 << 2) | (lsb0 << 1);
//			break;
//		}
//		case 5: {
//				const unsigned int lz1 = (unsigned int)(63 - _lzcnt_u64(m));
//				const unsigned int tz1 = (unsigned int)_tzcnt_u64(m);
//				m &= ~((1ULL << lz1) | (1ULL << tz1));
//				unsigned __int64 msb1 = _bextr_u64(v, lz1, 1);
//				unsigned __int64 lsb1 = _bextr_u64(v, tz1, 1);
//				const unsigned int lz0 = (unsigned int)(63 - _lzcnt_u64(m));
//				const unsigned int tz0 = (unsigned int)_tzcnt_u64(m);
//				m &= ~((1ULL << lz0) | (1ULL << tz0));
//				ret = (msb1 << 4) | lsb1;
//				unsigned __int64 msb0 = _bextr_u64(v, lz0, 1);
//				unsigned __int64 lsb0 = _bextr_u64(v, tz0, 1);
//				ret |= (msb0 << 3) | (lsb0 << 1);
//				unsigned __int64 csb = _bextr_u64(v, (unsigned int)_tzcnt_u64(m), 1);
//				ret |= csb << 2;
//			break;
//		}
//		case 6: {
//				const unsigned int lz2 = (unsigned int)(63 - _lzcnt_u64(m));
//				const unsigned int tz2 = (unsigned int)_tzcnt_u64(m);
//				unsigned __int64 msb2 = _bextr_u64(v, lz2, 1);
//				unsigned __int64 lsb2 = _bextr_u64(v, tz2, 1);
//				m &= ~((1ULL << lz2) | (1ULL << tz2));
//				const unsigned int lz1 = (unsigned int)(63 - _lzcnt_u64(m));
//				const unsigned int tz1 = (unsigned int)_tzcnt_u64(m);
//				ret = (msb2 << 5) | lsb2;
//				unsigned __int64 msb1 = _bextr_u64(v, lz1, 1);
//				unsigned __int64 lsb1 = _bextr_u64(v, tz1, 1);
//				m &= ~((1ULL << lz1) | (1ULL << tz1));
//				ret |= (msb1 << 4) | (lsb1 << 1);
//				unsigned __int64 msb0 = _bextr_u64(v, (unsigned int)(63 - _lzcnt_u64(m)), 1);
//				unsigned __int64 lsb0 = _bextr_u64(v, (unsigned int)_tzcnt_u64(m), 1);
//				ret |= (msb0 << 3) | (lsb0 << 2);
//			break;
//		}
//		case 7: {
//				const unsigned int lz2 = (unsigned int)(63 - _lzcnt_u64(m));
//				const unsigned int tz2 = (unsigned int)_tzcnt_u64(m);
//				unsigned __int64 msb2 = _bextr_u64(v, lz2, 1);
//				unsigned __int64 lsb2 = _bextr_u64(v, tz2, 1);
//				m &= ~((1ULL << lz2) | (1ULL << tz2));
//				const unsigned int lz1 = (unsigned int)(63 - _lzcnt_u64(m));
//				const unsigned int tz1 = (unsigned int)_tzcnt_u64(m);
//				ret = (msb2 << 6) | lsb2;
//				unsigned __int64 msb1 = _bextr_u64(v, lz1, 1);
//				unsigned __int64 lsb1 = _bextr_u64(v, tz1, 1);
//				m &= ~((1ULL << lz1) | (1ULL << tz1));
//				const unsigned int lz0 = (unsigned int)(63 - _lzcnt_u64(m));
//				const unsigned int tz0 = (unsigned int)_tzcnt_u64(m);
//				ret |= (msb1 << 5) | (lsb1 << 1);
//				unsigned __int64 msb0 = _bextr_u64(v, lz0, 1);
//				unsigned __int64 lsb0 = _bextr_u64(v, tz0, 1);
//				m &= ~((1ULL << lz0) | (1ULL << tz0));
//				ret |= (msb0 << 4) | (lsb0 << 2);
//				unsigned __int64 csb = _bextr_u64(v, (unsigned int)_tzcnt_u64(m), 1);
//				ret |= csb << 3;
//			break;
//		}
//		//case 5:
//		//case 6: {
//		//		unsigned __int64 lsb = 0, msb = 0;
//		//		do {
//		//			const unsigned int lz = (unsigned int)(63 - _lzcnt_u64(m));
//		//			const unsigned int tz = (unsigned int)_tzcnt_u64(m);
//		//			lsb = _rotr64(lsb | _bextr_u64(v, tz, 0x1), 1);
//		//			msb = ((msb << 1) | _bextr_u64(v, lz, 0x1));
//		//			m &= ~((1ULL << lz) | (1ULL << tz));
//		//		} while (m);
//		//		ret = _rotl64((lsb << (pc & 1)) | msb, (pc >> 1));
//		//	} break;
//		default: {
//			__m128i mm		= _mm_cvtsi64_si128(~m);
//			__m128i mtwo	= _mm_set1_epi64x((~0ULL) - 1);
//			__m128i clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
//	unsigned __int64 bit0	= _mm_cvtsi128_si64(clmul);
//	unsigned __int64 a		= v & m;
//					a		= (~bit0 & a) | ((bit0 & a) >> 1);
//					mm		= _mm_and_si128(mm, clmul);
//					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
//	unsigned __int64 bit1	= _mm_cvtsi128_si64(clmul);
//					a		= (~bit1 & a) | ((bit1 & a) >> 2);
//					mm		= _mm_and_si128(mm, clmul);
//					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
//	unsigned __int64 bit2	= _mm_cvtsi128_si64(clmul);
//					a		= (~bit2 & a) | ((bit2 & a) >> 4);
//					mm		= _mm_and_si128(mm, clmul);
//					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
//	unsigned __int64 bit3	= _mm_cvtsi128_si64(clmul);
//					a		= (~bit3 & a) | ((bit3 & a) >> 8);
//					mm		= _mm_and_si128(mm, clmul);
//					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
//	unsigned __int64 bit4	= _mm_cvtsi128_si64(clmul);
//					a		= (~bit4 & a) | ((bit4 & a) >> 16);
//					mm		= _mm_and_si128(mm, clmul);
//					clmul	= _mm_sub_epi64(_mm_setzero_si128(), mm);
//	unsigned __int64 bit5	= _mm_cvtsi128_si64(clmul);
//					bit5	+= bit5;
//					ret		= (~bit5 & a) | ((bit5 & a) >> 32);
//		} break;
//	}
//	return ret;
//};
//
//unsigned __int64 pdep64_emu(unsigned __int64 v, unsigned __int64 m) {
//	unsigned int pc =(unsigned int)__popcnt64(m);
//	unsigned __int64 ret = 0;
//	switch (pc) {
//		case 0:
//				ret = 0;
//			break;
//		case 1:
//				ret = (v & 1) << _tzcnt_u64(m);
//			break;
//		case 2:
//				ret = (((v << (64 - pc)) & 0x8000000000000000) >> _lzcnt_u64(m)) | ((v & 1) << _tzcnt_u64(m));
//			break;
//		case 3:
//		case 4:
//		case 5:
//		case 6:
//		case 7:
//		case 8: 
//		case 9:
//		case 10: 
//		case 11:
//		case 12:
//		case 13:
//		case 14:
//		case 15: {
//				unsigned __int64 lsb = 0, msb = 0;
//				unsigned __int64 v1 = v << (64 - pc);
//				for (unsigned int i = 0; i < pc / 2; i++) {
//					const unsigned __int64 tz = _tzcnt_u64(m);
//					const unsigned __int64 lz = _lzcnt_u64(m);
//					m &= ~((0x8000000000000000 >> lz) | (1ULL << tz));
//					msb = (v1 & 0x8000000000000000) >> lz;
//					lsb = (v & 1) << tz;
//					ret |= (msb | lsb);
//					v >>= 1;
//					v1 <<= 1;
//				}
//				ret |= ((pc & 1) & v) << _tzcnt_u64(m);
//			} break;
//		default: {
//			__m128i mtwo	= _mm_set1_epi64x((~0ULL) - 1);
//			__m128i mm		= _mm_cvtsi64_si128(~m);
//			__m128i bit0	= _mm_clmulepi64_si128(mm, mtwo, 0);
//					mm		= _mm_and_si128(mm, bit0);
//			__m128i bit1	= _mm_clmulepi64_si128(mm, mtwo, 0);
//					mm		= _mm_and_si128(mm, bit1);
//			__m128i bit2	= _mm_clmulepi64_si128(mm, mtwo, 0);
//					mm		= _mm_and_si128(mm, bit2);
//			__m128i bit3	= _mm_clmulepi64_si128(mm, mtwo, 0);
//					mm		= _mm_and_si128(mm, bit3);
//			__m128i bit4	= _mm_clmulepi64_si128(mm, mtwo, 0);
//					mm		= _mm_and_si128(mm, bit4);
//			__m128i bit5	= _mm_sub_epi64(_mm_setzero_si128(), mm);
//					bit5	= _mm_add_epi64(bit5, bit5);
//			__m128i a		= _mm_cvtsi64_si128(_bzhi_u64(v, pc));
//
//					bit5	= _mm_srli_epi64(bit5, 32);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit5, a),_mm_slli_epi64(_mm_and_si128(bit5, a), 32));
//					bit4	= _mm_srli_epi64(bit4, 16);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit4, a),_mm_slli_epi64(_mm_and_si128(bit4, a), 16));
//					bit3	= _mm_srli_epi64(bit3, 8);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit3, a),_mm_slli_epi64(_mm_and_si128(bit3, a), 8));
//					bit2	= _mm_srli_epi64(bit2, 4);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit2, a),_mm_slli_epi64(_mm_and_si128(bit2, a), 4));
//					bit1	= _mm_srli_epi64(bit1, 2);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit1, a),_mm_slli_epi64(_mm_and_si128(bit1, a), 2));
//					bit0	= _mm_srli_epi64(bit0, 1);
//					a		= _mm_add_epi64(_mm_andnot_si128(bit0, a),_mm_slli_epi64(_mm_and_si128(bit0, a), 1));
//			ret = _mm_cvtsi128_si64(a);
//		}
//		break;
//	}
//	return ret;
//};
#endif
